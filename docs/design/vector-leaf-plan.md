# Vector-leaf multi-target trees (CUDA) — design plan

**Status:** V1 (plumbing), V2 (training kernels, classic flow) and V3 (hybrid
level-batched prefix) landed ·
**Date:** 2026-08-02 (plan), 2026-08-28 (V2, V3) · **Prereq:** the round-robin
tier (`objective=multi_regression`) is the API on-ramp and the baseline these
models are judged against.

## 0. Implementation notes — where the implementation diverges from the plan below

- **Hybrid TWO-SYNC level prefix + classic fallback.** §2's "hybrid growth
  prefix support from day one" landed one release late (V3): vector training
  takes the hybrid level-batched prefix in the depth-limited regime
  (`2^max_depth <= num_leaves + 1`), the same regime plain level batching is
  leaf-wise-exact in for scalar training, and the classic one-split-at-a-time
  loop everywhere else. The level path adds three pieces to the scalar
  machinery: a batched vector find kernel that shares its whole body with the
  per-pair vector finder, a level plane fan-out kernel that refreshes every
  child's T leaf-splits structs from the primary structs the batched apply
  wrote, and the per-pair histogram loop the next bullet explains. The one-sync
  (speculative), selective (grow-then-prune) and graph-loop
  prefixes stay excluded: one-sync gates the children on device structs the
  fan-out has not refreshed yet, selective rebuilds the host tree from captured
  splits carrying target-0 leaf values only, and the graph controller models one
  construct node per level rather than one per plane.
- **The level flow batches the SEARCH, not the histograms.** The batched level
  construct is what makes plain level batching pay for scalar training, and it
  is the one piece vector mode does NOT take: with T gradient planes it measured
  110.8 ms/tree against the per-pair loop's 68.1 (200k x 200 features, T=5, 63
  leaves, depth 6) — the entire cost of the level flow on that shape, while the
  batched find was already cheaper (3.1 vs 4.7) and the batched apply plus the
  plane fan-out cost 0.6 together. So each pair's T planes construct through the
  per-pair path, back to back, and the level contributes ONE find + ONE sync +
  ONE apply and no per-split device syncs. That per-pair plane loop selects
  planes 1..T-1 gradient-only exactly as the classic loop does, so the two
  optimizations compose (measured 1.19-1.78x together, docs/performance.md §11).
- **Per-plane leaf-splits structs instead of widened structs.** §5's
  `CUDALeafSplitsStruct` stays untouched. The learner keeps a slab of 2·T
  structs (smaller/larger × plane): plane t carries target t's gradient sums
  and its histogram-plane pointer, so the existing construct/fix/subtract
  kernels run per plane completely unchanged. A tiny fan-out kernel refreshes
  the slab from the primary structs after every applied split.
- **`CUDASplitInfo` payload is a slab side-band, not inline arrays.** Four
  fields per target (left/right gradient sums, left/right leaf outputs) in a
  finder-owned slab, with the categorical-threshold pointer discipline. The
  finder recomputes parent gains from the plane sums (the scalar kernel does
  the same), so per-target child gains are not stored.
- **The finder is a separate kernel, not a `NUM_TARGETS` template.**
  `FindBestSplitsForLeafKernelVector` handles runtime T ≤ 16 with the scalar
  kernel's exact fp64 CPU-order folds; the scalar instantiations are untouched.
- **v1 fences (beyond §2):** no GOSS/query/balanced
  bagging (plain per-row bagging IS supported: the CUDA bagging path is
  index-based, so one shared row subset feeds every target's histogram plane
  and the per-plane leaf-splits init just takes the partition's index list),
  no L1/path-smooth/max_delta_step/extra-trees/monotone/interaction/
  forced-splits/CEGB, fp64 only (`cuda_precision=fp64`; quantized training IS
  supported, see §3a), `boost_from_average` forced off (per-target biases would
  need a per-target AddBias), max_bin ≤ 256 (shared-memory finder), single GPU,
  no `cat_random_search`.
- **Categorical splits (supported).**
  `FindBestSplitsForLeafKernelVectorCategoricalInner` mirrors the scalar
  categorical inner (one-hot + `max_cat_threshold`-sorted many-vs-many) over
  the T gradient streams with the shared plane-0 hessians; per-target child
  sums/outputs ride the slab payload exactly like numerical splits. The one
  designed divergence: the many-vs-many sort key is the summed-over-targets
  gradient over the smoothed hessian, `(Σ_t g_t) / (h + cat_smooth)` — the
  1-D output-ordering heuristic applied to the summed objective (there is no
  total category order that is optimal for every target at once). For
  duplicated targets the key is a positive multiple of the scalar key, so the
  sorted order — and the greedy structure — matches scalar training exactly.
- **Gradient-only planes.** The hessian is target-independent and every
  consumer reads it from plane 0 (the finder's `local_hess`, the categorical
  inner's `plane_hist[0]`, the `sum_hessians` gate). Planes 1..T-1 therefore
  construct GRADIENTS ONLY: `SelectGradientPlane(..., grad_only=true)`
  instantiates the deterministic dense construct with `GRAD_ONLY`, which drops
  the hessian read-modify-write from the inner slot loop. Their hessian cells
  stay zero (the slots are zeroed and merged as before) and carry no
  information — reading them is a bug. Accumulate work per (row, column) falls
  from 2T to T+1. The win depends on whether the slot rows are cache-resident:
  measured per-construct-launch cost of the grad-only kernel against the full
  one at 700k rows, 0.525× on 2400 five-valued features (`num_total_bin` ~
  12k, 192 KB slot rows) but 0.99× on 200 255-bin continuous features
  (`num_total_bin` 51k, 816 KB slot rows), where the gradient and hessian
  cells are adjacent doubles in the same sector so the second update is free.
  End to end per vector tree: 1.53× on real numerai 700k × 3555 (T=4, 250
  leaves, depth 12), 1.43× on the numerai-shaped synthetic, 1.03× on the wide
  continuous shape.
- **Verified by invariants, not md5** (non-quant CUDA is atomic-order
  nondeterministic): duplicated targets reproduce the scalar model's structure
  and outputs, negated targets predict antisymmetrically, per-target loss
  drops, text/FALB round-trips are exact. The hybrid level prefix is verified
  against the classic loop on the same config (T=2 and T=4): predictions match
  and the two leaf labelings of the same row partition form a bijection —
  level-batched growth numbers right children in level order where the
  per-split loop numbers them in best-gain order, exactly as on the scalar
  path. The scalar path is locked by the canonical md5 gates and the lattice.

## 1. What it is and why it wins

One tree STRUCTURE per iteration shared by all T targets; each leaf holds a
vector of T outputs. Split gain is the sum of per-target gains. Nobody ships
this CUDA-supported on asymmetric (leaf-wise) trees: XGBoost's
`multi_output_tree` is experimental and hist/CPU-oriented, CatBoost MultiRMSE
is symmetric-only. Two prizes:

- **Speed:** construct/partition/apply run once per iteration instead of once
  per target. For RMSE-family objectives the hessian is target-independent, so
  the incremental cost per extra target is one gradient-histogram plane and a
  wider finder read — estimated per-tree cost 1.3–1.6× single-target at T=5,
  i.e. ~3.6× per-target throughput vs round-robin.
- **Modeling:** shared structure is a regularizer (the MultiRMSE argument).
  Numerai-style era-stable multi-horizon targets are the motivating workload.
  Validated per-era, never assumed.

## 2. Scope for v1

- `objective=multi_regression` + `tree_mode=vector_leaf` (new param; default
  stays round-robin) — non-quantized CUDA only, L2 loss only, no bagging-by-
  query, no monotone constraints, no interaction constraints, no linear trees.
- Constant or per-row weights (hessian shared across targets either way).
- Hybrid growth prefix support from day one (the speed thesis lives there);
  classic fallback for excluded shapes as usual.
- Quantized vector-leaf was out of scope for v1 on the assumption that it
  needed a WIDER packed histogram word (one carrying T gradient fields), which
  would touch every codec. It does not, and it is now supported: see §3a.

## 3. Histogram layout — the load-bearing decision

Keep the (grad, hess)-pair layout untouched and give each target its own
histogram PLANE per leaf: plane t covers `num_total_bin` (g_t, h) pairs at
offset `t * num_total_bin * 2` in the leaf's hist slot. The hessian is
redundantly accumulated per plane.

Why not a packed (g_0..g_{T-1}, h) row per bin: every construct/subtract/fix/
finder kernel and both JIT template families assume the pair stride; a
per-model stride multiplies the template space and breaks the codec/compact
paths. Plane-per-target reuses ALL existing kernels unchanged for
construction:

- **Construct:** launch the existing dense construct once per target with the
  same leaf indices and per-target gradient pointers (`gradients_ + t * num_data`).
  The row→bin reads are cached hot across the T launches (same data, same
  partition); measured cost is the atomics, not the reads.
- **Fix/subtract:** existing kernels run per plane (T cheap launches, or the
  batched variants with a plane dimension in grid.z later).
- **Finder:** the only NEW kernel work. The per-task finder loads T grad
  prefix streams + 1 hess stream and computes
  `gain(bin) = Σ_t g_t(bin)²/(h(bin)+λ)` in registers (T ≤ ~16 for v1;
  refuse above). Output CUDASplitInfo carries per-target left/right sums —
  extend with a small fixed-size vector payload (slab-style, like the
  categorical thresholds: `leaf_value_vec` slab indexed per slot).

Memory: hist pool grows ×T for the planes. At numerai scale (10240 bins ×
2 × 8B × 1024 leaves ≈ 168MB single-target) T=5 → ~840MB; acceptable on
24–32GB cards, and the fp32-hist mode halves it.

## 3a. Quantized vector-leaf (supported)

The v1 scope note assumed quantized vector training meant widening the packed
histogram word to carry T gradient fields. It does not, and the plane layout of
§3 is exactly why: a quantized plane is a NORMAL `(grad32, hess32)` packed
histogram — the same word, the same codecs, the same fix/subtract/bit-width
machinery — just one per target.

Three decisions carry it:

- **Per-target gradient scales.** Each plane is discretized with its own
  `grad_scale`. Targets differ in gradient magnitude by orders of magnitude
  (that is the whole point of multi-target training), and a single shared scale
  would round the small-magnitude targets to zero. `grad_scale_ptr(t)` names
  target t's scale; the finder dequantizes plane t with it. The scales are
  SNAPSHOTS, not the discretizer's reduce scratch: a T-plane tree overwrites
  that scratch once per plane, so naming it would hand every consumer the last
  plane's scale.
- **One hessian, quantized once.** The hessian is target-independent, so plane
  0 owns it and planes 1..T-1 copy plane 0's already-quantized int16 hessians
  rather than re-rounding them under an independent dither. Every plane's
  packed word is then `(g_t, h)` with a bit-identical `h`, so the leaf's
  integer hessian total, the bin bit widths, and the row counts inferred from
  the hessian are the same object on every plane — the finder reads them from
  plane 0 and the invariant holds by construction, not by luck. There is one
  `hess_scale`.
- **Integer scans, no fp64 slab.** The fp64 vector finder needs a
  `(T+1) x blockDim` shared prefix buffer because its folds must run in a fixed
  CPU order to stay reproducible. Integer addition is exact and associative, so
  the quantized finder just runs the existing warp-shuffle prefix once per
  plane over one shared scratch. It is therefore the cheaper kernel in shared
  memory, and quantized vector models are bit-identical across execution
  strategies (batched level prefix vs per-split loop) where fp64 ones are not.

The child's exact integer parent total travels in the split payload as two
extra per-target fields (`kVecLeftGradInt` / `kVecRightGradInt`, int32 values
held as doubles, which is exact); the plane fan-out re-packs them against the
hessian half of the child's primary leaf-splits struct.

Fenced within quantized vector mode: categorical features (the quantized
vector search covers numerical tasks only) and `quant_train_renew_leaf` (it
reduces one exact gradient stream and overwrites a scalar leaf value).

Under bagging the discretized find kernels add one hessian quantum of L2
ridge, for the winner's-curse reason documented in the scalar quantized
finder — vector mode sums T such gains, so the same noise enters T times over.

**Not done, and why.** `grad_only` is passed for planes 1..T-1 in the quantized
construct loop as it is in the fp64 one, but it is inert there: a quantized row
is ONE packed int32 carrying gradient and hessian together, so skipping the
hessian saves no traffic. The apparent way to recover a saving — packing two
TARGETS' gradients into one word, halving the number of construct passes — does
not work with the existing field-wise accumulation. That trick is sound only
because the low field (the hessian) is non-negative and therefore never borrows
into the high one; two signed gradients in one word break it, and biasing the
low field to fix that requires a per-bin row count the histogram does not
carry.

## 4. Tree storage, model format, predict

- `Tree` gains `leaf_value_dim_` + a flat `leaf_values_vec_` array
  ([leaf * T + t]); `leaf_value_` keeps target 0 so untouched code paths stay
  well-defined but every vector-aware consumer reads the vec array.
- **Model text:** `leaf_value=` line carries `num_leaves * T` doubles with a
  `leaf_value_dim=T` header line per tree. This is a FORK EXTENSION: stock
  LightGBM cannot parse such models (documented; the stock-interop CI guard
  applies only to non-vector models). FALB's reserved `leaf_dim` covers the
  binary format from day one.
- **Predict:** one traversal per row, then add the leaf's T values to the T
  output planes. New `PredictMultiRaw` path in the predictor; Python reshape
  by `num_class` already yields [n, T]. `pred_contrib`/SHAP: refuse in v1
  with a clear error. FIL/treelite: fall back to CPU predict for vector
  models (treelite has no vector-leaf support).
- **Score updater:** one AddScore kernel that traverses once and updates T
  score planes (`scores + t * num_data`).

## 5. Learner changes (CUDA)

- `CUDALeafSplitsStruct` gains per-target sum_of_gradients slab pointer
  (shared hessian scalar stays). Leaf-splits init kernels fill T sums.
- Split apply: partition kernels are UNCHANGED (structure decisions don't
  depend on targets). Child leaf-splits seeding writes T sums from the
  finder's vector payload.
- Leaf output: `CalculateSplittedLeafOutput` per target from the vector sums;
  leaf-value upload writes T values.
- The selective/one-sync/graph prefixes treat the finder output generically;
  v1 fences: start with the two-sync batched prefix + classic fallback
  (mirroring how categoricals landed), lift the rest afterwards.

## 6. Milestones

- **V1** — plumbing: tree/model/predict vector storage behind
  `tree_mode=vector_leaf` with T=1 parity vs plain regression (bit-identical
  fingerprint gate).
- **V2** — finder + construct planes on the classic flow; parity harness vs
  round-robin on synthetic data (same seeds: vector-leaf trees are NOT
  expected identical — quality parity per target + speed measurement).
- **V3** — hybrid two-sync batched prefix (LANDED; see §0); numerai
  multi-target benchmark (v5 targets), per-era corr validation vs round-robin
  and vs sequential.
- **V4** — docs + gates (vector/nonquant lattice cell, perf entry).

## 7. Open questions (resolve during V2)

- Gain normalization: sum vs mean of per-target gains interacts with
  `min_gain_to_split` scaling; CPU MultiRMSE precedent says sum, with
  min_gain semantics documented.
- Whether `min_data_in_leaf`/`min_sum_hessian` gates stay per-structure
  (shared hessian ⇒ identical to single-target — the natural choice).
- Per-target learning-rate / target weights (numerai wants target weighting);
  cheap to add at the leaf-output step (`target_weights` param, applied as
  gain weights + output scaling).

## 8. What the classic-loop profile says V3 must do

Nsight Systems GPU-kernel breakdown of the classic vector loop against the
classic scalar loop on the same data (RTX 5090, non-quant fp64, per-kernel
totals over the whole training region). Two shapes, two row scales:

**Shape A** — synthetic 200 continuous features, T=5, `num_leaves=63`, no depth
cap, 10 rounds (620 splits). Vector / scalar-classic GPU ms:

| phase | 150k | ratio | 700k | ratio |
| --- | --- | --- | --- | --- |
| construct | 439.4 / 83.9 | 5.24 | 1824.3 / 335.6 | 5.44 |
| det merge | 82.4 / 15.8 | 5.22 | 115.4 / 22.8 | 5.07 |
| fix histogram | 50.1 / 10.0 | 4.99 | 50.6 / 10.2 | 4.99 |
| subtract | 6.3 / 1.3 | 4.91 | 7.1 / 1.5 | 4.92 |
| find split | 122.2 / 44.9 | 2.72 | 128.6 / 45.5 | 2.83 |
| apply + partition + tree | 26 / 26 | 1.00 | 29 / 29 | 1.00 |
| **total** | **726.3 / 177.1** | **4.10** | **2155.5 / 439.3** | **4.91** |

**Shape C** — numerai-shaped synthetic 2400 five-valued features, T=4,
`num_leaves=250`, `max_depth=12`, 8 rounds (1992 splits):

| phase | 150k | ratio | 700k | ratio |
| --- | --- | --- | --- | --- |
| construct | 1125.7 / 278.9 | 4.04 | 4461.3 / 1155.8 | 3.86 |
| det merge | 49.2 / 12.2 | 4.04 | 125.0 / 31.0 | 4.03 |
| subtract | 12.9 / 2.8 | 4.57 | 13.4 / 3.3 | 4.06 |
| find split | 480.3 / 268.4 | 1.79 | 482.3 / 271.6 | 1.78 |
| apply + partition + tree | 25 / 25 | 1.00 | 27 / 27 | 1.00 |
| **total** | **1770.7 / 645.9** | **2.74** | **5208.3 / 1578.9** | **3.30** |

Three facts fall out, and they set V3's scope.

1. **Construct is the whole story, and it is exactly T×.** Its share of vector
   GPU time is 60.5% → 84.6% (shape A) and 63.6% → 85.7% (shape C) from 150k to
   700k rows. Per-launch cost is within ±8% of the scalar construct's on the
   same leaf (shape C 700k: 560 µs vector vs 580 µs scalar), so a vector
   construct is a scalar construct — there are just T of them. Construct is the
   ONLY row-proportional phase: the finder costs 121 µs per launch at both 150k
   and 700k, and apply/partition are literally shared. That is the entire
   row-scaling law: `ratio = (T·C(N) + F) / (C(N) + F)` with `F` row-independent,
   so the vector/scalar-classic ratio climbs toward T as N grows.
2. **Launch and sync overhead is NOT the mechanism.** The vector loop issues
   the SAME NUMBER of host syncs as the classic scalar loop (3769 shape A,
   12003 shape C `cudaDeviceSynchronize` calls in both), and its extra
   `cudaLaunchKernel` host time is 41.6 ms of 2193 ms (1.9%) at shape A 700k
   and 104.8 ms of 5316 ms (2.0%) at shape C 700k — never the critical path
   (the GPU is busy the whole time). **A batched/one-sync/graph vector prefix
   that still issues T sequential per-plane constructs cannot move the T×
   term.** Batch the launches only for the reasons the scalar path batches
   them, not expecting the multi-target cost back.
3. **Memory bandwidth from the T gradient planes is second order.** At shape A
   700k the construct moves 140 MB of bin matrix in 541 µs = 259 GB/s, ~14% of
   the card's peak, while its slot read-modify-write traffic is 4.5 GB — 32×
   larger. The construct is accumulate-bound, not read-bound, so a fused
   plane-aware kernel that reads the bin row once and scatters into T planes
   buys only the bin re-reads (a few percent), NOT a factor of T. The
   accumulate-bound diagnosis is what made gradient-only planes (§0) worth
   1.35–1.53× on the numerai-shaped runs; the residual accumulate work is
   T+1 slot updates per (row, column) and there is no known way below it while
   each target keeps its own histogram plane. (Both tables above predate
   gradient-only planes, so their construct rows read T×; post-fix the shape C
   700k construct is 2846 ms — 3.86× → 2.46× — and total vector GPU time is
   3594 ms against the same 1579 ms scalar baseline.)

So V3's job is the scalar hybrid prefix's own win, no more and no less, and
that win is strongly `feature_fraction`-dependent. Measured scalar
classic/hybrid ratio, real numerai 700k × 3555, T=4, 250 leaves, depth 12:
0.90 at `feature_fraction=1.0` (hybrid LOSES on this shape), 2.00 at 0.3, 2.08
at 0.1 — the compact-view repack pays off through the batched level kernels far
better than through the per-split loop. That is why the production numerai A/B
(run at production `feature_fraction`) reported vector as 3× slower than T
independent scalar trainings while the same comparison at `feature_fraction=1.0`
makes vector a 1.3–1.9× WIN: the missing prefix is a ~2× multiplier that only
appears once features are subsampled.

The prefix landed on that basis (§0, `docs/performance.md` §11): it batches the
SEARCH — one find, one sync, one apply per level — and leaves each pair's T
planes to construct back to back through the per-pair path. Judge it against
`cuda_plan=auto,hybrid:off` on the SAME shape at the SAME `feature_fraction`,
not against a `feature_fraction=1.0` baseline.

### 8a. The fused all-T-planes construct is closed: it LOSES, 1.35–3.2×

The one lever the two independent V3 workstreams disagreed about was a construct
that reads each bin row once and accumulates all T gradient planes from it.
Settled by measurement, not by roofline argument.

**A priori**, once planes 1..T-1 are gradient-only (§0) the T gradient
accumulates are irreducible — there genuinely are T gradients to add per
(row, column). All fusing can remove is the (T-1) redundant passes over the bin
matrix and the (T-1) launches.

**The measurement.** A standalone probe mirrors
`ConstructHistogramDenseGMDeterministicInner` exactly (private per-thread slot
rows, global slot positions, cooperative zero, `__ldcs` bin reads, nibble
unpack) and runs at the geometry the learner actually launches, dumped from
`LaunchConstructHistogramDenseDeterministic`:

- wide continuous — 200 columns × 256 bins, 8-bit, 9 partitions,
  `grid=(9,122) block=(24,1)`, slot_stride 102400 doubles (800 KB per row);
- numerai-shaped — 2400 columns × 5 bins, 4-bit packed, 5 partitions,
  `grid=(5,128) block=(504,2)`, slot_stride 24000 doubles (188 KB per row).

Two fused arms: same `grid_y` as the sequential arm (same parallelism, T× the
slot slab) and `grid_y / T` (same 96 KB × 1024 slot budget, T× fewer tiles).
`seq / fused` — above 1.0 fusing would win:

| shape | rows | T | fused, same parallelism | fused, same slot budget |
| --- | --- | --- | --- | --- |
| wide200 | 200k | 4 | 0.331 | 0.649 |
| wide200 | 700k | 4 | 0.325 | 0.636 |
| wide200 | 200k | 5 | 0.319 | 0.568 |
| wide200 | 700k | 5 | 0.315 | 0.558 |
| numlike | 200k | 4 | 0.735 | 0.732 |
| numlike | 700k | 4 | 0.738 | 0.732 |
| numlike | 200k | 5 | 0.730 | 0.737 |
| numlike | 700k | 5 | 0.733 | 0.740 |

Fusing is slower everywhere, by 1.35× on the numerai-shaped geometry and 1.8–3.2×
on the wide one. Three readings settle it:

1. **The bin re-reads are already free.** T sequential grad-only constructs cost
   exactly T single-plane constructs: measured `seq / one-plane` is 4.00–4.21 at
   T=4 and 4.98–5.25 at T=5. The marginal cost of adding a plane IS a whole
   construct, so the read component of a construct is ≈ 0 and there is nothing
   for fusing to recover.
2. **Bin-matrix residency does not move the answer.** Sweeping the wide200
   geometry from an L2-resident bin matrix to several times L2, at T=5, same
   parallelism:

   | bin matrix | 9.5 MB | 38 MB | 133 MB | 381 MB |
   | --- | --- | --- | --- | --- |
   | seq / fused | 0.333 | 0.320 | 0.315 | 0.326 |

   A read-bound construct would show fusing gaining sharply as the matrix leaves
   L2, and gaining nothing when it fits. The ratio is flat to ±3%. Reads are not
   the constraint at any residency.
3. **Fusing makes the binding resource worse.** The dominant traffic is the slot
   read-modify-write, and a fused thread holds T private slot rows instead of
   one — 5 × 800 KB against 800 KB on wide200. Holding the slot budget fixed by
   cutting `grid_y` by T recovers about half the loss on wide200 (0.32 → 0.56)
   at the price of T× less parallelism, and recovers nothing on the numerai
   shape, where the slab was never the binding term.

**Verdict.** The construct is accumulate-bound, exactly as §8.3's Nsight reading
said, and the conclusion there was if anything too generous: a fused construct
does not buy "a few percent", it costs 1.35–3.2×. The competing claim that "a
construct accumulating all T planes in one pass is the remaining big lever" is
refuted — it confused the level flow's batched-construct regression (which is
about batching PAIRS, §0) with plane fusion. Not shipped. Re-open only if the
histogram layout changes so that a leaf's T planes share one slot row (e.g. an
interleaved `[bin][target]` cell layout that keeps one thread's working set at
one row), because that, not the launch count, is the term the measurement
indicts.

The remaining vector-leaf levers are therefore the ones that do not touch the
construct: selective and one-sync prefix support for vector mode, and the
modeling-side question of the leaf budget (§0).
