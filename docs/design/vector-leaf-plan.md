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
  ONE apply and no per-split device syncs.
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
  forced-splits/CEGB, fp64 only (`cuda_precision=fp64`), `boost_from_average`
  forced off (per-target biases would need a per-target AddBias), max_bin ≤
  256 (shared-memory finder), single GPU, no `cat_random_search`.
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
- Quantized vector-leaf is explicitly out of scope for v1 (the packed int
  hist carries one grad field; widening it interacts with every codec).

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
