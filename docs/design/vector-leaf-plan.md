# Vector-leaf multi-target trees (CUDA) — design plan

**Status:** draft for implementation · **Date:** 2026-08-02 · **Prereq:** the
round-robin tier (`objective=multi_regression`) is the API on-ramp and the
baseline these models are judged against.

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
  same leaf indices and per-target gradient pointers (`gradients_ + t*nd`).
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
  score planes (`scores + t*nd`).

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
- **V3** — hybrid two-sync batched prefix; numerai multi-target benchmark
  (v5 targets), per-era corr validation vs round-robin and vs sequential.
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
