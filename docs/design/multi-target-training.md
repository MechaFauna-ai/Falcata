# Design Spec: Multi-Target Training in the Falcata CUDA Learner

Status: **design only** (no code changes yet). Author: design agent.
Scope: single-GPU CUDA learner (`CUDASingleGPUTreeLearner`). Two variants are
specified; **variant 2 (vector-leaf) is the differentiator** and gets the depth.

---

## 1. Motivation

Numerai (and multi-task tabular problems generally) ship **many correlated
regression targets** — the classic Numerai target plus auxiliary targets, ~5–20
per era. Today Falcata trains them the way stock LightGBM does: one full GBDT
per target, sequentially. With N targets that is N independent training runs.

Two things changed that make multi-target worth building *now*:

1. **Tree construction on this fork is cheap.** The hybrid level-batched growth
   + CUDA-graph capture path (`HybridGrowthUsable`, `TrainLevelWisePrefixGraph`)
   collapses the per-split host round-trips that used to dominate. The remaining
   per-tree cost is dominated by histogram construction bandwidth, which is
   exactly what multi-target can *share*.
2. **The RMSE-family hessian is data-independent of the target.** For squared
   error `hess ≡ 1` (or `≡ weight`) for every target. So a shared tree over T
   targets needs **T gradient sums but only ONE hessian/count per histogram
   cell** — histogram bandwidth grows sub-linearly in T. This is the core
   efficiency result of this spec (§5).

There are two distinct products here, and they are *not* the same model:

- **Variant 1 — round-robin one-tree-per-target.** Reuses the existing
  `num_class` per-tree machinery (minus softmax). Produces a model
  **bit-identical** to sequential per-target training. Speedup comes only from
  amortizing fixed per-iteration overhead — modest (~1.05–1.1×/target).
- **Variant 2 — vector-leaf trees.** ONE shared tree structure; each leaf holds
  a vector of T values; split gain is the sum of per-target gains. A genuinely
  *different, regularized* model (shared structure across targets). This is the
  differentiator and where the shared-hessian bandwidth win lives.

---

## 2. Competitive landscape

### XGBoost `multi_strategy=multi_output_tree`
- Added in 2.0 as **experimental**; the docs still say "working-in-progress,
  most features are missing"
  (https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html).
- Leaf size = number of targets; split gain summed over targets. Tree structure
  is stored in a dedicated `MultiTargetTree` (vector leaves; the right-child slot
  is reused to index leaf-weight vectors —
  https://xgboost.readthedocs.io/en/stable/dev/classxgboost_1_1MultiTargetTree.html).
- **It was historically hist-CPU-mostly.** GPU-hist support for multi-target is
  still landing: PR **dmlc/xgboost#11798 "[mt] Split up gradient types for the
  GPU hist"** (https://github.com/dmlc/xgboost/pull/11798) introduces a
  `GradientContainer` with *two* gradient types and is described as
  "experimental." So a fast, asymmetric, GPU vector-leaf learner is **not** a
  solved, shipped thing in XGBoost.
- **SketchBoost / reduced gradient.** XGBoost adopts the SketchBoost idea
  (Iosipoi & Vakhrushev, NeurIPS 2022): use a *reduced-dimension* "split
  gradient" (e.g. mean or truncated-SVD over targets) to find structure, and the
  *full* "value gradient" to set leaf outputs
  (https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html,
  `split_grad` hook). This is a knob we should design *space* for but not build
  in phase 1.

### CatBoost `MultiRMSE`
- Native multi-target regression loss
  (https://catboost.ai/docs/en/concepts/loss-functions-multiregression).
- **But CatBoost trees are symmetric / oblivious** — the same (feature,
  threshold) is applied at every node of a level
  (https://catboost.ai/news/catboost-enables-fast-gradient-boosting-on-decision-trees-using-gpus).
  Symmetric trees make a vector leaf trivial (leaf = a level-path bitmask, values
  are just a table) but sacrifice the accuracy of asymmetric leaf-wise growth.

### The gap Falcata can fill
Nobody ships a **CUDA-supported vector-leaf learner on asymmetric leaf-wise
trees**: XGBoost's is experimental and its GPU path is only now landing;
CatBoost's is symmetric-only. Falcata already has the fast asymmetric
leaf-wise/hybrid CUDA growth engine. Adding a shared-hessian vector-leaf
histogram is a small, well-contained delta on top of it — that is the
differentiator.

---

## 3. Where per-target gradients already come from (the reuse anchor)

The multiclass path already lays gradients out exactly the way multi-target
needs, and GBDT already loops per output tree:

- **Gradient layout is `[class k][row i]`.** `MulticlassSoftmax::GetGradients`
  (`src/objective/multiclass_objective.hpp`) writes
  `idx = num_data_ * k + i`. Same layout the CUDA multiclass objective uses
  (`src/objective/cuda/cuda_multiclass_objective.cu`).
- **GBDT drives `num_tree_per_iteration_ == num_class_`.** In
  `GBDT::TrainOneIter` (`src/boosting/gbdt.cpp:399–427`) the outer loop is
  `for (cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id)`
  with `offset = cur_tree_id * num_data_` slicing the gradient array, then
  `TrainTree(... gradients_ + offset, hessians_ + offset ...)` and
  `UpdateScore(tree, cur_tree_id)`. `num_tree_per_iteration_` is set from
  `objective_function_->NumModelPerIteration()` (`gbdt.cpp:103–105`).
- **Score updater is already multi-output.** `ScoreUpdater`/`CUDAScoreUpdater`
  are constructed with `num_tree_per_iteration_` (`gbdt.cpp:133–136`) and
  `AddScore(tree, cur_tree_id)` writes into the `cur_tree_id` slice.

That is the entire skeleton variant 1 reuses. Variant 2 keeps the *objective*
producing T gradient slices but replaces the per-tree loop with **one** tree
that consumes all T slices at once.

---

## 4. Variant 1 — round-robin one-tree-per-target (small PR)

**Model:** identical to training N separate single-target regressors. No modeling
change; verifiable by md5-identity against sequential runs.

**Config / API surface:**
- New objective alias, e.g. `objective=regression` + `num_target=T`, or a
  dedicated `multi_output_regression` objective wrapper. Add `num_target` to
  `include/Falcata/config.h` near `num_class` (`config.h:925–927`) — a separate
  field so it does not collide with multiclass semantics.
- A thin `RegressionMultiTarget` objective (host + CUDA) whose
  `NumModelPerIteration()` returns `T`, and whose `GetGradients` fills the
  `[t][i]` layout by running the single-target L2 gradient per target slice. On
  CUDA this is T launches of the existing
  `CUDARegressionL2loss::LaunchGetGradientsKernel`
  (`src/objective/cuda/cuda_regression_objective.hpp:51`) into offset slices, or
  one fused kernel over `T * num_data` — either is trivial.
- **Label ingestion.** Falcata metadata carries a single `label_`. Multi-target
  labels need an `init_score`-style multi-column path or a documented
  convention (T contiguous label columns). Smallest change: accept a
  `label` matrix via the C API `num_target` and store T label columns in
  `Metadata` (mirror how `init_score` already stores `num_class * num_data`).

**Reuse of the num_class per-tree path:** none of `gbdt.cpp:399–427`,
`ScoreUpdater`, or the tree learner changes. `num_tree_per_iteration_ = T`, the
loop trains T trees per iteration, softmax is simply never invoked (regression
objective). Trees serialize exactly as multiclass trees do today (T trees per
iteration in the model file).

**Effort:** ~1 objective class (+CUDA), 1 config field, 1 metadata label-matrix
path, docs. No tree-learner or histogram changes. Good first PR; de-risks the
label/config plumbing that variant 2 also needs.

**Expected speedup:** ~1.05–1.1×/target — only fixed per-iteration overhead is
amortized (data partition reset, graph reuse). Each tree still builds its own
full histograms. This variant is about *convenience + a stepping stone*, not
throughput.

---

## 5. Variant 2 — vector-leaf trees (the project)

### 5.1 Model definition
One shared tree per boosting iteration. Internal nodes split on a single
(feature, threshold) as usual. Each **leaf stores a vector `v ∈ ℝ^T`**. Split
gain of a candidate = **sum over targets** of the usual single-target gain:

```
gain(split) = Σ_{t=1..T} [ G_L,t² / (H_L + λ)  +  G_R,t² / (H_R + λ) ]
```

where `G_·,t` is the per-target gradient sum in the child and `H_·` is the
child's hessian sum — **shared across targets** for the RMSE family. Leaf value
for target t: `v_t = -G_t / (H + λ)` (the existing
`SplitGainMath::CalculateLeafOutput`, evaluated per target with the shared H).

Because structure is shared, this is a *regularized* multi-task model, not the
sequential model — a genuine modeling change (see risks §8).

### 5.2 The shared-hessian histogram — the core efficiency insight

**Current cell layout** (`src/treelearner/cuda/cuda_histogram_constructor.cu`,
construct kernel ~lines 355–415): each bin is a **(grad, hess) pair**. In the
kernel `pos = bin << 1`, `atomicAdd_block(pos_ptr, grad)` /
`atomicAdd_block(pos_ptr+1, hess)`; global histogram is
`hist_in_leaf + (partition_hist_start << 1)`. Types: `hist_t = double`
(`include/Falcata/bin.h:34`), `kHistEntrySize = 2*sizeof(hist_t)`,
`kHistOffset = 2` (`bin.h:40,43`). The quantized path packs (grad16, hess16)
into a 32-bit word.

**Vector-leaf cell layout (proposed):** `T` gradient accumulators + **1** shared
hessian/count accumulator per bin:

```
cell = [ g_0, g_1, ..., g_{T-1}, h ]   →  (T + 1) accumulators, not 2T
```

The naïve "T independent single-target histograms" costs `2T` accumulators/cell
and reads the row's bin `T` times. The shared-hessian cell costs `T+1` and reads
each row's bin **once**, accumulating T gradients + 1 hessian in the same pass.

### 5.3 Bandwidth / cost model

Let a single-target histogram cell = 2 units (grad+hess). Define per-cell cost:

| Layout | accumulators / cell | relative to 1 target |
|---|---|---|
| single-target | 2 | 1.0× |
| T × single-target (round-robin, variant 1) | 2T | T× |
| **vector-leaf shared-hessian** | **T + 1** | **(T+1)/2 ×** |

So the histogram-phase cost (global-memory writes, shared-memory zero+merge, and
the split-finder's histogram reads, all of which scale with accumulators/cell)
grows as **(T+1)/2** instead of **T**.

There is *also* a per-row read saving: the row's bin index and the compact bin
matrix are read **once** for all T targets in the vector-leaf pass, vs T times in
round-robin. The input-scan bandwidth (bin bytes) is therefore ~1× regardless of
T, only the accumulate traffic grows.

**Shared-memory pressure.** The construct kernel accumulates into a per-block
shared histogram (`__shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE]`,
`cuda_histogram_constructor.cu:370`). A cell goes from 2 to T+1 slots, so the
per-feature shared footprint grows ~(T+1)/2×. For T=5 that is 3× the shared
histogram. This is the binding constraint: it forces smaller `block_dim_x`
(fewer features per block) and may exceed the 48–96 KB shared cap for wide-bin
features. **Mitigation:** (a) tile targets — accumulate ⌈T/G⌉ groups of G
gradients + the shared hessian in separate passes over the same shared budget;
(b) fall back to the global-memory construct path
(`use_global_memory_`) for large T. The design must keep the shared-hist size a
compile-time-tiled quantity, not linear in T.

**L2 residency.** The global histogram pool (`cuda_hist_`) grows by
(T+1)/2×. Single-target histograms comfortably fit the ~96 MB L2 on the target
GPU; at T=5 the pool is ~3× larger, so a given leaf's histogram may spill L2 and
the split-finder read then hits HBM. Still, 3× the data at HBM bandwidth beats
5× the data (round-robin) — the shared-hessian layout is strictly better and the
sub-linear scaling is the headline.

### 5.4 T = 5 Numerai estimate

Per-tree histogram cost (dominant term) scales as (T+1)/2:

- **Vector-leaf, T=5:** (5+1)/2 = **3.0×** a single-target tree's histogram cost.
- **Sequential / round-robin, T=5:** **5.0×** (five full trees).

Assuming histogram construction + split-find is ~70–80% of per-tree wall time on
this fork (the graph path having removed most host overhead), and the remaining
~20–30% (partition, apply, graph launch) is **shared once** across all T targets
in the vector-leaf tree (vs paid 5× in round-robin):

```
sequential per-tree cost   ≈ 5 × (0.75 hist + 0.25 other)          = 5.00
vector-leaf per-tree cost  ≈ 3.0 × 0.75 hist + 1.0 × 0.25 other    = 2.50
```

**Estimated per-tree speedup at T=5: ~2.0× (5.00 / 2.50)**, i.e. **~2× the
throughput of sequential per-target training, ≈0.4 tree-equivalents per target**.
The win grows with T (asymptotically → 2× from the hessian sharing alone, more
once the shared "other" overhead is counted). This is an *estimate*; the shared
hessian is exact, the 0.75/0.25 split is the number to measure first (§7).

Caveat: this is a *structure-shared* model, so the per-target *quality* is not
identical to sequential — the throughput win must be weighed against a possible
per-era correlation change (§8). The honest pitch is "≈2× faster to a
*comparable* model, and often better-regularized on correlated targets."

### 5.5 Split-gain and leaf-output seams

- **Gain = sum over targets.** `CUDALeafSplits::GetSplitGains`
  (`cuda_leaf_splits.hpp:144–160`) currently returns
  `GetLeafGain(left) + GetLeafGain(right)` for scalar sums. Vector-leaf wraps
  this in a `Σ_t`: loop t, read `(G_L,t, G_R,t)` from the T-gradient cell, reuse
  the shared `(H_L, H_R)`. Keep the existing `GAIN_T` templating
  (`LaunchFindBestSplitsForLeafKernelInner3<..., GAIN_T>`,
  `cuda_best_split_finder.hpp:301`) — add a compile-time `int T` (or a small
  `NUM_TARGETS` template) so T=1 is byte-identical to today and the T>1
  instantiations are separate kernels (no runtime branch in the inner loop).
- **`CUDALeafSplitsStruct` gains a vector value.** Today it has scalar
  `sum_of_gradients` / `leaf_value` (`cuda_leaf_splits.hpp:37–48`). Vector-leaf
  needs `sum_of_gradients[T]` and `leaf_value[T]` with one shared
  `sum_of_hessians`. Options: (a) fixed-max `T ≤ kMaxTargets` inline array (keeps
  the struct POD and pooled-slab-friendly — the hybrid pair slots are a raw
  device slab, `hybrid_pair_slots_`); (b) a side buffer indexed by leaf. Prefer
  (a) with a modest `kMaxTargets` (e.g. 16) to keep the struct trivially
  copyable for the batched pair-descriptor H2D copy.
- **`CUDASplitInfo`** (`include/Falcata/cuda/cuda_split_info.hpp`) must carry the
  per-target left/right sums (or at least the per-target leaf outputs) so
  `ApplySplit` can write both children's vector values.

### 5.6 Tree structure / data partition seams

- **Data partition is unchanged.** `cuda_data_partition` routes *rows* to leaves
  by the single (feature, threshold) split; it is target-agnostic. Vector-leaf
  changes *what a leaf stores*, not *which rows land there*. This is why
  vector-leaf is cheap on an existing asymmetric learner — the expensive
  partitioning machinery is reused verbatim.
- **`CUDATree` leaf value becomes a vector.** Today `Tree::leaf_value_` is
  `std::vector<double>` of length `num_leaves` (`include/Falcata/tree.h:509`),
  `LeafOutput(leaf)` returns a scalar (`tree.h:92`). Vector-leaf makes it
  `num_leaves * T` (row-major `[leaf][t]`). The CUDA side
  `CUDATree::cuda_leaf_value_` (device array carved from the pooled slab,
  `PooledDeviceBufferSize`, `cuda_tree.hpp`) grows by ×T; `ToHost()` reads back
  `num_leaves * T` doubles. `SetLeafOutput`/`AddBias`/`Shrinkage` (`tree.h:95,
  190–237`) must iterate over the T-vector per leaf.
- **`Split` / `SplitBatch`** (`cuda_tree.hpp`) already take a `CUDASplitInfo*`;
  they must write T leaf values per child instead of 1.

### 5.7 Interaction with hybrid / graph fast paths

The hybrid growth and CUDA-graph gates are structure-only and stay green for
vector-leaf **if** the histogram/gain kernels are made T-aware:

- `HybridGrowthUsable` (`cuda_single_gpu_tree_learner.cpp`) gates on
  `num_leaves`, categorical features, forced splits, NCCL, node-feature-select —
  **none target-dependent**. Vector-leaf passes it unchanged.
- `CUDAHistogramConstructor::SupportsBatchedLevel`
  (`cuda_histogram_constructor.hpp:196`) gates on dense/non-sparse layout,
  bit_type, compact-view — target-independent. **But** the batched construct
  kernel's shared-histogram sizing (`SHARED_HIST_SIZE`,
  `CalcConstructHistogramBatchedKernelDim`) is per-cell-size dependent; the
  (T+1)/2× shared growth (§5.3) must be folded into the grid/shared sizing and
  the tiling fallback, or `SupportsBatchedLevel` must return false for T that
  overflows shared memory (clean degrade to per-pair / global-mem path).
- `CUDABestSplitFinder::SupportsBatchedLevel`
  (`cuda_best_split_finder.hpp:122`) gates on template combo (no L1/smoothing/
  extra_trees/global-mem/by-node/categorical) and task count — target-
  independent. The gain summation over T is inside the kernel body, so the gate
  is unchanged; only new `NUM_TARGETS` instantiations are added.
- **Graph capture** bakes per-tree pointers into the graph key
  (`HybridGraphKeyPointers`, `HybridGraphCompactShapeKey`). T is a *static*
  per-run constant, so it becomes part of the compile-time kernel selection, not
  the runtime key — graphs capture and replay exactly as today. **Recommendation:
  land vector-leaf behind the classic (non-graph) hybrid path first**
  (`EXABOOST_GRAPH_LEVEL_LOOP=0`), then extend graph capture once the T-aware
  kernels are proven.

### 5.8 Model format & prediction

- **Serialization.** `Tree` gains a `num_targets_` (mirror XGBoost's
  `size_leaf_vector`) written to the model header; `leaf_value=` lines carry
  `num_leaves * T` values. This is a **new model schema** — bump a model version
  tag and keep the loader back-compatible (T absent ⇒ T=1). `GBDT` must know a
  vector-leaf iteration is **one tree producing T outputs**, distinct from
  multiclass's **T trees**: `num_tree_per_iteration_` stays 1 while
  `num_output_ = T`.
- **Predict.** One traversal yields T outputs. `GBDT::PredictRaw`
  (`src/boosting/gbdt_prediction.cpp:16–29`) currently does
  `output[k] += models_[i*num_tree_per_iteration_ + k]->Predict(features)` — a
  scalar `Tree::Predict` per class. Vector-leaf needs a `Tree::PredictVector`
  that returns the leaf's T-vector from a single traversal and accumulates into
  `output[0..T)`. Cheaper than T traversals.
- **⚠ FIL / Treelite fallback.** GPU-accelerated inference (Treelite/FIL, used by
  `bst.predict` fast paths) **does not support vector-leaf trees** — Treelite's
  model IR is scalar-leaf. Models trained vector-leaf will **fall back to the CPU
  (native Falcata) predictor**. This must be flagged loudly in docs and the
  Numerai upload pickle path (predict must use the native traversal, not a FIL
  export). Round-robin (variant 1) models are ordinary scalar-leaf trees and
  keep full FIL support.

### 5.9 Objective seam

New `RegressionMultiTargetVector` objective (host + `src/objective/cuda/`):
- `NumModelPerIteration()` returns **1** (unlike multiclass) — GBDT builds one
  vector tree per iteration.
- Emits the same `[t][row]` gradient layout the multiclass path already uses
  (`num_data_ * t + i`), hessian slice is constant (all 1s or weights) so the
  histogram kernel can *ignore the per-target hessian entirely* and use the
  shared count — the key that makes T+1 (not 2T) correct. Assert `hess_t ≡ hess`
  across t at init; refuse non-RMSE multi-target losses in phase 1.
- **Design space for SketchBoost (later):** allow the objective to expose a
  reduced "split gradient" (mean/SVD over T) for structure + full "value
  gradient" for leaf outputs, exactly like XGBoost's `split_grad`
  (dmlc/xgboost#11798). This decouples histogram width (reduced dim r ≪ T) from
  T. Not phase 1, but the `CUDALeafSplitsStruct`/histogram-cell layout should not
  hard-assume "histogram width == output width."

---

## 6. Per-file change list (file : function/struct)

**Variant 1 (round-robin):**
- `include/Falcata/config.h` — add `int num_target = 1;` near `num_class`
  (`:925–927`).
- `src/objective/regression_objective.hpp` + `src/objective/cuda/
  cuda_regression_objective.{hpp,cu}` — new `RegressionMultiTarget`:
  `NumModelPerIteration()→T`, `GetGradients` filling `[t][i]` slices
  (reuse `CUDARegressionL2loss::LaunchGetGradientsKernel`, `:51`).
- `src/io/metadata.cpp` / `include/Falcata/dataset.h` — T-column label ingestion
  (mirror `init_score`'s `num_class * num_data` storage).
- No changes to `gbdt.cpp` per-tree loop, `ScoreUpdater`, tree learner.

**Variant 2 (vector-leaf) — additive, T=1 stays byte-identical:**
- `include/Falcata/tree.h` — `leaf_value_` → length `num_leaves*T`; add
  `num_targets_`; vectorize `LeafOutput`/`SetLeafOutput`/`AddBias`/`Shrinkage`
  (`:92,95,190–237`); add `PredictVector`.
- `include/Falcata/cuda/cuda_tree.hpp` — `cuda_leaf_value_` ×T;
  `PooledDeviceBufferSize` (+`(T-1)*sizeof(double)*L`); `Split`/`SplitBatch`/
  `ToHost` write/readback T values.
- `src/treelearner/cuda/cuda_leaf_splits.hpp` — `CUDALeafSplitsStruct`: add
  `sum_of_gradients[kMaxTargets]`, `leaf_value[kMaxTargets]`, keep one
  `sum_of_hessians` (`:37–48`); `GetSplitGains`/`GetLeafGain`
  wrapped in `Σ_t` with shared H (`:135–160`).
- `src/treelearner/cuda/cuda_histogram_constructor.{cu,hpp}` — cell layout
  `[g_0..g_{T-1}, h]`; construct kernels (`:355–415`, batched
  `:450–590`) accumulate T grads + 1 shared hess in one row pass; shared-hist
  sizing/tiling in `CalcConstructHistogramBatchedKernelDim`,
  `SupportsBatchedLevel` (`:196`); subtract kernel over T grads + 1 hess.
- `src/treelearner/cuda/cuda_best_split_finder.{cu,hpp}` — add `int NUM_TARGETS`
  template to `LaunchFindBestSplitsForLeafKernelInner3` (`:301`); gain loop over
  T reading the T-gradient cell; `CUDASplitInfo` carries per-target sums.
- `include/Falcata/cuda/cuda_split_info.hpp` — per-target left/right grad sums /
  leaf outputs.
- `src/treelearner/cuda/cuda_single_gpu_tree_learner.{cpp,cu}` — `ApplySplit` /
  `LaunchCalcLeafValuesGivenGradStat` (`.hpp:343`) write T leaf values;
  `RenewTreeOutput`/`ReduceLeafStat` vectorized.
- `src/objective/cuda/cuda_regression_objective.{hpp,cu}` — new
  `RegressionMultiTargetVector`: `NumModelPerIteration()→1`, `[t][i]` gradients,
  constant shared hessian; init-time assert `hess` target-invariant.
- `src/boosting/gbdt.cpp` / `gbdt_prediction.cpp` — vector-leaf iteration =
  1 tree, `num_output_=T`; `PredictRaw` uses `PredictVector` single traversal
  (`prediction.cpp:16–29`).
- Model I/O (`src/io/tree.cpp` `Tree::ToString`/parse) — `num_targets` header +
  `num_leaves*T` leaf values; back-compat T=1.

---

## 7. Phased implementation plan

1. **Phase 0 — measure.** Instrument a single-target Numerai tree on this fork:
   what fraction of per-tree wall time is histogram construct + split-find vs
   partition/apply/graph? This fixes the 0.75/0.25 assumption in §5.4 and the
   real T=5 speedup. (No code churn; profiling only.)
2. **Phase 1 — Variant 1 (round-robin), small PR.** Config `num_target`, label
   matrix, `RegressionMultiTarget` objective. Ships a usable multi-target API and
   de-risks label/config plumbing. md5-identical to sequential ⇒ trivially
   verified.
3. **Phase 2 — Variant 2 kernels behind an env flag, T=1 identity.** Add the
   `NUM_TARGETS` template + `[g..g,h]` cell but compile/run T=1 first; assert
   **bit-identical** histograms and models to the pre-change build (the whole
   fork's md5-gate discipline). This proves the refactor is inert at T=1.
4. **Phase 3 — T>1 vector-leaf, classic hybrid path** (`EXABOOST_GRAPH_LEVEL_LOOP=0`).
   Shared-hessian construct + summed-gain find + vector leaf output + vector
   serialization + `PredictVector`. Validate quality per-era (§8).
5. **Phase 4 — graph capture for T>1**, shared-memory tiling for large T, and
   the SketchBoost reduced-gradient hook (`split_grad`).

---

## 8. Verification strategy (md5-gate + quality)

- **T=1 identity gate (structural).** This fork's core discipline is
  md5-identical model files under fixed seeds. Phase 2/3 must reproduce the
  pre-change model md5 exactly at T=1 for the standard fraud/Numerai fixtures —
  the vector-leaf refactor is only allowed to *widen*, never to perturb, the T=1
  path.
- **Round-robin identity (variant 1).** Train `num_target=T` round-robin vs T
  sequential single-target runs ⇒ **identical md5 per tree**. Direct proof of
  no modeling change.
- **Vector-leaf: NOT md5-identical to sequential** (different model). Verify
  instead:
  - *Determinism gate:* two vector-leaf runs same seed ⇒ identical md5 (rules out
    the usual atomic-order / float-reduction nondeterminism the fork already
    guards).
  - *Numeric gate:* CPU reference vector-leaf (small, exact) vs CUDA vector-leaf
    ⇒ leaf values within tolerance, identical tree structure (mirror the
    quantized-gradient CPU/GPU parity harness already in the repo).
  - *Quality gate:* **per-era Numerai correlation / MMC**, per target, vs
    sequential baseline — this is a *modeling* change and must be judged on
    validation, not on md5. Report mean and per-era spread.

---

## 9. Risks

1. **Quality is a modeling change (top risk).** Vector-leaf ties all T targets
   to one structure. On weakly-correlated Numerai targets this can *hurt* the
   primary target's per-era correlation even while total loss drops. **Must be
   validated per-era, per-target against sequential**, not accepted on speed
   alone. Mitigation path: SketchBoost reduced-gradient (structure from a
   reduced/mean gradient, values full) recovers most of the sequential quality
   while keeping the histogram narrow.
2. **FIL / Treelite predict fallback.** Vector-leaf models cannot use GPU FIL
   inference (scalar-leaf IR only) and fall back to the CPU native predictor.
   Breaks any Numerai upload path that assumes a Treelite/FIL export — the
   upload pickle must pin the native traversal. Round-robin models are unaffected.
3. **Quant interaction.** The quantized-gradient path packs (grad16, hess16) into
   one 32-bit word and has its own int16/int32 histogram bit-width machinery
   (`cuda_gradient_discretizer`, `HybridQuantConstructMaxRowsPerThread`,
   `hist_buffer_for_num_bit_change_`). A T-gradient shared-hessian cell needs a
   *new* packed layout (T×int16 grads + 1 shared int16 hess) and a re-derivation
   of the overflow guard. **Phase 1–3 should require non-quantized training for
   vector-leaf**; quant is a follow-up. Getting this wrong silently corrupts
   histograms (the fork's fraud 63/6 exact-repro tests would catch it).
4. **Shared-memory overflow at large T.** The (T+1)/2× shared-hist growth caps
   the batched construct path; without target-tiling, `SupportsBatchedLevel`
   must degrade to global-memory construct for large T (losing some of the win).

---

## 10. Open questions

1. **`kMaxTargets` inline-array vs side-buffer** for `CUDALeafSplitsStruct` /
   `CUDASplitInfo` — inline keeps the batched pair-descriptor H2D copy trivially
   POD; side-buffer scales to arbitrary T. Pick a cap (16?) that covers Numerai.
2. **Where does the shared hessian actually equal 1?** Weighted RMSE has
   `hess = weight` (target-invariant ✓). But if any target has NaN/missing labels
   with per-target masking, the hessian becomes target-dependent and the T+1
   trick breaks for those rows. Need a policy: shared mask across targets, or
   fall back to 2T cells when masks differ.
3. **Leaf-value gradient vs split gradient dimensionality** — commit now to the
   layout not hard-assuming `histogram_width == num_targets`, so SketchBoost drops
   in later without another histogram refactor.
4. **Model-format version bump** — coordinate with the fork's existing model
   version tags so a T>1 file is rejected by old loaders rather than
   mis-parsed as T=1.
5. **Does the per-tree "other" overhead (§5.4) really amortize?** Phase 0 must
   confirm partition/apply/graph-launch is genuinely shared, not re-paid per
   target inside the vector tree.
