# What makes Falcata fast — and how we know

This is the verbose companion to the README's feature bullets: each landed
optimization explained for readers who do not live inside CUDA, with the
measured gains that justify it. The evidence comes from two instruments:

- **Leave-one-out ablation** (`benchmarks/ablation.py`): train with the full
  auto plan, then turn each feature off one at a time and measure how much
  slower training gets. "+300%" below means *turning the feature off makes
  training 4× slower* — i.e. the feature is worth 4×. Because every
  mechanical feature is required to produce bit-identical models, the
  ablation doubles as a correctness gate. Raw tables:
  `benchmarks/ablation_2026-07-30.txt`.
- **The cross-library benchmark suite** (`~ benchmarks/README.md`): Falcata
  vs upstream LightGBM, XGBoost, and CatBoost on the gbm-bench datasets plus
  the real numerai workload.

Failed ideas are deliberately NOT here — they live in
[perf-dead-ends.md](perf-dead-ends.md). Open ideas live in
[../ROADMAP.md](../ROADMAP.md).

Datasets referenced below: **covtype** (581k×54, 7-class), **year** (515k×90
regression), **fraud** (285k×28, imbalanced binary), **higgs** (11M×28
binary), **epsilon** (500k×2000 binary), **numerai** (6.7M×3555 int8
regression; "example" = 2k trees/32 leaves, "deep" = 30k trees/1024 leaves).

---

## 1. Hybrid level-batched growth — the biggest single win

**The problem.** A leaf-wise GBDT learner classically grows a tree one split
at a time: build histograms for one leaf, find its best split, tell the CPU,
apply the split, repeat. On a GPU each of those steps is a separate kernel
launch plus a CPU⇄GPU synchronization. The GPU spends most of its time
waiting for the next tiny instruction rather than computing — the loop is
*latency-bound*.

**The idea.** While the leaf budget cannot yet bind, every profitable leaf
will eventually be split anyway — so the ORDER of splits doesn't change the
final tree. Falcata therefore grows whole *levels* of sibling pairs at once:
one batched histogram pass, one batched split search, one batched apply per
level. Dozens of launches and syncs collapse into three. The resulting tree
is provably identical to the leaf-wise tree (and the ablation verifies the
models match).

**Measured (ablation, "what turning it off costs"):**

| covtype-deep | fraud-deep | numerai-deep | year | covtype-shallow | numerai-example | epsilon | higgs |
|---|---|---|---|---|---|---|---|
| +1004% | +360% | +324% | +303% | +345% | +200% | +150% | +55% |

Deep trees benefit most (more levels, more launches saved); higgs least (its
huge rows make each kernel long enough that launch latency matters less).

Two sub-features extend the same idea:

- **Batched split kernels (`batch_kernels`)** — one find/sync launch per
  level instead of per pair: +274% covtype-deep, +241% numerai-deep, +171%
  numerai-example, +92% fraud-deep, +61–81% year/covtype-shallow.
- **Batched apply (`batch_apply`)** — the split-application phase batched the
  same way: +281% covtype-deep, +102% covtype-shallow, +79% year, +28%
  numerai-deep.
- **Selective grow-then-prune (`selective`)** and the **speculative one-sync
  pipeline (`one_sync`)** extend level batching to budget-limited and
  non-quantized configurations; on shapes where they don't apply they cost
  nothing (±2% noise in every cell), which is exactly why they can default on.

## 2. CUDA-graph level loops (`graph_loop`)

**The problem.** Even batched, each level's launch sequence is issued by the
CPU. For shallow trees the levels are so short that CPU launch overhead
returns.

**The idea.** CUDA lets you record a sequence of kernel launches once (a
"graph") and replay it from the device itself. Falcata captures the per-level
sequence and lets a device-side controller replay it, removing the CPU from
the inner loop entirely.

**Measured:** +10% on year (shallow trees on a mid-size dataset — the target
shape); neutral on deep configurations, where the fixed controller latency
would hurt — so the planner disables it there automatically.

## 3. Per-tree compact column view (`compact_quant`)

**The problem.** With `feature_fraction < 1`, each tree randomly samples a
subset of columns. But the bin matrix is row-major: even if a tree uses 10%
of the columns, reading a row drags the other 90% through the memory system,
because unused columns share the same cache lines.

**The idea.** Once per tree, gather ONLY the sampled columns into a dense
"compact" matrix. Histogram passes then read purely useful bytes. The gather
costs one pass; the histogram kernels run 11+ passes per tree over the
result.

**Measured:** +236% on numerai-deep, +153% on numerai-example — the two
feature-sampled workloads; exactly neutral elsewhere (it only activates when
sampling is on). The win scales with the excluded fraction: ~3.4× at
`feature_fraction=0.1`, tapering to ~1.1× at 0.6.

## 4. GPU-native ingestion (`gpu_construct`, `fast_rowdata`, `efb_precheck`, `rowdata_4bit`)

**The problem.** Before training starts, raw features must be binned and laid
out for the GPU. Upstream does this on the CPU, then uploads — minutes of
setup for large datasets (numerai Booster creation was 13.9s even after
earlier fixes; originally far worse).

**The ideas.**
- **`gpu_construct`**: dense binning runs on the device; CuPy /
  `__cuda_array_interface__` inputs never round-trip through host memory.
- **`fast_rowdata`**: build the row-major training matrix directly from
  column bins, skipping the CPU multi-value-bin machinery: +297% on
  numerai-example, +26% on year, +20% on higgs (throughput effect via
  construct-time inclusion in those cells).
- **`efb_precheck`**: a cheap density check that skips Exclusive Feature
  Bundling's ~7.7s no-op search on provably-unbundlable dense data.
- **`rowdata_4bit`**: datasets whose features all fit 16 bins store two
  values per byte, halving the training matrix (numerai: 19GB → 9.5GB on the
  current cache) and the bytes every kernel reads.

## 5. Quantized training: `quant_mode` (the speed/quality dial)

**The problem.** Histogram accumulation is the hot loop, and accumulating
double-precision gradient/hessian pairs is memory-heavy.

**The idea.** Quantize gradients to small integers (a published technique —
see the NeurIPS'22 reference in the README) and accumulate integers instead.
Falcata ships two modes: `stochastic` (4 bins, seeded stochastic rounding —
the aggressive end) and `fixedpoint` (deterministic rounding with an
outlier-robust gradient scale — the near-lossless end). Both are
bit-reproducible run to run; integer atomics also make results
order-invariant, which is what lets the md5 regression gates exist.

**Measured (benchmark suite, numerai-deep, 30k trees):**

| mode | trees/s | holdout corr |
|---|---|---|
| stochastic | 31.7 | 0.0234 |
| fixedpoint | 32.4 | 0.0232 |
| none | 15.9 | 0.0229 |

2× the speed of full precision at equal (here: marginally better) quality.
The outlier-robust scale is what makes fixedpoint safe on imbalanced data:
without it, fraud/deep AUC drops 0.9825 → 0.8001.

## 6. Precision modes (`cuda_precision=fp32`)

For NON-quantized training, storing global histograms as float pairs instead
of double pairs halves their bandwidth. Measured per-tree wins at
equal-or-better quality: epsilon-deep −36% time, year −18%, covtype −16%,
fraud-deep −14%, higgs-deep −12%; numerai neutral (sampling-dominated).
Quality-gated rather than bit-identical, hence a config parameter and not a
plan key.

## 7. Memory-layout micro-optimizations (each small, all free)

- **`gh_interleave`** — gradient and hessian interleaved as one float2 so a
  row costs one scattered 32-byte read instead of two: +21% numerai-deep.
- **`split_packed_read`** — split kernels read the 4-bit packed matrix
  directly instead of materializing a ~1.5GB per-tree column copy: +12%
  numerai-deep.
- **`batch_reghist`** — for ≤8-bin datasets, accumulate a thread's rows in
  registers and flush once instead of two shared-memory atomics per row.
- **`batch_wide`** — wide-shape batched search for many-column datasets
  (+9.9% on epsilon in the smoke ablation), plus wide leaf-splits init
  batching.
- **`small_leaf_construct`** — a cheaper construct body for very small
  leaves.

The ablation shows each of these within noise on shapes they don't target —
the planner's "default on, individually ablatable" contract in action.

## 8. GPU inference via NVIDIA FIL

`Booster.predict()` on a CUDA-trained model routes through cuML's Forest
Inference Library when available: numerai predict 0.90s → **0.046s** (CuPy
in/out), higgs 0.37s → 0.004s. See the README for precision notes and the
opt-out.

## 9. How this adds up against other libraries

From the 2026-07 benchmark suite (RTX 5090, aligned hyperparameters, medians
of 3, failures flagged rather than hidden):

- **numerai-deep**: Falcata 32.4 trees/s; upstream LightGBM, XGBoost and
  CatBoost all fail (crash/OOM/timeout) at this scale.
- **numerai-example**: Falcata 51.3 trees/s vs CatBoost 17.5 (2.9×), with
  better holdout correlation; LightGBM-CUDA and XGBoost failed at warmup
  (XGBoost's failure was our harness's predict OOM, since fixed).
- **Classic datasets (deep)**: 1.8× vs best competitor on covtype, 1.2× on
  higgs, 1.7× on epsilon, 2.2× on year (shallow).
- Peak memory: Falcata trains numerai-deep in ~14GB VRAM; CatBoost needed
  ~30GB VRAM + 112GB anonymous host RAM for the same regime.

Full tables, charts, environment manifest and the failure ledger:
`~/Documents/exaboost-bench/report/REPORT.md`.

---

*Footnote on the 2026-07-30 ablation snapshot: the numerai-deep
`batch_kernels` row is marked "MD5 DIFFERS" in the raw table; this was
subsequently investigated and verified benign — the fallback path breaks
exact-gain ties in a different order at 5.4M-row scale (holdout corr
identical to 5 decimals). It is reclassified as a tie-break key, not an
equality key (commit `2f30f043`).*
