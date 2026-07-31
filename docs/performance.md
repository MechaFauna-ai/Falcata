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
Falcata ships two modes: `stochastic` (seeded stochastic rounding — the
aggressive end) and `fixedpoint` (deterministic rounding with an
outlier-robust gradient scale — the near-lossless end). Bin count is the
`quant_bins` dial for both modes (defaults: 4 stochastic, 64 fixedpoint;
any value in [2, 65534]). Both are
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

Four more landed 2026-07-31 (battery: `benchmarks/top4_2026-07-31.txt`; all
bit-identical in every cell):

- **`wide_partitions`** — on datasets wider than 504 columns, each construct
  thread handles two columns, halving the partition count and its per-partition
  zero/merge overhead: **+8.8% numerai-deep**; mechanically inert on narrower
  data.
- **`l2_policy`** — pins the gradient/hessian and data-index buffers into a
  persisting L2 window so each level's bin-matrix stream stops evicting them
  (the construct kernel is latency-bound on exactly those scattered re-reads):
  +2.8% numerai-deep and covtype-deep, neutral on year at real run lengths.
- **`colmajor_fill`** — a one-time column-major copy of the packed bin matrix
  serves as the compact-fill gather source, so the per-tree fill reads
  contiguous columns instead of dragging ~10× its bytes through row-major
  cache lines: +2.2% numerai-deep (the fill is genuinely bandwidth-bound, but
  it is a small slice of tree time — the +14% pre-measurement estimate did not
  survive contact with the profiler). VRAM-gated by the planner.
- **`tuner`** — a per-tree bandit over the batched-construct saturation floor
  (candidates {80,160,320,640}, best-of-15 timing, re-probe every 3000 trees):
  +2.1% numerai-deep, +2.7% year. Quantized training only — integer histograms
  make the model schedule-invariant, so retuning cannot change results. Under
  `auto` it engages only at ≥300 rounds (the ~60-tree probe phase costs ~2% on
  a 100-round run); `cuda_plan=auto,tuner:on` forces it regardless.

All four compose: **+10.5% on numerai-deep combined**. A methodology note the
battery re-taught us: 100-round probe cells on fast datasets (year runs 0.4s)
sit inside clock/thermal noise — the year "regressions" the battery first
reported all vanished under interleaved A/B at 500 rounds.

The ablation shows each of these within noise on shapes they don't target —
the planner's "default on, individually ablatable" contract in action.

## 8. GPU inference via NVIDIA FIL

`Booster.predict()` on a CUDA-trained model routes through cuML's Forest
Inference Library when available: numerai predict 0.90s → **0.046s** (CuPy
in/out), higgs 0.37s → 0.004s. See the README for precision notes and the
opt-out.

## 9. How this adds up against other libraries

From the 2026-07 benchmark suite: RTX 5090, aligned hyperparameters
(gbm-bench convention), medians of 3 timed runs, held-out quality reported
for every number. Falcata's `quant_mode` is a user-facing dial, so each row
quotes the variant that matches or beats the competitor's quality — the
speedup is never bought with quality the user wouldn't accept.

**The flagship workload (numerai, official example params — 2000 trees on
5.4M×3555):**

| vs | speedup | quality (era corr) |
|---|---|---|
| upstream LightGBM (OpenCL GPU) | **6.2×** | 0.0198 vs 0.0198 — identical |
| XGBoost (CUDA) | **7.8×** | 0.0198 vs 0.0194 — better |
| CatBoost (CUDA) | **2.9×** | 0.0198 vs 0.0182 — better |

**Classic gbm-bench datasets** (speedup at matched-or-better quality;
falcata variant in parentheses):

| dataset / regime | vs LightGBM-OCL | vs XGBoost | vs CatBoost |
|---|---|---|---|
| higgs shallow (stoch) | **15.7×** (=AUC) | 1.2× (=AUC) | 3.0× (better) |
| higgs deep (stoch) | **12.2×** (=AUC) | 1.2× (=AUC) | 2.4× (better) |
| year deep (fixed) | **7.9×** | **2.9×** (better RMSE) | 2.8×¹ |
| epsilon shallow (stoch) | 3.3× (=AUC) | 1.9× (=AUC) | 2.3× (=AUC) |
| covtype deep (fixed) | —² | **1.4×** (better acc: .971 vs .963) | 1.6× (better) |
| fraud deep (fixed) | —² | ~par speed, **best AUC of any library** (.9846) | **10.5×** (better) |

¹ CatBoost reaches slightly better RMSE on year-deep (8.93 vs 8.97) at 2.8×
the time; falcata-noquant closes most of the gap at still-lower time.
² Upstream LightGBM produced invalid models on these cells (see last bullet).

**The deep end nobody else finishes**: on the numerai deep example config
(30k trees, 1024 leaves, 5.4M×3555), falcata-fixed trains at **32.4
trees/s** (~15 minutes total) with corr 0.0232. LightGBM and CatBoost crash
internally on this regime; XGBoost's re-measurement is pending (its earlier
failure was a since-fixed harness bug, not XGBoost).

**Resources**: falcata trains numerai-deep in ~14GB VRAM with reclaimable
memmap reads; CatBoost's working numerai run needed ~30GB VRAM (the whole
card) plus ~112GB of anonymous host RAM.

**Competitor failures** (flagged, never silently averaged): upstream
LightGBM 4.7.0's CUDA backend is broken on this hardware/scale — garbage
models on fraud/covtype-deep and hard CUDA crashes on higgs/numerai — and
its quantized mode produces invalid models everywhere; its legacy OpenCL
backend (benchmarked above) works but trails by 3–16×. CatBoost dies on
numerai-deep with an internal CUDA kernel timeout (error 702). XGBoost
trains the numerai regimes fine — an earlier recorded failure there was our
harness's oversized predict allocation, since fixed and re-measured. Full
failure ledger with per-cell errors: the benchmark report.

---

*Footnote on the 2026-07-30 ablation snapshot: the numerai-deep
`batch_kernels` row is marked "MD5 DIFFERS" in the raw table; this was
subsequently investigated and verified benign — the fallback path breaks
exact-gain ties in a different order at 5.4M-row scale (holdout corr
identical to 5 decimals). It is reclassified as a tie-break key, not an
equality key (commit `2f30f043`).*
