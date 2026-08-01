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
bit-reproducible — run to run, across GPU models, and across host
machines: stochastic rounding noise is Philox-generated in-kernel as a pure
function of (seed, tree, row) — an idea borrowed from XGBoost 3.3's Philox
sampling — so machine-independence holds by construction (no tables; also
freed 8 bytes/row of VRAM and their per-tree reads, worth ~+4% on
numerai-deep). Integer atomics make results order-invariant, which is what
lets the md5 regression gates exist and makes them portable to any machine
(cross-arch verified sm_89 vs sm_120 on the table-era build; Philox is pure
integer math and inherits the guarantee).

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

A second, separately-measured mechanism (2026-08-01): on DEEP trees the
histogram pool halves from ~248MB (doesn't fit the 5090's 96MB L2) to
~124MB (mostly fits), so subtraction's parent-histogram re-reads start
hitting cache. Isolated on covtype non-quant: fp32 gains **+26% deep** vs
+1.7% shallow — the cache cliff, not bandwidth, dominates the deep win.
Practical guidance: on deep non-quantized configs, `cuda_precision=fp32` is
the single highest-leverage switch available.

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
  leaves. Extended 2026-07-31 to quantized training: when a deep level's
  largest sibling pair is under 1024 rows, a dedicated kernel adds packed
  integer gradients straight to the global histogram — the shared-memory
  zero + sync + merge (whose cost is proportional to partition *bins*, not
  rows) is skipped entirely: **+11.2% covtype-deep**, neutral on numerai
  (its 10k-row min-leaf never triggers it). Bit-identical by integer
  order-invariance — unlike the float direct body, which remains permanently
  disabled. Kept as a separate kernel deliberately: an in-kernel branch
  version cost the never-taken numerai path measurable register pressure.

Four more landed 2026-07-31 (battery: `benchmarks/top4_2026-07-31.txt`; all
bit-identical in every cell):

- **`wide_partitions`** — on datasets wider than 504 columns, each construct
  thread handles two columns, halving the partition count and its per-partition
  zero/merge overhead: **+8.8% numerai-deep**; mechanically inert on narrower
  data.
- **`l2_policy`** — pins the gradient/hessian and data-index buffers into a
  persisting L2 window so each level's bin-matrix stream stops evicting them
  (the construct kernel is latency-bound on exactly those scattered re-reads):
  +2.8% numerai-deep and covtype-deep at the original fixed 64MB carve-out,
  improved to **+5.3%** once the carve-out was sized to the buffer actually
  pinned (device-proportional sizing, 2026-07-31 follow-up — a fixed carve
  stranded 42MB of L2 the streaming reads could have used); neutral on year
  at real run lengths. Its companion: bin-matrix loads are marked
  **evict-first** (`__ldcs`) since bin bytes have no intra-level reuse —
  deprioritizing them frees L2 for histogram-subtraction re-reads: +8%
  covtype-deep, +10% year via the dense-path loads, and **+9.7%
  numerai-deep** once extended to the compact-view pack-codec reads (the
  per-tree compact matrix re-reads 11+ passes/level against a 96MB L2 —
  streaming priority stops it evicting the pinned gradient window).
  (Prediction inverted by measurement: the win is largest on the "already
  L2-resident" shapes.)
- **`colmajor_fill`** — a one-time column-major copy of the packed bin matrix
  serves as the compact-fill gather source, so the per-tree fill reads
  contiguous columns instead of dragging ~10× its bytes through row-major
  cache lines: +2.2% numerai-deep (the fill is genuinely bandwidth-bound, but
  it is a small slice of tree time — the +14% pre-measurement estimate did not
  survive contact with the profiler). VRAM-gated by the planner.
- **`tuner`** — a per-tree bandit over behavior-preserving execution knobs,
  best-of-15 timing, re-probe every 3000 trees: +2.1% numerai-deep, +2.7%
  year from the saturation-floor knob alone. Quantized training only —
  integer histograms make the model schedule-invariant, so retuning cannot
  change results. Under `auto` it engages only at ≥300 rounds;
  `cuda_plan=auto,tuner:on` forces it. Extended 2026-08-01 into the full
  three-tier stack: **tier-0** seeds the candidate sets from the device
  (floor candidates scale with SM count relative to the 5090 they were tuned
  on); **tier-1** runs coordinate descent over two knobs (saturation floor
  with an elastic bracket, and the quant small-leaf row threshold); and
  **tier-3** persists the chosen values per (shape, device) signature to
  `~/.cache/falcata/wisdom.txt` — retrains of the same workload skip the
  ~130-tree probe phase and start at the known-best point (measured: +3% on
  a numerai-deep 300-round retrain, covtype-deep 83.8 → 88.2 t/s), while the
  periodic re-probe still verifies the cached choice against reality. (A
  histogram-pipeline-count knob was considered and rejected: it only affects
  the per-pair fallback path — the batched flow every real workload uses
  runs on a single stream.)

All four compose: **+10.5% on numerai-deep combined**. A methodology note the
battery re-taught us: 100-round probe cells on fast datasets (year runs 0.4s)
sit inside clock/thermal noise — the year "regressions" the battery first
reported all vanished under interleaved A/B at 500 rounds.

The ablation shows each of these within noise on shapes they don't target —
the planner's "default on, individually ablatable" contract in action.

## 7b. Runtime-JIT construct kernels (`construct_jit`)

The NVRTC infrastructure from the July arc (shape-keyed compile cache, AOT
fallback, self-test-then-promote) now serves ALL mask-free quantized dense
shapes, not just the compact view. The specialized kernel strips the runtime
branches the AOT kernel must carry (feature/bin masks, graph state,
speculative sizing, wide-partition predication) — on an issue-bound kernel
those branches are the remaining fat. Measured (bit-identical everywhere;
the canonical 700-round locks reproduce exactly with JIT live):

| numerai-deep | covtype-deep | year | higgs |
|---|---|---|---|
| **+4.0%** | +2.4% | +2.2% | +0.7% |

The numerai number required syncing the JIT template with the evict-first
(`__ldcs`) loads first — an unsynced template measured at parity, which
earlier led to a premature dead-end verdict (since corrected). Default ON
for quantized runs of ≥300 rounds (the ~230ms one-time compile+self-test
amortizes); `construct_jit:on` forces it, unsupported shapes (graph capture,
speculative levels, masked trees, wide partitions) fall back to AOT
automatically.

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

## 10. Categorical features on the hybrid fast paths (phases 1a-2b, 2026-08-02)

Categorical datasets previously fell back to the classic one-split-at-a-time
loop for every hybrid stage, and quantized training refused them outright.
Four commits lifted the whole class (c3b27ae7, d6934128, b235de83, 0a839325):

- **Batched apply** (1a): variable-length categorical bitsets travel through
  the fixed-size batched split inputs via a per-level side-band INNER-bitset
  arena in the data partition. The arena-build kernel constructs bitsets and
  patches most-frequent-bin default directions on device, removing the
  classic flow's three per-split D2H round trips. Tree recording interleaves
  per-split categorical recording with numerical SplitBatch chunks.
- **Selective (grow-then-prune) flow** (1b): applied-record snapshots of the
  inner threshold bins (the finder's per-leaf slab is recycled before
  finalize) plus a categorical replay branch in RebuildFromHostSplits with
  host-built bitsets. Selective vs classic produces identical structure and
  categorical bitsets (leaf-value fp noise only, same class as numerical).
- **Quantized training** (2a): the categorical search runs its per-bin math
  in double either way, so one reader-templatized body serves both
  pipelines; quantized readers unpack the packed int32/int64 integer bins
  per bin (exact) and the writers fill the packed int64 child totals the
  quantized pipeline seeds child leaves from. fixedpoint-vs-none rmse delta
  at 200k rows with card-3 + card-120 categoricals: 0.198356 vs 0.198346.
- **Batched level kernels** (2b): both level find kernels (non-quantized and
  discretized) run the categorical body per (task, pair, role) block; the
  per-slot categorical-threshold slabs grow with the level output buffer.

Measured (RTX 5090, 400k rows x (5 numeric + card-3 + card-150 categorical),
63 leaves, 200 rounds, identical rmse and categorical split counts across
all flows):

| flow | train time |
|---|---|
| classic loop | 1.76s |
| hybrid, per-pair fallback (after 1b) | 1.45s |
| hybrid, batched level kernels (after 2b) | **0.48s (3.7x)** |

Still excluded for categoricals: the one-sync speculative prefix and the
graph loop (two-sync batched carries the win); >256-category features use
the 255 most frequent categories per split on the shared-memory finder
(Init warns) because upstream never wrote a global-memory discretized
finder. Open items tracked on the ROADMAP.

---

## 11. Batched-apply partition overhaul: deferred leaf map + flat-grid kernels (2026-08-02, a23fe633)

Profiling the 92M-row airline-cat deep regime (1023 leaves, depth 10) showed
the level-batched apply's partition kernels at 77% of GPU time -- the
data-partition-memory-bound class first profiled on higgs. Two structural
fixes, both bit-parity (canonical md5 locks reproduce identically):

- **Deferred row->leaf map.** The gen-bit-vector kernel wrote
  `data_index_to_leaf_index` for every row at every level: a 4-byte random
  scatter, one full DRAM sector per row, ~55% of the kernel's traffic at deep
  levels. Every consumer of the map is a tree-end operation, so it is now
  written once per tree by `MaterializeLeafMapKernel` from the final leaf
  windows (the selective flow materializes from the final classic layout,
  replacing its remap).
- **Flat-grid apply kernels.** The 2D `(largest leaf's blocks x num_splits)`
  grid is mostly empty blocks at skewed deep levels (millions per launch).
  Host-launched levels now run a 1D grid-stride loop over the level's real
  chunk count with a binary-searched `(descriptor, local block)` mapping
  (`flat_block_start` prefix in the descriptor). The graph-captured device
  loop keeps the controller-resized 2D form.

Airline-cat, 500 rounds, RTX 5090 (xgboost native-categorical as reference):

| regime | falcata-noquant | falcata-stoch | xgboost |
|---|---|---|---|
| deep (1023 leaves) | 95.1s -> **47.0s** | 93.4s -> **41.7s** | 43.8s |
| shallow (63 leaves) | 28.7s -> **23.5s** | 26.8s -> **20.5s** | 22.8s |

Per-level partition cost fell from ~17.4ms to ~2.2ms; the remaining per-tree
profile is construct 26.5ms, leaf-map materialize 17ms, partition 22ms,
fixed-overhead kernels ~9ms, plus ~30ms host sync gaps (the two-sync flow's
per-level readbacks -- lifting the one-sync categorical exclusion is the next
lever if more is needed).

---

*Footnote on the 2026-07-30 ablation snapshot: the numerai-deep
`batch_kernels` row is marked "MD5 DIFFERS" in the raw table; this was
subsequently investigated and verified benign — the fallback path breaks
exact-gain ties in a different order at 5.4M-row scale (holdout corr
identical to 5 decimals). It is reclassified as a tie-break key, not an
equality key (commit `2f30f043`).*
