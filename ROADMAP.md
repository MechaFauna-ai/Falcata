# ExaBoost CUDA roadmap

Ideas from the hybrid-growth + benchmark session (2026-07-13/14, PRs #32/#33), roughly
priority-ordered within groups. Measurements refer to an RTX 5090, per-tree/per-100-tree
figures from the profiles in the PR discussions.

## Performance

- [ ] **Graphs L1 — device-driven level loop.** Wrap the per-level pipeline in a CUDA
  conditional-WHILE graph node; a level-controller kernel builds the next level's pair
  descriptors on-device, updates grid dims via device-updatable node params, and sets the
  loop condition; ONE host sync per tree. Est. ~1.6-1.7x on small-tree workloads
  (fraud-class: ~344 us/tree turnaround idle + ~234 us launch API measured), ~1.3x mid
  (covtype/year), ~nil on roofline-bound data (higgs/epsilon). Depth-limited exact regime
  first; the budget-limited selective mode keeps its host loop until the selection +
  tie-break replication is ported to device and re-verified against the quant md5 gates.
- [ ] **Graphs L2 — per-parent dependency chaining.** A child pair only depends on *its
  parent's* partition + histogram, not on its level. Launch each pair's chain from the
  device (fire-and-forget device graph launch) when its parent finishes — a
  self-scheduling tree that removes phase-boundary tail effects on unbalanced trees.
  Needs: device-atomic budget reservation + a global final-level top-K join, per-node
  (not per-level) task/scratch regions, and an order-independent split log (the host
  rebuild already renumbers canonically, absorbing completion-order nondeterminism).
  Est. +10-20% over L1 on unbalanced small trees.
- [ ] **Static planner (auto-tuner tier 0).** At Init the dataset shape (rows,
  features, actual bins/feature) and config (num_leaves, max_depth, num_class,
  iterations) determine: expected level geometry (leaf sizes ~ rows/2^level) -> grid
  configs, small-leaf threshold, speculative bounds; histogram partition packing and
  quant bit thresholds from the real bin histogram; pipeline selection; and a compact
  per-tree histogram layout for feature_fraction runs (static version of the 161fe88b
  dead-entry mask). Precedent: CPU LightGBM's row/col-wise chooser, EFB, multi-val bin
  packing. Decides only provable shape functions; supplies priors for the ambiguous
  constants below (GPU cost models are brittle: the construct-floor cap gained
  year/higgs 35% and regressed covtype 45% -- only measurement caught it).
- [ ] **Runtime auto-tuner ("JIT optimizer") tier 1 — online policy tuning.** Boosting
  runs thousands of near-identical trees: measure per-tree wall time (CUDA events) +
  feedback stats (churn, level widths, imbalance) and bandit-tune the existing
  dataset-dependent knobs: batched-construct grid sizing (EXABOOST_BATCH_CONSTRUCT_FLOOR
  -- covtype is latency-bound, year/higgs merge-bound), small-leaf construct threshold
  (EXABOOST_SMALL_LEAF_ROWS), hist pipeline count, and the selective-growth speculation
  policy (e.g. gain-margin gating for unbalanced trees). Speculation is model-invariant
  by monotonicity, and quant-mode integer histograms keep md5 locks valid under any
  schedule retuning -- the tuner cannot break exactness gates. Hysteresis vs noise;
  decision logging.
- [ ] **Runtime auto-tuner tier 2 -- NVRTC shape-specialized kernels.** JIT-compile
  construct/find kernels at Dataset construction with columns / per-feature bin counts
  baked in (precedent: LightGBM's OpenCL backend JIT-compiled with #defined bin counts).
  Star case: numerai's ~5.5-bin features (5 quintiles + a missing-marker bin on ~half
  of them) waste >90% of the fixed 12288-entry shared histogram; a specialized kernel
  packs ~10x more features per partition -> fewer
  partitions, less shared->global merge traffic. One-time ~0.5s compile amortized over
  thousands of trees; needs AOT fallback.
- [ ] **Runtime auto-tuner tier 3 -- persisted tuning cache**: store best-found configs
  keyed by dataset-shape signature (FFTW-wisdom style) so retrains skip exploration.
- [ ] **Selective-growth churn reduction.** covtype 64/12 applies 2.09x the final split
  count (52% displaced-then-pruned). Smarter speculation — e.g. only apply candidates
  with a selection margin / hysteresis — to cut wasted search+apply.
- [x] **Wide-shape batched search** (161fe88b): multi-block batched find/sync for
  num_tasks > 1024 (bit-identical reduction), compact-view batched construct,
  bin-used-mask skipping of dead histogram entries, learner gather from the compact
  matrix. numerai 4096/13: 39.4 -> 27.2s; example shape per-tree 41.6 -> 34.8ms.
- [x] **Numerai first-train setup** (22331b18): Booster create 13.9s -> 2.3s. Dense CUDA
  row data now builds directly from column bins (tiled transpose, no zero-fills) and the
  host multi-val bin is skipped for dense non-multi-val datasets (-14.6 GB peak RSS).
  Remaining: ~0.8s pageable 15 GB H2D (pinned staging ring / transpose into pinned) and
  ~1.2s host transpose (device-side transpose candidate); sparse row-wise CUDA datasets
  still build the host multi-val bin. Kill switch EXABOOST_FAST_ROWDATA=0.
- [x] **Native small-int ingestion (int8/int16)** (9c0f5ffa): int8/int16 numpy
  matrices (row- or col-major) pass zero-copy through the C API and are binned by
  per-column LUTs via Bin::PushBlock; bins identical to the float path by construction
  (md5-verified CUDA quant + CPU deterministic). numerai fed int8: construct 38.9 ->
  15.8s, peak RSS 86.4 -> 43.9 GB. Measured separately from the cross-library matrix
  (which stays float32-fed for fairness). Completed in ff77c701: uint8/uint16
  dtypes, pandas plain-small-int frame passthrough (nullable/float/categorical keep
  the float path), and float16 via a 65536-entry bit-pattern LUT (exhaustive
  all-pattern md5 gate). Remaining: EFB FindGroups is now the dominant construct cost
  (~9s of 15.8s on 2748 features; see EFB density precheck below); CSR/CSC +
  streaming push still use the double path.
- [x] **4-bit packed CUDA row/compact data for <=16-bin datasets** (354ceef3): row
  matrix AND per-tree compact matrix packed two cells/byte; all construct/fill/gather
  kernels unpack via IS_4BIT variants. numerai: 2000-tree train 87.6 -> 66.3s (-24%),
  device memory 19.3 -> 11.5 GB, Booster create 2.25 -> 1.44s, gather source at 1.47
  TB/s. Kill switch EXABOOST_ROWDATA_4BIT=0; verify env checks unpacked equality.
  Both follow-ups landed in 45eea4c6 (numerai per-tree 34.4 -> 30.7 ms, -10.6%;
  2000-tree ~60s): (a) float2 grad/hess interleave (EXABOOST_GH_INTERLEAVE,
  compile-time-template dispatch -- a runtime branch cost ~1 ms in both modes),
  modest (-0.5..1 ms) since grad/hess was the smaller half of scattered traffic;
  (b) split kernels read the packed compact matrix directly
  (EXABOOST_SPLIT_PACKED_READ), dropping the per-tree 1.5 GB row-to-col gather
  (-4.2 ms/tree); classic per-split consumers lazily materialize the old view once
  per tree. Next on this path: CUDAFillCompactData4BitKernel is now the #2 kernel
  (5.3 ms/tree); the construct kernel (22.7 ms, ~74% of tree time) remains
  latency-bound on the packed-bin row gathers themselves. Measurement hygiene:
  desktop wall clocks drift up to ~15% between sessions -- trust same-session A/B
  per-tree medians (or lock clocks) for future numerai perf work.
- [x] **GPU-native dataset construction for device_type=cuda + CuPy ingestion**
  (ca5bca30 GPU dense binning, 633bba9e __cuda_array_interface__). Sampling + host
  GreedyFindBin kept (identical boundaries); dense push binned ON DEVICE via pinned
  staging ring (raw never fully resident), packed bins copied back so all host
  consumers work unchanged; byte-identity verify env; cupy-cuda12x works on sm_120.
  numerai construct: f32 37.5 -> 7.2-7.8s, int8 15.2 -> 4.5-4.9s, cupy int8 3.4-3.8s;
  higgs 0.65s, epsilon ~3s. Kill switch EXABOOST_GPU_CONSTRUCT=0 (that path is also
  ~7s faster now: md5-safe parallel FeatureGroup creation + column-parallel sampling
  are unconditional). Follow-ups: device-side bin finding (~0.9s), single pinned bin
  slab (~0.5s), stage 3 = device-resident-only bins without host copy-back (pairs
  with removing CreateCUDAColumnData's re-upload); cuDF users pass .values.
- [x] **EFB density precheck** (c9b00820): skips the ~7.7s no-op FindGroups on dense
  data (numerai/higgs/epsilon/fraud), provably refuses where bundling exists
  (covtype, sparse); group-structure verify env; EXABOOST_EFB_PRECHECK=0.
- [ ] **Latency-bound construct on tiny-bin wide data**: post-161fe88b numerai construct
  is scattered-read latency-bound (19ms/tree) -- candidate for NVRTC shape
  specialization (auto-tuner tier 2) or layout changes.
- [ ] **Quant one-sync parity.** The quantized path still uses the two-sync level flow;
  extend the one-sync speculative pipeline to it. Also investigate the per-tree gradient
  discretization cost on many-tree/small-tree configs (numerai-quant is slower than
  non-quant today).
- [x] **FP32 gain (EXABOOST_FP32_GAIN) + FP32-atomic histogram (EXABOOST_FP32_HIST)
  modes** (161dc901). Measured per-tree wins at equal-or-better quality on dense
  hist/find-bound data: epsilon deep -36% (both flags), year -18%, covtype -16%,
  fraud deep -14%, higgs deep -12%; numerai flat (feature_fraction-dominated).
  Defaults OFF: covtype shows a seed-noise-scale accuracy dip and numerai gains
  nothing -- the auto-tuner should own per-shape enablement. fp32-hist auto-falls
  back to fp64 for quant/sparse/large-bin/categorical/forced-splits/NCCL/global-mem
  finder. Follow-up: fp32 hists still occupy fp64-sized slots (bandwidth halved,
  memory unchanged) -- shrink the pool arithmetic to also halve hist memory.

## Correctness / determinism

- [ ] **Deterministic non-quant CUDA mode** (deprioritized: determinism is a
  verification tool here, not a production requirement -- quant mode already provides
  it for md5 gates). Fixed-point integer histogram accumulation (XGBoost-style) would
  eliminate float-atomic run-to-run jitter (measured +-2.6pp on covtype multiclass).
  Related: `deterministic=true` silently no-ops on CUDA — make it work or warn; it also
  doesn't pin the timing-based col/row-wise auto-choice on CPU (bimodal md5s; pin
  force_col_wise in gates).

## Upstream (lightgbm-org/LightGBM) bugs found (documented here for reference;
## we do not contribute upstream)

- Quantized CUDA int32 histogram-index overflow on wide data (fixed here in
  6f8402f5; upstream segfaults at scale and silently corrupts below it).
- Quantized multiclass per-tree random-offset buffer overrun (fixed here in
  1b28ba03).
- CUDA-vs-CPU growth parity: upstream CUDA over-grows trees ~3.5x vs its own CPU at
  identical params (854 vs 237 leaves/tree on covtype; lambda_l2 sweep shows it is
  systematic).
- Latent race in the classic loop: child leaf-splits structs point into per-split
  scratch that the next split overwrites; masked only by per-split syncs (fixed here
  via point_structs_at_main + copy-event ordering).

## Benchmark

- [ ] Airline (115M rows): kt.ijs.si down; retry, find a mirror, or substitute the
  benchm-ml 10M-row variant.
- [ ] Optional: xgboost-lossguide as a seventh benchmark config (leaf-wise
  apples-to-apples column).
- [ ] Harness: host-memory guard for runs near RAM limits (xgboost/numerai peaked at
  102 GB and OOM-killed the host session; its curve cell is skipped).
