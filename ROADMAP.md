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
  Star case: numerai's ~7-bin features waste >90% of the fixed 12288-entry shared
  histogram; a specialized kernel packs ~10x more features per partition -> fewer
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
- [ ] **Numerai first-train setup (~14s)**: dataset->CUDA init dominates short runs on
  the example shape; outside the search path. Profile and trim (row-data upload,
  compact gather warmup).
- [ ] **Latency-bound construct on tiny-bin wide data**: post-161fe88b numerai construct
  is scattered-read latency-bound (19ms/tree) -- candidate for NVRTC shape
  specialization (auto-tuner tier 2) or layout changes.
- [ ] **Quant one-sync parity.** The quantized path still uses the two-sync level flow;
  extend the one-sync speculative pipeline to it. Also investigate the per-tree gradient
  discretization cost on many-tree/small-tree configs (numerai-quant is slower than
  non-quant today).
- [ ] **FP32 gain math in the find kernels.** find ~= 136 us/tree, FP64-bound and
  leaf-size-independent; evaluate FP32/mixed gains behind a flag.
- [ ] **FP32/fixed-point histogram option for wide data.** Non-quant histogram entries
  are FP64 pairs (16 B/bin); epsilon non-quant loses to XGBoost (8 B/bin fixed point)
  for exactly this reason.

## Correctness / determinism

- [ ] **Deterministic non-quant CUDA mode.** Fixed-point integer histogram accumulation
  (XGBoost-style) to eliminate float-atomic run-to-run jitter (measured +-2.6pp on
  covtype multiclass); alternatively bless quant as the deterministic mode. Related:
  `deterministic=true` silently no-ops on CUDA — make it work or warn.

## Upstream (lightgbm-org/LightGBM) bug reports to file

- [ ] Quantized CUDA int32 histogram-index overflow on wide data (fixed here in
  6f8402f5; upstream segfaults at scale and silently corrupts below it).
- [ ] Quantized multiclass per-tree random-offset buffer overrun (fixed here in
  1b28ba03).
- [ ] CUDA-vs-CPU growth parity: upstream CUDA over-grows trees ~3.5x vs its own CPU at
  identical params (854 vs 237 leaves/tree on covtype; lambda_l2 sweep shows it is
  systematic).
- [ ] Latent race in the classic loop: child leaf-splits structs point into per-split
  scratch that the next split overwrites; masked only by per-split syncs (fixed here
  via point_structs_at_main + copy-event ordering).

## Benchmark

- [ ] Airline (115M rows): kt.ijs.si down; retry, find a mirror, or substitute the
  benchm-ml 10M-row variant.
- [ ] Optional: xgboost-lossguide as a seventh benchmark config (leaf-wise
  apples-to-apples column).
- [ ] Harness: host-memory guard for runs near RAM limits (xgboost/numerai peaked at
  102 GB and OOM-killed the host session; its curve cell is skipped).
