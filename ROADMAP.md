# Falcata CUDA roadmap

**Open ideas only.** Landed optimizations are explained (with measured
cross-dataset gains) in [docs/performance.md](docs/performance.md); attempts
that failed live in [docs/perf-dead-ends.md](docs/perf-dead-ends.md) with
their re-open conditions. An idea leaves this file in one of those two
directions — nothing is deleted.

> **Historical note:** entries below may reference `FALCATA_*` environment
> variables. Those were replaced by typed config params (`quant_mode`,
> `quant_bins`, `cuda_precision`) and `cuda_plan` keys; the names are kept
> here as written at the time. See `include/Falcata/falcata_plan.h`.
Ideas from the hybrid-growth + benchmark session (2026-07-13/14, PRs #32/#33), roughly
priority-ordered within groups. Measurements refer to an RTX 5090, per-tree/per-100-tree
figures from the profiles in the PR discussions.

## Performance

- [ ] **Static planner (auto-tuner tier 0).** At Init the dataset shape (rows,
  features, actual bins/feature) and config (num_leaves, max_depth, num_class,
  iterations) determine: expected level geometry (leaf sizes ~ rows/2^level) -> grid
  configs, small-leaf threshold, speculative bounds; histogram partition packing and
  quant bit thresholds from the real bin histogram; pipeline selection; and a compact
  per-tree histogram layout for feature_fraction runs (static version of the 161fe88b
  dead-entry mask). Precedent: CPU Falcata's row/col-wise chooser, EFB, multi-val bin
  packing. Decides only provable shape functions; supplies priors for the ambiguous
  constants below (GPU cost models are brittle: the construct-floor cap gained
  year/higgs 35% and regressed covtype 45% -- only measurement caught it).
  UPDATE (investigated the spec's flagged §4 "highest-value win", compact-layout
  pre-sizing): NOT pursued -- (a) the LRU-eviction cliff it targets does not occur on
  realistic shapes (numerai 1000 trees: 23 instantiations vs 64 cache limit, 0 evictions,
  0 disables; feature_fraction 0.05-0.5 all <64); (b) pre-sizing max_num_compact_cols
  changes block_dim_y -> reorders non-quant float atomicAdds -> NOT bit-identical
  (verified md5 flip). The other 12 tier-0 knobs remain valid. A bit-neutral variant
  needs decoupling block_dim_y from block_dim_x (a launch refactor), only if a future
  very-low-fraction/high-feature workload ever pushes distinct shape keys past 64.
- [ ] **Runtime auto-tuner tier 1 — remaining knobs.** The saturation-floor bandit
  landed 2026-07-31 (`tuner` plan key, default on at >=300 rounds: probe
  {80,160,320,640}, best-of-15, re-probe every 3000 trees — +2.1% numerai-deep,
  +2.7% year; see docs/performance.md). Still open: extending the same bandit to
  the small-leaf construct threshold (mechanism currently hardcoded off), hist
  pipeline count, and the selective-growth speculation policy (gain-margin
  gating) — plus hysteresis vs noise and decision logging for multi-knob runs.
- [ ] **Runtime auto-tuner tier 2 -- NVRTC shape-specialized kernels.** JIT-compile
  construct/find kernels at Dataset construction with columns / per-feature bin counts
  baked in (precedent: Falcata's OpenCL backend JIT-compiled with #defined bin counts).
  Star case: numerai's ~5.5-bin features (5 quintiles + a missing-marker bin on ~half
  of them) waste >90% of the fixed 12288-entry shared histogram; a specialized kernel
  packs ~10x more features per partition -> fewer
  partitions, less shared->global merge traffic. One-time ~0.5s compile amortized over
  thousands of trees; needs AOT fallback.
  IN PROGRESS (session goal): NVRTC JIT construct infra WORKING (60cfa129:
  cuda_construct_jit.{hpp,cpp}, compile shape-consts->PTX->module, shape-keyed cache,
  AOT fallback, self-tests bit-identity, ~160ms one-time compile; cuda_plan=auto,construct_jit:on,
  not yet the live batched path). The big win landed via the compact-view-for-quant lever
  (was hard-disabled): numerai-quant construct 2.46x (32/5) / 1.96x (1024/10),
  BIT-IDENTICAL (independently verified ff=0.1 compact on/off both = 8f0f9f915449),
  default-on, FALCATA_CONSTRUCT_COMPACT_QUANT=0 kill-switch. Phase 3: wire JIT as live
  construct path + score all benchmarks + test whether the bin-cap benches (higgs/epsilon/
  year) have REAL headroom (phase-1 NO-GO was on upper-bound roofline estimates) or are
  genuinely at roofline (then bit-identical no-regression is the honest outcome there).
- [ ] **Runtime auto-tuner tier 3 -- persisted tuning cache**: store best-found configs
  keyed by dataset-shape signature (FFTW-wisdom style) so retrains skip exploration.
- [ ] **Selective-growth churn reduction.** covtype 64/12 applies 2.09x the final split
  count (52% displaced-then-pruned). Smarter speculation — e.g. only apply candidates
  with a selection margin / hysteresis — to cut wasted search+apply.
- [ ] **Latency-bound construct on tiny-bin wide data**: post-161fe88b numerai construct
  is scattered-read latency-bound (19ms/tree). The non-JIT 2-columns-per-thread step
  landed 2026-07-31 (`wide_partitions`, +8.8% numerai-deep); the remaining headroom
  (~10x feature packing with per-feature bin counts baked in) is NVRTC tier 2.
- [ ] **Multi-target training (not urgent, per Felix).** Two variants: (1)
  round-robin one-tree-per-target (multiclass machinery minus softmax) -- identical
  models to sequential training, but only ~1.1x/target now that construct is cheap;
  API-convenience tier. (2) Vector-leaf trees (the differentiator: nobody ships
  this CUDA-supported on asymmetric trees -- XGBoost's multi_output_tree is
  experimental/hist-CPU-mostly, CatBoost MultiRMSE is symmetric-only): shared
  structure, gain = sum of per-target gains, leaf outputs vectors; RMSE-family
  hessians are identical across targets so hist entries are T grads + 1 shared
  hess -- est. per-tree cost ~1.3-1.6x single-target at T=5 => ~3.6x per-target
  speedup vs sequential, plus the shared-structure regularization numerai folks
  want (MultiRMSE-style). Modeling change: validate per-era, don't assume. FIL
  predict falls back to CPU for vector-leaf models initially (treelite support).

- [ ] **L2 residency tuning, remaining part (5090: 96 MB L2).** Parts (a)
  cudaAccessPolicyWindow persistence (`l2_policy`, +5.3% numerai-deep after
  device-proportional sizing) and (b) __ldcs evict-first bin loads (+8%
  covtype-deep, +10% year) landed 2026-07-31. Still open: (d) deep configs:
  fp32-hist halves the hist pool 248 -> 124 MB = from doesn't-fit to
  mostly-fits L2 -- a cache argument for fp32-hist on deep trees on top of
  the bandwidth one (subtraction re-reads parent hists).

- [ ] **Hybrid coverage extensions.** The hybrid/graph fast paths currently fall
  back to the classic loop for: categorical features (variable-length bitset
  payloads vs the fixed 18-int split slabs -- first one worth lifting), NCCL
  multi-GPU (level batching would mean ONE all-reduce per level instead of one per
  split -- promising but unverifiable on a single-GPU box), interaction
  constraints/select_features_by_node (per-node masks; the 161fe88b bin-used-mask
  machinery is the natural vehicle), forced splits (a depth-wise prescription that
  bypasses gain selection), linear trees (leaf renumbering vs per-leaf model
  bookkeeping). Each lift needs its own verification gates; fallbacks are the
  md5-reference classic loop, so correctness is never at risk.

## Inference

## Correctness / determinism

- [ ] **Deterministic non-quant CUDA mode** (deprioritized: determinism is a
  verification tool here, not a production requirement -- quant mode already provides
  it for md5 gates). Fixed-point integer histogram accumulation (XGBoost-style) would
  eliminate float-atomic run-to-run jitter (measured +-2.6pp on covtype multiclass).
  Related: `deterministic=true` silently no-ops on CUDA — make it work or warn; it also
  doesn't pin the timing-based col/row-wise auto-choice on CPU (bimodal md5s; pin
  force_col_wise in gates).

## Upstream (MechaFauna-ai/Falcata) bugs found (documented here for reference;
## we do not contribute upstream)

- Packed 16+16-bit quantized histogram overflow (fixed here in 3afe7c62): with
  num_grad_quant_bins >= 16, blocks accumulating > 65534/bins rows into one shared
  bin carry the hessian field into the gradient field (signature dg=+K,
  dh=-K*65536) -- silent binary-objective collapse at >~2.2M rows; upstream shares
  the kernels.
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
