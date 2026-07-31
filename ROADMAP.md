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

- **Static planner (auto-tuner tier 0).** At Init the dataset shape (rows,
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
- **Runtime auto-tuner tier 1 — remaining knobs.** The saturation-floor bandit
  landed 2026-07-31 (`tuner` plan key, default on at >=300 rounds: probe
  {80,160,320,640}, best-of-15, re-probe every 3000 trees — +2.1% numerai-deep,
  +2.7% year; see docs/performance.md). Still open: extending the same bandit to
  the quant small-leaf threshold (landed 2026-07-31 as a fixed 1024-row cut,
  +11.2% covtype-deep -- the bandit could tune the cut per shape) and hist
  pipeline count — plus hysteresis vs noise and decision logging for
  multi-knob runs. (The selective speculation-policy knob is dead: see the
  churn-deferral entry in perf-dead-ends.md.)
- **Runtime auto-tuner tier 3 -- persisted tuning cache**: store best-found configs
  keyed by dataset-shape signature (FFTW-wisdom style) so retrains skip exploration.
- **Multi-target training.** Two variants: (1)
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

- **Hybrid coverage extensions.** The hybrid/graph fast paths currently fall
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
- CPU quantized training (``use_quantized_grad`` on device_type=cpu) produces
  constant/garbage models at num_grad_quant_bins >= 512 on any data tried
  (AUC 0.500 flat; found 2026-08-01 while investigating the CUDA high-bins
  corruption). Not investigated further -- we do not use the CPU quant path.
- Latent race in the classic loop: child leaf-splits structs point into per-split
  scratch that the next split overwrites; masked only by per-split syncs (fixed here
  via point_structs_at_main + copy-event ordering).

## Benchmark

- Airline (115M rows): kt.ijs.si down; retry, find a mirror, or substitute the
  benchm-ml 10M-row variant.
