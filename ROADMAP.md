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

- **Static planner (tier-0, remaining shape priors).** Device-side seeding
  landed 2026-08-01 (SM-scaled tuner candidates + per-(shape,device) wisdom
  cache; see docs/performance.md). Still open: purely shape-derived priors
  decided at Init — expected level geometry (leaf sizes ~ rows/2^level) for
  grid configs and speculative bounds, and quant bit thresholds from the
  real bin histogram. Cost-model caution stands: the construct-floor cap
  gained year/higgs 35% and regressed covtype 45% — only measurement catches
  such reversals, which is why the runtime bandit is the primary mechanism
  and static priors only seed it.
- **NVRTC tier-2, general case (non-compact kernels) — REOPENED 2026-08-01.**
  The compact-quant instance measured at parity and is closed
  (perf-dead-ends.md), but that kernel is the only one the JIT engages —
  higgs/epsilon/year/covtype-class regimes (no feature sampling, no compact
  view) have never run specialized code, so phase 3's real question ("do the
  bin-cap benches have headroom, or are they at roofline?") is untested.
  Work: extend the JIT template to the non-compact discretized + non-quant
  batched construct kernels (bake per-feature bin offsets, column count,
  row stride; sync the template with wide_partitions/__ldcs first), then
  A/B on year/covtype/higgs quant + nonquant. The infra (shape-keyed cache,
  AOT fallback, bit-identity self-tests) is live and reusable.
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

- **Hybrid coverage extensions.** The hybrid/graph fast paths fall back to
  the classic loop for several feature classes; scoped 2026-08-01:
  - **Categorical features — the priority lift, prize now MEASURED: ~4.8×**
    (controlled same-data hybrid vs classic at 1M×30, 255 leaves: 250 vs
    52.5 t/s — this machinery gap is what every categorical workload pays).
    Worse than the fallback: categorical + quantized training is a hard
    REFUSAL ("not supported yet"), so categorical users get neither of the
    fork's two headline wins. Lift plan: (1) variable-length categorical
    bitset payloads through the batched apply path (the fixed 18-int split
    slabs need either a side-band bitset arena indexed per split, or a
    cap-and-fallback for pathological cardinalities); (2) quant support
    needs the categorical split finder taught to read integer histograms
    (same dequant scheme as numerical); (3) own equality gates vs the
    classic loop (the lattice `categorical/nonquant` cell exists; add a
    categorical fingerprint cell once quant-categorical exists).
  - NCCL multi-GPU level batching (ONE all-reduce per level): promising but
    unverifiable on this single-GPU box; revisit with a rented multi-GPU
    instance.
  - Interaction constraints / select_features_by_node (per-node masks via
    the bin-used-mask machinery), forced splits, linear trees: fallback-
    covered, no measured demand; lift on request.
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
