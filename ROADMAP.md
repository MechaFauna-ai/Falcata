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
  the classic loop for several feature classes; scoped 2026-08-01.
  Categorical features LANDED 2026-08-02 (phases 1a c3b27ae7, 1b d6934128,
  2a b235de83, 2b 0a839325): batched-apply bitset arena, selective-flow
  replay, quant-categorical finder, batched level kernels — measured 3.7×
  vs classic on 400k×(5 num + 2 cat); details in docs/performance.md §10.
  Remaining categorical items:
  - One-sync speculative prefix + graph loop keep categorical exclusions
    (two-sync batched is live; lift on measured demand).
  - Categorical fingerprint gate: add a quant-categorical lattice/canonical
    cell so the quant-cat path is regression-locked like the numerical one.
  - >256-category features: shared-memory finder considers the 255 most
    frequent categories per split (Init warns); exact full-cardinality
    support means a discretized global-memory finder (upstream never wrote
    one — the branch was an empty TODO).
  - NCCL multi-GPU level batching (ONE all-reduce per level): promising but
    unverifiable on this single-GPU box; revisit with a rented multi-GPU
    instance.
  - Interaction constraints / select_features_by_node (per-node masks via
    the bin-used-mask machinery), forced splits, linear trees: fallback-
    covered, no measured demand; lift on request.
## Inference

## Serialization

- FALB binary model format -- plan in docs/design/binary-model-format-plan.md.
  M1 + compression LANDED 2026-08-02 and MEASURED on a real numerai production
  artifact (45k trees, 3555 features): 471.6 MB text -> 45.6 MB zlib-6 default
  (10.3x, bit-identical predictions, load 0.188s -> 0.160s), 29.6 MB with
  opt-in f32 leaves (15.9x, ~2.7e-08 rel), 65.5 MB uncompressed/mmap-able.
  gzip -6 of the text is 148 MB, so the default is 3.2x better than gzip. Not
  the 31x originally projected -- that assumed a far more verbose text per
  node; see the measured table in the plan. leaf_value f64 is incompressible
  (1.15x shuffled) and was 54.8% of the raw file, which is exactly why f32
  leaves exist as an option and why nothing else would have moved the number.
  - Reserve a per-model ``leaf_dim`` in the v1 leaf-value layout (leaf_value is
    [num_leaves x leaf_dim]); writer emits dim=1 until vector-leaf multi-target
    lands, but the format must not need a v2 for it. Same reservation for linear
    trees (flag bit + section id): both are on this roadmap, so "refuse forever"
    is not an option -- v1 readers must reject them cleanly, not misparse.
  - Per-array dtype tags instead of boolean flags: every tree-data array header
    carries a dtype enum (u8/u16/u32/f32/f64, room for f16/bf16). New precisions
    become additive; old readers fail with "unknown dtype", never misread.
  - Thresholds: per-feature DICTIONARY of the distinct doubles occurring in the
    trees, nodes store an index (u8/u16/u32 by dictionary size). Supersedes the
    bin-index + verify-on-write design, which could not work: the bin bounds it
    meant to reference are not in the model at all (``feature_infos`` is only
    ``[min:max]``, bin.h:236 -- the bounds live in the Dataset), so text-loaded
    models could never re-encode. The dictionary is exact by construction (no
    grid to fall off, so no whole-model f64 fallback for one refit/text-edited
    threshold), Dataset-free, and never larger than the bin table it replaces.
    Raw-f32 thresholds remain an IMPORT fidelity mode (XGBoost is f32-native, so
    f32 is exact there), NOT a compression knob: on f64 thresholds it moves
    decision boundaries.
  - Sections split three ways: core (structure + thresholds + leaf values),
    structural stats (internal/leaf counts + weights -- TreeSHAP / pred_contrib
    reads data_count() on every internal node, trees_to_dataframe and
    dump_model need them), training diagnostics (split_gain, internal_value --
    opt-in, clear error from feature_importance("gain") when absent). Stats are
    BEHIND A FLAG in M1, not default-on as first amended: naive encoding adds
    ~37 MB to a ~21 MB core on the numerai reference (~1.5M internal + ~1.55M
    leaf entries), so the headline is a core-only number. Default-on is
    reconsidered in M3 against a measurement, with f32 weights (hessian sums;
    leaf VALUES stay f64), varint counts and zstd on that section.
  - Codec default raw, so core sections stay mmap-able and per-tree access is
    O(1); zstd opt-in for archival. The loader is an untrusted-input parser:
    every section offset/length bounds-checked against real file size before
    any read, clean failure never UB, with the fuzzing as verification.
  - M1 includes categoricals (the 2026-08-01 categorical lift makes them a
    headline feature; a format that cannot store an airline-class model is not
    shippable). Importers M4/M5 are independent of the format -- they can build
    models through canonical text -- so they need not wait on it.
  - Header carries the exact predict-semantics set, enumerated: objective string
    verbatim (sigmoid, quantile alpha, ...), num_class / num_tree_per_iteration,
    average_output, init/base score, missing-handling flags, feature names +
    feature_infos. Plus the full param blob (tiny) so continued training /
    refit from a FALB-loaded booster either works or refuses explicitly --
    "predict-only params" would break init_model continuation silently.
  - Pickle flips to FALB only with an escape hatch (falcata.set_pickle_format /
    env var) and __setstate__ accepts text-payload pickles forever. New pickles
    are unreadable by stock lightgbm and older Falcata by design; numerai prod
    stays *.txt.gz until the container ships Falcata (plan section 8).
  - C API is FLC_BoosterSaveModelBinary / FLC_BoosterCreateFromBinary with
    LGBM_* shim aliases (plan predates the rename). Python/C++/CLI only in v1;
    R/SWIG explicitly deferred.
  - Fixed-width section-relative per-tree offsets (not varint): O(1) tree access
    keeps mmap + start_iteration/num_iteration slicing cheap. Little-endian
    declared, raw sections 8-byte aligned.
  - Text writer stays byte-stable: canonical md5 gates hash model text, and
    treelite/FIL + stock-lightgbm interop consume it. FALB is purely additive;
    the stock-interop CI guard (plan section 6.2) is mandatory before M2 ships.
  - Importer caveats to settle with parity tests: xgboost strict-less-than vs
    our <= (nextafter(t, -inf) is exact on f64; validate against f32 input
    pipelines), xgboost transformed base_score -> raw margin, catboost
    scale/bias, pinned source-library versions in CI.

## Correctness / determinism

- OPEN: hybrid categorical corruption at max_cat_threshold >= 48 (FENCED to
  classic since 2026-08-02). Symptom: leaf-cache cat_threshold entries contain
  invalid bins (e.g. bin 255 for a 255-bin feature) at small indices with
  plausible counts -> OOB reads of categorical_bin_to_value (the batched
  recording kernel now crashes loudly where the old per-split path consumed
  the same garbage silently). Bisected on airline-cat 5M subsample, 63
  leaves/depth 6, noquant: classic CLEAN; hybrid DIRTY with both the batched
  level finder AND the per-pair finder (batch_kernels:off), one-sync and
  two-sync alike; onset between max_cat_threshold 32 (clean) and 48. Repro:
  scratchpad crash_edge.py. Suspects: pipelined pair searches sharing the
  task-out cat slabs, or the level-capacity slab re-allocation; per-pair-
  finder involvement points at the former.

- Airline-cat quality gap vs upstream lightgbm CUDA: at identical params
  (500r, both regimes) upstream posts 0.850/0.886 AUC where we and xgboost
  sit at ~0.843/0.873. NOT fp32 gain (fp64 measured identical: 0.87221 vs
  0.87231). Suspects: categorical split-selection defaults or the hybrid
  level-batched top-K selection differing from pure leaf-wise on this data
  shape. Worth a bisect: same seed classic-vs-upstream split-by-split on a
  subsample.

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
- CPU split finder infers row counts from hessians, which quantization
  invalidates (found 2026-08-02 via the nightly fuzz; upstream 4.7.0 reproduces
  identically, so inherited). ``FeatureHistogram`` recovers per-bin counts as
  ``cnt = RoundInt(hess * num_data / sum_hessian)`` instead of counting rows --
  exact only while every row contributes the same hessian to the histogram.
  Quantized gradients break that for EVERY objective (stochastic rounding
  randomizes even L2's constant hessian once binned), so the inferred count
  drifts from the real one and either (a) reaches num_data, leaving an
  estimated 0 rows on one side -- ``min_data_in_leaf`` gates the ESTIMATE, so
  the split passes and ``CHECK_GT(count, 0)`` then aborts training in
  serial_tree_learner 886/898 -- or (b) leaves no candidate clearing the guard,
  so boosting stops early with fewer trees than requested. Repro (both
  symptoms, falcata and upstream):
  ``tests/gates/fuzz.py --spec`` on any corpus entry with device_type=cpu +
  quant_mode=stochastic. OPEN in this fork: reachable by any user setting
  quant_mode with device_type=cpu. The nightly fuzz classifies the signature as
  KNOWN rather than failing on it (~238 hits/run were drowning the gate); fix
  would be to carry true counts in the quantized CPU histogram, or to fence the
  combination. We do not develop the CPU path, so this is parked, not planned.
- Latent race in the classic loop: child leaf-splits structs point into per-split
  scratch that the next split overwrites; masked only by per-split syncs (fixed here
  via point_structs_at_main + copy-event ordering).
- The whole CUDA >256-bin family (any feature with more than 256 histogram bins:
  max_bin > 256, or a categorical with > 256 non-trivial categories) was broken
  upstream, five distinct defects (all fixed here 2026-08-01):
  (1) ``ShufflePrefixSumExclusive`` indexed the warp hand-off buffer by warpLane
  instead of warpID -- every warp overwrote one slot and lane 0 read shared memory
  at index -1 (uint underflow), so every global-memory prefix scan was garbage;
  the same helper corrupts the weighted L1/quantile leaf-renewal kernels.
  (2) ``GlobalMemoryPrefixSum`` had no trailing barrier while consumers read the
  scanned array with a different thread mapping than the writers.
  (3) The global-memory reverse scan visits bin num_bin-1 (mfb_offset 0), encoding
  threshold -1 (0xFFFFFFFF); when it wins, the host indexes bin_upper_bound with it
  (garbage split or segfault). The shared-memory kernel has the missing bound.
  (4) ``FixHistogram(Discretized)Inner`` read one bin per thread with a 512-thread
  block: bins >= 512 were excluded from the partial sum, so the most-frequent-bin
  patch re-injected their mass (histogram total inflated by exactly the missed
  tail; num_bin_aligned > blockDim also made the reduction read stale shared mem).
  (5) Quantized + global-memory finder: upstream's launch branch is an empty
  ``TODO(shiyu1994)`` -- no kernel runs, training silently uses stale split info
  (we now refuse with an actionable error).
  Additionally the forward+reverse tasks of one feature share a staging region
  across concurrent blocks (fixed here with per-direction regions), and the hybrid
  per-pair fallback overlapped pairs on the shared staging (hybrid now routes
  global-memory shapes to the classic flow).

## Benchmark

- Airline (115M rows): kt.ijs.si down; retry, find a mirror, or substitute the
  benchm-ml 10M-row variant.
