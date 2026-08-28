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
Roughly priority-ordered within groups. Measurements refer to an RTX 5090.

## Performance

- **Multi-target training.** Design spec:
  [docs/design/multi-target-training.md](docs/design/multi-target-training.md);
  vector-leaf plan + landed-V2 notes in
  [docs/design/vector-leaf-plan.md](docs/design/vector-leaf-plan.md).
  Round-robin (variant 1, `objective=multi_regression`) and the vector-leaf
  training core (variant 2 V1/V2: `tree_mode=vector_leaf`, CUDA classic loop,
  plane-per-target histograms, summed-gain finder, vector apply/serialize/
  predict) are LANDED. Open items:
  - Hybrid/one-sync/graph prefix support for vector mode (currently classic
    per-split loop only; the speed thesis lives in the batched flows) and a
    perf measurement vs round-robin at numerai-like T=5.
  - Lift v1 fences on demand: bagging/GOSS, categorical features,
    L1/path-smooth/max_delta_step/extra-trees/monotone/CEGB, fp32 modes,
    quantized training (needs a T-grad packed cell layout), per-target
    boost_from_average bias, T > 16, multi-GPU.
  - Per-era numerai validation of the shared-structure model vs round-robin
    (modeling change: validate, don't assume); SketchBoost-style reduced
    split-gradient hook.
  - FIL predict falls back to CPU for vector-leaf models (treelite has no
    vector-leaf IR).

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
    [num_leaves x leaf_dim]); scalar writers emit dim=1 and vector-leaf models
    now emit their target dimension without requiring a binary-format v2. Same
    reservation for linear trees (flag bit + section id): both are on this
    roadmap, so "refuse forever" is not an option -- v1 readers must reject
    them cleanly, not misparse.
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
  - M2 LANDED 2026-08-02: save_model(format="auto"|"txt"|"falb") picking by
    extension, model_to_binary/Booster(model_bin=), model files loaded by MAGIC
    rather than extension (a renamed model still loads), pickle emitting FALB
    by default with falcata.set_pickle_format("text") / FALCATA_PICKLE_FORMAT
    as the escape hatch and __setstate__ accepting both payloads forever, and
    `python -m falcata convert`. Conversion is LOSSLESS by default (--drop-stats
    / --drop-diagnostics to shrink), because converting an existing artifact
    must not silently cost it pred_contrib and gain importance.
  - M4 LANDED 2026-08-02: falcata.from_xgboost() converts XGBoost gbtree models
    (reg:squarederror, binary:logistic, reg:logistic, binary:logitraw,
    multi:softprob/softmax, count:poisson) from a Booster, sklearn wrapper,
    .json path or parsed dict. Parity vs XGBoost ~2e-7 (float32 rounding) on
    all objectives incl. missing values; gblinear, native categorical splits
    and unsupported objectives refuse explicitly. Two conversion subtleties are
    load-bearing: thresholds must be recovered THROUGH float32 (XGBoost's JSON
    stores the shortest decimal round-tripping a float32, and parsing it as
    float64 reroutes rows in the gap), and multiclass base_score is per-class
    (softmax is only shift-invariant under a uniform shift).
  - M5 LANDED 2026-08-02: falcata.from_catboost() converts CatBoost models over
    numeric features (RMSE, MAE, Quantile, Logloss, Poisson), unrolling the
    oblivious trees into ordinary binary ones. Parity is exact to machine
    precision (~1e-16) since CatBoost's JSON carries float64 borders. Refuses
    multiclass, categorical/CTR features and non-oblivious grow policies. The
    leaf-index bit order (LSB-first over the listed splits) and the
    scale_and_bias convention were both determined by measurement, not docs.
  - M3 DECIDED 2026-08-02 by measurement on the 45k-tree numerai artifact:
    structural stats and training diagnostics stay OFF by default. Each nearly
    doubles the file (core 45.6 MB / 10.3x; +stats 87.9 MB / 5.4x; +diagnostics
    90.8 MB / 5.2x; both 133.1 MB / 3.5x), which is the wrong price for a
    feature most models never invoke. This supersedes the earlier "DEFAULT ON"
    amendment. The default is safe because a stats-less model now REFUSES
    pred_contrib with a clear message rather than returning NaN/Inf.
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

## Multi-GPU (NCCL) -- WORKS for non-quantized training (2026-08-03)

Verified on TWO different pairs, 2M x 50 binary, 200 trees:

| hardware | interconnect | 1 GPU | 2 GPU | auc 1 / 2 |
|---|---|---|---|---|
| 2x RTX 3090 | PCIe SYS (no P2P) | 20.5 ms/tree | 64.9 ms/tree | 0.80870 / 0.80935 |
| 2x A100-SXM4-40GB | PCIe NODE (NVLink INACTIVE) | 14.8 ms/tree | 64.7 ms/tree | 0.80870 / 0.80935 |

Multi-GPU had never been run. Six findings, five fixed:

1. FIXED cross-device illegal access (rank read the full dataset's column
   pointers, which live on another GPU; no peer access, and GeForce forbids
   P2P outright).
2. FIXED CalcBlockDim CHECK on empty leaves (normal when splits come from
   all-reduced histograms).
3. FIXED global child counts never written: split-info slots 16/17 are READ to
   pick the smaller leaf and feed global_num_data_in_leaf_, but nothing wrote
   them, so each rank read stale garbage independently.
4. FIXED the first deadlock: the all-reduce ran on nccl_stream_ without
   waiting on the construct-done event recorded on the CONSTRUCT stream, so it
   reduced half-built histograms.
5. FIXED (worked around) the second deadlock: NCCL's P2P/direct-pointer
   transport hangs this trainer at the first collective. It only showed on the
   A100 pair, because GeForce forbids P2P and so silently used host staging --
   the 3090 success was luck of topology, not correctness. NCCL_P2P_DISABLE=1
   is now set by default (with a warning, overridable); root cause not yet
   found.
6. OPEN, FENCED: quantized gradients + num_gpu > 1 still hangs even with P2P
   off. The global per-leaf histogram bit widths were never maintained past
   the root (now fixed), but that is not sufficient. Refuses at Init.

OPTIMIZATION 1 of 3 LANDED (colsample-aware reduce): with feature_fraction < 1
the per-leaf histogram still spans EVERY feature's bins, but only the sampled
features' bins are ever read, so the all-reduce was shipping (1 - fraction) of
the message for nothing. The live bins are now gathered into a contiguous
staging buffer, all-reduced once, and scattered back. Measured on 2x RTX 3090,
1.5M x 200, 255 leaves: 2-GPU overhead falls from ~68 ms/tree (ff=1.0) to
~48 ms/tree (ff=0.1) -- the byte count drops 10x (202 KB -> 20 KB per
collective) but only ~20 ms of the overhead goes with it.

THE REMAINING COST IS LATENCY, NOT BANDWIDTH: ~254 collectives per tree at
~190 us each. That is what optimization 2 (level-batched all-reduce, ~254 ->
~10 collectives per tree) has to fix; it is now clearly the decisive one, and
bandwidth tricks cannot substitute for it.

PERFORMANCE: 2 GPUs are ~3.2x SLOWER than 1 on BOTH pairs, and the A100 pair
is no better than the 3090 pair despite far faster GPUs -- the cost is the
per-split collective, not compute. NVLink could not be obtained on vast.ai:
the only 2x A100-SXM4 offers report "all links inActive" (2 GPUs sliced out of
a larger node), so NVLink remains unmeasured. Removing the per-split
host-blocking sync via event ordering was measured NEUTRAL (66.6 vs 64.3
ms/tree) -- the classic loop is a serial chain, so there is nothing to
overlap. The real lever is the level-batched all-reduce in
docs/design/nccl-level-allreduce-plan.md.

## Correctness / determinism

- OPEN (minor): NEGATIVE categorical codes -- documented as "treated as
  missing" -- train measurably differently on CUDA vs CPU: fuzz shapes whose
  int8 storage wrapped high category codes negative showed cuda consistently
  worse (+6..35% multiclass NLL) while the same shapes with intact codes agree
  to +-1%. Real data should never hit this (the codes are invalid input), but
  the missing-category routing in the CUDA cat finder evidently differs from
  the CPU one; worth aligning whenever the cat finder is next open.

- FIXED 2026-08-11: fixedpoint rounding bias at low bin budgets, by
  error-feedback accumulation (plan key `quant_ef`, default on): each row
  carries its rounding residual (int8, 1/128-bin units, 2B/row/class) into the
  next tree's quantization, so deterministic round-to-nearest bias telescopes
  away. Under bagging, residuals only move for in-bag rows -- an out-of-bag
  carry "corrects" error that was never injected, and measurably made things
  worse before that gate existed. Fuzz corner seed20260811#431: 6-seed mean
  +15.8% mlogloss vs full precision -> -6.1%; lattice imbalanced/fixedpoint
  AUC 0.9368 -> 0.9626; the 6 fixedpoint lattice cells re-baselined, all 41
  other cells and the canonical (stochastic) locks bit-identical. Remaining:
  symmetric seed variance in that pathological corner (fuzz classifies it,
  hard ceiling 0.35).

- FIXED 2026-08-11, fence lifted: the "hybrid categorical corruption" was the
  out-of-bag/validation traversal (AddPredictionToScoreKernel) running against
  a tree whose categorical bitset arrays were absent on device. ToHost frees
  the per-tree structure arrays and the traversal's restore path re-uploaded
  five of them but not cuda_bitset_inner_/cuda_cat_boundaries_inner_; the
  selective grow-then-prune flow never populates the device copies at all
  (RebuildFromHostSplits builds the tree on host), so a categorical decision
  dereferenced null bases. Restore now covers the categorical arrays, gated so
  a classic-flow tree's live copies are never freed under it.
  Validation on sm_120: the previously-crashing local repro sweep (36 cells)
  and a 9-shape hostile matrix (4095 leaves, min_data 1, mct 8/64, one-hot,
  35% NaN, both quant modes, extreme skew) all pass; hybrid-vs-classic
  prediction parity is at fp-accumulation noise (<2e-6) on every non-bagging
  shape, matching the numerical baseline; airline-cat deep (92M rows, the
  original crash regime) trains clean, AUC .8722; memcheck 0 errors. The
  sm_86 rental crash in BatchConstructCatBitsetsKernel was never reproduced
  on sm_120 before or after the fix -- if it resurfaces, cat_hybrid:off is
  the escape hatch and the fuzzer now generates categorical specs (~25%) to
  keep hunting. Bagging-mode hybrid-vs-classic divergence exists equally for
  numerical and categorical data (butterfly of near-tied gains, quality-
  neutral) and is tracked separately below.

- OPEN: hybrid-vs-classic model divergence under bagging (numerical AND
  categorical equally; max|pred delta| ~0.3 after 40 rounds on tie-heavy
  synthetic data, end quality equal). Not cat-specific and not new -- the
  no-bagging paths agree to fp noise. Likely near-tied gains resolving
  differently after bagged hessian sums; worth a tie-break audit of the
  bagged level flow someday.

- RESOLVED 2026-08-04: the airline-cat "quality gap vs upstream" was never a
  quality gap -- **upstream's CUDA learner silently ignores max_depth**. On a
  same-box, same-data A/B (rented 3090, 5M subsample, 500r): upstream trees
  under max_depth=6 average DEPTH 14.7 (max 20) while ours honor 6.0 exactly;
  its 63 leaves therefore sit wherever gain is best instead of inside 6
  levels. Giving falcata identical semantics (max_depth=-1) matches upstream
  to the 5th decimal: shallow 0.83466 == 0.83466, deep 0.86642 vs 0.86650 --
  at 1.7x/3.7x our speed. Isolation chain: gap present numeric-only (not
  categorical), unchanged by hybrid:off (not the level-batched selection),
  both sides fp64. Filed under upstream bugs below; benchmark comparisons
  against upstream CUDA must either drop max_depth or note that upstream
  does not enforce it.

## Upstream LightGBM bugs found

Documented here for reference; we do not contribute them upstream.

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
  systematic). 2026-08-04: the mechanism is (at least partly) that upstream's
  CUDA learner DOES NOT ENFORCE max_depth at all -- measured depth 14.7 avg /
  20 max under max_depth=6 on airline 5M. This also explained our entire
  apparent airline-cat AUC deficit (see Correctness section).
- Before the CPU quantization fence, upstream and this fork produced
  constant/garbage models with ``use_quantized_grad`` on device_type=cpu at
  num_grad_quant_bins >= 512 on any data tried (AUC 0.500 flat; found
  2026-08-01 while investigating the CUDA high-bins corruption). See the
  resolved count-inference defect below.
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
  so boosting stops early with fewer trees than requested. Historical
  pre-fence corpus entries with device_type=cpu + quant_mode=stochastic
  reproduced both symptoms in Falcata and upstream. FENCED 2026-08-18 in this
  fork: configuration rejects every resolved quantized mode with
  device_type=cpu and directs callers to
  CUDA quantization or CPU full-precision training. The nightly fuzz now uses
  full-precision CPU as the quality reference for quantized CUDA cells instead
  of suppressing the inherited CPU crashes. A full fix would carry true counts
  in the CPU quantized histogram; we do not develop that path.
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
