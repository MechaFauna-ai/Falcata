# ExaBoost CUDA roadmap

Ideas from the hybrid-growth + benchmark session (2026-07-13/14, PRs #32/#33), roughly
priority-ordered within groups. Measurements refer to an RTX 5090, per-tree/per-100-tree
figures from the profiles in the PR discussions.

## Performance

- [x] **Graphs L1/L1.5/L2 arc — device-driven level loop** (99734bae -> a1d4aa43
  unroll -> 2e047b0d grid-update cache -> c6628cdf per-body controller sizing ->
  2f37bd37 single-sync post-graph readback). Conditional-graph design shipped in
  full, then iterated: cumulative same-session A/B vs host loop: fraud 63/6 -11.9%,
  fraud 1023/10 -10.9%, covtype 63/6 -8.8%, year -5.8%, numerai parity
  (construct-bound). Launch API 132 -> ~30 us/tree; controller ~42 us/tree; sync
  D2H 3 -> 1 per tree. Non-quant depth-limited only; EXABOOST_GRAPH_LEVEL_LOOP=0.
  Multiclass crash ROOT-CAUSED AND FIXED (84db39cd): device-updatable graphs
  must be cudaGraphUpload-ed before first launch (the controller's first device
  update raced the lazy upload; multiclass's per-class instances multiplied
  exposure). 40/40 soak, gate lifted, multiclass graph ON ~4% faster. A2 done (199dc6fa): all 10 roles
  pow2-bucketed with device cur_n guards -- but the SetGridDim hypothesis was
  FALSIFIED (the 2e047b0d cache already absorbed them; a zero-update body still
  costs 5.2us vs the 2.0us epilogue). Controller cost is intrinsic serial latency:
  dependent descriptor-chain reads + ~20 block barriers. Next lever: prefetch the
  read chains + single-warp fast path for bodies <=32 splits (~20us/tree bound);
  bucket-shrink hysteresis if flipping shows up. cross-tree
  overlap needs a GBDT-contract change (host tree mirror feeds
  CUDATree::Shrinkage sizing + the leaf-wise tail arbitration -- call chain
  documented); per-parent chaining (original L2 idea) still open.
  Quant graph support landed bit-exact (a2279763: device hist-bits, guarded quant
  construct grids, full lock matrix incl. multiclass-quant) but is OPT-IN
  (EXABOOST_GRAPH_QUANT=1, 5c61a0ed): quant levels are cheap so the controller's
  fixed serial latency nets -10.8% covtype 1023/10 / -7% year; wins only
  fraud-class (+4%). Fixing the controller-latency frontier flips the default.
- [~] **Graphs L2 — per-parent dependency chaining — INVESTIGATED, honest negative.**
  The reclaimable barrier-induced tail idle (apply-end -> first-construct-start gap a
  chained first pair could skip) measures **<1% of tree span** (fraud 0.59%, covtype
  0.77%), far under the 5% bar. The earlier "18-33% potential" was a misread of
  min(apply,search) = apply/search PIPELINING, which per-parent chaining does NOT
  deliver (within a body every pair shares the same apply->construct dependency).
  Not pursued. The one real gap is on covtype full/deep trees (~2.7% apply->construct
  + ~21% apply-INTERNAL idle) and that is intra-apply grid serialization / launch
  bubbles across batched pairs -- a DIFFERENT lever (better grid packing / fewer
  batched-kernel launches per apply), not per-parent barriers. See new item below.
- [~] **Intra-apply grid serialization — INVESTIGATED, honest negative; graph-perf
  frontier now fully mapped.** The batched apply is already a single grid-strided launch
  per stage over all pairs (no per-pair launch loop); measured intra-apply idle is
  covtype-deep 2.88% / numerai-deep 0.10% of tree time -- below the 5% bar, and it's
  dependency-chained kernels (Gen->Aggregate->Inner->TreeStructure) that can't overlap
  without breaking bit-identity, not reclaimable slack. Combined with #25 (per-parent
  <1%) and #32 (compact-layout cliff-doesn't-occur), the graph-perf frontier is MAPPED:
  the only remaining idle is inter-stage + controller SERIAL LATENCY, already
  characterized and rejected (needs the descriptor-prefetch / single-warp-small-body
  work, a launch refactor of uncertain payoff). Graph-perf cheap+medium wins are
  exhausted. (The investigating agent also raised a FALSE 'covtype data drift' alarm from
  a non-canonical gate invocation -- DISPROVEN: canonical gate reproduces 5f4e7bdfff1e /
  0.91952 / 291.6 exactly, data intact. See memory canonical-md5-gates.)
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
  UPDATE (investigated the spec's flagged §4 "highest-value win", compact-layout
  pre-sizing): NOT pursued -- (a) the LRU-eviction cliff it targets does not occur on
  realistic shapes (numerai 1000 trees: 23 instantiations vs 64 cache limit, 0 evictions,
  0 disables; feature_fraction 0.05-0.5 all <64); (b) pre-sizing max_num_compact_cols
  changes block_dim_y -> reorders non-quant float atomicAdds -> NOT bit-identical
  (verified md5 flip). The other 12 tier-0 knobs remain valid. A bit-neutral variant
  needs decoupling block_dim_y from block_dim_x (a launch refactor), only if a future
  very-low-fraction/high-feature workload ever pushes distinct shape keys past 64.
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
  IN PROGRESS (session goal): NVRTC JIT construct infra WORKING (60cfa129:
  cuda_construct_jit.{hpp,cpp}, compile shape-consts->PTX->module, shape-keyed cache,
  AOT fallback, self-tests bit-identity, ~160ms one-time compile; EXABOOST_CONSTRUCT_JIT=1,
  not yet the live batched path). The big win landed via the compact-view-for-quant lever
  (was hard-disabled): numerai-quant construct 2.46x (32/5) / 1.96x (1024/10),
  BIT-IDENTICAL (independently verified ff=0.1 compact on/off both = 8f0f9f915449),
  default-on, EXABOOST_CONSTRUCT_COMPACT_QUANT=0 kill-switch. Phase 3: wire JIT as live
  construct path + score all benchmarks + test whether the bin-cap benches (higgs/epsilon/
  year) have REAL headroom (phase-1 NO-GO was on upper-bound roofline estimates) or are
  genuinely at roofline (then bit-identical no-regression is the honest outcome there).
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

- [~] **Exact small-int bin finding (EXABOOST_EXACT_INT_BINS) — IMPLEMENTED, parked on
  branch `exact-int-bins-wip` (07c70a72), not on the merge branch.** Works: exact
  per-column counts -> exact bins, deterministic, min_data_in_bin on TRUE counts,
  rare-value-gets-own-bin mechanism proven on synthetic (sampled 3 bins -> exact 4).
  Default path byte-identical (covtype 5f4e7bdfff1e verified). BUT no value on numerai:
  (a) construct 4.5 -> 7.9s (EFB still needs the per-row sample gather, so the exact
  count is pure extra scan -- the ~1.6s saving premise was wrong); (b) numerai has NO
  sample-invisible rare values (rarest -1 has >2000 rows), so 0 features gain a bin,
  quality flat/marginally lower. Revive only if: a construct win is found (fuse the
  count into the GPU construct pass, or make EFB consume exact counts), OR a dataset
  with genuinely rare (<min_data, missed by sampling) small-int values appears.
- [ ] **L2 residency tuning (5090: 96 MB L2).** (a) cudaAccessPolicyWindow
  persistence on grad/hess float2 (43 MB) + data indices (22 MB): each level's
  compact-matrix stream (~375 MB) currently evicts them, and the construct kernel is
  latency-bound on exactly those scattered re-reads; (b) evict-first/__ldcs hints on
  bin-matrix loads (zero reuse within a level -- stop polluting L2); (c) note:
  fraud/covtype/year datasets (4/31/46 MB) are already fully L2-resident (why they
  profile latency-bound); (d) deep configs: fp32-hist halves the hist pool 248 ->
  124 MB = from doesn't-fit to mostly-fits L2 -- a cache argument for fp32-hist on
  deep trees on top of the bandwidth one (subtraction re-reads parent hists). Few
  lines each, cleanly A/B-able.

- [~] **Quant quality — PARTIALLY LANDED (3fbe9050 renew fix, 438d8e9e fixed-point mode).**
  (1) renew_leaf multi-block reduction (3fbe9050): RenewDiscretizedTreeLeavesKernel was
  1 block/leaf (SMs idle) -> 16 blocks/leaf grid-strided; kernel 500->67us/tree, renew
  overhead 38%->4% on higgs-shallow. Default-off path (quant_train_renew_leaf) unchanged.
  (2) **fixed-point quant mode (438d8e9e, EXABOOST_FIXEDPOINT_QUANT=1, default OFF)**:
  round-to-nearest at high bins (64) via the exact-int-accumulation packed path --
  near-lossless at any depth, DETERMINISTIC, no per-tree stochastic buffer. VERIFIED
  (independent clean-build A/B): year/deep rmse fp 8.970 == non-quant 8.977 vs default
  quant 9.344, at quant speed (1.81 vs 1.80s). Known limitation: heavily-imbalanced
  fraud/deep regresses (global max|grad| scale is outlier-sensitive) -- hence opt-in;
  a percentile-clipped/per-class scale would fix it (follow-up). CAVEAT ON THE COMMIT
  MESSAGES: the #28 agent worked from a contaminated baseline build and mis-recorded the
  covtype quant lock as 22c0ff5e95de in both commit bodies + follow-ups -- that is WRONG.
  The lock WAS covtype 1023/10 quant GROWTH=1 = 5f4e7bdfff1e / GROWTH=0 = fcb9f6c2ab87
  (do NOT propagate the bogus 22c0ff5e95de). RE-BASELINED by #13 (tolerance gain tie-break,
  2026-07-16): its CPU-parity plateau tie-break makes CUDA pick CPU's lowest-index bin on a
  gain plateau instead of the FP-noise bin, moving covtype to GROWTH=1 = 1bfd2d7aed5f /
  GROWTH=0 = 26852449fbac, quality 0.91952->0.91800 (plateau-choice delta, not a regression;
  numerai unchanged at 763c75c0d9cb; TreeSHAP 0.048->1e-16, OOS 0.45->1e-17). These
  (1bfd2d7aed5f / 26852449fbac) are now the canonical covtype locks.
  fixed-point outlier-robust scale LANDED (d24ab00b, EXABOOST_FIXEDPOINT_ROBUST, default-on within fixedpoint): gap-gated bulk re-anchoring recovers fraud/deep 0.940->0.973 (near non-quant 0.975), balanced cases bit-identical, speed-neutral, deterministic -- the fixed-point mode is now complete for both balanced and imbalanced data.
  Quant one-sync parity INVESTIGATED -> honest-negative (parked on branch
  one-sync-quant-wip, EXABOOST_HYBRID_ONE_SYNC_QUANT opt-in): bit-correct but 1-4%
  slower -- the quant per-level sync is already cheap/overlapped and the speculative
  construct's parent-bounded grid costs more than the sync saves. Real lever (shared
  with graph-quant): a tighter device-side CHILD-size construct grid (parent-bounded
  is ~2x the smaller child).
  Constant-hessian regression special case: CONFIRMED STRUCTURAL NEGATIVE (two
  agents; second one built a working prototype + measured deep configs). Can't drop a
  histogram tier: the gradient field is SIGNED int16 (histogram_constructor.cu:1122,
  static_cast<int16_t>(packed>>16)), so signed grad ±(bins/2)*nd needs the SAME width
  as unsigned hess bins*nd -- the /2 and the sign bit cancel. Even FORCING tier drops
  (numerai-deep: 564 leaves 64->16-bit) yields <0.2% construct delta because construct
  is bound by the per-row BIN READ + shared-hist update, NOT accumulation width, and
  the batched launch is gated by the largest leaf. Count-mode is a bit-identical no-op.
  Broader lesson: construct (92.5% of GPU) is bin-read-bound -- accumulation-width and
  sync ideas are tapped; real construct levers are bin-byte reduction (4-bit packing +
  int8 ingestion, both DONE) and partition efficiency (NVRTC shape-specialization,
  tier-2). bins=2 max-aggressive mode remains a separate future lever.)
  num_grad_quant_bins default DECIDED: keep 4. With fixed-point mode covering the
  near-lossless (XGBoost ~30-bit) end, the two quant modes are deliberate EXTREMES
  (aggressive bins=4 stochastic vs fixed-point near-lossless); bumping to 16 was only
  a patch for the aggressive mode's quality, now served by switching mode. Keeps modes
  distinct + upstream-compatible. Deep/small-leaf quality is a documented characteristic
  of the extreme mode -> point users to fixed-point/non-quant. fixed-point outlier-robust scale;
  constant-hessian special case for regression quant; quant one-sync parity.
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

- [x] **FIL soft-integration for Booster.predict** (013c77ed): with cuml-cu12
  installed and device_type=cuda, predict() routes through cuML FIL/nvforest via a
  fully in-memory treelite handoff (no temp files); numpy/CuPy in -> matching
  residency out; cache invalidated on model change; opt-outs use_fil=False /
  EXABOOST_FIL=0; graceful CPU fallback (pred_contrib/pred_leaf/exotic postprocessors
  stay CPU). numerai predict 0.90s (CPU 32T) -> 0.046s (cupy-in), higgs 0.37 ->
  0.004s; host-in loses only on very wide inputs with small models (PCIe-bound).
  Default fp32 (EXABOOST_FIL_PRECISION=double is bit-exact; fp32 flips ~0.01% of
  rows one leaf at split-threshold rounding gaps, AUC unchanged). Follow-ups:
  directed-rounding of thresholds in the treelite handoff would eliminate the fp32
  leaf-flip class; width-based host-input heuristic to auto-skip FIL on
  numerai-shaped host inputs.
- [x] **In-training validation eval: already device-side** (verified, no work
  needed): CUDAScoreUpdater covers valid sets, CUDATree applies trees to device
  valid scores each iteration, rmse/l2/binary metrics reduce on device (auc/multi
  fall back to CPU at eval points). Measured 0.22 ms/iter (~2.6%) at numerai
  curve-flow scale.

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
