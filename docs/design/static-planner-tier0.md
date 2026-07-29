# Design spec: static planner (auto-tuner tier 0)

Status: design only. No code in this document. Scope: the CUDA single-GPU tree
learner (`src/treelearner/cuda/*`) and its Dataset-side inputs
(`src/io/cuda/cuda_row_data.cpp`, `src/treelearner/gradient_discretizer.cpp`).

## 1. Motivation

At Dataset construction and learner `Init`, Falcata already knows everything
that fixes the *shape* of the work:

- `num_data` (rows), `num_features`, `num_feature_groups`,
- the ACTUAL per-feature bin counts from the built `BinMapper`s
  (`FeatureBinMapper(f)->num_bin()`), hence `num_total_bin_` and the
  `feature_hist_offsets` prefix sums,
- config: `num_leaves`, `max_depth`, `num_class`, `num_iterations`,
  `feature_fraction`, `use_quantized_grad`, `num_grad_quant_bins`.

Today a scatter of hardcoded constants and threshold logic in the histogram
constructor, the hybrid graph controller and `cuda_row_data.cpp` *guess* things
that are in fact provable functions of that shape. Others are guesses of a
genuinely *measurement-dependent* perf constant. **The whole point of tier 0 is
to separate those two classes.** Tier 0 computes the provable ones exactly, once,
at init; it must NOT try to pick the measurement-ambiguous ones — it only hands
tier 1 (the measured auto-tuner) a *prior*.

### The construct-floor cautionary tale (why tier 0 must be conservative)

`FALCATA_BATCH_CONSTRUCT_FLOOR`
(`cuda_histogram_constructor.hpp:275`, default 160) is the device-saturation
floor of the batched construct grid's total y-blocks. It is exactly the kind of
knob that *looks* shape-derivable — "just set it to fill the SM count". An
earlier attempt built a shape cost model that scaled this floor (and the
per-thread row cap) from `num_data`, bin count and SM occupancy. On `year` and
`higgs` it gained ~35%. On `covtype` it regressed ~45%. The floor trades launch
overhead against tail-block idle time, and the crossover depends on the *runtime*
leaf-size distribution and the SM's actual achieved occupancy — neither of which
is a pure function of dataset shape. A shape formula that is right for one
distribution is wrong for another.

Lesson encoded in this spec: **tier 0 only decides knobs that are bit-neutral
AND provably optimal from shape (layout, packing, eligibility, capacity). Every
performance-crossover constant stays a tier-1 job; tier 0 hands it a documented
prior and nothing more.** The verification gate is bit-identity: the planner may
only move layout/scheduling, never arithmetic. Every existing `FALCATA_*` env
override must remain, so any tier-0 decision is A/B-testable against the
historical constant.

## 2. Knob inventory

Legend for the "class" column:

- **P** = shape-provable. Tier 0 computes an exact closed form; it is either a
  correctness/capacity bound or a bit-neutral layout choice whose optimum is a
  pure function of shape. Tier 0 decides.
- **A** = measurement-ambiguous. The optimum depends on runtime distribution or
  achieved occupancy. Tier 0 does NOT decide it; it hands tier 1 a prior (the
  current constant, annotated with the shape range where it is known good).

| # | Knob | Current value / heuristic (file:line) | Class | Tier-0 formula, or the prior handed to tier 1 |
|---|------|----------------------------------------|:---:|-----------------------------------------------|
| 1 | `construct_reg_bins_` (register-hist ≤8-bin body eligibility) | `reg_hist_enabled && max_num_bin <= 8 && !feature_num_bins_.empty()` — `cuda_histogram_constructor.cpp:98-104`, `kRegHistMaxBins == 8` at `cuda_histogram_constructor.cu:431` | **P** | Already shape-provable and already computed at init from `feature_num_bins_`. Systematize: planner owns `max_num_bin = max_f num_bin(f)` and the predicate `max_num_bin <= kRegHistMaxBins`. Bit-neutral (chooses a kernel body, not arithmetic). Keep `FALCATA_BATCH_REGHIST=0` override. |
| 2 | `shared_hist_size_` → `max_num_bin_per_partition = shared_hist_size_/2` | `DP_SHARED_HIST_SIZE 5176/6144`, `SP_SHARED_HIST_SIZE = 2×` (`cuda_row_data.hpp:25-29`); used `cuda_row_data.cpp:241` | **P** | Pure function of `gpu_use_dp` and the target arch's shared-mem budget. Already a compile/arch constant; planner should read it, not re-derive. Provable, decides the partition packing below. |
| 3 | DivideCUDAFeatureGroups partition packing (`feature_partition_column_index_offsets_`, `partition_hist_offsets_`, `max_num_column_per_partition_`) | greedy fill to `max_num_bin_per_partition` OR `cur_partition_columns >= 504` (half `NUM_THREADS_PER_BLOCK`) — `cuda_row_data.cpp:240-359` | **P** | Fully determined by per-feature bin counts + `shared_hist_size_` + the 504 column cap. Deterministic bin-packing over ACTUAL bins; output is a pure layout with no arithmetic effect. This is *already* a tier-0 decision made implicitly; the planner should own it and expose `max_num_column_per_partition_` / `num_feature_partitions_` as planned facts other knobs read. |
| 4 | 504 column cap (`cur_partition_columns >= 504`) | literal `504` (= half `NUM_THREADS_PER_BLOCK`) — `cuda_row_data.cpp:290,330`; `NUM_THREADS_PER_BLOCK (504)` at `cuda_histogram_constructor.hpp:28` | **P** | Provable: it is `NUM_THREADS_PER_BLOCK/2` so `block_dim_y=2` is possible. Derive from `NUM_THREADS_PER_BLOCK`, do not hardcode `504` in two places. Bit-neutral. |
| 5 | `SetNumBitsInHistogramBin` thresholds (8/16/32-bit histogram width) | `max_stat_per_bin = num_data_in_leaf × num_grad_quant_bins`; `<256 → 8`, `<65536 → 16`, else 32 — `gradient_discretizer.cpp:173-198` | **P** (root) / **runtime** (children) | The bit width is a **correctness bound**, not a perf guess: it is the number of bits needed so a bin's accumulator cannot overflow. The ROOT width is fully shape-provable at init: `root_bits = width(num_data × num_grad_quant_bins)`. Per-child widths depend on the runtime leaf size, so they stay runtime — but tier 0 can precompute the *level-wise upper bound* `width(ceil(num_data / 2^level) × num_grad_quant_bins)` (see §3) to pre-size the graph's frozen-bucket bit-change scratch. Never a bit-identity risk: the width only chooses accumulator storage, results are identical. |
| 6 | `QuantConstructMaxRowsPerThread` / `HybridQuantConstructMaxRowsPerThread` (65534/bins overflow guard) | `65534 / num_grad_quant_bins` then `/ block_dim_y` — `cuda_histogram_constructor.hpp:87-92`, `.cpp:741-744` | **P** | Correctness bound: caps rows-per-thread so the packed 16+16-bit shared accumulator cannot overflow. Pure function of `num_grad_quant_bins` and `block_dim_y`. Already computed on demand; planner should cache the per-`block_dim_y` value at init (the inputs are fixed after init). Bit-neutral (only affects grid sizing / row grouping, results identical). |
| 7 | `kNumHistPipelines` (# concurrent stream/event pipelines) | `static constexpr int = 4` — `cuda_histogram_constructor.hpp:148`; hist scratch sized `num_total_bin_ × 4` at `.cpp:630-632` | **A** (count) / **P** (scratch size given count) | The *number* 4 is measurement-ambiguous (concurrency benefit vs scratch memory vs SM contention) — hand tier 1 the prior "4, good across tested datasets". But once the count is fixed, the scratch sizing (`num_total_bin_ × pipelines`) is shape-provable and tier 0 owns it. Splitting the knob is the key move: capacity is P, the count is A. |
| 8 | `hist_buffer_for_num_bit_change_` sizing | `max(num_total_bin_×2, (num_total_bin_×kNumHistPipelines+1)/2)` — `cuda_histogram_constructor.cpp:630-632` | **P** | Pure function of `num_total_bin_` and the (tier-0-owned) pipeline count. Capacity bound, bit-neutral. Planner owns. |
| 9 | Batched-vs-per-pair level gate (`SupportsBatchedLevel`) | dense && `!is_sparse` && `NumLargeBinPartition()==0` && (compact ⇒ batch_wide && !col_major && bit_type==8) — `cuda_histogram_constructor.hpp:196-203` | **P** | Pure eligibility predicate over the built layout (sparsity, large-bin partitions from knob #3, bit_type). No perf crossover — it is "is this path even correct/supported here". Tier 0 computes it once at init (layout is fixed) and caches, instead of recomputing per tree. Keep `FALCATA_BATCH_WIDE=0`. |
| 10 | `UseOneSyncPrefix` / hybrid-batch eligibility | `!use_quantized_grad && batch_kernels && batch_apply && SupportsBatchedLevel (hist & finder)` — `cuda_single_gpu_tree_learner.cpp:536-543` | **P** | Same character as #9: a shape/config eligibility conjunction, not a perf guess. Precompute at init. All the `FALCATA_HYBRID_*` overrides (`cuda_single_gpu_tree_learner.cpp:84-104`) must remain as the A/B path. |
| 11 | Hybrid-graph eligibility (depth-limited regime) | `max_depth>0 && max_depth<31 && 2^max_depth <= num_leaves+1 && num_leaves <= 2·kHybridGraphMaxSplitsPerLevel-1 && max_depth < kHybridGraphMaxLevels && max_depth < kHybridGraphMaxBodies` — `cuda_single_gpu_tree_learner.cpp:1277-1284` | **P** | Entirely a function of `max_depth` and `num_leaves` vs compile-time capacities (`kHybridGraphMaxLevels 32`, `kHybridGraphMaxNodes 16`, `kHybridGraphMaxBodies`). Tier 0 evaluates it once and stores an "eligible" flag + the reason if not. Bit-neutral (graph replays the host loop exactly). |
| 12 | Graph LRU cache size | `const size_t cache_limit = 64` — `cuda_single_gpu_tree_learner.cpp:1557` | **P** (lower bound) / **A** (headroom) | The *necessary* key count is provable: steady state is `(#gradient buffers = max(num_class,1)) × (compact double-buffer = 2)` distinct keys (comment at `:1552-1554`). So the provable minimum is `2·num_class`. The literal 64 is padding above that. Tier 0 should size the cache to `max(2·num_class·small_pad, provable_min)` — turning a magic 64 into a shape function — and hand tier 1 the pad factor as a prior. Memory-only, bit-neutral. |
| 13 | A2 pow2 grid buckets (`NextPow2` frozen grids) | `grid = NextPow2(blocks)` per role — `cuda_hybrid_graph.cu:528-569` | **A** (bucket policy) / **P** (bound correctness) | That the frozen grid must be an *upper bound* on live blocks is provable and already guaranteed (the kernel derives exact work on-device; idle blocks early-out via the A2 guard, `cuda_hybrid_graph.hpp:293-320`). The pow2 *bucketing* trades device-graph `SetGridDim` update frequency (~4.5µs) against idle-block waste (<2×). That crossover is measurement-ambiguous → tier-1 prior: "pow2, keeps idle < 2×, update on bucket flip only". Tier 0 can precompute the *max* bucket per level from the geometry model (§3) to pre-freeze grids. |
| 14 | `ControllerThreadsForBody` (per-body controller block size) | `body>=10 ? kMaxSplits : (body<=4 ? 64 : 2·2^body)` — `cuda_hybrid_graph.cu:44-46` | **P** | Pure function of the body index (= level) and the per-level split bound (`≤2^level`). It is a provable cover of the four intra-block reductions (descriptors, gaps, sort lanes, nodes). Already closed-form; planner just confirms it against the tier-0 geometry model and the `num_leaves ≤ 2047` gate. Bit-neutral. |
| 15 | `FALCATA_BATCH_CONSTRUCT_FLOOR` (device-saturation floor) | default `160` — `cuda_histogram_constructor.hpp:275-281` | **A** | The cautionary tale (§1). Launch-overhead vs tail-idle crossover, distribution-dependent. Tier-1 prior only: "160; do NOT scale from shape — regressed covtype 45% when tried". |
| 16 | `FALCATA_BATCH_CONSTRUCT_MINROWS` (min rows/thread cap) | default `64` — `cuda_histogram_constructor.hpp:264-270` | **A** | Same family as #15. Perf crossover between per-thread work and parallelism. Prior "64" to tier 1. |
| 17 | `FALCATA_SMALL_LEAF_ROWS` (direct small-leaf construct threshold) | default `0` (disabled); enable at `8192` — `cuda_histogram_constructor.hpp:315-321` | **A** | Explicitly perf-ambiguous AND arithmetic-sensitive (per-row double adds vs per-block float partials change the fp accumulation; ~1-2% fraud gain, breaks fraud 63/6 exact reproduction — see the comment). Tier 0 must NOT touch this: it is not bit-neutral. Leave OFF by default; tier-1 prior "0". Flagged as a bit-identity hazard. |
| 18 | `SmallLeafConstructEnabled` kill switch | default on — `cuda_histogram_constructor.hpp:299-305` | **A** | Path selection with a perf crossover; keep as tier-1 prior "on". Bit-neutral (unlike #17). |
| 19 | Compact-view dead-entry mask vs static per-tree layout | runtime `host_bin_used_bytree_` mask built every `SetFeatureUsedBytree` — `cuda_histogram_constructor.cpp:164-189`; compact build `BuildCompactView` `:236` | **P** (opportunity) | See §4. Under low `feature_fraction`, the sampled-column layout is a shape-provable static per-tree layout, not a runtime mask recomputed per tree. Tier 0 owns the layout planning; the runtime mask stays as the fallback for the non-sampled path. |
| 20 | `GHInterleaveEnabled` (float2 g/h interleave) | on iff `sizeof(score_t)==float` — `cuda_histogram_constructor.hpp:288-294` | **P** | Pure type predicate; bit-identical (the comment states values are identical). Already provable at init. Planner confirms; keep `FALCATA_GH_INTERLEAVE=0`. |

**Tally: 20 candidate knobs. Provable (P, tier-0 decides): #1, #2, #3, #4, #5(root+level-bound), #6, #8, #9, #10, #11, #14, #19, #20 → 13. Measurement-ambiguous (A, prior to tier-1): #15, #16, #17, #18 → 4. Split knobs (capacity/eligibility provable, one embedded count/policy ambiguous): #7, #12, #13 → 3.** So the "clean provable" set is 13, the "clean ambiguous" set is 4, and the interesting 3 are the split-the-knob cases where tier 0 takes the capacity/bound half and hands tier 1 only the crossover half.

## 3. The expected-tree-geometry model

Tier 0 builds one shared model at init and every level-indexed precomputation
reads from it. For a tree grown to a leaf budget `B = num_leaves` with depth cap
`D = max_depth`:

- **level count** `L = min(D, ceil(log2(B)))`. In the depth-limited exact regime
  (`2^D <= B+1`, the same predicate used at knobs #10/#11) every level `l`
  (`0..L-1`) is full: it holds `2^l` leaves and splits all of them.
- **expected leaf size at level `l`**: `rows_l ≈ ceil(num_data / 2^l)` (root =
  `num_data`; balanced-split assumption). This is a *bound-shaped* estimate — the
  true split is data-dependent, so use it only where an upper bound is what is
  needed, never as a perf constant.
- **max smaller-leaf size at level `l`** ≤ `rows_l` (the smaller child is ≤ half
  the parent, but the level's largest smaller-leaf is bounded by `rows_l`).

What this lets tier 0 precompute *exactly* (bound-correct, bit-neutral):

1. **Level-wise histogram bit widths (#5)**: `level_bits[l] = width(rows_l ×
   num_grad_quant_bins)` with the same `256 / 65536` thresholds. This is an
   *upper bound* on any leaf's runtime width at that level, so the graph's
   frozen bit-change scratch and the pow2 bit-change buckets can be pre-sized
   from it without any per-tree readback. The runtime per-child width
   (`SetNumBitsInHistogramBin`) still fires and remains the source of truth for
   arithmetic — the precompute only sizes buffers.
2. **Pre-frozen A2 grid buckets (#13)**: the max live block count per level is a
   function of `rows_l`, `num_pairs_l = 2^l`, `block_dim_y` and the row cap
   (#6). Tier 0 computes `NextPow2` of that bound per level once, so the graph
   can capture already-frozen grids and skip early `SetGridDim` churn. Still a
   bound (the kernel derives exact work on-device), so bit-neutral.
3. **Controller block sizes (#14)**: confirm `ControllerThreadsForBody(l)`
   covers `2^l` splits + gaps for every `l < L`; assert against the geometry
   model at init rather than trusting the `num_leaves ≤ 2047` gate implicitly.
4. **Speculative launch bounds**: the single-sync path sizes its construct grid
   from `max_num_data_in_smaller_leaf` as an upper bound
   (`cuda_histogram_constructor.hpp:247-259`). Tier 0 supplies `rows_l` as that
   bound directly from the model, so the host does not need a per-level max
   readback to size the launch (the exact per-thread grouping is still derived
   on-device from the actual sizes array — bit-identical).
5. **Hist pool / scratch capacities (#7,#8,#12)**: all are `num_total_bin_`,
   `num_class` and pipeline-count functions — evaluated once from the model.

Explicitly NOT derivable from the geometry model: the construct floor/minrows
(#15,#16). The model gives *sizes*, not the launch-overhead-vs-idle crossover.

## 4. Low-`feature_fraction` compact-layout opportunity

Today two mechanisms handle `feature_fraction` sampling:

- the **runtime dead-entry mask** (`host_bin_used_bytree_`, rebuilt every
  `SetFeatureUsedBytree`, `cuda_histogram_constructor.cpp:164-189`) that tells
  the elementwise batched fix/subtract kernels to skip bins of unused features;
- the **compact column view** (`BuildCompactView`,
  `cuda_histogram_constructor.cpp:236`) that gathers only the sampled columns
  into a dense per-tree matrix (~10× at `feature_fraction=0.1`, per the header
  comment at `:412-418`).

The tier-0 observation: the *number* of sampled columns per tree is a shape fact
— `k = round(feature_fraction × num_features)` (or `feature_fraction_bynode`) —
and the compact matrix's per-partition column count, byte stride and even-column
4-bit padding are pure functions of `k` and the per-feature bin counts. The
compact metadata buffers already "grow to the running-max sampled column count"
and then stabilize (comment at `cuda_histogram_constructor.hpp:361-364`,
`:390-392`), which is precisely why the graph key stabilizes after a few trees.

Proposed tier-0 move: at init, when `feature_fraction < 1`, **pre-size the
compact layout to the provable worst-case `k`** (`ceil(feature_fraction ×
num_features)`, rounded up by Falcata's sampling rule) and pre-allocate the
compact metadata / staging buffers to that size. Benefits, all bit-neutral:

- the compact metadata pointers stabilize on tree 0 instead of after "the first
  few trees", so the hybrid-graph key (`HybridGraphKeyPointers`,
  `:365-383`) stops churning early — directly attacking the LRU eviction that
  caps the cache at 64 (#12) and can trip `hybrid_graph_disabled_` after 8
  build failures (`cuda_single_gpu_tree_learner.cpp:1558-1564`).
- the `HybridGraphCompactShapeKey` (`:390-392`) collapses to one shape immediately.
- the runtime dead-entry mask remains as the fallback for the non-compact /
  non-batched path; tier 0 does not remove it, it just makes the compact path's
  layout static instead of running-max.

This is the single most self-contained provable win (see §7). It changes *when*
buffers reach their final size and *which* graph key is used, never any
arithmetic — bit-identity holds trivially.

## 5. Where the planner lives and how knobs flow

**Object**: a `CUDAStaticPlanner` (header + cpp under
`src/treelearner/cuda/`), constructed inside
`CUDASingleGPUTreeLearner::Init` (`cuda_single_gpu_tree_learner.cpp:52`)
*after* `share_state_->feature_hist_offsets()` and the constructor/data-partition
are built (so `num_total_bin_`, the partition packing and the built bin counts
are all available), and *before* the best-split-finder and discretizer are
wired. Inputs it captures once:

- from `train_data_`: `num_data`, `num_features`, per-feature `num_bin`;
- from `share_state_`: `feature_hist_offsets`, `column_hist_offsets`;
- from `cuda_row_data_` (via the constructor): `shared_hist_size_`,
  `num_feature_partitions_`, `max_num_column_per_partition_`, `bit_type`,
  `is_sparse`, `NumLargeBinPartition`;
- from `config_`: `num_leaves`, `max_depth`, `num_class`, `num_iterations`,
  `feature_fraction`, `use_quantized_grad`, `num_grad_quant_bins`, `gpu_use_dp`.

The earliest shape is actually available at `Dataset::FinishLoad`
(`src/io/dataset.cpp:754`) and `DivideCUDAFeatureGroups`
(`cuda_row_data.cpp:240`, called from `CUDARowData::Init` at `:114`) — the
partition packing (#3) is *already* a tier-0 computation happening there. The
planner does not move that code; it reads the result. The learner `Init` seam is
where the planner assembles the cross-cutting decisions (#5 level bounds, #7/#8
capacities, #9/#10/#11 eligibility, #12 cache size, #13 pre-frozen buckets, #19
compact pre-size).

**Flow to kernels — env-override-compatible.** Every existing `FALCATA_*`
override MUST remain the ground truth for A/B. The rule: **each planner field is
computed as `env_override.value_or(planner_default)`**, i.e. the planner supplies
the *default* that used to be the hardcoded literal, and the env still wins when
set. Concretely:

- knobs that are already `static` env-latched getters (#7 minrows, #15 floor,
  #16, #17, #18, #20, `SupportsBatchedLevel`'s `FALCATA_BATCH_WIDE`) keep their
  getter signature; the planner only *replaces the literal fallback* with a
  planned value where the knob is P (#1, #6, #8, #9, #10, #11, #14), and leaves
  the literal fallback untouched where the knob is A (#15-#18).
- the planner exposes a small struct of plain fields (`max_num_bin`,
  `root_hist_bits`, `level_hist_bits[]`, `pipeline_count`,
  `bit_change_scratch_bins`, `graph_cache_size`, `batched_level_eligible`,
  `graph_eligible`, `compact_worst_case_k`, per-level pre-frozen grid buckets).
  Constructors that currently read a literal read the field instead.
- `FALCATA_PLANNER=0` (new, optional) restores every literal fallback wholesale,
  giving one master A/B switch in addition to the per-knob envs.

No planner field may feed an *arithmetic* input (accumulator width beyond the
correctness bound, rounding mode, add order). Fields only feed grid sizing,
buffer capacity, path eligibility and layout — the bit-neutral set.

## 6. Phased plan

**Phase 0 — read-only planner (no behavior change).** Introduce
`CUDAStaticPlanner`, populate it from the shape at `Init`, and have it *recompute*
the current values of #1-#14, #19-#20 and log them under `FALCATA_HYBRID_DIAG`.
Assert each planned value equals the value the existing code computes. This is
the safety net: it proves the formulas match before anything switches over.

**Phase 1 — provable layout / packing / eligibility (the clean-P knobs).** Route
#3/#4 (partition packing constants), #9/#10/#11 (eligibility conjunctions,
computed once instead of per tree), #14 (controller block assert), #20 through
the planner as their default source. All bit-neutral, all already computed
elsewhere — this is consolidation, lowest risk.

**Phase 2 — capacity precompute from the geometry model.** #5 level bit-width
bounds, #6 cached row cap, #7-capacity/#8/#12 scratch and cache sizing, #13
pre-frozen buckets. These change buffer sizes and graph capture, still
bit-neutral. Gate each behind its env.

**Phase 3 — compact-layout pre-sizing (§4, the high-value win).** Pre-size the
compact metadata/staging to the worst-case `k` when `feature_fraction < 1`.
Verify the graph key stabilizes on tree 0 and the LRU no longer churns.

**Deferred to tier 1 (never in tier 0).** #15 floor, #16 minrows, #17 small-leaf
rows (also a bit-identity hazard — must stay off), #18 kill switch, the #7 *count*
(4), the #13 *bucket policy*. Tier 0 ships each of these a documented prior
(current constant + known-good shape range) and stops.

## 7. Highest-value tier-0 win

**§4 compact-layout pre-sizing (#19 + its knock-on to #12/graph eligibility).**
It is provable (worst-case `k` is a shape function), bit-neutral (only changes
when buffers reach final size and which graph key is used), and it unblocks the
hybrid-graph path under `feature_fraction` sampling by killing the early
metadata-pointer churn that currently forces LRU eviction and can disable the
graph after 8 failures. Every other P-knob is a consolidation; this one removes
a real performance cliff.

## 8. Verification strategy

**Bit-identity is the only gate.** The planner is defined to touch exclusively
performance-neutral layout/scheduling/capacity — never arithmetic. Therefore:

- **md5 locks must be UNCHANGED.** The existing per-dataset model md5 locks are
  the acceptance test. Any planner phase that moves an md5 is a bug in the
  planner (it leaked into arithmetic), not an accepted trade-off. Run the lock
  suite after each phase, with `FALCATA_PLANNER=0` (must match baseline) and
  with the planner on (must match the SAME md5).
- **Phase 0 assert harness**: the read-only planner asserting planned == current
  for every knob is the first line of defense and should run in CI-shape (small
  synthetic datasets covering: `max_num_bin ≤ 8` and `> 8` for #1; sparse and
  dense for #9; depth-limited and budget-limited for #11; `feature_fraction < 1`
  and `= 1` for #19; `num_class = 1` and multiclass for #12).
- **A/B via envs**: for each P-knob, flip its existing env and confirm the md5 is
  identical whether the value comes from the planner default or the env — proving
  the planner only changed the *source* of a bit-neutral value.
- **#17 stays quarantined**: it is the one knob that is NOT bit-neutral; the
  spec forbids tier 0 from setting it, and the md5 suite would catch it if a
  future change wired it into the planner by mistake.

Any knob whose planner value would change an md5 is, by definition,
measurement-ambiguous-or-arithmetic and belongs in tier 1 — the md5 suite is the
mechanical classifier that keeps the P/A boundary honest.
