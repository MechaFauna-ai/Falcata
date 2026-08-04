# Performance dead ends

Every performance idea we tried that did NOT earn its place, with the reason
it failed and — crucially — the condition under which it would be worth
trying again. Negatives are not absolute: most failed because of what the
bottleneck happened to be at the time. If the bottleneck moves, entries here
come back to life.

The counterpart document is [performance.md](performance.md), which explains
the optimizations that DID land. The forward-looking idea list lives in
[../ROADMAP.md](../ROADMAP.md).

Entry format: what we tried → why it seemed promising → what we measured →
why it actually failed → **re-open when**.

---

## Compact-view packing codecs (bit3 / radix5 / radix6 / radix7)

- **Tried:** pack bins tighter than 4 bits in the per-tree compact matrix —
  3-bit fields, or base-5/6/7 arithmetic coding into uint32 words (up to
  1.63× fewer bytes for numerai-class 5–7-value features). Full
  policy-template implementation, bit-identical, gate-covered (`5ce1c75f`).
- **Promising because:** histogram construction reads the compact matrix on
  every level pass; fewer bytes looked like direct speed.
- **Measured (numerai-deep, canonical config):** bit3 −2.5%, radix7 −27%
  versus a matched baseline. radix5/radix6 cannot even engage (every
  feature-sampled tree contains a span-7 column).
- **Why it failed:** the discretized construct kernel is **ALU/issue-bound,
  not read-bound** — extraction instructions scale the kernel linearly while
  byte savings buy almost nothing. nsys attributed the entire radix7 loss to
  the hist kernel itself (~2× its nibble time); the fill was cost-neutral.
- **Re-open when:** the shared-memory-atomic / instruction bottleneck of the
  construct kernel is removed, making it read-bound again. Also note the
  synthetic result: on small single-partition shapes radix6 was 2.2× FASTER —
  a construct-dominated small-model regime could use this today.
- Full record: [design/pack-codecs.md](design/pack-codecs.md). Traps caught
  on the way (worth rereading before any similar work): EFB bundle columns
  invalidate per-feature bin bounds; runtime-indexed constexpr tables live in
  thread-local memory (19× penalty); silent eligibility-fallback mimics wins.

## Compact-view prefill (next-tree fill overlap)

- **Tried:** double-buffer the compact matrix and fill the NEXT tree's column
  sample on a non-blocking side stream while the current tree trains
  (`631e1bc3`; the header fields for this had existed as dormant plumbing).
- **Promising because:** the per-tree fill was ~5ms of a ~29ms tree and has
  no dependency on the current tree's splits — textbook overlap.
- **Measured:** exactly 0% wall change, bit-identical, engagement verified.
- **Why it failed:** steady-state training is ~97% device-busy, and the ~41
  synchronous D2H readbacks per tree act as legacy-stream barriers that
  serialize the fill into its own slot regardless of stream. There is no idle
  device time to hide work in, and no stream trick crosses those barriers.
- **Re-open when:** the readback/barrier structure is reduced (see quant
  one-sync below). Key `compact_prefill` stays in the tree, default off,
  gate-covered.

## Quant one-sync parity (single-sync level pipeline for quantized training)

- **Tried:** extend the speculative one-sync level flow (which serves the
  non-quantized path) to quantized training. Parked on archive branch
  `archive/one-sync-quant-wip`.
- **Promising because:** it halves per-level host synchronization; and (in
  hindsight) it is the gatekeeper for every overlap-based idea above.
- **Why it failed:** the quantized path's per-level bit-width readbacks and
  histogram-bit bookkeeping resisted the restructuring; the attempt was an
  honest negative at the time.
- **Re-open when:** someone is willing to spend serious effort — the
  41-barriers finding raised the prize: it now unlocks prefill (~+8%) on top
  of its own win, and possibly re-opens the codec question after that.

## Graphs L2 — per-parent dependency chaining

- **Tried:** finer-grained CUDA-graph dependencies (per-parent chains instead
  of level barriers).
- **Why it failed:** measured no win; the graph controller's level barrier
  was not the limiter. Honest negative recorded in ROADMAP.

## Intra-apply grid serialization

- **Tried:** serializing the apply phase's grid to reduce contention.
- **Why it failed:** no win; superseded by the batched apply design.

## Controller JIT (phase 6 of the hybrid arc)

- **Tried:** JIT-specializing the device-side graph controller.
- **Measured:** no win (archive branch `archive/controller-jit`).
- **Re-open when:** probably never — the controller is not on the critical
  path at any measured shape.

## Construct-floor shape model (the cautionary tale)

- **Tried:** deriving the batched-construct saturation floor and rows-per-
  thread from dataset shape and SM count instead of baked constants.
- **Measured:** +35% on year/higgs, **−45% on covtype** with the same model.
- **Why it failed:** the floor trades launch overhead against tail-block idle
  time, and the crossover depends on the runtime leaf-size distribution and
  achieved occupancy — not pure shape. A formula right for one distribution
  is wrong for another.
- **Re-open when:** as MEASUREMENT (auto-tuner tier 1 bandit), never as a
  shape formula. This entry is why tier 0 of the planner only decides
  provable layout/capacity questions.

## Static planner §4 — compact-layout pre-sizing

- **Tried:** pre-sizing max compact columns to stabilize graph shape keys.
- **Why it failed:** (a) the LRU-eviction cliff it targets does not occur on
  realistic shapes (numerai: 23 instantiations vs 64 cache slots, zero
  evictions); (b) pre-sizing changes block_dim_y, which reorders non-quant
  float atomics → NOT bit-identical (verified md5 flip).
- **Re-open when:** a very-low-fraction/high-feature workload pushes distinct
  shape keys past 64 — and only with a launch refactor that decouples
  block_dim_y.

## Exact small-int bin finding

- **Tried:** exact integer bin boundaries for small-int features
  (`archive/exact-int-bins-wip`).
- **Why it failed:** unfinished; the underlying need was better served by the
  quant_mode/quant_bins config work.
- **Re-open when:** a dataset with genuinely rare (< min_data, missed by
  sampling) small-int values appears.

## construct_jit as a blanket default

- **Status nuance:** the NVRTC JIT infrastructure WORKS (self-test-then-
  promote, bit-identical) and stays in the tree, but flipping it on measured
  −30% on numerai-deep and ~neutral elsewhere — the current JIT specializes
  the wrong thing. The unbuilt "star case" (shared-histogram packing with
  baked bin counts) is a separate, still-open roadmap item.
- **Re-open when:** the tier-2 shared-hist packing is built; the JIT is its
  natural delivery vehicle.

## graph_quant (CUDA-graph level loop for quantized training)

- **Tried:** extending the graph level loop to the quant path (key exists,
  default off).
- **Measured:** −8% on numerai-deep, −2.5% on covtype-shallow when forced on
  (ablation 2026-07-30 measured +4.2% on covtype-deep — shape-dependent,
  net-negative or noise on the shapes that matter).
- **Re-open when:** deep-config controller latency changes materially.

## Tensor-core histogram construction (one-hot MMA)

**Hypothesis.** Recast the hist pass as H = Bᵀ·G (B = rows×bins one-hot built
on the fly, G = int8 quant grad/hess) and let int8 MMA replace shared-memory
atomics: no atomic serialization, fixed accumulation order, and quantized
grads already fit int8.

**Measured** (`benchmarks/tc_hist_prototype.cu`, 5090, 2M×356×8-bin synthetic,
all exactly verified): three escalating implementations — WMMA with
shared-staged fragments (0.32 T upd/s), register-direct PTX `mma.m16n8k32`
with `__vcmpeq4` one-hot build (0.45 T), plus packed dual-column loads and
4 column-pairs per warp amortizing B fragments (0.58 T). The atomic baseline
does **1.12 T upd/s** — TC lost by ~2× after all tuning.

**Mechanism of failure.** (1) The MMAs were only ~4% of tensor throughput —
the kernel is bound by the ~10 fragment-construction instructions per MMA
(loads, nibble masks, vector compares), which the formulation cannot avoid;
scaling analysis puts the ceiling at atomic *parity*. (2) The premise
overestimated the atomic path's pain: thread-per-column shared atomics
self-serialize per column and never conflict, so there is no serialization
to eliminate. (3) Integration reality is worse than the synthetic: below the
root level, rows arrive through the `data_indices` leaf-partition gather, so
the contiguous 32-row tiles the vectorized one-hot build needs don't exist
(~2× additional penalty).

**Re-open when:** (a) targeting datacenter silicon where the int8-TC :
int-ALU throughput ratio is several times the consumer ratio (H100/B200 —
combine with the DSMEM cluster-merge idea), (b) a layout change makes
leaf-partitioned rows contiguous in the bin matrix, or (c) multi-target
vector-leaf training lands (accumulate width T+1 amortizes the dead n
columns, the formulation's biggest fixed waste).

## Selective-growth apply-churn deferral (gain-margin gating)

**Hypothesis.** Selective grow-then-prune applies ~2.1× the splits it keeps
on budget-bound shapes (covtype 64/12: 52% displaced-then-collapsed);
deferring the displacement-prone selected candidates (gain within a margin
of the selection cutoff, kept visible across levels) should recover that
wasted apply+construct work while provably preserving the final tree.

**Measured** (covtype 64 leaves / depth 12 / quant, 500 rounds, 5090; full
mechanism implemented with frontier persistence + progress guarantee, all
gates green — the trees were bit-identical as designed):

| margin | applied | displaced | deferred | levels | t/s |
|---|---|---|---|---|---|
| off | 67,310 | 35,810 | 0 | 6,490 | **93.2** |
| 0.25 | 62,482 | 30,982 | 10,657 | 7,341 | 87.5 |
| 1.0 | 59,291 | 27,791 | 20,852 | 7,712 | 85.8 |
| 4.0 | 58,805 | 27,305 | 23,072 | 7,326 | 91.9 |

**Mechanism of failure.** The premise mispriced the churn: applies are
BATCHED (an extra pair in a level costs almost nothing) while LEVELS carry
the fixed launch/sync/readback overhead. Every deferral variant cut applies
by 8–13% but paid 13–19% more levels — a strictly bad trade at every margin
tried. The 2.09× apply churn is cheap churn.

**Re-open when:** the per-level fixed cost shrinks dramatically (e.g. the
one-sync-quant rework lands and level overhead halves), or a predictor
exists that avoids doomed applies WITHOUT extra levels (i.e., defers only
candidates that would otherwise be applied and displaced within the SAME
level budget — requires lookahead the greedy stream does not have).

## NVRTC specialization of the COMPACT-QUANT construct — SUPERSEDED
(2026-08-01: the parity finding was an unsynced template; with __ldcs added
the same shape measures +4.0%. Kept for the methodology lesson: keep JIT
templates in lockstep with the AOT kernel before comparing. The general
tier-2 case LANDED — see performance.md §7b.)

**Hypothesis** (from the July JIT arc): numerai's ~5.5-bin features waste
>90% of the fixed 12288-entry shared histogram; a kernel with per-feature
bin counts baked in packs ~10x more features per partition, cutting
shared→global merge traffic.

**Re-scored 2026-08-01, post the July-31 lever batch:** the headroom the
specialization targeted no longer exists on the flagship shape. The compact
column view reduces numerai-deep to 356 live columns and `wide_partitions`
packs up to 1008 columns per partition — the whole tree trains in a SINGLE
partition, so "fewer partitions, less merge traffic" has nothing left to
remove. Measured directly: the working JIT construct path (shape-keyed
NVRTC cache, bit-identical, 160ms compile) scores **-0.5% vs the current
AOT kernel** on numerai-deep at 4 interleaved reps (an earlier +2.2%
reading was warmup noise). Profile: the remaining hist kernel is
ALU/issue-bound at ~58% of steady-state tree time with evict-first loads
already applied.

**Scope note (2026-08-01, after review):** this entry covers ONLY the
compact-quant instance — the single kernel family the JIT engages today.
The GENERAL tier-2 question (specialize the non-compact construct/find
kernels for higgs/epsilon/year/covtype-class shapes) remains OPEN in the
ROADMAP: those regimes never touched the JIT because it is gated to
compact-quant, so parity here says nothing about them.

**Re-open when:** a workload trains WIDE data without feature sampling
(ff=1.0 at 3555+ cols = 4+ partitions — the packing argument returns), or
the AOT inner loop grows shape-dependent branches that specialization could
constant-fold (check the JIT template is synced to the AOT kernel first —
it predates wide_partitions/__ldcs).

## Static planner tier-0: shape-derived knob priors (superseded)

**Hypothesis.** Decide tuner priors at Init from dataset shape (expected
level geometry → grid configs and speculative bounds; bin-histogram-derived
packing thresholds).

**Why it closed without implementation (2026-08-02):** the runtime
machinery landed first and confined the priors' value to ~nothing. With
the wisdom cache, a shape's second-ever run starts at the known-best knob
values; the priors could only improve the first ~130 probe trees of the
FIRST run on a shape — probe candidates measure within ~20% of optimum on
average, so the entire addressable win is ~26 tree-equivalents (~0.2% of
one long run), once per (shape, device), ever. Against that, the measured
brittleness of GPU cost models (the construct-floor cap: +35% year, −45%
covtype — same formula) makes fitted shape formulas net-risk. The
bin-threshold sub-item was moot: every current packing decision already
reads real bin-mapper counts. Device-side seeding (SM-scaled candidates)
did land and stays.

**Re-open when:** a deployment pattern emerges with many DISTINCT one-shot
shapes on fresh machines (wisdom never warm), or a knob appears whose probe
cost is large relative to run length.

## Multi-GPU beyond level batching: transport and scale are not the levers (2026-08-04)

**Claim tested.** After the level-batched all-reduce (1.54x over per-split),
the remaining 2-GPU gap (0.68x vs a single GPU on 2x3090/PCIe) might close
with a faster interconnect, or at larger data where construct dominates the
collectives.

**Result: both refuted, same day, same workload (4M x 400 int8, max_bin=5).**

- NVLink-bridged 2x3090 (NV4, 56 GB/s, verified topology; `NCCL_P2P_DISABLE=0`
  so the collectives actually ride the bridge): 13.57 s vs 13.13 s
  host-staged — NO transport win (ratios 0.64x vs 0.66x of that box's 1-GPU
  8.7 s). The per-level messages are ~1.5 MB; the binding term is fixed
  per-level cost (collective rendezvous, the host-blocking reduce sync, the
  two-sync flow's per-level readbacks), which faster wires do not shrink.
  Notably the historical P2P deadlock did NOT reproduce over NVLink — it
  appears specific to PCIe P2P.
- 3x the data (12M x 400, 100 trees): ratio got WORSE, 0.54x — at fixed
  min_data_in_leaf the deeper/wider levels add pairs (bigger payloads, same
  barrier count per tree), so overhead grows with data instead of amortizing.
  There is no crossover on this workload class.

**Conclusion.** Data-parallel level-synchronous GBDT on 2 GPUs does not beat
one fast GPU on dense tabular workloads of this shape, independent of
interconnect. The collective COUNT was already cut ~10x by level batching;
what remains is the per-level barrier structure itself.

**Re-open if:** (a) the flow becomes device-driven end-to-end across ranks
(graph-captured multi-GPU with device-initiated collectives — a redesign,
not a tune); (b) histograms are quantized into integer collectives (4-8x
smaller payloads AND cheap ncclInt sums) once quantized multi-GPU plumbing
is finished; or (c) the workload is not data-parallel-per-tree at all
(multi-target: one tree per target per GPU has zero per-level collectives
and is the natural 2-GPU win).

**2026-08-04, same day: (b) was implemented** (`e65313ca` — integer level
all-reduce, global bit-width table, quantized multi-GPU un-fenced and
verified exact on an NVLink 2x3090). Outcome: correctness delivered,
throughput verdict UNCHANGED — 2gpu-quant is 0.72x of 1gpu-quant at 4M rows
and 0.62x at 10M, because quantized construct is so fast that the per-level
barrier structure dominates again, and the barrier cost scales with
pairs-per-level (host-side readbacks and bookkeeping), not with payload or
transport. Two refinements to the record:

- The non-quantized crossover EXISTS: at feature_fraction=1.0 on 400
  features (double the construct load of the earlier ff=0.5 benches),
  2gpu-fp beat 1gpu-fp for the first time — 19.53s vs 24.43s (1.25x) at 4M
  rows, shrinking to 1.11x at 10M. Data-parallel 2-GPU pays off only when
  per-level construct time clearly exceeds the fixed per-level barrier cost
  — heavy-construct, moderate-leaf-count shapes. Quantization removes that
  regime by making construct cheap: 1gpu-quant beats every other arm at
  every size tested.
- What binds now is neither transport (NVLink == host-staged) nor payload
  (integer reduce halved it: no ratio change) but the per-level HOST
  machinery scaling with leaf count (best-split readback, splittable
  arbitration, apply bookkeeping). Removing it is re-open condition (a),
  the device-driven loop — the only remaining lever, and a redesign.

(c) multi-target remains the practical 2-GPU win and is unaffected.


## Feature-parallel multi-GPU: the last tree-level variant, also closed (2026-08-04)

**Claim tested.** Splitting FEATURES instead of rows (every rank holds the
full dataset, searches its feature stripe, ranks merge per-leaf winners
host-side — kilobytes per level instead of megabytes, and no local-vs-global
count hazards by construction) should win on wide data where histogram
construction dominates.

**Result: refuted** (`cbce5d82`, correct but slower everywhere). On
2M x 2000 int8: 0.80x of 1 GPU at ff=1.0, 0.22x at ff=0.1. The premise fails
because the OPTIMIZED single-GPU flow (graph-captured level loop, JIT
construct, compact view) runs a whole level in ~3.4ms at the numerai-style
regime — halving construct saves ~1.5ms while the coordination (thread
barrier, ~1MB leaf-cache re-upload, forced two-sync instead of the graph
flow) costs several ms. The faster the single-GPU path gets, the less room
any per-level cross-GPU scheme has; ours got too fast.

**Standing conclusion.** Tree-level multi-GPU parallelism is closed in every
variant tried: data-parallel per-split, data-parallel level-batched,
integer-quantized collectives, NVLink transport, bigger data, and
feature-parallel. The one remaining multi-GPU win is MODEL-level
parallelism: multi-target training with one tree per GPU (zero per-level
coordination) — task #67. The feature-parallel mode stays in-tree (opt-in
`tree_learner=feature`, off by default, zero-cost when unused): its
merge-state machinery is the seed of the multi-target orchestration.

**Operational note:** two fresh vast.ai boxes on driver 590.48.01 threw a
cold-start CUDA "invalid argument" on the first multi-GPU touch after boot
(old wheel and new alike); warm runs pass consistently.
`CUDA_MODULE_LOADING=EAGER` is the rental hedge.
