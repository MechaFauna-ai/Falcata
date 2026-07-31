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
