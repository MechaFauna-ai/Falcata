# Growing the trees of a random forest together (forest stage 2)

Status: design spike and first implementation (cuda_plan key `forest_batched`, default on), 2026-10-15. **Removed
in 1.2** (2026-10-23): the slots are deleted and every forest tree trains on the booster's learner, one after another,
taking the device flows a gbdt tree takes (`docs/performance.md` §10r). This note is kept as the record of the design.
Plan: kernel-evo `runs/forest/PLAN_batched_forest.md` §3 stage 2. Numbers: one RTX 5090, relaxed (desktop on),
master 929fdbea (after #113 `level_bynode_masks` and #114) unless a line says otherwise.

## 1. Question

Stage 1 put `feature_fraction_bynode < 1` on the batched level flows. A random forest (`boosting=rf`) still grows one
tree at a time: `RF::TrainOneIter` trains the `num_class` trees of a round one after the other through the booster's
single `CUDASingleGPUTreeLearner`, and the next round starts after the last tree of this one. cuML grows a forest's
trees together, level by level. The plan asked for the same here: "T bags, T partitions, one fill / one finder launch
/ one apply per level across the trees of a group". This note checks that design against where the time goes,
amends it, and records the data structures, the loop, the memory budget and the determinism rule of what was built.

## 2. Where a forest round spends its time

Profiles of 5-10 rounds with `nsys --trace=cuda,osrt --sample=process-tree` (scripts and traces in kernel-evo
`runs/forest/stage2/nsys/`), steady-state window (the last 60 % of the trace):

| cell (forest_bench config) | wall per round | GPU busy (kernel union) | where the rest goes |
|---|---:|---:|---|
| covtype-rf-deep (7 trees, 4095 leaves, depth -1, quant stochastic, √F per node) | 112 ms | 14 ms (12 %) | host: CPU samples on the training thread are 30 % `RunSelectiveLevel` (the greedy simulation: heap pushes/pops, `SelectiveTieBreak`), 8 % `cudaMemsetAsync`; API: 1900 memsets per tree (`ZeroHistSlots`, one per recycled histogram slot), ~440 launches and ~90 stream syncs per tree |
| higgs-rf-deep (1 tree, depth 12, quant none) | 63 ms | ~65 ms | GPU: the deterministic fp64 construct (`CUDAConstructHistogramDenseGMDeterministicBatchedKernel`) is 582 of 648 ms of kernel time in 10 trees |
| numerai-rf-deep (1 tree, depth 12, quant none, 3555 features) | 95 ms | ~94 ms | GPU: the same construct, 740 of 940 ms |

So the plan's premise holds for one of the three cells only. covtype-rf-deep is host- and API-bound: 12 % GPU, the
rest spent per tree on the host (the selective simulation over thousands of leaves, ~46 levels per tree) and in API
calls. higgs-rf-deep and numerai-rf-deep (quant none, depth 12) are GPU-bound in one kernel; growing their trees
together cannot shorten them, only less histogram work can (plan stage 3).

## 3. Two ways to grow T trees together

**(a) One learner with a tree dimension** (the plan's design). Every per-learner structure gets a tree index: leaf ids
become (tree, leaf) (global leaf id `t * S + l`), the partition holds T bags back to back with T sets of leaf windows,
the histogram pool, the best-split cache (`cuda_leaf_best_split_info_`), the node-mask slab, the quantized per-leaf bit
widths and the selective-growth records get a tree slice, the construct kernels read a per-task gradient pointer (a
multiclass tree reads its class's column; quantized trees each have their own discretized gradients), and one level
step enqueues the leaf tasks of all T trees. The functions this generalises: `EnqueueLevelBestSplitSearch` /
`...Speculative` (descriptors, `CUDAHybridPairDescriptor`), `CUDAHistogramConstructor::ConstructHistogramsForLevel`
and `SubtractHistogramsForLevel` (all construct variants: dense, sparse, 4-bit, quantized, deterministic fp64, small
leaf, pair-code views), `CUDABestSplitFinder::FindBestSplitsForLevel` and the level sync kernels, `ApplyLevelBatched` /
`ApplyLevelBatchedSelective` and `CUDADataPartition::SplitLevelBatched` (apply descriptors, the leaf map),
`CUDATree::SplitBatch`, `RunSelectiveLevel` / `SelectiveFinalize`, `ArbitrateLevelBudget`, `FinishLevelBookkeeping`,
`BeforeTrain` (root init, discretizer, column views), across four flows (prefix two-sync / one-sync, selective two-sync
/ one-sync). It removes launches and syncs per tree (T trees' level in one launch, one readback), and fills the GPU
with larger kernels. It does not remove the per-tree host work (the selective simulation is per tree), which on
covtype is the larger part.

**(b) One learner per tree in flight ("slots"), grown concurrently by host threads.** Slot 0 is the booster's
learner; slots 1..G-1 are learners of the same config and data. A group of G trees runs as G host threads, each
calling its slot's unchanged `Train` (every level flow, every tree shape). The per-tree host work runs in parallel on
the CPU cores, the per-tree API calls and syncs overlap, and the GPU interleaves the slots' kernels. Its limits: the
CUDA driver serializes API calls of one context (under 7-28 threads a launch costs 13-37 µs instead of 2.5 µs), and
the learner's kernels run on the legacy default stream or on blocking streams, so the GPU runs the slots' kernels one
after the other: the GPU-busy time is a floor (covtype: ~14 ms per round of 7 trees).

**Decision: (b) now, (a) only where (b) leaves the GPU-serial floor.** (b) attacks the dominant cost of the
launch-bound cell (the per-tree host work) and reuses every flow as it is; (a) is a rewrite of the level flows whose
gain on top of (b) is bounded by the GPU-busy floor (covtype: from ~24 to ≥14 ms per round), and does nothing for
the GPU-bound cells. Measured (§8): covtype-rf-deep 112 → 24 ms per round with (b). A later stage can merge the slots'
launches (the cheapest form of (a): the same per-tree learners, one level step of all slots enqueued as one launch
per kernel family) if the floor matters.

## 4. Data structures

Per slot (a full `CUDASingleGPUTreeLearner`): its `CUDADataPartition` (the tree's data-index array, filled from the
tree's bag by `SetBaggingData`, and leaf windows: the plan's "a CUDADataPartition per tree"), histogram pool, best
split finder and leaf cache, gradient discretizer, CUDA tree record and pooled slab, selective-growth records, node
masks, and a private `CUDAColumnData` view (`CUDAColumnData::InitViewOf`: the dataset's column buffers, not copied,
with a pointer table and per-tree compact/packed view of its own, so a slot installing its tree's view does not
replace the one another slot's kernels read; slot 0 keeps the dataset's).

Shared: the dataset's device columns (read through the views), the bin matrix (`CUDARowData`, built once by slot 0's
histogram constructor and shared read-only, `CUDAHistogramConstructor::ShareRowData`), the gradients and hessians
(the device objective computes them once; class c's column at `c * num_data`), the round's bag (a copy per round of
the group: in-bag first, then out-of-bag).

Host side: `CUDAForestTrainer` (src/treelearner/cuda/cuda_forest_trainer.*) owns slots 1..G-1, draws the per-tree
inputs in tree order and runs the threads; `RF` keeps the rounds grown ahead (`ForestRound`: trees by class, the slot of
each, the bag) and hands them out one round per `TrainOneIter`.

## 5. The loop

```
RF::TrainOneIter (round r)
  if no round grown ahead: GrowForestGroup
      R rounds r..r+R-1 (forest_group trees / auto size, capped by num_iterations, memory, error feedback)
      for each round: data_sample_strategy_->Bagging(round) in round order, copy the bag
      jobs: one per (round, trained class), slot = round_in_group * classes + class_index, in tree order
      CUDAForestTrainer::Train(jobs)
          main thread, in tree order: counters (bynode node keys, stochastic rounding) = booster learner's + k,
                                      feature_fraction sample = the booster learner's sampler's next draw
          one thread per job: slot.SetBaggingData(bag); tree = slot.Train(grad + class * N, hess + class * N)
  pop round r: per class tree, as the sequential loop does, with the tree's own slot:
      RenewTreeOutput(slot), AddBias, deferred train score (or the per-tree update with the slot's leaf map and the
      round's out-of-bag rows), validation scores, models_.push_back
```

A slot grows one tree per group, so its partition still holds that tree's leaves when the round is popped (renewal,
the per-tree train score update). Host `Tree` objects are built at each slot's tree end as before (`ToHost`, or
`RebuildFromHostSplits` under selective growth); the train score is replayed lazily (`rf_lazy_train_score`, #103).

Group size (`cuda_plan=auto,forest_group:N`, 0 = auto). Auto starts with one round (its class trees in parallel) and
doubles the rounds per group while a size's measured time per tree (the mean of two steady-state groups: no slot's
first tree, not the training's first group) drops below 0.85 of the previous size's, up to 32 trees and while four
groups of the next size fit in the rounds left; a doubling that gains less is kept if it gained at all, else the size
goes back. The model does not
depend on the size, so the runtime may choose it. A slot's level buffers are sized for any level when the slot is made
(`PresizeLevelBuffers`), so a slot's steady state has no allocation and no device synchronize, which would stall every
slot.

## 6. Memory budget

A slot's footprint is what its Init allocates plus what its first tree allocates (lazily sized buffers, views); both
are measured (`[forest-diag]` lines under `FALCATA_DEBUG=diag`). Measured per slot (after sharing the bin matrix):

| cell | slot Init | first tree | footprint | slots that fit beside an eighth of 32 GB |
|---|---:|---:|---:|---:|
| covtype-rf-deep (465k rows, 54 features, 4095 leaves) | 174 MiB, 15 ms | 27 MiB | 0.20 GiB | > 100 |
| higgs-rf-deep / higgs stochastic (3.2M rows, 28 features) | 576 MiB, 35 ms | 234 MiB | 0.79 GiB | ~34 |
| numerai-rf-deep (400k rows, 3555 features) | 2230 MiB, 180 ms | 1036 MiB | 3.2 GiB | ~8 |

The histogram pool (num_leaves x total bins x 16 B: 1.16 GiB on numerai) and the per-tree column views dominate.
`EnsureSlots` creates a slot only while the free memory covers the reserve (max(1 GiB, an eighth of the device, twice
a slot's first-tree bytes); forest stage 3), the first-tree allocations still pending for the slots not yet grown, and
the new slot's footprint (its
first tree assumed as large as its Init until one was measured). The group size is capped by the slots that fit:
this is the group-of-G fallback; at least one round's class trees must fit, otherwise the round trains sequentially.
The bag copies are R x num_data x 4 B (higgs, 16 rounds: 205 MB).

Stage 3 (PR #121 review): the measured footprint missed the level buffers that grow after a slot's first tree
(numerai's level finder output, ~2 GiB, was presized only up to 256 MiB, and a growing Resize holds the old block
beside the new). With the count construct's faster trees auto settled on 6 numerai slots and peaked at 31.9 GiB of 32
(one OOM). Now the level buffers are presized up to an eighth of the device (in the Init footprint: numerai 4.56 GiB per
slot, 5 slots, peak 25.6 GiB); a group that leaves the device below the reserve halves the next group and frees the
slots beyond it; a group that runs out of device memory is grown again once on one round's slots.

## 7. Determinism: every tree is the sequential tree

What a tree depends on beyond the data and config, and how the group gives it the sequential value:

- **bag**: the device sampler is keyed by the round (`CUDABaggingSample(..., iter, seed)`, Philox); the group calls
  the strategy's `Bagging` for its rounds in round order. Only every-round bagging (`bagging_freq` 1) or none: with
  `bagging_freq` > 1 a round reuses the bag the learner's partition kept, state a dropped group would leave wrong.
  Balanced and by-query bagging (the host sampler's stream) stay sequential.
- **feature_fraction sample**: the booster learner's `ColSampler` stream, one draw per tree; the trainer draws them
  in tree order on the main thread and each slot installs its tree's (`ColSampler::SetByTreeSample`).
- **bynode masks** (`level_bynode_masks`): keyed by (seed, tree count, node path); the trainer sets each slot's tree
  count to the sequential one (`SetTreeCounters`). Per-node draws from the shared stream (interaction constraints,
  `level_bynode_masks:off`) stay sequential.
- **stochastic rounding**: Philox of (seed, discretizer tree counter, row); set like the bynode count.
- **fixedpoint error feedback**: a residual per class carried from one tree of the class to the next: groups hold one
  round (its class trees), each class on the same slot every round.
- **cross-tree learner state** that changes trees: CEGB's coupled feature penalty (sequential), compact prefill
  (draws the next sample early; forest off), the graph loop (slots never take it: its stream capture would be joined
  by another thread's legacy-stream work; sequential training may, and its det/atomic histogram choice then differs).

Dropped trees (rollback, a parameter reset, a reset of the training data, early stopping, model merge/shuffle/load):
the booster learner's counters and sampler go back to the first dropped tree's (`CUDAForestTrainer::RewindTo`), so the
next tree, grouped or sequential, still gets the sequential inputs. Under these rules a forest trained with the key is
md5-identical to the forest trained without it and independent of the group size, under quantized gradients and in
every non-quantized shape that is run-to-run deterministic without the key (the fp64 level loops with
`feature_fraction` < 1 regression and the graph loop are not; those differ run to run with the key off as well).
The plan only required run-to-run determinism and G-independence.

## 8. Measured (relaxed, 100 rounds, total seconds of `flc.train` and steady-state ms per round)

| shape | key off | key on (auto) | steady state off -> on |
|---|---:|---:|---|
| covtype-rf-deep | 10.13 s | 3.81 s | 93.7 -> 23.7 ms (28 slots) |
| higgs d-1 stochastic | 1.14 s | 1.07 s | 8.8 -> 4.4 ms (8 slots) |
| higgs-rf-deep (d12 none) | 6.60 s | 6.66 s | 63.5 -> 63.8 ms (2 slots, no gain) |
| numerai-rf-deep (d12 none) | 10.04 s | 9.57 s | 94.5 -> 87.4 ms (2 slots) |

(key off includes the slot-zeroing kernel of this change: covtype-rf-deep 115 -> 96 ms per round sequentially.)

## 9. Reused and new

Reused unchanged: every level flow and kernel, the bagging sampler, the device objective's gradients, the bynode
keyed masks, the lazy train score. New: `CUDAForestTrainer`; `RF::GrowForestGroup` / `TrainOneIterForest` /
`ForestAutoSize` / `ResetForest`; `CUDAColumnData::InitViewOf`; `CUDAHistogramConstructor::ShareRowData`;
`ColSampler::SetByTreeSample`; the learner's `EnableForestSlot`, `SetTreeCounters`, `SetNextTreeSample`; the plan keys
`forest_batched`, `forest_group` and `zero_slots_kernel` (one kernel zeroing the recycled histogram slots of a
selective level instead of one memset per slot: gbdt's selective growth gains too, bit-identical).

## 10. Out of scope, risks

- Multi-GPU (num_gpu > 1, NCCL) keeps the sequential learner; so do the host objective, GOSS, linear trees, forced
  splits, vector leaves, CEGB, interaction constraints with bynode, `bagging_freq` > 1, balanced/by-query bagging.
- Thread safety: the slots share only read-only device data and the process-global plan, which `Init` writes (slots
  are initialised on the main thread before any group runs) and `Train` writes only on a change
  (`batch_construct_saturation_floor`, the same value in every slot). Debug statics and `thread_local` caches are
  benign. The log level and log callback are per thread (`Log`): each slot thread installs the caller's, so
  `verbose` and a registered logger (Python, R) apply to its messages. Errors in a slot thread are rethrown on the main
  thread after the group's rounds are dropped and the counters and sampler rewound. The once-per-process copy streams
  of the host-mapped compact path are created under a `std::once_flag`.
- Known, not caused by this: on master 929fdbea rf on CUDA with a class absent from the labels (`class_need_train_`
  false) segfaults, key off or on, so the md5 identity under fixedpoint error feedback with a skipped class (where
  sequential training's residual slot `counter % num_class` drifts across classes and the slots do not) is untested.
- The memory budget measures footprints with `cudaMemGetInfo` deltas, which other processes on the GPU disturb (the
  reserve of an eighth of the device absorbs ordinary noise).
- Speed only: a `reset_parameter` callback that changes a parameter every round drops the group every round (with a
  fixed `forest_group` the group's other rounds are grown and discarded); `update()` past `num_iterations` makes and
  frees the slots every round.
- The GPU-serial floor (§3): the slots' kernels interleave on the legacy default stream. Removing it needs either
  merged launches across slots (§3 (a), the cheap form) or per-slot non-blocking streams throughout the learner.
- GPU-bound shapes (fp64 deterministic construct) gain nothing: stage 3 (count histograms) is their lever.
