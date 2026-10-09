# Falcata 1.1: keeping the GPU busy, from the first row to the last tree

> **Release article draft, 9 October 2026.** This is a preview of the proposed
> 1.1 release, including the pending non-quantized training changes in
> [PR #71](https://github.com/MechaFauna-ai/Falcata/pull/71). Version 1.1 has not
> been published. Features already shipped in the 1.0 series are identified below.

When we [introduced Falcata](https://www.mechafauna.ai/blog/falcata-gpu-gradient-boosting),
the main idea was to change how a tree asks the GPU for work. A conventional
leaf-wise learner grows the most promising leaf, waits, then does it again.
Falcata batches split work across a level while the growth constraints permit it,
and handles a binding leaf budget separately. That gives the GPU more useful work
between conversations with the CPU.

That made the training loop much faster. It also made the next bottlenecks easier
to see: moving the same columns twice, issuing a dozen tiny copies, searching
thresholds in a serial chain, and preparing histograms that no split would ever
use. Once a tree takes milliseconds, small costs repeated thousands of times
become the costs that matter.

The work towards 1.1 follows those costs from dataset construction to prediction.
It also strengthens the contracts around model round trips, live parameter
changes, and detecting a fast model that has stopped being useful.

## The new measurements

The latest quiet-machine comparison targets **non-quantized training**:
`quant_mode="none"`, with **FP64 still the default**. Here, non-quantized means
the gradients and Hessians are not packed into small integer values; feature
values still use the tree learner's bins. FP64 means 64-bit floating
point for histogram storage and split arithmetic. Objective gradients and Hessians
still use their established representation; these gains come from organizing
the FP64 work more efficiently.

Both builds ran on the same RTX 5090, using the same cached data, configuration,
seed and 32 CPU threads. GPUQ admitted them serially under its strict quiet-machine
policy. Short workloads use three interleaved timed runs per build after warmup.
Dataset construction and prediction are outside the training times below.

| Workload | Before | After | Less training time |
| --- | ---: | ---: | ---: |
| Year, deep | 3.50 s | 2.74 s | **21.8%** |
| Epsilon, deep | 57.77 s | 46.64 s | **19.3%** |
| Higgs, deep | 15.93 s | 9.66 s | **39.4%** |
| Numerai example, 2,000 rounds | 54.81 s | 53.81 s | **1.8%** |
| Numerai deep, 30,000 rounds | 22.88 min | 20.61 min | **9.9%** |

![FP64 non-quantized training takes less time on the five measured workloads, comparing October development builds](falcata-1.1/fp64-training-time.png)

The [chart data](falcata-1.1/training-time-data.json) and
[plotting script](falcata-1.1/plot_training_time.py) reproduce this figure.

The full Numerai-deep result is **one complete run per build**: two minutes and
sixteen seconds saved, with holdout CORR moving from 0.0238562 to 0.0238004.
One pair cannot establish repeatability or equivalent quality. Year and Epsilon's
quality ranges overlap; Higgs has a similar median AUC and one lower-scoring draw.

**What “before” means matters.** This table compares development commit
`09b6be97` with `d0bb8c47`. The former already contains the public October
optimizations described later in this article. The table measures the additional
work in PR #71. Tagged 1.0 and competing engines were not rerun; the launch post's
cross-library numbers belong to its original experiment.

The [dated timing report](https://github.com/MechaFauna-ai/Falcata/blob/a478c3f613f838e952b163c16e38c500cedc2a64/docs/2026-10-09_noquant-timings.md)
retains configurations, loaded-library hashes, endpoints and failures. All 35
strict queue jobs finished: 33 done, two failed, zero recorded contention.

## Faster split search without a precision shortcut

A decision tree chooses a split by asking which feature threshold reduces the
loss most. To answer efficiently, it builds a **histogram**: for each feature bin,
it sums gradients, which say which way predictions should move, and Hessians,
which describe the local curvature of the loss.

Testing successive thresholds requires **prefix sums**. The first threshold gets
bin 1; the next gets bins 1 and 2; the next gets bins 1, 2 and 3. A literal loop
makes each result wait for the previous addition.

The new FP64 path uses parallel scans, exchanging partial sums within a **warp**,
a group of 32 GPU threads. A short tree of steps makes more thresholds ready
together. The arithmetic stays FP64, but a parallel sum
can add values in a different order. Floating-point addition is not associative,
so “same precision” does not mean “identical bits.”

That is why the change is restricted to the graph route that already builds
histograms with floating-point atomics, whose accumulation order can vary.
Eligible deterministic histogram routes retain CPU-order scans. Host-launched
and classic routes also retain those scans, although an atomic histogram fallback
still prevents a general CPU-parity guarantee.

Earlier work in the development line sped up the CPU-order finder while keeping
its arithmetic order: independent serial chains run in separate lanes, finite
gain comparisons use ordered integer keys, and cheap FP32 bounds screen candidates
before expensive exact work. A bound is a filter, not the final answer: candidates
that can win still receive the FP64 calculation. These changes were already in
the baseline of the new table.

## Giving captured graphs the same efficient bookkeeping

A **CUDA graph** records a sequence of GPU operations and replays it with less CPU
launch overhead. Recording an efficient sequence matters just as much as replaying
it efficiently.

Three existing optimizations now also serve the captured level loop:

- **Process more rows per thread.** Row-partition kernels can keep more rows in
  flight. The planner uses the GPU's occupancy calculation to choose suitable
  builds: enough work resident on a compute unit can hide memory waits.
- **Fuse apply bookkeeping.** Applying a split means routing rows, recording the
  tree structure and preserving rows in terminal leaves. Combining compatible
  work removes intermediate launches and passes over those rows.
- **Skip leaves that cannot split.** A conservative row-count bound identifies
  leaves too small for two legal children, allowing for count rounding. Forced
  splits keep their established path. Unusable histograms and searches are skipped.

The ablations put limits on the story. On Numerai example, the row-kernel port
saves about 1.9% in a same-build ON/OFF comparison; the fused apply port is
effectively unchanged. Count pruning saves about 7% of elapsed time in the
declared 500-round Numerai-deep probe. That probe is not a prediction of a
30,000-round gain.

Forced `graph_quant:on` was slower than OFF in all seven historical quantized
cases we reran, despite matching timed model and prediction fingerprints.
Quantized graph mode stays opt-in: the stopwatch decides whether fewer launches
help the complete workload.

## Moving less data, and making each load do more

Much of the work since the launch concerns wide, low-cardinality data such as
Numerai: thousands of columns, but only a few values per column.

### One persistent matrix, with a view that fits the tree

**Column sampling** means a tree uses only a fraction of the features. Reading a
full row anyway wastes bandwidth on columns the tree will never inspect.
Falcata's compact view gathers the selected columns once, then uses that smaller
matrix throughout the tree.

The development line keeps either a column-major store plus a compact per-tree
view, or a full row-major view with a feature mask when almost every column is
used. A runtime probe chooses the crossover and caches its decision.

In the documented 6.79-million-row, 3,555-feature experiment at
`feature_fraction=0.1`, peak device memory over idle went from **25.0 to 13.7 GiB**.
The live layout probe can temporarily use the larger candidate layout; this
specific comparison describes steady training memory.

### Histograms for pairs of columns

The quantized path can accumulate a **joint histogram** for each pair of feature
columns. Each cell describes one combination of their bins; summing rows or
columns recovers the ordinary per-feature histograms used by split search.

For two five-valued columns there are only 25 combinations. Their joint-table
index therefore fits in **five bits**. The newer compact representation stores six
such indices in a 32-bit word. On the documented Numerai shape that makes a
compact row 128 bytes instead of 178 bytes, within the same allocated buffer.

The larger gain comes from how it is read: one thread handles a code word, so a
single row index and gradient load serve six column pairs. The measured construct
kernel used 63% fewer load/store instructions and fell from 2.05 to 1.27 ms per
round. This is an example of an **issue bottleneck**: the GPU was spending too
many instructions requesting data, even before the memory bandwidth was exhausted.

The [documented pair-code experiment](../performance.md#10d-the-numerai-round-after-58-a-5-bit-pair-code-view-of-the-compact-rows)
measured 4.67 to 3.84 ms per training round with identical models. It used the
5.46-million-row benchmark split and a quantized configuration. These earlier
gains are already represented in the baseline of the FP64 table above.

### Removing waits around useful work

The same series batches small tree metadata uploads, overlaps safe transfers,
starts row partitioning before host bookkeeping, and lets producing kernels write
their own readback buffers. Constant-Hessian objectives avoid reading an array
whose entries are all the same, and terminal levels avoid partition work that
only a later level would use.

Stream dependencies keep the ordering intact; waits move to where their answers
are needed. The [performance notes](../performance.md#10b-the-numerai-round-after-the-pair-joint-construct-occupancy-the-tree-boundary-level-pruning)
record profiles and ablations, including optimizations that lost and remained off.

## Starting sooner, and predicting without a giant allocation

Training time begins after a dataset has been built. For a large input, that
preparation can be substantial, so it deserves its own measurements.

Dataset construction reuses **page-locked staging buffers**, host memory prepared
for GPU transfers, so chunk uploads can overlap binning. CPU preparation uses radix
sorting where appropriate, skips redundant initialization, and creates the CUDA
context during sampling. Numerical matrices with categorical columns can bin
those columns on the device too.

Native int8 ingestion remains useful for five-valued data. Our latest isolated
probe measured median construction of **3.89 s for int8 versus 31.34 s for
float32**, with peak host RSS **32.3 versus 64.9 GiB**. That is a comparison of input
representations within the selected build, without held-out quality evaluation.
Older grouped runs reused datasets and cannot establish a construction speedup
against these fresh-process measurements.

Prediction through NVIDIA's **Forest Inference Library (FIL)** now scores large
host inputs in bounded slabs, avoiding an allocation for the entire wide test
matrix. The benchmark records the actual backend and stages device slabs when
CuPy is available. The overnight table measures training; inference speed needs
its own comparison.

## More kinds of trees, with clearer contracts

The 1.0 series has also grown beyond single-output numerical trees. These features
already shipped in **1.0.5** and form part of what has changed since the first post:

- **Vector leaves** put several target predictions in each leaf of one shared
  tree. Low-cardinality, sampled features benefit most from shared work;
  continuous-feature histogram construction can dominate. CUDA multi-regression
  supports up to 16 targets, with documented precision and multi-GPU restrictions
  and a CPU fallback for FIL prediction.
- **ObliquePool** supplies sparse random projections as ordinary features. A
  threshold on a projection can describe a slanted boundary across several
  original features. The fitted pool can be saved beside the model and must be
  applied again at prediction time.
- **Random categorical search** tries seeded random subsets of categories instead
  of only the sorted search. It offers another regularization choice for features
  with many categories; `cat_random_search=0` keeps the existing search.
- **Packed four-bit storage and chunked CUDA input** reduce representation costs
  and let datasets arrive as a list of device arrays.

The [1.0.5 release notes](https://github.com/MechaFauna-ai/Falcata/releases/tag/v1.0.5)
also document correctness repairs: round trips now preserve model statistics and
handle constant trees; quantized training gets protection against tiny Hessians
under bagging and synchronized rounding errors. Those quantization repairs
intentionally change affected trained models.

More recent development hardens model lifecycle and importer behavior, loads the
CUDA driver lazily so CPU use can import without an NVIDIA driver, and adds
vector-leaf gate coverage. Fixedpoint training with per-row Hessians also gets a
small, quantization-scale **ridge**: a stabilizing denominator term that prevents
tiny rounded Hessians from producing unreasonable split gains and leaf updates.
Its covtype quality change is intentional and has updated regression fingerprints.

PR #71 also makes live parameter changes safer. A CUDA learner allocates resources
for its precision and quantization mode when training starts. Changing those modes
without rebuilding the learner could make the configuration disagree with its
actual buffers. Unsupported changes now fail before altering the live state;
ordinary learning-rate updates and settings that resolve to the current mode
remain supported. Partial updates preserve the original backend and automatic
quantization intent.

## What the failures taught us

Covtype accuracy varies between draws. Fraud's unregularized binary configuration
produced near-random probability AUC in several runs; both learning curves were
rejected by the existing harness. Those records remain excluded from the headline
table.

The follow-up diagnostics explain both problems. On severely imbalanced data,
tiny Hessians let uncapped Newton updates become huge while remaining finite.
Uncapped shallow and deep runs reached margins above 550,000, with almost every
holdout probability rounded to 0 or 1. Native AUC matched CPU raw-margin AUC;
cached probabilities matched CPU tree predictions exactly. The apparent curve
disagreement came from comparing raw-margin ranks with saturated probabilities.

With explicit `max_delta_step=1`, both diagnostic runs completed all 500 trees,
had no saturated probabilities, and reached AUC 0.97926 shallow and 0.97916 deep.
The subsequent strict rerun passed **all ten endpoints**: a warmup, three timed
draws and a curve for each capped configuration. All accepted 500 trees, with
AUC remaining around 0.979 and no recorded contention. This is a separately
labelled [follow-up experiment](https://github.com/MechaFauna-ai/Falcata/blob/523ed775ad2eec9eeca147c4815c39a8f9400f70/docs/2026-10-09_fraud-followup.md);
the archived uncapped failures remain unchanged. The harness correction compares
AUC on the same representation and records actual completed trees when training
stops. The cap bounds each leaf update before applying the learning rate; it
repairs this unstable configuration without changing the library's defaults.

**FP64 stays the default.** Fresh precision checks found small Epsilon AUC changes
and unstable covtype outcomes. Explicit `cuda_precision="fp32"` remains available;
`"auto"` opts into FP32 on eligible non-quantized CUDA routes and retains FP64 for
deterministic, double-precision histogram and vector-leaf requirements.

## How we keep the claims testable

The measured runtime passed four canonical model gates, a 357-cell verification
lattice and the touched CUDA suite: 284 passed, 10 skipped, four expected failures.
Graph ports have separate identity and lifecycle tests; numerical changes have
fresh-seed quality evidence, including failed seeds.

Much of this work is agent-assisted: separate coding agents inspect profiles,
implement candidates and review the evidence. Their proposals go through the
same identity, validity and performance gates. A profile can suggest an idea;
an interleaved experiment and a regression test decide whether it earns a place
in the library.

Matching fingerprints establish identity on a tested route. Overlapping quality
ranges and uncontended timings answer different questions. The Numerai cache is
a pinned historical reproduction with an unidentified label, scored with
`NumeraiEvaluator(cpu)`; it compares implementations on fixed data rather than
selecting a current target or production model.

The proposed release keeps the familiar `falcata.train()` API. Across the pipeline,
the goal is consistent: give GPU operations useful work, move only the data they
need, and wait where the algorithm needs an answer.

The code is MIT licensed at [MechaFauna-ai/Falcata](https://github.com/MechaFauna-ai/Falcata).
The [performance notes](../performance.md),
[dead ends](../perf-dead-ends.md) and
[latest quality report](https://github.com/MechaFauna-ai/Falcata/blob/a478c3f613f838e952b163c16e38c500cedc2a64/docs/2026-10-09_noquant-validation.md)
contain the detailed experiments. Workloads that disagree with these results are
welcome: they are how we find the next bottleneck, or the next broken assumption.

Falcata derives from LightGBM (MIT, Microsoft Corporation and the LightGBM
developers). The measurements above are for a single RTX 5090 and the stated
data splits and configurations.
