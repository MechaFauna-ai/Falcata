# Numerai training throughput with five values and a missing state

Falcata trained the six-state Numerai deep benchmark at **192.65 trees/s**:
30,000 accepted trees in **155.721 s**, versus **266.19 trees/s** and
**112.702 s** with the original five numeric states. The sixth state increased
elapsed training time by **38.2%** in this single full-length pair.

The current Numerai recipe behaves differently. Its 2,000-round six-state
probe reached **10.29 trees/s**, versus **88.21 trees/s** with five states.
Diagnostic warmups show that its larger memory reservation selects a full-width
mask view instead of the compact sampled view. This observed planner threshold
accompanies a much larger timing penalty than the standard recipe's path change.

All **27 expected records passed**: one representation-equivalence gate, six
diagnostic warmups, eighteen short timed draws and two full-length draws.
Failed, pending, blocked and contended/excluded counts are **zero**. Strict
GPUQ job 3653 finished successfully; its contention log is empty. No failed
draw, seed or configuration was replaced.

![Paired training throughput with five states and a distinct missing state](blog/falcata-1.1/six-state-throughput.png)

## Measured throughput

The short probes are medians of three fresh-process draws per arm, ordered
five/six, six/five, five/six. They repeat seed 42 for timing; they are not three
independent model-quality seeds. Full-length results are one draw per arm.
Every timed model accepted its entire requested tree budget and passed finite,
nonconstant canonical scoring sanity.

| Recipe | Requested and accepted trees | Five-state seconds | Six-state seconds | Five-state trees/s | Six-state trees/s | Six/five elapsed time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard stochastic, FP64 default | 2,000, median of 3 | 8.120793 | 11.088747 | 246.28 | 180.36 | 1.365× |
| Standard fixed point, FP64 default | 2,000, median of 3 | 9.444341 | 11.428993 | 211.77 | 174.99 | 1.210× |
| Current recipe, fixed point, explicit FP32 | 2,000, median of 3 | 22.673740 | 194.446197 | 88.21 | 10.29 | 8.576× |
| Standard stochastic, FP64 default | 30,000, one pair | 112.702339 | 155.721482 | 266.19 | 192.65 | 1.382× |

The corresponding short-probe throughput decreases are 26.8%, 17.4% and
88.3%; the full stochastic pair loses 27.6%. Time ratios and throughput
percentages use different denominators. A 1.382× time ratio means 38.2% more
elapsed time, rather than 38.2% less throughput.

These are boosting-loop timings. The full five/six pair separately measured
construction at **3.900/4.088 s**, CPU prediction at **17.420/19.693 s**, and
peak process GPU memory at **13,586/13,921 MiB**. Construction and scoring are
excluded from trees/s. Short probes do not predict a 15,000-round current-recipe
run or replace the measured full 30,000-round pair.

## What the sixth state means

The Numerai loader identifies feature columns that are constant at **2 over a
complete source era**, then marks those entries missing. It preserves ordinary
twos and zeros. Conversion happens before label filtering, avoiding a false
constant-column decision on an incomplete era. The five-state arm retains
numeric 0–4; the six-state arm uses native int8 `-1` with
`missing_sentinel=-1`, `use_missing=true`, and `zero_as_missing=false`.
CPU prediction restores that sentinel to real NaN.

Both arms use the same frozen build **1230**, **5,502,748 training rows**,
**3,555 features**, labels, feature order and split. All 6,846,250 source rows
were verified; the two retained matrices differ at exactly **1,554,367,577**
entries, all `2 → -1`. There are 1,292,366 held-out rows over eras 1031–1230;
training ends at era 1022, with eras 1023–1030 embargoed using the named
`Target("target_ender_20")` contract. All 200 held-out eras are complete.

Build 1230 is deliberately pinned for this representation comparison because
the historical benchmark's full build 1226 source is no longer available.
The new times are **not a before/after comparison against that old cache**.
There was no data refresh or production retrain.

Actual constructed bin counts, identical across recipes, are:

| Arm | Features with five bins | Features with six bins | Filtered constant features | Total |
| --- | ---: | ---: | ---: | ---: |
| Five states | 3,525 | 0 | 30 | 3,555 |
| Five plus missing | 1,583 | 1,942 | 30 | 3,555 |

Before timing, a 32,768-row, eight-tree gate required native int8 sentinel,
float16 NaN and float32 NaN to produce exactly equal tree fingerprints, bin
metadata and CPU predictions. All three accepted eight trees and produced
finite, nonconstant predictions. This establishes representation fidelity;
it does not assert that the five-state and six-state models are identical.

## Which optimized paths survive

The standard recipes retain the compact sampled view, tiled fill, pair-joint
histograms and the narrow FP64 warp split finder with missing values. A warp is
the GPU's group of 32 cooperating threads. The five-bit pair-code view declines:
two six-bin features need 36 joint cells, beyond its 32-cell limit. A six-by-five
pair needs only 30 cells, but one oversized selected pair disables that tree's
code view.

| Diagnostic warmup | View and fill | Histogram construction | Split finder | Five-bit pair view |
| --- | --- | --- | --- | --- |
| Standard five-state recipes | Compact, tiled | Pair-joint and code-word paths observed | Warp | ON in 191/200 trees |
| Standard six-state recipes | Compact, tiled | Pair-joint path observed | Warp | OFF in all 200 trees |
| Current recipe, five states | Compact, per-cell fill | `pair_hist` | Block | Not engaged |
| Current recipe, six states | Full-width mask view | `row_batch` | Block | Not applicable |

These are observations from separate 200-round diagnostic warmups. Timed
draws suppress diagnostic output; the table does not pretend to trace every
launch inside their training loops. The current recipe's sampled view is too
wide for the tiled/code-word route even with five states, so its slowdown
cannot be attributed to losing that route.

## The current recipe crosses a memory threshold

The standard recipes use depth 10, 1,024 leaves, learning rate 0.001,
feature fraction 0.1, leaf floor 10,000, 32 CPU threads and `max_bin=255`.
Stochastic mode resolves to four gradient levels; fixed point resolves to 64.
FP64 is the library default.

The current `best.textproto` probe uses its declared depth 15, 16,384 leaves,
learning rate 0.003375, feature fraction 0.15, leaf floor 101,079 and 16 threads,
with fixed point 64 and **explicit FP32**. Its leaf floor is not rescaled for
production rows. Its 15,000-round declared model budget is not run here.

The planner reserves room for future split output and quantized histogram
scratch, rather than only the leaves already grown. With the large declared
leaf budget, the six-state bin layout raises that reservation enough to reject
the compact view:

| Logged warmup admission, MiB | Five states | Six states |
| --- | ---: | ---: |
| Column store | 9,250 | 9,250 |
| Compact view | 1,402 | 1,402 |
| Additional reservation | 10,701 | 15,922 |
| Available at the check | 24,784 | 24,281 |
| Decision | Compact fits | Mask fallback |

At 16,384 declared leaves, the reservation allows **8,194 possible sibling
pairs**, even though the warmups grow about 41 leaves per tree. Missing-valued
features require an extra split-search task: the logged task count rises from
3,525 to 5,467. Those tasks and the additional bins enlarge future split-output
and histogram scratch allocations. The standard 1,024-leaf recipe reserves
only 514 pairs and retains its compact view in both arms.

The logged decision is direct evidence of the path change and a strong
explanation for the 8.576× elapsed-time difference. The arms also learn
different models. No forced-view ablation or new runtime was used to assign
every millisecond to that mechanism or claim a repaired performance result.
Reducing a declared capacity or changing memory planning would require its own
separately recorded correctness and timing comparison.

Source at the measured revision:
[future allocation reservation](https://github.com/MechaFauna-ai/Falcata/blob/d0bb8c47789e062acbf6b770918a0589a74ae9b2/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp#L466),
[view admission and fallback](https://github.com/MechaFauna-ai/Falcata/blob/d0bb8c47789e062acbf6b770918a0589a74ae9b2/src/treelearner/cuda/cuda_histogram_constructor.cpp#L1580).
The [split-task metadata](https://github.com/MechaFauna-ai/Falcata/blob/d0bb8c47789e062acbf6b770918a0589a74ae9b2/src/treelearner/cuda/cuda_best_split_finder.cpp#L199)
and [future split-output sizing](https://github.com/MechaFauna-ai/Falcata/blob/d0bb8c47789e062acbf6b770918a0589a74ae9b2/src/treelearner/cuda/cuda_best_split_finder.hpp#L393)
show why the missing state increases this reserve.

## Quality and provenance limits

`NumeraiEvaluator(cpu)` scored the same complete held-out eras **1226–1230**
outside training timing. The full stochastic pair's mean CORR is
**0.035861/0.037087** for five/six states. These five-era descriptive sanity
scores support neither equal quality nor a deployment, target selection,
production or prequential selection verdict. Every per-era score and model
fingerprint remains in the compact evidence.

The existing shared tuner wisdom was preserved and copied before/after each
worker. Cache reuse and initial tuner effects remain inside measured training
time; no fresh-cache or confidence-interval claim is made. The two full runs
have no repeat-based uncertainty estimate.

Measured runtime: `d0bb8c47789e062acbf6b770918a0589a74ae9b2`; installed library
SHA256 `e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c`.
Blueprint SHA256 `0f6eaf8fc6a8404fcdc501070bdad93b61af22c67ecc2031dbfed97f09312b49`;
measurement plan SHA256 `067b3f7e3ebb55c7a9e0652f52cc415e3b1256fdbdbf880ef3ca6017125492c0`;
data manifest SHA256 `6a5943f3657fe126886e3626021c30672f5627db48514ca21959ae0c5e80c972`.

[Compact raw evidence](2026-10-10_numerai-six-state-throughput.json),
[memory diagnostic](2026-10-10_numerai-six-state-diagnostic.json),
[numbered startup excerpts](2026-10-10_numerai-six-state-log-excerpts.txt),
[chart data](blog/falcata-1.1/six-state-data.json), and
[chart generator](blog/falcata-1.1/plot_six_state.py) preserve the endpoint
identities and reproduce the chart. The measurements were made on one RTX 5090
under strict GPUQ admission on 2026-10-10, Europe/Zurich.
