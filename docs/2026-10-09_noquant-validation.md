Falcata noquant validation — 2026-10-09

The selected FP64 policy passes the completed correctness gates. The fresh quality comparison has large
seed variation and recorded fraud sanity failures; it establishes neither a quality improvement nor a
cost-free quality tradeoff. This dated verdict and its evidence are immutable. Strict timing jobs await
admission and will receive a separate report; no speed gain is claimed here.

The measured final runtime is `d0bb8c47789e062acbf6b770918a0589a74ae9b2`, with `cuda_precision=fp64`
retained by user choice. Source `63a1fc93` adds documentation only after that build. The reference runtime is
`09b6be9724f330f947c86de6e819654b077f70f2`.

New work routes atomic-path FP64 threshold prefixes through parallel scans while preserving CPU-order
routing for eligible deterministic histograms. Paired CPU-order folds, integer gain keys, FP32 gain bounds
and grouped most-frequent-bin fixes were already in baseline `09b6be97` (PR65). Three independently
ablatable graph ports are new: `graph_apply_rows` preserves chunk/ballot/partition order;
`graph_apply_fused` combines independent metadata writes and disjoint terminal-window copies; and
`graph_skip_unsplittable` ports the conservative host count bound and forced-split exclusion. Existing host
switches still control their components.

FP64 alone does not guarantee CPU bit parity. Eligible deterministic constructs retain CPU-order scans and
CPU split parity, including supported `graph_det:on` shapes. Atomic graph histograms are order-dependent;
their parallel scans can choose another bin on an equal-gain plateau. Exact quantized/deterministic
plan-key checks remain mandatory, and atomic variability is not evidence of bit identity. Partial resets
preserve backend, precision and automatic quant-bin intent; unsupported live CUDA mode changes fail before
altering state. Ordinary config/data resets release graphs and rebuild static thresholds, budgets and pointers.

Final-runtime correctness results, with `FALCATA_VERIFY=1`: canonical **4/4**, lattice **357/357**, and
**284 CUDA tests passed**, 10 skipped, 4 expected failures (GPUQ jobs 3619/3620, relaxed admission).
All previous fingerprint entries are unchanged. Added coverage comprises 56 graph cells with 13 new
fingerprint bases plus one explicit-FP64 validity cell. The 23 graph lifecycle cases check actual capture,
fusion node counts, exact on/off model/prediction hashes, tails, gaps, count boundaries, resets and intended
fallbacks. Pinned checks passed. [Compact correctness results](2026-10-09_noquant-validation/correctness.json)
retain each lattice outcome and the canonical locks.

The policy was declared before scoring fresh seeds **9–16**. Both arms use FP64, the same fixed caches and
the 500-round deep regime. Every one of the eight paired endpoints is finite; there are no missing draws or
unscored infrastructure errors. Mean ± sample SD includes finite scores from sanity-failed draws; no seed
was replaced. A positive paired difference favors accuracy/AUC; a negative one favors log loss.

| Metric | Baseline mean ± SD | Selected FP64 mean ± SD | Paired mean | Paired median | Paired SD | Worse pairs |
|---|---:|---:|---:|---:|---:|---:|
| Covtype accuracy | 0.845091 ± 0.072755 | 0.866198 ± 0.044199 | +0.021106 | −0.003954 | 0.094276 | 5/8 |
| Covtype log loss | 3.473879 ± 2.639676 | 2.662988 ± 1.154900 | −0.810891 | +0.415928 | 3.225610 | 6/8 |
| Fraud AUC | 0.626190 ± 0.090022 | 0.670312 ± 0.125216 | +0.044122 | +0.046611 | 0.151954 | 3/8 |

Covtype's favorable mean changes are outlier-sensitive: most pairs and both paired medians move adversely.
Fraud's positive mean/median comes with wide dispersion and sanity failures on baseline seed 15 (**1/8**)
and selected-runtime seeds 10/12 (**2/8**). Covtype has none. These eight draws do not prove improvement or
quality equivalence. [Fresh endpoints and complete summaries](2026-10-09_noquant-validation/acceptance-seeds9-16.json)
retain the ranges, failure flags and per-seed values.

Earlier seeds **1–8** remain descriptive precision attribution. Their candidate was `f38de5c4`, evaluated in
FP32 and explicit FP64; neither arm is the final `d0bb8c47` runtime, and these draws are not pooled with
the fresh policy block.

| Metric, seeds 1–8 | Baseline FP64 mean ± SD | `f38de5c4` explicit FP64 | `f38de5c4` FP32 |
|---|---:|---:|---:|
| Covtype accuracy | 0.844845 ± 0.013678 | 0.849197 ± 0.019437 | 0.827086 ± 0.032536 |
| Covtype log loss | 3.283803 ± 0.596114 | 3.163458 ± 0.913554 | 3.988093 ± 1.316941 |
| Fraud AUC | 0.642722 ± 0.140123 | 0.685618 ± 0.123056 | 0.645905 ± 0.101598 |

Fraud sanity failures were baseline seeds 1/6/7, explicit-FP64 candidate seeds 6/7, and FP32 candidate seed 8;
all finite failed endpoints remain included. Covtype had none. This is not a universal precision ranking.
[Attribution evidence](2026-10-09_noquant-validation/precision-attribution-seeds1-8.json) preserves the actual
runtime, precision and library digest of each draw.

All reported quality jobs used relaxed admission. Desktop idleness was not measured, and `contended=0`
does not establish it. Active scheduling can alter floating histogram atomic order and thereby quality,
but these fixed-budget paired holdouts cannot quantify or attribute a desktop effect. That possibility
does not establish that FP32 has no quality cost. Quiet-GPU performance requires the separate strict timing
report.

The single RTX 5090 is serialized through GPUQ. Builds match CUDA 13.0.88, architecture `120-real`, GCC
15.2.0 and Python 3.12.13. [Reproduction details](2026-10-09_noquant-validation/reproduction.md) and
[machine-readable provenance](2026-10-09_noquant-validation/reproduction.json) pin the library digests,
regime, cache shapes, harness identities and evaluator. Numerai quality metrics use `NumeraiEvaluator(cpu)`;
its historical build1226 cache has an unidentified label, so these results compare implementations on fixed
data and do not establish current Numerai target selection, prequential selection or a production release.

Float row batching/fused-root and pair-joint/code-word histogram regrouping remain deferred because they
cannot guarantee general bit identity; continuous-gradient probes varied even with the recovered keys off.
Graph `final_map_only` needs a shared pending-partition/buffer-ownership protocol for later readers, reset
and refit. These limitations are recorded in [perf-dead-ends.md](perf-dead-ends.md); no histogram-regrouping
or map-only speed claim is made.
