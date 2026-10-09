# Fraud failure diagnosis and capped reruns — 2026-10-09

The user requested a diagnosis, repair and rerun of the failed fraud cells after
the [overnight report](2026-10-09_noquant-timings.md). This follow-up preserves
that immutable evidence. It confirms unregularized numerical collapse and a
raw-score/probability AUC mismatch in the benchmark check; it found no prediction
cache inconsistency in these fresh probes. The library artifact is unchanged.

## Diagnosis, same uncapped recipe and explicit capped comparison

Strict GPUQ jobs 3647–3650 ran the pinned d0bb runtime, FP64/noquant, seed 42,
32 threads, the same cached fraud train/test rows, and 500 requested updates.
The two uncapped probes reproduce the original configuration. The two cap-1
probes explicitly change only `max_delta_step`; none replaces an original draw.
Extra CPU predictions make these correctness probes unsuitable for timing claims.
All four jobs completed with zero recorded contention.

| Configuration | Accepted trees | Native AUC | CPU raw AUC | CPU probability AUC | Saturated probabilities / 56,962 | Largest absolute raw score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shallow-uncapped | 131 | 0.355680363 | 0.355680363 | 0.545312554 | 56,950 | 1,046,978.82 |
| deep-uncapped | 277 | 0.674529023 | 0.674529023 | 0.514929642 | 56,956 | 551,711.80 |
| shallow-capped | 500 | 0.979257527 | 0.979257527 | 0.979257527 | 0 | 32.81 |
| deep-capped | 500 | 0.979155960 | 0.979155960 | 0.979155960 | 0 | 29.70 |

Native binary AUC ranks the raw validation score buffer. CPU `predict()` normally
applies a sigmoid to those scores. On these collapsed models almost every
probability rounded to exactly zero or one, turning distinct raw ranks into
ties. Native AUC matches CPU raw-score AUC at every diagnostic checkpoint,
whereas probability AUC can differ substantially. CPU and cached probabilities
are bit-identical for training and validation at all checked points (maximum
absolute difference zero). Direct raw validation-cache values are not exposed
by the public API, so direct per-row raw-cache equality is not claimed.

Uncapped shallow/deep first returned finished at update calls 132/278, with
131/277 accepted trees. The model remained unchanged through the remaining
requested calls. This is ordinary no-split termination, not a tree silently
removed after scoring. The old curve loop ignored that return value and drew
flat points out to the requested 500 updates.

The instability has prior evidence in commit `568ad02c`: tiny logistic Hessians
with zero L2 and no leaf-step cap allow huge finite Newton steps on severely
imbalanced data. It is not explained by desktop contention. The existing
binary-objective warning already recommends a cap. `max_delta_step=1` bounds the
leaf update before learning-rate shrinkage; it is a documented configuration
repair, not a new precision default or an automatic change to other datasets.
The cap comparison is not an unbiased configuration-selection experiment.

## Benchmark corrections and new frozen reruns

The corrected harness retains probability AUC and its sanity threshold. For
LightGBM-family binary curves it additionally recomputes CPU raw-score AUC after
the training timer, records its separate scoring overhead, and checks the native
curve against that matching representation. The integrity tolerance is not
loosened. A wrong raw series still fails. Curve loops stop on finished and record
requested updates, actual accepted iterations/trees and stop reason; throughput
uses the actual model count. An explicit `--set max_delta_step=1` reaches the
training parameters. Default shallow/deep recipes remain uncapped.

Two further strict jobs rerun the separately frozen cap-1 shallow/deep recipes,
each in five fresh processes: warmup, three timed draws, and curve. All ten
endpoints passed probability sanity and native/raw curve agreement, with no
recorded contention. Every endpoint is retained in the sidecar.

| New configuration | Endpoints passing | Median training seconds, 3 draws | Probability AUC range, all 5 draws | Last curve iteration |
| --- | ---: | ---: | ---: | ---: |
| shallow, cap 1 | 5/5 | 0.473757 | 0.979257527–0.979258065 | 500 |
| deep, cap 1 | 5/5 | 0.524587 | 0.979156139–0.979157036 | 500 |

These absolute times describe capped configurations. They do not establish a
before/after speedup against uncapped failed models or an equivalent-quality
comparison between runtime versions. Original sanity failures and rejected
curves remain failures in their original report; the metric fix does not
retroactively relabel models whose scores were not saved.

## Evidence and reproduction

- [Compact raw evidence](2026-10-09_fraud-followup/evidence.json): all diagnostic
  checkpoints, summaries, parameters, cache/source/library hashes, all ten new
  endpoints and queue admission/contended state.
- [Frozen rerun manifest](2026-10-09_fraud-followup/capped-rerun-manifest.txt),
  SHA256 `c9269eeda0931207873e14f9d17069486848f240bf453d119831983f1e253b03`.
- Exact [diagnostic](2026-10-09_fraud-followup/diagnose_fraud.py.txt) and
  [fresh-process rerun](2026-10-09_fraud-followup/run_capped_reruns.py.txt) scripts.
  Full models and prediction arrays remain under
  `/tmp/falcata-fraud-debug-20261009/{shallow,deep}-{uncapped,capped}`.

The measured runtime remains `d0bb8c47789e062acbf6b770918a0589a74ae9b2`, library
SHA256 `e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c`.
The manifest separately pins the corrected Python harness. Source and library
identities are different contracts; no runtime bug was repaired by substituting
a different binary. CPU regression and pinned static-check results accompany
the source change in PR #71.

All ten CPU regression cases pass on the pinned runtime, including native AUC
saturation, retained probability sanity, rejection of a wrong raw curve, stopped
curve axes, actual-tree throughput and explicit cap propagation. The complete
82-checkpoint [diagnostic audit](2026-10-09_fraud-followup/diagnostic-audit.json)
passes. All applicable pinned static hooks pass; the
[check output](2026-10-09_fraud-followup/static-checks.txt) records the result.
