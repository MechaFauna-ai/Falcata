# Deep classification stability: CPU on 12 October, FP64 CUDA on the release candidate

Updated 21 October 2026. The CPU half below is the 12 October validation, repeated on
`1408ee79` (the measured base of release candidate `49c5622a`). The FP64
CUDA half is rerun on `1408ee79`: since then
[#122](https://github.com/MechaFauna-ai/Falcata/pull/122) made non-quantized CUDA
histograms deterministic fixed-point sums, and
[#154](https://github.com/MechaFauna-ai/Falcata/pull/154) corrected column sampling on
bundled data (covtype's sampled recipe uses `feature_fraction=0.8`), so the October
CUDA models no longer describe the release ([checklist](../NUMBERS.md) for the
source of every rerun value).

This extends the earlier single-seed diagnosis with
full-data coverage on fraud and covtype. [PR #91](https://github.com/MechaFauna-ai/Falcata/pull/91)
adds automatic Newton-step stabilization; [PR #93](https://github.com/MechaFauna-ai/Falcata/pull/93)
fixes a separate CPU empty-child failure found during this sweep. Both are merged.
The final candidate is
[`08c37d95`](https://github.com/MechaFauna-ai/Falcata/commit/08c37d950deab5fcf868edaa0564b97534a435ab),
merged as [`a65d34e8`](https://github.com/MechaFauna-ai/Falcata/commit/a65d34e80a7c337342fba6886d0a1cbd69eda079)
with an identical source tree.

## What failed and what changed

The original non-quantized collapse occurred on CPU and FP64 CUDA. Confident
classification predictions can leave tiny Hessians, so `-G / (H + lambda_l2)`
produces an enormous Newton step. In one reproduced fraud leaf, `G=27.9989` and
`H=0.00112228` produced an update around `-2494.82` after learning rate `0.1`.
Extra precision preserves that ratio; it does not make the step appropriate.
Probability clipping can conceal the severity of the resulting loss, so this
validation evaluates logloss directly from raw scores using stable log-domain math.

`max_delta_step=-1` now automatically bounds the learned constant-leaf update to
one unit of the linked score for non-quantized binary, softmax and one-vs-all
classification with GBDT (including GOSS) or DART. It accounts for learning rate
and binary/OVA sigmoid; the initial class-prior offset is excluded. Quantized
training, regression, custom objectives and random forests are unconstrained by
this automatic policy. Fitted linear-leaf predictions are outside the bound.
Explicit `0` restores unbounded behavior. Positive values retain the existing raw
leaf cap before shrinkage: manual `1` at learning rate `0.1` permits a learned raw
step of `0.1`, and is not equivalent to automatic `-1`.

The sweep found two additional CPU assertions at `min_data_in_leaf=1` and
`min_sum_hessian_in_leaf=0`: fraud at round 3 and covtype at round 2. Histogram
estimates selected a child that contained no actual rows. PR #93 checks the real
partition before committing tree, constraint or CEGB state, rolls back an empty
split and continues with other leaves. Invalid forced splits recover an ordinary
candidate. The original error observations remain in the evidence beside their
separate successful final-build reruns.

The automatic constraint changes non-quantized classification models, including
possible split ties when the bound is inactive, and can require more trees. No
universal bit identity or monotonically decreasing loss is claimed.

## Coverage and observed quality

**CPU, 12 October: 66 guarded full-data configurations plus eight additional
tiny-leaf seed cases completed all 500 boosting rounds and passed the recorded
gates. FP64 CUDA, release candidate: 66 of 66.**
Explicit `max_delta_step=0` controls reproduced runaway margins on CPU (2 of 2) and
on the release candidate's CUDA build (2
of 2); they stop once that negative-control condition is observed and do not count
as 500-round passes. Tree fingerprints equal the October records in
2 of 68 CUDA cases: #122 and #154 change these
models, so the checked claim is the gates, not bit identity.

The CPU half was repeated on `1408ee79` (its CUDA-enabled build with
`device_type=cpu`; October used a CPU-only build): 74
of 74 cases passed and 2 of 2 controls
reproduced; fraud AUC over the 20 seeds 0.9722–0.9840,
covtype accuracy 0.9692–0.9702.

Each of the four dataset/backend combinations covers:

- Ten training seeds without sampling: `0, 1, 2, 3, 4, 5, 7, 11, 19, 23`.
- Ten seeds with row and feature sampling both `0.8`: `42, 67, 101, 123, 271, 509, 997, 2026, 4099, 65521`.
- Five fresh stratified 80/20 splits, with split seeds `13, 37, 83, 211, 1009` and training seeds equal to the split seed plus 10,000.
- Eight stress profiles: learning rates `0.03` and `0.5`, 15 bins, one-row leaves with zero Hessian floor, sigmoid `2` for fraud / OVA for covtype, GOSS, DART, and an alternate route (CPU column-wise / CUDA hybrid disabled).

The eight extra CPU cases repeat the tiny-leaf profile with seeds `0, 7, 42, 1009`
on each dataset. The ordinary recipe uses learning rate `0.1`, depth `10`, 1,023
leaves, 255 bins, leaf floor 20, Hessian floor `0.001`, L2 zero, eight CPU threads,
`quant_mode="none"`, no early stopping and explicit CUDA FP64. CPU uses its default
float32 gradient/Hessian type with double histogram accumulation; it is not an
all-FP64 CPU implementation. Every case's full parameters are retained.

Fraud has 227,845 training rows × 29 features and 56,962 test rows; covtype has
464,809 × 54 and 116,203 test rows, with seven classes. Data file hashes and fresh
split-index hashes are included in the [portable evidence](stability-data.json).

Across the 20 training seeds on the original cached split, combining the ordinary
and sampled recipes:

| Backend | Fraud raw-score AUC range | Covtype accuracy range |
| --- | ---: | ---: |
| CPU (12 October) | 0.97224–0.98396 | 0.96916–0.97022 |
| FP64 CUDA (release candidate) | 0.9697–0.9850 | 0.9694–0.9704 |

Across fresh splits, CPU fraud AUC ranges from 0.96120 to 0.98575 and covtype
accuracy from 0.97025 to 0.97108; on the release candidate's FP64 CUDA build,
0.9647–0.9849 and
0.9702–0.9710. (The October CUDA ranges were
0.97078–0.98479 and 0.96812–0.97072 on the original split.) Changed stress recipes can trade quality
for capacity or convergence speed: CPU covtype at learning rate `0.03` reaches
0.93902 accuracy at 500 rounds, and the 15-bin profile reaches 0.94684. Passing
stability gates does not imply that every profile matches the ordinary recipe.

The gates require completion, finite native loss at every round, finite predictions
at checkpoints `1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 500`, bounded learned
constant-leaf linked updates, raw-score magnitude below 100, and held-out logloss
below the class-prior baseline. Binary and softmax cases additionally audit
unclipped logloss from raw scores. The two covtype OVA cases instead retain
logloss from normalized independent-sigmoid probabilities; no unclipped raw-loss
claim is made for those records. Ordinary, sampled, resplit, alternate,
GOSS, tiny-leaf and high-learning-rate profiles also require unclipped logloss
below `0.03` for fraud or `0.18` for covtype. The lower-capacity profiles are checked
against the prior instead. Saturated probabilities can still occur; bounded
updates and the reported loss measure are the relevant checks. Loss need not improve
at every iteration, particularly with DART.

## Build provenance and final validation

The release-candidate CUDA rerun uses one library, SHA256 `c1902dce58ac356c9964a708e700e438062c596104617f4d91d243bcce26839a`,
for all 68 CUDA cases (66 guarded, two controls). The paragraphs below describe the
October cohort; its CUDA records are retained in the evidence as history, labelled
by their own library hashes.

The October cases are an audited coverage cohort across identified native builds,
not fresh runs on the final PR head. Most original cases use merged
`aa5f00e7`, which already contains PR #91. The two failing CPU cases were rerun
on final `08c37d95`; eight extra CPU tiny-leaf cases use the earlier partition fix
`498b3538`. Final forced-split recovery is covered by separate final-build tests.

The initial CUDA runtime used a symlink to a build output. Rebuilding replaced
that target after 35 completed CUDA cases; later cases carry the `498b3538`
library hash. The CUDA learner algorithm was unchanged, and the runtime was then
frozen as an independent file. Per-case `library_sha256` and the SHA-to-source map
are authoritative; a blanket claim that every sweep case used one final binary
would be wrong. The evidence preserves this correction.

Final `08c37d95` validation includes 39 focused CPU/CUDA regressions (27 CPU and
12 CUDA), 1,195 passed CPU/C API CI tests, and all 21 PR checks. All five triggered
post-merge master workflows passed: Build, Tests, SWIG, C++, and Static Analysis.
The evidence contains the exact GitHub check/run links. OpenCL validation was
source compilation only, without linked OpenCL training.

Two additional default-model byte-identity checks failed and remain recorded.
Fraud differed by tiny floating-point values and a threshold sign that changed no
observed row partition. Covtype selected an equivalent-feature near tie before
later model divergence, with close held-out quality; the experiment does not
isolate its cause. A separate final deterministic column-wise covtype comparison
matched all 3,500 tree topologies and raw predictions exactly over 500 rounds.
That establishes the tested deterministic comparison, not universal identity.

## How to read the evidence

[stability-data.json](stability-data.json) retains 146 compact October observations:
136 original observations (132 guarded configurations and four controls), two final
reruns and eight extra seed cases. The release-candidate CUDA records are
68 further observations in the same schema. It keeps every case's parameters, checkpoints,
model fingerprint, loss summary, original-record hash and native-library hash,
plus summary groups, retained compatibility failures and final CI provenance.
Full 500-round trajectories, model files and prediction arrays are retained in
the local audit bundle and omitted from this public compact report. Portable
record identifiers refer to that retained bundle; they are not downloadable links.
Original-record hashes identify the unmodified source records.

These results test two datasets and the stated recipes. They do not establish
behavior for arbitrary datasets, all parameters, fitted linear leaves, or a
current cross-framework leaderboard. They also provide no speed measurements:
new quality scores must not be paired with older uncapped timing rows. The
[historical benchmark tables](benchmarks.md) remain tied to their measured builds.
