Falcata noquant strict timings — 2026-10-09

This report records the settled strict GPUQ series. All endpoint metrics, failures, actual runtime/library
hashes, invocation arguments and contention flags are preserved in [strict-timing-evidence.json](strict-timing-evidence.json).
Initial precision defaults are unoverridden. Quality metrics are taken from the native harness and its NumeraiEvaluator wrapper.

Retained 90/90 records in 24/24 files: 68 timed draws and 22 discarded warmups.
The baseline contributes 33 records and the selected d0bb artifact contributes 57 (including the six graph-key arms).

Times show median [minimum, maximum] in seconds; warmups are excluded. Performance headline ratios require
the complete planned set of sanity-passing runs and a strict job with no recorded contention. Finite failed
quality endpoints remain in the quality ranges. Admission flags establish the scheduler's quiet criterion.

| Workload | Baseline training seconds | Selected training seconds | Baseline/selected | Contended | Job state |
|---|---:|---:|---:|---:|---|
| year-noquant | 3.50122 [3.49875, 3.50526] | 2.73836 [2.73204, 2.74384] | 1.2786× | False | done |
| epsilon-noquant | 57.7655 [57.7042, 57.778] | 46.6361 [46.6053, 46.6669] | 1.2386× | False | done |
| higgs-noquant | 15.9336 [15.9277, 15.9409] | 9.65527 [9.5558, 9.6729] | 1.6502× | False | done |
| covtype-noquant | 5.31367 [5.21347, 5.50655] | 4.21961 [4.18245, 4.27304] | 1.2593× | False | done |
| fraud-noquant | 0.606037 [0.504075, 0.620739] | 0.453849 [0.442745, 0.538669] | 1.3353× | False | done |
| numerai-noquant | 54.8088 [54.7328, 54.9182] | 53.8071 [53.7586, 53.8219] | 1.0186× | False | done |
| epsilon-stoch | 11.2253 [11.1932, 11.2495] | 11.2201 [11.1928, 11.2417] | 1.0005× | False | done |
| numerai-stoch | 5.63179 [5.6052, 5.65418] | 5.61594 [5.61227, 5.64238] | 1.0028× | False | done |

The full Numerai-deep workload uses 30,000 rounds, one run per build. These values are single-run
measurements; no partial-run extrapolation or repeated-run precision is reported. Their CORR endpoints are
0.0238562 baseline and 0.0238004 selected; this single pair does not establish quality equivalence.

| Build | Training seconds | Sanity-passing runs | Contended | Job state |
|---|---:|---:|---:|---|
| baseline | 1372.53 [1372.53, 1372.53] | 1 | False | done |
| candidate | 1236.51 [1236.51, 1236.51] | 1 | False | done |

Graph-key comparisons use three interleaved measured runs per arm. The skip key uses the exact
Numerai-deep parameters with 500 rounds as a bounded probe; its timing is not a 30,000-round estimate.

| Key | On seconds | Off seconds | Off/on | Contended | Job state |
|---|---:|---:|---:|---:|---|
| graph_apply_rows | 53.7878 [53.7617, 53.9228] | 54.8055 [54.74, 54.9449] | 1.0189× | False | done |
| graph_apply_fused | 53.7921 [53.7098, 53.8034] | 53.7921 [53.7391, 53.8171] | 1.0000× | False | done |
| graph_skip_unsplittable | 21.3436 [21.3076, 21.3561] | 22.9577 [22.9358, 22.9608] | 1.0756× | False | done |

The fused-key ranges overlap and its median ratio is approximately one; no measured benefit is established for that workload.
Quality medians and ranges below include finite timed endpoints, including any sanity-failed timed models.
Each median3 arm contributes three timed endpoints; full Numerai-deep contributes one. The 22 warmup
endpoints are preserved in the raw sidecar and excluded from both the timing and quality tables.
Repeated noquant covtype/fraud metrics vary substantially even at the same seed; faster training does not establish quality equivalence.
Identical recorded metric values in other cells do not prove identical trees/predictions: this original series did not record their fingerprints.
Numerai uses a historical build1226 cache whose label is unidentified; its measurements compare fixed-cache
implementations and do not establish current production or target selection.

| File/build | Metric | Median [minimum, maximum] | Finite endpoints | Sanity failures |
|---|---|---:|---:|---:|
| strict-ablate-graph_apply_fused-off.jsonl / d0bb8c47789e | corr_mean | 0.0193116 [0.0193116, 0.0193116] | 3 | 0 |
| strict-ablate-graph_apply_fused-off.jsonl / d0bb8c47789e | rmse | 0.223644 [0.223644, 0.223644] | 3 | 0 |
| strict-ablate-graph_apply_fused-on.jsonl / d0bb8c47789e | corr_mean | 0.0193116 [0.0193116, 0.0193116] | 3 | 0 |
| strict-ablate-graph_apply_fused-on.jsonl / d0bb8c47789e | rmse | 0.223644 [0.223644, 0.223644] | 3 | 0 |
| strict-ablate-graph_apply_rows-off.jsonl / d0bb8c47789e | corr_mean | 0.0193116 [0.0193116, 0.0193116] | 3 | 0 |
| strict-ablate-graph_apply_rows-off.jsonl / d0bb8c47789e | rmse | 0.223644 [0.223644, 0.223644] | 3 | 0 |
| strict-ablate-graph_apply_rows-on.jsonl / d0bb8c47789e | corr_mean | 0.0193116 [0.0193116, 0.0193116] | 3 | 0 |
| strict-ablate-graph_apply_rows-on.jsonl / d0bb8c47789e | rmse | 0.223644 [0.223644, 0.223644] | 3 | 0 |
| strict-ablate-graph_skip_unsplittable-off.jsonl / d0bb8c47789e | corr_mean | 0.0161257 [0.0161257, 0.0161257] | 3 | 0 |
| strict-ablate-graph_skip_unsplittable-off.jsonl / d0bb8c47789e | rmse | 0.223659 [0.223659, 0.223659] | 3 | 0 |
| strict-ablate-graph_skip_unsplittable-on.jsonl / d0bb8c47789e | corr_mean | 0.0161257 [0.0161257, 0.0161257] | 3 | 0 |
| strict-ablate-graph_skip_unsplittable-on.jsonl / d0bb8c47789e | rmse | 0.223659 [0.223659, 0.223659] | 3 | 0 |
| strict-covtype-noquant-baseline.jsonl / 09b6be9724f3 | accuracy | 0.860615 [0.836984, 0.86157] | 3 | 0 |
| strict-covtype-noquant-baseline.jsonl / 09b6be9724f3 | mlogloss | 2.82122 [2.73723, 3.54857] | 3 | 0 |
| strict-covtype-noquant-candidate.jsonl / d0bb8c47789e | accuracy | 0.854995 [0.807862, 0.862818] | 3 | 0 |
| strict-covtype-noquant-candidate.jsonl / d0bb8c47789e | mlogloss | 2.95728 [2.65894, 4.75024] | 3 | 0 |
| strict-epsilon-noquant-baseline.jsonl / 09b6be9724f3 | auc | 0.943311 [0.94331, 0.943311] | 3 | 0 |
| strict-epsilon-noquant-candidate.jsonl / d0bb8c47789e | auc | 0.94331 [0.94331, 0.943311] | 3 | 0 |
| strict-epsilon-stoch-baseline.jsonl / 09b6be9724f3 | auc | 0.94243 [0.94243, 0.94243] | 3 | 0 |
| strict-epsilon-stoch-candidate.jsonl / d0bb8c47789e | auc | 0.94243 [0.94243, 0.94243] | 3 | 0 |
| strict-fraud-noquant-baseline.jsonl / 09b6be9724f3 | auc | 0.637685 [0.566066, 0.815755] | 3 | 0 |
| strict-fraud-noquant-candidate.jsonl / d0bb8c47789e | auc | 0.738508 [0.66799, 0.815992] | 3 | 0 |
| strict-higgs-noquant-baseline.jsonl / 09b6be9724f3 | auc | 0.84864 [0.848639, 0.84864] | 3 | 0 |
| strict-higgs-noquant-candidate.jsonl / d0bb8c47789e | auc | 0.84864 [0.848567, 0.84864] | 3 | 0 |
| strict-numerai-deep-baseline.jsonl / 09b6be9724f3 | corr_mean | 0.0238562 [0.0238562, 0.0238562] | 1 | 0 |
| strict-numerai-deep-baseline.jsonl / 09b6be9724f3 | rmse | 0.223696 [0.223696, 0.223696] | 1 | 0 |
| strict-numerai-deep-candidate.jsonl / d0bb8c47789e | corr_mean | 0.0238004 [0.0238004, 0.0238004] | 1 | 0 |
| strict-numerai-deep-candidate.jsonl / d0bb8c47789e | rmse | 0.223696 [0.223696, 0.223696] | 1 | 0 |
| strict-numerai-noquant-baseline.jsonl / 09b6be9724f3 | corr_mean | 0.0193116 [0.0193116, 0.0193116] | 3 | 0 |
| strict-numerai-noquant-baseline.jsonl / 09b6be9724f3 | rmse | 0.223644 [0.223644, 0.223644] | 3 | 0 |
| strict-numerai-noquant-candidate.jsonl / d0bb8c47789e | corr_mean | 0.0193116 [0.0193116, 0.0193116] | 3 | 0 |
| strict-numerai-noquant-candidate.jsonl / d0bb8c47789e | rmse | 0.223644 [0.223644, 0.223644] | 3 | 0 |
| strict-numerai-stoch-baseline.jsonl / 09b6be9724f3 | corr_mean | 0.0194266 [0.0194266, 0.0194266] | 3 | 0 |
| strict-numerai-stoch-baseline.jsonl / 09b6be9724f3 | rmse | 0.223643 [0.223643, 0.223643] | 3 | 0 |
| strict-numerai-stoch-candidate.jsonl / d0bb8c47789e | corr_mean | 0.0194266 [0.0194266, 0.0194266] | 3 | 0 |
| strict-numerai-stoch-candidate.jsonl / d0bb8c47789e | rmse | 0.223643 [0.223643, 0.223643] | 3 | 0 |
| strict-year-noquant-baseline.jsonl / 09b6be9724f3 | rmse | 8.9744 [8.97342, 8.97443] | 3 | 0 |
| strict-year-noquant-candidate.jsonl / d0bb8c47789e | rmse | 8.97434 [8.9734, 8.97442] | 3 | 0 |

Actual measured artifacts:

- Runtime `09b6be9724f330f947c86de6e819654b077f70f2`, library SHA256 `cb6de00388a1d85e32c8a8d1144310777d73767e1e56a8b32ab810e47d636e08`, resolved precision `fp64`.
- Runtime `d0bb8c47789e062acbf6b770918a0589a74ae9b2`, library SHA256 `e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c`, resolved precision `fp64`.
