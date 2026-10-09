# Full framework comparison

Each Falcata column reports the latest relevant **measured** build, not the fastest mode or a fresh measurement of every final-build cell. Competitor columns retain the original August 2026 launch-snapshot measurements. This is a descriptive historical table, not a rerun against current competing releases or a matched-quality experiment.

Cells show training seconds and primary held-out quality. Short workloads use the median of three timed endpoints; Numerai deep/leaf use one complete requested-round endpoint. Ranges appear where repeated quality endpoints materially vary. Every finite endpoint, including failed models, contributes to its interval; `FAIL` bars are unsuitable for timing headlines. Warmups and curves do not enter medians.

| Workload | Falcata stochastic | Falcata fixedpoint | Falcata FP64 noquant | XGBoost | CatBoost | LightGBM CUDA | LightGBM OpenCL |
|---|---|---|---|---|---|---|---|
| Fraud shallow | 0.394 s · B<br>AUC .9365 | 0.459 s · B<br>AUC .9808 | 0.347 s · D<br>AUC .5146 [.5048–.7534]<br>**FAIL / incomplete** | 0.460 s<br>AUC .9794 | 1.34 s<br>AUC .9792 | 2.29 s<br>AUC .7947 [.7752–.8159] | **FAILED** (skipped_warmup_failed) |
| Fraud deep | 0.392 s · B<br>AUC .9630 | 0.553 s · B<br>AUC .9658 | 0.454 s · D<br>AUC .7385 [.6680–.8160] † | 0.516 s<br>AUC .9747 | 5.86 s<br>AUC .9785 | **FAILED** (skipped_warmup_failed) | **FAILED** (skipped_warmup_failed) |
| Covtype shallow | 1.44 s · R<br>Acc .9204 | 1.50 s · R<br>Acc .9221 | 2.43 s · D<br>Acc .8789 [.8630–.8836] | 3.64 s<br>Acc .8942 | 1.97 s<br>Acc .8499 | 28.07 s<br>Acc .8341 [.8084–.8427] | **FAILED** (skipped_warmup_failed) |
| Covtype deep | 2.50 s · R<br>Acc .9581 | 2.80 s · R<br>Acc .9687 | 4.22 s · D<br>Acc .8550 [.8079–.8628] | 8.70 s<br>Acc .9631 | 9.76 s<br>Acc .9177 [.9169–.9177] | 117.65 s<br>Acc .4990 [.4461–.5346]<br>**FAIL / incomplete** | **FAILED** (skipped_warmup_failed) |
| Year shallow | 0.481 s · B<br>RMSE 9.000 | 0.487 s · B<br>RMSE 8.971 | 0.767 s · D<br>RMSE 8.961 | 0.889 s<br>RMSE 8.955 | 1.31 s<br>RMSE 9.005 | 3.94 s<br>RMSE 8.974 [8.973–8.975] | **FAILED** (skipped_warmup_failed) |
| Year deep | 1.000 s · B<br>RMSE 8.972 | 1.02 s · B<br>RMSE 8.971 | 2.74 s · D<br>RMSE 8.974 [8.973–8.974] | 5.79 s<br>RMSE 9.009 | 5.44 s<br>RMSE 8.931 | 60.77 s<br>RMSE 9.031 [9.030–9.039] | **FAILED** (skipped_warmup_failed) |
| Higgs shallow | 1.69 s · B<br>AUC .8310 | 1.96 s · B<br>AUC .8307 | 3.16 s · D<br>AUC .8306 | 3.65 s<br>AUC .8306 | 9.06 s<br>AUC .8225 [.8224–.8227] | 6.09 s<br>AUC .8347 | **FAILED** (skipped_warmup_failed) |
| Higgs deep | 2.70 s · B<br>AUC .8486 | 3.33 s · B<br>AUC .8486 | 9.66 s · D<br>AUC .8486 | 9.69 s<br>AUC .8484 | 19.33 s<br>AUC .8374 [.8373–.8374] | 51.73 s<br>AUC .8499 [.8499–.8505] | **FAILED** (skipped_warmup_failed) |
| Epsilon shallow | 2.55 s · B<br>AUC .9489 | 2.97 s · B<br>AUC .9485 | 8.02 s · D<br>AUC .9485 | 9.86 s<br>AUC .9490 | 12.69 s<br>AUC .9485 | 30.77 s<br>AUC .9510 | **FAILED** (skipped_warmup_failed) |
| Epsilon deep | 11.22 s · D<br>AUC .9424 | 12.42 s · B<br>AUC .9435 | 46.64 s · D<br>AUC .9433 | 52.26 s<br>AUC .9440 | 101.48 s<br>AUC .9508 | 462.83 s<br>AUC .9432 | **FAILED** (skipped_warmup_failed) |
| Airline shallow | 15.44 s · B<br>AUC .8263 | 15.55 s · B<br>AUC .8262 | 19.97 s · D<br>AUC .8263 | 22.83 s<br>AUC .8256 | 79.99 s<br>AUC .8024 [.8022–.8026] | 32.56 s<br>AUC .8352 | 638.15 s<br>AUC .8265 [.8263–.8266] |
| Airline deep | 23.73 s · B<br>AUC .8641 | 25.85 s · B<br>AUC .8639 | 64.58 s · D<br>AUC .8640 | 39.75 s<br>AUC .8633 | 138.36 s<br>AUC .8222 | 114.80 s<br>AUC .8782 [.8780–.8783] | **FAILED** (skipped_warmup_failed) |
| Numerai example | 5.62 s · D<br>CORR .01943 | 6.26 s · B<br>CORR .01896 | 53.81 s · D<br>CORR .01931 | 286.35 s<br>CORR .01972 † | 112.10 s<br>CORR .01800 † | **FAILED** (skipped_warmup_failed) | **FAILED** (skipped_warmup_failed) |
| Numerai deep | 110.80 s · B<br>CORR .02385 | 129.28 s · B<br>CORR .02332 | 1,237 s · D<br>CORR .02380 | 7,007 s<br>CORR .02345 | 5,240 s<br>CORR .02171 † | **FAILED** (crashed) | 10,362 s<br>CORR .02384 |
| Numerai leaf-limited | 263.81 s · B<br>CORR .01999 | 289.56 s · B<br>CORR .01979 | 2,789 s · B<br>CORR .02008 | 10,094 s<br>CORR .02012 | **FAILED** (crashed) | **FAILED** (crashed) | **FAILED** (timeout) |

## Build and source legend

- **B:** Oct 8 runtime `b4da8135`, package version 1.0.6, GPUQ3494 strict admission with no recorded contention. Its runtime/lib identity is recovered from the contemporaneous build record and matching library MD5 `58897acd3df952c1989b8e354ed09a17`. It predates PR68’s fixedpoint Hessian ridge.
- **R:** Post-PR68 Covtype refresh on runtime `0b170e0a`. All three Falcata Covtype modes and both regimes were refreshed. This build has no archived library digest or recorded quiet-admission flags in those raw rows.
- **D:** Oct 9 selected runtime `d0bb8c47`, unoverridden FP64 default, actual library SHA256 `e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c`. Strict GPUQ endpoint provenance is recorded. The two stochastic controls use this build; the remaining D cells are noquant.
- **L / competitor archive:** August 2026 snapshot. Recorded versions: XGBoost 3.3.0, CatBoost 1.2.10, upstream LightGBM CUDA/quant 4.7.0 and OpenCL 4.6.0. Falcata’s original snapshot prints 1.0.0 but is not a controlled tagged-v1.0.0 binary measurement. Exact record IDs, versions and hashes remain in the [matrix](benchmark-matrix.json).

A dagger (†) means another endpoint in that selected group failed, despite a complete sane timed set. In particular, Fraud deep noquant has a rejected curve and unstable finite quality; its timed median must not imply that all 500 requested updates produced valid trees. Original uncapped failures stay in this table. The separately labelled cap-1 follow-up changes the configuration and is not substituted here.

## Recipe and quality limits

- Classic shallow/deep: 500 requested rounds, learning rate .1, depth6/10 and leaves63/1023. Numerai example: 2,000 rounds, learning rate .01, depth5, 32 leaves, feature fraction .1. Numerai deep: 30,000 rounds, learning rate .001, depth10, 1,024 leaves, minimum leaf10,000, feature fraction .1. Numerai leaf-limited: same budget/rate/fraction, unbounded depth, 1,024 leaves and minimum leaf1,000. Seed42, 32 CPU threads and max_bin255 are the declared recipes.
- Framework L2 defaults differ: Falcata/LightGBM0, XGBoost1, CatBoost3. Growth policies also differ; upstream CUDA does not enforce the requested depth cap. “Sanity passing” is a coarse floor, not proof of equivalent quality.
- B fixedpoint binary endpoints are historical pre-ridge measurements. The ridge changes models with varying/subquantum Hessians; these old values do not certify final-build fixedpoint quality. Unweighted constant-Hessian Numerai/Year regression is unchanged by that ridge. R fixedpoint Covtype is measured after the repair.
- The historical Numerai cache has an unidentified target label. Shapes agree at5,463,797 training rows ×3,555 features, but historical source/split byte hashes are absent. B/launch loaders used float32; D Numerai uses int8. These are implementation workload records, not production or honest configuration-selection evidence.
- The timer is training only, excluding Dataset construction and CPU prediction/scoring. No failure is assigned an infinite or fabricated duration. Source-accurate failures, duplicate historical retries, excluded auxiliary regimes and actual metric ranges are retained in the machine matrix.

## Additional archived variants

The loss-guided XGBoost variant uses the LightGBM-family depth/leaves pair. Its Numerai leaf-limited row is absent because the ordinary XGBoost row already uses that policy. Upstream gradient quantization is a separate mode with its own coarse quality floor; its fast Year result does not imply comparable RMSE.

| Workload | Archived XGBoost loss guide | Archived LightGBM CUDA quantized |
|---|---|---|
| Fraud shallow | 0.643 s<br>AUC .9794 | **FAILED** (skipped_warmup_failed) |
| Fraud deep | 0.754 s<br>AUC .9747 | 0.360 s<br>AUC .8514 † |
| Covtype shallow | 7.90 s<br>Acc .8932 | **FAILED** (skipped_warmup_failed) |
| Covtype deep | 38.96 s<br>Acc .9631 | **FAILED** (skipped_warmup_failed) |
| Year shallow | 1.79 s<br>RMSE 8.963 | 0.341 s<br>RMSE 10.850 † |
| Year deep | 21.00 s<br>RMSE 9.009 | 0.510 s<br>RMSE 10.851 † |
| Higgs shallow | 4.58 s<br>AUC .8308 | **FAILED** (skipped_warmup_failed) |
| Higgs deep | 25.23 s<br>AUC .8485 | **FAILED** (skipped_warmup_failed) |
| Epsilon shallow | 10.74 s<br>AUC .9487 | **FAILED** (skipped_warmup_failed) |
| Epsilon deep | 67.51 s<br>AUC .9440 | **FAILED** (skipped_warmup_failed) |
| Airline shallow | 24.14 s<br>AUC .8259 | **FAILED** (skipped_warmup_failed) |
| Airline deep | 56.54 s<br>AUC .8632 | **FAILED** (skipped_warmup_failed) |
| Numerai example | 287.67 s<br>CORR .01972 † | **FAILED** (skipped_warmup_failed) |
| Numerai deep | 7,170 s<br>CORR .02345 | **FAILED** (crashed) |
| Numerai leaf-limited | — (not run) | **FAILED** (crashed) |

The [45-row mode appendix](benchmark-appendix.json) retains original Falcata launch timings/quality alongside the later per-mode measurements. The [full matrix](benchmark-matrix.json) additionally preserves source-line identities, all finite metrics/statuses, failures, raw-source hashes and strict runtime pairs.
