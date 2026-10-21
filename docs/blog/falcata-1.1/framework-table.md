# Full framework comparison

Updated 21 October 2026. The Falcata columns are the release candidate, measured on
`1408ee79` with the [mode appendix](appendix-table.md)'s
controlled runs ([checklist](../NUMBERS.md)).
The competitor columns are the **August archive**: the original August 2026
launch-snapshot measurements, not rerun against current releases of those engines.
This is a descriptive table, not a matched-quality experiment.

Cells show training seconds and primary held-out quality: AUC for Fraud, Higgs,
Epsilon and Airline, accuracy (Acc) for Covtype, RMSE for Year and CORR (mean
per-era `numerai_corr`) for Numerai. Release-candidate times
are medians of two fresh-process runs; competitor short workloads use the median of
three timed endpoints and Numerai deep/leaf one complete requested-round endpoint.
Ranges appear where repeated quality endpoints materially vary. Every finite
endpoint, including failed models, contributes to its interval; `FAIL` bars are
unsuitable for timing headlines.

| Workload | Falcata stochastic | Falcata fixedpoint | Falcata non-quantized | XGBoost | CatBoost | LightGBM CUDA | LightGBM OpenCL |
|---|---|---|---|---|---|---|---|
| Fraud shallow | 0.203 s<br>AUC .9365 | 0.274 s<br>AUC .9808 | 0.379 s<br>AUC .9771 | 0.460 s<br>AUC .9794 | 1.34 s<br>AUC .9792 | 2.29 s<br>AUC .7947 [.7752–.8159] | **FAILED** (skipped_warmup_failed) |
| Fraud deep | 0.250 s<br>AUC .9630 | 0.323 s<br>AUC .9774 | 0.477 s<br>AUC .9695 | 0.516 s<br>AUC .9747 | 5.86 s<br>AUC .9785 | **FAILED** (skipped_warmup_failed) | **FAILED** (skipped_warmup_failed) |
| Covtype shallow | 1.133 s<br>Acc .9204 | 1.169 s<br>Acc .9221 | 2.696 s<br>Acc .9243 | 3.64 s<br>Acc .8942 | 1.97 s<br>Acc .8499 | 28.07 s<br>Acc .8341 [.8084–.8427] | **FAILED** (skipped_warmup_failed) |
| Covtype deep | 1.893 s<br>Acc .9581 | 2.060 s<br>Acc .9687 | 7.496 s<br>Acc .9704 | 8.70 s<br>Acc .9631 | 9.76 s<br>Acc .9177 [.9169–.9177] | 117.65 s<br>Acc .4990 [.4461–.5346]<br>**FAIL / incomplete** | **FAILED** (skipped_warmup_failed) |
| Year shallow | 0.285 s<br>RMSE 9.000 | 0.306 s<br>RMSE 8.971 | 0.740 s<br>RMSE 8.961 | 0.889 s<br>RMSE 8.955 | 1.31 s<br>RMSE 9.005 | 3.94 s<br>RMSE 8.974 [8.973–8.975] | **FAILED** (skipped_warmup_failed) |
| Year deep | 0.690 s<br>RMSE 8.972 | 0.706 s<br>RMSE 8.971 | 3.690 s<br>RMSE 8.973 | 5.79 s<br>RMSE 9.009 | 5.44 s<br>RMSE 8.931 | 60.77 s<br>RMSE 9.031 [9.030–9.039] | **FAILED** (skipped_warmup_failed) |
| Higgs shallow | 1.305 s<br>AUC .8310 | 1.457 s<br>AUC .8303 | 3.936 s<br>AUC .8306 | 3.65 s<br>AUC .8306 | 9.06 s<br>AUC .8225 [.8224–.8227] | 6.09 s<br>AUC .8347 | **FAILED** (skipped_warmup_failed) |
| Higgs deep | 2.101 s<br>AUC .8486 | 2.236 s<br>AUC .8482 | 17.509 s<br>AUC .8484 | 9.69 s<br>AUC .8484 | 19.33 s<br>AUC .8374 [.8373–.8374] | 51.73 s<br>AUC .8499 [.8499–.8505] | **FAILED** (skipped_warmup_failed) |
| Epsilon shallow | 2.330 s<br>AUC .9489 | 2.558 s<br>AUC .9487 | 9.401 s<br>AUC .9485 | 9.86 s<br>AUC .9490 | 12.69 s<br>AUC .9485 | 30.77 s<br>AUC .9510 | **FAILED** (skipped_warmup_failed) |
| Epsilon deep | 9.822 s<br>AUC .9424 | 10.373 s<br>AUC .9432 | 72.703 s<br>AUC .9430 | 52.26 s<br>AUC .9440 | 101.48 s<br>AUC .9508 | 462.83 s<br>AUC .9432 | **FAILED** (skipped_warmup_failed) |
| Airline shallow | 13.317 s<br>AUC .8263 | 13.859 s<br>AUC .8262 | 27.439 s<br>AUC .8261 | 22.83 s<br>AUC .8256 | 79.99 s<br>AUC .8024 [.8022–.8026] | 32.56 s<br>AUC .8352 | 638.15 s<br>AUC .8265 [.8263–.8266] |
| Airline deep | 22.098 s<br>AUC .8641 | 22.443 s<br>AUC .8639 | 140.881 s<br>AUC .8637 | 39.75 s<br>AUC .8633 | 138.36 s<br>AUC .8222 | 114.80 s<br>AUC .8782 [.8780–.8783] | **FAILED** (skipped_warmup_failed) |
| Numerai example | 5.370 s<br>CORR .01943 | 6.127 s<br>CORR .01896 | 55.743 s<br>CORR .01931 | 286.35 s<br>CORR .01972 † | 112.10 s<br>CORR .01800 † | **FAILED** (skipped_warmup_failed) | **FAILED** (skipped_warmup_failed) |
| Numerai deep | 105.823 s<br>CORR .02385 | 127.390 s<br>CORR .02332 | not run | 7,007 s<br>CORR .02345 | 5,240 s<br>CORR .02171 † | **FAILED** (crashed) | 10,362 s<br>CORR .02384 |
| Numerai leaf-limited | 239.700 s<br>CORR .01999 | 259.415 s<br>CORR .01979 | not run | 10,094 s<br>CORR .02012 | **FAILED** (crashed) | **FAILED** (crashed) | **FAILED** (timeout) |

## Build and source legend

- **Falcata:** release candidate `49c5622a`, measured on `1408ee79`, library SHA256
  `c1902dce58ac356c9964a708e700e438062c596104617f4d91d243bcce26839a`, default plan, one otherwise idle RTX 5090. The
  non-quantized column is deterministic by default since #122. Numerai deep and
  leaf-limited are not run without quantization.
- **August archive (competitors):** recorded versions XGBoost 3.3.0, CatBoost 1.2.10,
  upstream LightGBM CUDA/quant 4.7.0 and OpenCL 4.6.0. Exact record IDs, versions and
  hashes remain in the [matrix](benchmark-matrix.json). Falcata's original launch
  snapshot prints 1.0.0 but is not a controlled tagged-v1.0.0 binary measurement.

A dagger (†) means another endpoint in that selected group failed, despite a
complete sane timed set.

## Recipe and quality limits

- Classic shallow/deep: 500 requested rounds, learning rate .1, depth6/10 and leaves63/1023. Numerai example: 2,000 rounds, learning rate .01, depth5, 32 leaves, feature fraction .1. Numerai deep: 30,000 rounds, learning rate .001, depth10, 1,024 leaves, minimum leaf10,000, feature fraction .1. Numerai leaf-limited: same budget/rate/fraction, unbounded depth, 1,024 leaves and minimum leaf1,000. Seed42, 32 CPU threads and max_bin255 are the declared recipes.
- Framework L2 defaults differ: Falcata/LightGBM0, XGBoost1, CatBoost3. Growth policies also differ; upstream CUDA does not enforce the requested depth cap. “Sanity passing” is a coarse floor, not proof of equivalent quality.
- The release-candidate fixedpoint rows include the Hessian-quantum ridge (merged before `a65d34e8`). The dated October 8 fixedpoint binary rows, kept in the [mode appendix](appendix-table.md), predate it.
- The historical Numerai cache has an unidentified target label. Shapes agree at5,463,797 training rows ×3,555 features, but historical source/split byte hashes are absent. Launch Numerai deep and leaf-limited loaders used float32; release-candidate Numerai rows use int8. These are implementation workload records, not production or honest configuration-selection evidence.
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
