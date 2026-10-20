# Full framework comparison

Updated 20 October 2026. The Falcata columns are release candidate
`{{NUM:rc_commit}}`, measured with the [mode appendix](appendix-table.md)'s
controlled runs (`{{NUM:…}}` marks a value from them; [checklist](../NUMBERS.md)).
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
| Fraud shallow | {{NUM:ab_fraud_shallow_stoch_rc_s}} s<br>AUC {{NUM:q_fraud_shallow_stoch_rc}} | {{NUM:ab_fraud_shallow_fixed_rc_s}} s<br>AUC {{NUM:q_fraud_shallow_fixed_rc}} | {{NUM:ab_fraud_shallow_none_rc_s}} s<br>AUC {{NUM:q_fraud_shallow_none_rc}} | 0.460 s<br>AUC .9794 | 1.34 s<br>AUC .9792 | 2.29 s<br>AUC .7947 [.7752–.8159] | **FAILED** (skipped_warmup_failed) |
| Fraud deep | {{NUM:ab_fraud_deep_stoch_rc_s}} s<br>AUC {{NUM:q_fraud_deep_stoch_rc}} | {{NUM:ab_fraud_deep_fixed_rc_s}} s<br>AUC {{NUM:q_fraud_deep_fixed_rc}} | {{NUM:ab_fraud_deep_none_rc_s}} s<br>AUC {{NUM:q_fraud_deep_none_rc}} | 0.516 s<br>AUC .9747 | 5.86 s<br>AUC .9785 | **FAILED** (skipped_warmup_failed) | **FAILED** (skipped_warmup_failed) |
| Covtype shallow | {{NUM:ab_covtype_shallow_stoch_rc_s}} s<br>Acc {{NUM:q_covtype_shallow_stoch_rc}} | {{NUM:ab_covtype_shallow_fixed_rc_s}} s<br>Acc {{NUM:q_covtype_shallow_fixed_rc}} | {{NUM:ab_covtype_shallow_none_rc_s}} s<br>Acc {{NUM:q_covtype_shallow_none_rc}} | 3.64 s<br>Acc .8942 | 1.97 s<br>Acc .8499 | 28.07 s<br>Acc .8341 [.8084–.8427] | **FAILED** (skipped_warmup_failed) |
| Covtype deep | {{NUM:ab_covtype_deep_stoch_rc_s}} s<br>Acc {{NUM:q_covtype_deep_stoch_rc}} | {{NUM:ab_covtype_deep_fixed_rc_s}} s<br>Acc {{NUM:q_covtype_deep_fixed_rc}} | {{NUM:ab_covtype_deep_none_rc_s}} s<br>Acc {{NUM:q_covtype_deep_none_rc}} | 8.70 s<br>Acc .9631 | 9.76 s<br>Acc .9177 [.9169–.9177] | 117.65 s<br>Acc .4990 [.4461–.5346]<br>**FAIL / incomplete** | **FAILED** (skipped_warmup_failed) |
| Year shallow | {{NUM:ab_year_shallow_stoch_rc_s}} s<br>RMSE {{NUM:q_year_shallow_stoch_rc}} | {{NUM:ab_year_shallow_fixed_rc_s}} s<br>RMSE {{NUM:q_year_shallow_fixed_rc}} | {{NUM:ab_year_shallow_none_rc_s}} s<br>RMSE {{NUM:q_year_shallow_none_rc}} | 0.889 s<br>RMSE 8.955 | 1.31 s<br>RMSE 9.005 | 3.94 s<br>RMSE 8.974 [8.973–8.975] | **FAILED** (skipped_warmup_failed) |
| Year deep | {{NUM:ab_year_deep_stoch_rc_s}} s<br>RMSE {{NUM:q_year_deep_stoch_rc}} | {{NUM:ab_year_deep_fixed_rc_s}} s<br>RMSE {{NUM:q_year_deep_fixed_rc}} | {{NUM:ab_year_deep_none_rc_s}} s<br>RMSE {{NUM:q_year_deep_none_rc}} | 5.79 s<br>RMSE 9.009 | 5.44 s<br>RMSE 8.931 | 60.77 s<br>RMSE 9.031 [9.030–9.039] | **FAILED** (skipped_warmup_failed) |
| Higgs shallow | {{NUM:ab_higgs_shallow_stoch_rc_s}} s<br>AUC {{NUM:q_higgs_shallow_stoch_rc}} | {{NUM:ab_higgs_shallow_fixed_rc_s}} s<br>AUC {{NUM:q_higgs_shallow_fixed_rc}} | {{NUM:ab_higgs_shallow_none_rc_s}} s<br>AUC {{NUM:q_higgs_shallow_none_rc}} | 3.65 s<br>AUC .8306 | 9.06 s<br>AUC .8225 [.8224–.8227] | 6.09 s<br>AUC .8347 | **FAILED** (skipped_warmup_failed) |
| Higgs deep | {{NUM:ab_higgs_deep_stoch_rc_s}} s<br>AUC {{NUM:q_higgs_deep_stoch_rc}} | {{NUM:ab_higgs_deep_fixed_rc_s}} s<br>AUC {{NUM:q_higgs_deep_fixed_rc}} | {{NUM:ab_higgs_deep_none_rc_s}} s<br>AUC {{NUM:q_higgs_deep_none_rc}} | 9.69 s<br>AUC .8484 | 19.33 s<br>AUC .8374 [.8373–.8374] | 51.73 s<br>AUC .8499 [.8499–.8505] | **FAILED** (skipped_warmup_failed) |
| Epsilon shallow | {{NUM:ab_epsilon_shallow_stoch_rc_s}} s<br>AUC {{NUM:q_epsilon_shallow_stoch_rc}} | {{NUM:ab_epsilon_shallow_fixed_rc_s}} s<br>AUC {{NUM:q_epsilon_shallow_fixed_rc}} | {{NUM:ab_epsilon_shallow_none_rc_s}} s<br>AUC {{NUM:q_epsilon_shallow_none_rc}} | 9.86 s<br>AUC .9490 | 12.69 s<br>AUC .9485 | 30.77 s<br>AUC .9510 | **FAILED** (skipped_warmup_failed) |
| Epsilon deep | {{NUM:ab_epsilon_deep_stoch_rc_s}} s<br>AUC {{NUM:q_epsilon_deep_stoch_rc}} | {{NUM:ab_epsilon_deep_fixed_rc_s}} s<br>AUC {{NUM:q_epsilon_deep_fixed_rc}} | {{NUM:ab_epsilon_deep_none_rc_s}} s<br>AUC {{NUM:q_epsilon_deep_none_rc}} | 52.26 s<br>AUC .9440 | 101.48 s<br>AUC .9508 | 462.83 s<br>AUC .9432 | **FAILED** (skipped_warmup_failed) |
| Airline shallow | {{NUM:ab_airline_shallow_stoch_rc_s}} s<br>AUC {{NUM:q_airline_shallow_stoch_rc}} | {{NUM:ab_airline_shallow_fixed_rc_s}} s<br>AUC {{NUM:q_airline_shallow_fixed_rc}} | {{NUM:ab_airline_shallow_none_rc_s}} s<br>AUC {{NUM:q_airline_shallow_none_rc}} | 22.83 s<br>AUC .8256 | 79.99 s<br>AUC .8024 [.8022–.8026] | 32.56 s<br>AUC .8352 | 638.15 s<br>AUC .8265 [.8263–.8266] |
| Airline deep | {{NUM:ab_airline_deep_stoch_rc_s}} s<br>AUC {{NUM:q_airline_deep_stoch_rc}} | {{NUM:ab_airline_deep_fixed_rc_s}} s<br>AUC {{NUM:q_airline_deep_fixed_rc}} | {{NUM:ab_airline_deep_none_rc_s}} s<br>AUC {{NUM:q_airline_deep_none_rc}} | 39.75 s<br>AUC .8633 | 138.36 s<br>AUC .8222 | 114.80 s<br>AUC .8782 [.8780–.8783] | **FAILED** (skipped_warmup_failed) |
| Numerai example | {{NUM:ab_numerai_numerai_stoch_rc_s}} s<br>CORR {{NUM:q_numerai_numerai_stoch_rc}} | {{NUM:ab_numerai_numerai_fixed_rc_s}} s<br>CORR {{NUM:q_numerai_numerai_fixed_rc}} | {{NUM:ab_numerai_numerai_none_rc_s}} s<br>CORR {{NUM:q_numerai_numerai_none_rc}} | 286.35 s<br>CORR .01972 † | 112.10 s<br>CORR .01800 † | **FAILED** (skipped_warmup_failed) | **FAILED** (skipped_warmup_failed) |
| Numerai deep | {{NUM:ab_numerai_numerai_deep_stoch_rc_s}} s<br>CORR {{NUM:q_numerai_numerai_deep_stoch_rc}} | {{NUM:ab_numerai_numerai_deep_fixed_rc_s}} s<br>CORR {{NUM:q_numerai_numerai_deep_fixed_rc}} | not run | 7,007 s<br>CORR .02345 | 5,240 s<br>CORR .02171 † | **FAILED** (crashed) | 10,362 s<br>CORR .02384 |
| Numerai leaf-limited | {{NUM:ab_numerai_numerai_leaf_stoch_rc_s}} s<br>CORR {{NUM:q_numerai_numerai_leaf_stoch_rc}} | {{NUM:ab_numerai_numerai_leaf_fixed_rc_s}} s<br>CORR {{NUM:q_numerai_numerai_leaf_fixed_rc}} | not run | 10,094 s<br>CORR .02012 | **FAILED** (crashed) | **FAILED** (crashed) | **FAILED** (timeout) |

## Build and source legend

- **Falcata:** release candidate `{{NUM:rc_commit}}`, library SHA256
  `{{NUM:rc_library_sha256}}`, default plan, one otherwise idle RTX 5090. The
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
