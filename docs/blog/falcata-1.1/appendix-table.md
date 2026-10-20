# Mode appendix

Every Falcata workload and mode of the [benchmark matrix](benchmark-matrix.json),
updated 20 October 2026. The first table is the controlled release comparison:
`a65d34e8` (12 October master) against release candidate `{{NUM:rc_commit}}`, two
interleaved fresh-process pairs per row, training seconds as medians of the two runs
per arm. `{{NUM:…}}` marks a value from that run ([checklist](../NUMBERS.md)). Launch
times are the August archive. Numerai rows use native int8 input; the launch Numerai
deep and leaf-limited rows used float32.

## `a65d34e8` → release candidate

| Dataset / regime | Mode | Launch s | `a65d34e8` s | Release candidate s | Speedup | Same model |
|---|---|---:|---:|---:|---:|---|
| fraud/shallow | stoch | 0.389116 | {{NUM:ab_fraud_shallow_stoch_a65_s}} | {{NUM:ab_fraud_shallow_stoch_rc_s}} | {{NUM:ab_fraud_shallow_stoch_x}} | {{NUM:ab_fraud_shallow_stoch_md5}} |
| fraud/shallow | fixed | 0.445616 | {{NUM:ab_fraud_shallow_fixed_a65_s}} | {{NUM:ab_fraud_shallow_fixed_rc_s}} | {{NUM:ab_fraud_shallow_fixed_x}} | {{NUM:ab_fraud_shallow_fixed_md5}} |
| fraud/shallow | none | missing | {{NUM:ab_fraud_shallow_none_a65_s}} | {{NUM:ab_fraud_shallow_none_rc_s}} | {{NUM:ab_fraud_shallow_none_x}} | {{NUM:ab_fraud_shallow_none_md5}} |
| fraud/deep | stoch | 0.376829 | {{NUM:ab_fraud_deep_stoch_a65_s}} | {{NUM:ab_fraud_deep_stoch_rc_s}} | {{NUM:ab_fraud_deep_stoch_x}} | {{NUM:ab_fraud_deep_stoch_md5}} |
| fraud/deep | fixed | 0.558055 | {{NUM:ab_fraud_deep_fixed_a65_s}} | {{NUM:ab_fraud_deep_fixed_rc_s}} | {{NUM:ab_fraud_deep_fixed_x}} | {{NUM:ab_fraud_deep_fixed_md5}} |
| fraud/deep | none | 0.3789 | {{NUM:ab_fraud_deep_none_a65_s}} | {{NUM:ab_fraud_deep_none_rc_s}} | {{NUM:ab_fraud_deep_none_x}} | {{NUM:ab_fraud_deep_none_md5}} |
| covtype/shallow | stoch | 2.35175 | {{NUM:ab_covtype_shallow_stoch_a65_s}} | {{NUM:ab_covtype_shallow_stoch_rc_s}} | {{NUM:ab_covtype_shallow_stoch_x}} | {{NUM:ab_covtype_shallow_stoch_md5}} |
| covtype/shallow | fixed | 2.58746 | {{NUM:ab_covtype_shallow_fixed_a65_s}} | {{NUM:ab_covtype_shallow_fixed_rc_s}} | {{NUM:ab_covtype_shallow_fixed_x}} | {{NUM:ab_covtype_shallow_fixed_md5}} |
| covtype/shallow | none | 2.71079 | {{NUM:ab_covtype_shallow_none_a65_s}} | {{NUM:ab_covtype_shallow_none_rc_s}} | {{NUM:ab_covtype_shallow_none_x}} | {{NUM:ab_covtype_shallow_none_md5}} |
| covtype/deep | stoch | 5.15619 | {{NUM:ab_covtype_deep_stoch_a65_s}} | {{NUM:ab_covtype_deep_stoch_rc_s}} | {{NUM:ab_covtype_deep_stoch_x}} | {{NUM:ab_covtype_deep_stoch_md5}} |
| covtype/deep | fixed | 5.68891 | {{NUM:ab_covtype_deep_fixed_a65_s}} | {{NUM:ab_covtype_deep_fixed_rc_s}} | {{NUM:ab_covtype_deep_fixed_x}} | {{NUM:ab_covtype_deep_fixed_md5}} |
| covtype/deep | none | 5.09616 | {{NUM:ab_covtype_deep_none_a65_s}} | {{NUM:ab_covtype_deep_none_rc_s}} | {{NUM:ab_covtype_deep_none_x}} | {{NUM:ab_covtype_deep_none_md5}} |
| year/shallow | stoch | 0.570685 | {{NUM:ab_year_shallow_stoch_a65_s}} | {{NUM:ab_year_shallow_stoch_rc_s}} | {{NUM:ab_year_shallow_stoch_x}} | {{NUM:ab_year_shallow_stoch_md5}} |
| year/shallow | fixed | 0.576381 | {{NUM:ab_year_shallow_fixed_a65_s}} | {{NUM:ab_year_shallow_fixed_rc_s}} | {{NUM:ab_year_shallow_fixed_x}} | {{NUM:ab_year_shallow_fixed_md5}} |
| year/shallow | none | 0.736454 | {{NUM:ab_year_shallow_none_a65_s}} | {{NUM:ab_year_shallow_none_rc_s}} | {{NUM:ab_year_shallow_none_x}} | {{NUM:ab_year_shallow_none_md5}} |
| year/deep | stoch | 2.15412 | {{NUM:ab_year_deep_stoch_a65_s}} | {{NUM:ab_year_deep_stoch_rc_s}} | {{NUM:ab_year_deep_stoch_x}} | {{NUM:ab_year_deep_stoch_md5}} |
| year/deep | fixed | 1.97925 | {{NUM:ab_year_deep_fixed_a65_s}} | {{NUM:ab_year_deep_fixed_rc_s}} | {{NUM:ab_year_deep_fixed_x}} | {{NUM:ab_year_deep_fixed_md5}} |
| year/deep | none | 3.33884 | {{NUM:ab_year_deep_none_a65_s}} | {{NUM:ab_year_deep_none_rc_s}} | {{NUM:ab_year_deep_none_x}} | {{NUM:ab_year_deep_none_md5}} |
| higgs/shallow | stoch | 2.36564 | {{NUM:ab_higgs_shallow_stoch_a65_s}} | {{NUM:ab_higgs_shallow_stoch_rc_s}} | {{NUM:ab_higgs_shallow_stoch_x}} | {{NUM:ab_higgs_shallow_stoch_md5}} |
| higgs/shallow | fixed | 2.54355 | {{NUM:ab_higgs_shallow_fixed_a65_s}} | {{NUM:ab_higgs_shallow_fixed_rc_s}} | {{NUM:ab_higgs_shallow_fixed_x}} | {{NUM:ab_higgs_shallow_fixed_md5}} |
| higgs/shallow | none | 3.86487 | {{NUM:ab_higgs_shallow_none_a65_s}} | {{NUM:ab_higgs_shallow_none_rc_s}} | {{NUM:ab_higgs_shallow_none_x}} | {{NUM:ab_higgs_shallow_none_md5}} |
| higgs/deep | stoch | 4.23726 | {{NUM:ab_higgs_deep_stoch_a65_s}} | {{NUM:ab_higgs_deep_stoch_rc_s}} | {{NUM:ab_higgs_deep_stoch_x}} | {{NUM:ab_higgs_deep_stoch_md5}} |
| higgs/deep | fixed | 4.9944 | {{NUM:ab_higgs_deep_fixed_a65_s}} | {{NUM:ab_higgs_deep_fixed_rc_s}} | {{NUM:ab_higgs_deep_fixed_x}} | {{NUM:ab_higgs_deep_fixed_md5}} |
| higgs/deep | none | 15.3027 | {{NUM:ab_higgs_deep_none_a65_s}} | {{NUM:ab_higgs_deep_none_rc_s}} | {{NUM:ab_higgs_deep_none_x}} | {{NUM:ab_higgs_deep_none_md5}} |
| epsilon/shallow | stoch | 4.9727 | {{NUM:ab_epsilon_shallow_stoch_a65_s}} | {{NUM:ab_epsilon_shallow_stoch_rc_s}} | {{NUM:ab_epsilon_shallow_stoch_x}} | {{NUM:ab_epsilon_shallow_stoch_md5}} |
| epsilon/shallow | fixed | 5.35324 | {{NUM:ab_epsilon_shallow_fixed_a65_s}} | {{NUM:ab_epsilon_shallow_fixed_rc_s}} | {{NUM:ab_epsilon_shallow_fixed_x}} | {{NUM:ab_epsilon_shallow_fixed_md5}} |
| epsilon/shallow | none | 8.73323 | {{NUM:ab_epsilon_shallow_none_a65_s}} | {{NUM:ab_epsilon_shallow_none_rc_s}} | {{NUM:ab_epsilon_shallow_none_x}} | {{NUM:ab_epsilon_shallow_none_md5}} |
| epsilon/deep | stoch | 39.2823 | {{NUM:ab_epsilon_deep_stoch_a65_s}} | {{NUM:ab_epsilon_deep_stoch_rc_s}} | {{NUM:ab_epsilon_deep_stoch_x}} | {{NUM:ab_epsilon_deep_stoch_md5}} |
| epsilon/deep | fixed | 39.2276 | {{NUM:ab_epsilon_deep_fixed_a65_s}} | {{NUM:ab_epsilon_deep_fixed_rc_s}} | {{NUM:ab_epsilon_deep_fixed_x}} | {{NUM:ab_epsilon_deep_fixed_md5}} |
| epsilon/deep | none | 59.1271 | {{NUM:ab_epsilon_deep_none_a65_s}} | {{NUM:ab_epsilon_deep_none_rc_s}} | {{NUM:ab_epsilon_deep_none_x}} | {{NUM:ab_epsilon_deep_none_md5}} |
| airline/shallow | stoch | 18.0865 | {{NUM:ab_airline_shallow_stoch_a65_s}} | {{NUM:ab_airline_shallow_stoch_rc_s}} | {{NUM:ab_airline_shallow_stoch_x}} | {{NUM:ab_airline_shallow_stoch_md5}} |
| airline/shallow | fixed | 19.5105 | {{NUM:ab_airline_shallow_fixed_a65_s}} | {{NUM:ab_airline_shallow_fixed_rc_s}} | {{NUM:ab_airline_shallow_fixed_x}} | {{NUM:ab_airline_shallow_fixed_md5}} |
| airline/shallow | none | 27.8943 | {{NUM:ab_airline_shallow_none_a65_s}} | {{NUM:ab_airline_shallow_none_rc_s}} | {{NUM:ab_airline_shallow_none_x}} | {{NUM:ab_airline_shallow_none_md5}} |
| airline/deep | stoch | 29.472 | {{NUM:ab_airline_deep_stoch_a65_s}} | {{NUM:ab_airline_deep_stoch_rc_s}} | {{NUM:ab_airline_deep_stoch_x}} | {{NUM:ab_airline_deep_stoch_md5}} |
| airline/deep | fixed | 31.9576 | {{NUM:ab_airline_deep_fixed_a65_s}} | {{NUM:ab_airline_deep_fixed_rc_s}} | {{NUM:ab_airline_deep_fixed_x}} | {{NUM:ab_airline_deep_fixed_md5}} |
| airline/deep | none | 145.053 | {{NUM:ab_airline_deep_none_a65_s}} | {{NUM:ab_airline_deep_none_rc_s}} | {{NUM:ab_airline_deep_none_x}} | {{NUM:ab_airline_deep_none_md5}} |
| numerai/numerai | stoch | 30.9813 | {{NUM:ab_numerai_numerai_stoch_a65_s}} | {{NUM:ab_numerai_numerai_stoch_rc_s}} | {{NUM:ab_numerai_numerai_stoch_x}} | {{NUM:ab_numerai_numerai_stoch_md5}} |
| numerai/numerai | fixed | 31.1184 | {{NUM:ab_numerai_numerai_fixed_a65_s}} | {{NUM:ab_numerai_numerai_fixed_rc_s}} | {{NUM:ab_numerai_numerai_fixed_x}} | {{NUM:ab_numerai_numerai_fixed_md5}} |
| numerai/numerai | none | 65.6925 | {{NUM:ab_numerai_numerai_none_a65_s}} | {{NUM:ab_numerai_numerai_none_rc_s}} | {{NUM:ab_numerai_numerai_none_x}} | {{NUM:ab_numerai_numerai_none_md5}} |
| numerai/numerai-deep | stoch | 742.769 | {{NUM:ab_numerai_numerai_deep_stoch_a65_s}} | {{NUM:ab_numerai_numerai_deep_stoch_rc_s}} | {{NUM:ab_numerai_numerai_deep_stoch_x}} | {{NUM:ab_numerai_numerai_deep_stoch_md5}} |
| numerai/numerai-deep | fixed | 763.635 | {{NUM:ab_numerai_numerai_deep_fixed_a65_s}} | {{NUM:ab_numerai_numerai_deep_fixed_rc_s}} | {{NUM:ab_numerai_numerai_deep_fixed_x}} | {{NUM:ab_numerai_numerai_deep_fixed_md5}} |
| numerai/numerai-deep | none | 1566.2 | not run | not run | — | — |
| numerai/numerai-leaf | stoch | 1808.84 | {{NUM:ab_numerai_numerai_leaf_stoch_a65_s}} | {{NUM:ab_numerai_numerai_leaf_stoch_rc_s}} | {{NUM:ab_numerai_numerai_leaf_stoch_x}} | {{NUM:ab_numerai_numerai_leaf_stoch_md5}} |
| numerai/numerai-leaf | fixed | 1964.14 | {{NUM:ab_numerai_numerai_leaf_fixed_a65_s}} | {{NUM:ab_numerai_numerai_leaf_fixed_rc_s}} | {{NUM:ab_numerai_numerai_leaf_fixed_x}} | {{NUM:ab_numerai_numerai_leaf_fixed_md5}} |
| numerai/numerai-leaf | none | 3037.97 | not run | not run | — | — |

"Same model" compares the model MD5 across both arms and all runs. Non-quantized
`a65d34e8` models differ from run to run (float atomics), so for `none` rows it
reports whether the release candidate's two runs agree. Numerai deep and
leaf-limited are not run without quantization (20–45 minutes per run). Interim
values for `a65d34e8` → `4641c9f0` (16 October) are in the kernel-evo `since_1_1`
report; they are not release-candidate results.

## Dated October 8–9 quantized measurements

These are the measurements of the earlier draft, kept with their builds:
`b4da8135` (8 October), `0b170e0a` (post-ridge covtype refresh) and `d0bb8c47`
(9 October strict controls). Its non-quantized rows came from the unmerged
[PR #71](https://github.com/MechaFauna-ai/Falcata/pull/71) prototype and are removed;
the matrix keeps them as history. Original quality and timing pairs are preserved.

| Dataset / regime | Mode | Launch s | Latest measured s | Metric median | Source/build | Timed failures |
|---|---|---:|---:|---|---|---|
| fraud/shallow | falcata-stoch | 0.389116 | 0.394486 | auc 0.936503 | oct8 / b4da8135 | False |
| fraud/shallow | falcata-fixed | 0.445616 | 0.458947 | auc 0.980774 | oct8 / b4da8135 | False |
| fraud/deep | falcata-stoch | 0.376829 | 0.391923 | auc 0.962973 | oct8 / b4da8135 | False |
| fraud/deep | falcata-fixed | 0.558055 | 0.552874 | auc 0.965829 | oct8 / b4da8135 | False |
| covtype/shallow | falcata-stoch | 2.35175 | 1.44383 | accuracy 0.920407 | corrected_snapshot / 0b170e0a | False |
| covtype/shallow | falcata-fixed | 2.58746 | 1.49788 | accuracy 0.922102 | corrected_snapshot / 0b170e0a | False |
| covtype/deep | falcata-stoch | 5.15619 | 2.50403 | accuracy 0.958134 | corrected_snapshot / 0b170e0a | False |
| covtype/deep | falcata-fixed | 5.68891 | 2.79802 | accuracy 0.968727 | corrected_snapshot / 0b170e0a | False |
| year/shallow | falcata-stoch | 0.570685 | 0.481481 | rmse 8.99965 | oct8 / b4da8135 | False |
| year/shallow | falcata-fixed | 0.576381 | 0.486683 | rmse 8.97145 | oct8 / b4da8135 | False |
| year/deep | falcata-stoch | 2.15412 | 0.999961 | rmse 8.97162 | oct8 / b4da8135 | False |
| year/deep | falcata-fixed | 1.97925 | 1.01889 | rmse 8.97136 | oct8 / b4da8135 | False |
| higgs/shallow | falcata-stoch | 2.36564 | 1.69359 | auc 0.830953 | oct8 / b4da8135 | False |
| higgs/shallow | falcata-fixed | 2.54355 | 1.95663 | auc 0.830734 | oct8 / b4da8135 | False |
| higgs/deep | falcata-stoch | 4.23726 | 2.69824 | auc 0.848643 | oct8 / b4da8135 | False |
| higgs/deep | falcata-fixed | 4.9944 | 3.3254 | auc 0.848587 | oct8 / b4da8135 | False |
| epsilon/shallow | falcata-stoch | 4.9727 | 2.54885 | auc 0.948852 | oct8 / b4da8135 | False |
| epsilon/shallow | falcata-fixed | 5.35324 | 2.96884 | auc 0.948542 | oct8 / b4da8135 | False |
| epsilon/deep | falcata-stoch | 39.2823 | 11.2201 | auc 0.94243 | strict_epsilon-stoch_candidate / d0bb8c47 | False |
| epsilon/deep | falcata-fixed | 39.2276 | 12.4175 | auc 0.943536 | oct8 / b4da8135 | False |
| airline/shallow | falcata-stoch | 18.0865 | 15.4401 | auc 0.826321 | oct8 / b4da8135 | False |
| airline/shallow | falcata-fixed | 19.5105 | 15.5505 | auc 0.826167 | oct8 / b4da8135 | False |
| airline/deep | falcata-stoch | 29.472 | 23.7317 | auc 0.864116 | oct8 / b4da8135 | False |
| airline/deep | falcata-fixed | 31.9576 | 25.8491 | auc 0.863894 | oct8 / b4da8135 | False |
| numerai/numerai | falcata-stoch | 30.9813 | 5.61594 | corr_mean 0.0194266 | strict_numerai-stoch_candidate / d0bb8c47 | False |
| numerai/numerai | falcata-fixed | 31.1184 | 6.26179 | corr_mean 0.018964 | oct8 / b4da8135 | False |
| numerai/numerai-deep | falcata-stoch | 742.769 | 110.799 | corr_mean 0.0238513 | oct8 / b4da8135 | False |
| numerai/numerai-deep | falcata-fixed | 763.635 | 129.28 | corr_mean 0.0233243 | oct8 / b4da8135 | False |
| numerai/numerai-leaf | falcata-stoch | 1808.84 | 263.814 | corr_mean 0.0199908 | oct8 / b4da8135 | False |
| numerai/numerai-leaf | falcata-fixed | 1964.14 | 289.564 | corr_mean 0.0197929 | oct8 / b4da8135 | False |
| numerai/numerai-leaf | falcata-noquant | 3037.97 | 2789.39 | corr_mean 0.0200821 | oct8 / b4da8135 | False |
