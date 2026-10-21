# Mode appendix

Every Falcata workload and mode of the [benchmark matrix](benchmark-matrix.json),
updated 21 October 2026. The first table is the controlled release comparison:
`a65d34e8` (12 October master) against the release candidate, measured on
`1408ee79` ([checklist](../NUMBERS.md)), two interleaved fresh-process pairs
per row (Numerai deep: three), training seconds as medians of each arm's runs; speedup is `a65d34e8`
seconds over release-candidate seconds. "Same model" compares the two builds' model
MD5s for quantized rows. For `none` rows it says whether the release candidate's two
runs agree: `a65d34e8`'s non-quantized runs differed from run to run, so its models
differ from the release candidate's on every `none` row (#122). Fraud fixed point
differs because of a fixed-point accumulation change since `a65d34e8` (most likely #146, which keeps the quantized CUDA root reduction's packed totals in int64); the deep run stops at 357
trees. Launch
times are the August archive. Numerai rows use native int8 input; the launch Numerai
deep and leaf-limited rows used float32.

## `a65d34e8` → release candidate

| Dataset / regime | Mode | Launch s | `a65d34e8` s | Release candidate s | Speedup | Same model |
|---|---|---:|---:|---:|---:|---|
| fraud/shallow | stoch | 0.389116 | 0.289 | 0.203 | 1.424 | yes |
| fraud/shallow | fixed | 0.445616 | 0.353 | 0.274 | 1.291 | no |
| fraud/shallow | none | missing | 0.430 | 0.379 | 1.135 | yes |
| fraud/deep | stoch | 0.376829 | 0.309 | 0.250 | 1.237 | yes |
| fraud/deep | fixed | 0.558055 | 0.463 | 0.323 | 1.433 | no |
| fraud/deep | none | 0.3789 | 0.525 | 0.477 | 1.099 | yes |
| covtype/shallow | stoch | 2.35175 | 1.493 | 1.133 | 1.319 | yes |
| covtype/shallow | fixed | 2.58746 | 1.532 | 1.169 | 1.310 | yes |
| covtype/shallow | none | 2.71079 | 3.057 | 2.696 | 1.134 | yes |
| covtype/deep | stoch | 5.15619 | 2.450 | 1.893 | 1.294 | yes |
| covtype/deep | fixed | 5.68891 | 2.665 | 2.060 | 1.294 | yes |
| covtype/deep | none | 5.09616 | 7.799 | 7.496 | 1.040 | yes |
| year/shallow | stoch | 0.570685 | 0.391 | 0.285 | 1.370 | yes |
| year/shallow | fixed | 0.576381 | 0.408 | 0.306 | 1.333 | yes |
| year/shallow | none | 0.736454 | 0.794 | 0.740 | 1.073 | yes |
| year/deep | stoch | 2.15412 | 0.918 | 0.690 | 1.331 | yes |
| year/deep | fixed | 1.97925 | 1.012 | 0.706 | 1.434 | yes |
| year/deep | none | 3.33884 | 3.428 | 3.690 | 0.929 | yes |
| higgs/shallow | stoch | 2.36564 | 1.516 | 1.305 | 1.162 | yes |
| higgs/shallow | fixed | 2.54355 | 1.749 | 1.457 | 1.200 | yes |
| higgs/shallow | none | 3.86487 | 4.042 | 3.936 | 1.027 | yes |
| higgs/deep | stoch | 4.23726 | 2.488 | 2.101 | 1.184 | yes |
| higgs/deep | fixed | 4.9944 | 3.107 | 2.236 | 1.390 | yes |
| higgs/deep | none | 15.3027 | 15.566 | 17.509 | 0.889 | yes |
| epsilon/shallow | stoch | 4.9727 | 2.539 | 2.330 | 1.090 | yes |
| epsilon/shallow | fixed | 5.35324 | 2.925 | 2.558 | 1.144 | yes |
| epsilon/shallow | none | 8.73323 | 9.274 | 9.401 | 0.987 | yes |
| epsilon/deep | stoch | 39.2823 | 11.460 | 9.822 | 1.167 | yes |
| epsilon/deep | fixed | 39.2276 | 12.583 | 10.373 | 1.213 | yes |
| epsilon/deep | none | 59.1271 | 64.739 | 72.703 | 0.890 | yes |
| airline/shallow | stoch | 18.0865 | 14.436 | 13.317 | 1.084 | yes |
| airline/shallow | fixed | 19.5105 | 14.831 | 13.859 | 1.070 | yes |
| airline/shallow | none | 27.8943 | 27.225 | 27.439 | 0.992 | yes |
| airline/deep | stoch | 29.472 | 23.157 | 22.098 | 1.048 | yes |
| airline/deep | fixed | 31.9576 | 24.094 | 22.443 | 1.074 | yes |
| airline/deep | none | 145.053 | 132.977 | 140.881 | 0.944 | yes |
| numerai/numerai | stoch | 30.9813 | 5.620 | 5.370 | 1.046 | yes |
| numerai/numerai | fixed | 31.1184 | 6.324 | 6.127 | 1.032 | yes |
| numerai/numerai | none | 65.6925 | 54.837 | 55.743 | 0.984 | yes |
| numerai/numerai-deep | stoch | 742.769 | 112.969 | 105.823 | 1.068 | yes |
| numerai/numerai-deep | fixed | 763.635 | 133.469 | 127.390 | 1.048 | yes |
| numerai/numerai-deep | none | 1566.2 | not run | not run | — | — |
| numerai/numerai-leaf | stoch | 1808.84 | 265.021 | 239.700 | 1.106 | yes |
| numerai/numerai-leaf | fixed | 1964.14 | 282.766 | 259.415 | 1.090 | yes |
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
