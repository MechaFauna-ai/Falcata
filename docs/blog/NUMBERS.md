# Falcata 1.1 article: numbers to re-measure

Internal checklist for the release-candidate measurement, drafted 20 October 2026.
Delete this file before the article is published. Every `{{NUM:name}}` in
`docs/blog/falcata-1.1.md` and `docs/blog/falcata-1.1/*.md` is listed here with the
run that fills it, the file it is read from and the magnitude the interim data
predicts. A filled value far from its expectation needs an explanation before it is
published. `{{NUM:…}}` in the draft notes is a reference to this convention, not a
placeholder.

Paths: `kevo/` is the kernel-evo repository; `src/` is the Falcata checkout. Run
Python from a run directory, never from a Falcata repository root. All GPU jobs go
through strict `gpuq` with `--timeout`.

## Counts

| Group | Unique placeholders | Measurement |
| --- | ---: | --- |
| Release identity | 2 | Release commit and library hash |
| Headline Numerai deep | 6 | (b) three strict draws per arm |
| Article matrix, per row | 172 | (a) `article_ab.py`, 43 rows × 4 |
| Article matrix, summaries | 5 | (a), derived |
| Quality per row | 43 | (a′) `score.py` on the saved A/B models |
| August-archive ratios | 7 | Derived from (a)/(b) and the archive |
| Six-state | 14 | (c) six-state driver, to be rebuilt |
| Stability, FP64 CUDA | 7 | (d) stability driver, to be rebuilt |
| Validation counts | 8 | (e) gates, test suites, CI |
| **Total** | **264** | |

## Release identity

| Placeholder | Source | Expected |
| --- | --- | --- |
| `rc_commit` | The frozen RC: master HEAD at step 2 of `kevo/runs/release_1_1/PLAN.md` | 8-hex short SHA; same value everywhere |
| `rc_library_sha256` | `sha256sum` of the RC venv's `lib_falcata.so` used by (a)–(d) | One hash for all RC runs |

## (b) Headline: Numerai deep, 30,000 rounds

Run `kevo/runs/since_1_1/article_ab.py --dataset numerai --regime numerai-deep
--mode stoch --py-old <a65d34e8 venv> --py-new <RC venv> --pairs 3` (old venv:
`kevo/work/falcata/wt/since11/venv`; strict gpuq, `--timeout` ≥ 7200 s per job,
about 20 minutes). Output JSON: `old.train_s`, `new.train_s` (medians), `md5s`.

| Placeholder | Value | Expected (interim) |
| --- | --- | --- |
| `numerai_deep_30k_a65_s` | `old.train_s` | ~114 s (since_1_1: 114.6 / 114.1) |
| `numerai_deep_30k_rc_s` | `new.train_s` | ~105–109 s (`4641c9f0`: 108.5 s; #123 ~1.00× on Numerai) |
| `numerai_deep_30k_rc_over_a65` | old / new | ~1.05× |
| `numerai_deep_30k_rc_min` | new / 60, one decimal | ~1.8 |
| `numerai_deep_30k_launch_over_rc` | 742.769 / new | ~6.8–7.1× (launch used float32; see benchmarks.md) |
| `numerai_deep_md5_rc_vs_a65` | "identical" or "different" from `md5_identical` | identical (a65d34e8 = 4641c9f0 here); #146 is the change that could flip it |

If (b) is folded into (a), `ab_numerai_numerai_deep_stoch_*` and these six must
come from the same JSON.

## (a) Article matrix: `a65d34e8` → RC

Driver: `kevo/runs/since_1_1/article.sh` with `PYN` pointed at the RC venv and a
new output directory (it skips files that exist). Cells: the 15 workloads × `stoch`,
`fixed`, plus `none` on the 12 classic shapes and Numerai example (43 rows; Numerai
deep/leaf `none` not run). Two pairs per row, about 2 hours in total. One JSON per
row: `<dataset>-<regime>-<mode>.json`.

Per-row placeholders, used in `appendix-table.md` (and `framework-table.md` for
`rc_s`): `ab_<dataset>_<regime>_<mode>_<field>`, where `<regime>` replaces `-` with
`_` (Numerai example is `numerai_numerai`, deep `numerai_numerai_deep`, leaf-limited
`numerai_numerai_leaf`).

| Field | Value from the row JSON | Expected (interim, `kevo/runs/since_1_1/REPORT.md`) |
| --- | --- | --- |
| `a65_s` | `old.train_s` | Equal to since_1_1's old column within ~3% |
| `rc_s` | `new.train_s` | Quantized: ≤ since_1_1's new column; `none`: ~1.0–1.08× since_1_1's new column (#122 cost) |
| `x` | `speedup_train` | Quantized 1.01–1.37× (+ up to ~1.05× from #123 on covtype/Higgs); `none` ~0.94–1.04× |
| `md5` | Quantized: "yes" if `md5_identical`; `none`: "yes" if the two `new` runs agree | Quantized: yes; `none`: RC yes, a65 runs differ |

Summaries:

| Placeholder | Derivation | Expected |
| --- | --- | --- |
| `ab_geomean_stoch` | Geometric mean of `x`, 15 stoch rows | ≥ 1.13 (interim 1.126) |
| `ab_geomean_fixed` | Same, 15 fixed rows | ≥ 1.16 (interim 1.162) |
| `ab_geomean_none` | Same, 13 none rows | ~0.98–1.00 (interim 0.995 × #122's 1.010 cost) |
| `ab_quant_md5_identical` | "N of 30" quantized rows with `md5_identical` | 30 of 30 (classic shapes set no `feature_fraction`, so #154 does not apply; check #146) |
| `ab_none_rc_deterministic` | "N of 13" none rows whose RC runs share one MD5 | 13 of 13 |

`summarize.py` in `kevo/runs/since_1_1/` already prints the table and geometric
means; point it at the new directory.

## (a′) Quality per row

`q_<dataset>_<regime>_<mode>_rc` (43, in `framework-table.md`; three also in the
article and `benchmarks.md`). Produced by `kevo/runs/release_1_1/score.py`
(`release_ab.sh score --new RC`), which predicts each model `article_ab.py
--save-models` kept on the CPU and scores it as `benchmarks/bench.py` at `a65d34e8`
does, on the bench cache's held-out split. Read from
`results/quality_<rc8>.json` `placeholders` (RC arm, median over its runs). Values
are bare numbers, so every cell that uses one names the metric in front of it:

| Rows | Prefix in the text | Metric | Value format |
| --- | --- | --- | --- |
| Fraud, Higgs, Epsilon, Airline (24) | `AUC` | AUC of the predicted probability | 4 decimals |
| Covtype (6) | `Acc` | Accuracy of the argmax class | 4 decimals |
| Year (6) | `RMSE` | RMSE | 3 decimals |
| Numerai example, deep, leaf-limited (7) | `CORR` | Mean per-era `numerai_corr`, 200 test eras | 5 decimals |

The same JSON's `display` object holds the prefixed strings ("AUC .9794") for
copying into prose; it is not a placeholder source. Old-arm values (`*_a65`) are
under `old_arm`.

Expected: within the October ranges of `benchmark-matrix.json` for quantized rows
(Epsilon deep stochastic AUC ~0.942; Numerai deep CORR ~0.0239; example ~0.0194).
Non-quantized classification rows follow the automatic step limit (#91) and need
no comparison with the PR #71 prototype's failed fraud rows.

## August-archive ratios (derived)

Competitor seconds from `benchmark-matrix.json` `archived_competitors`; RC seconds
from (b) for Numerai deep, (a) otherwise.

| Placeholder | Formula | Expected |
| --- | --- | --- |
| `aug_xgb_over_rc_numerai_deep` | 7,007 / RC | ~65 |
| `aug_catboost_over_rc_numerai_deep` | 5,240 / RC | ~49 |
| `aug_lgbm_opencl_over_rc_numerai_deep` | 10,362 / RC | ~96 |
| `aug_xgb_over_rc_numerai_example` | 286.35 / RC | ~53 |
| `aug_catboost_over_rc_numerai_example` | 112.10 / RC | ~21 |
| `aug_xgb_over_rc_epsilon_deep` | 52.26 / RC | ~5.4 |
| `aug_catboost_over_rc_epsilon_deep` | 101.48 / RC | ~10.5 |

Optional: a 2 October competitor rerun exists in `kevo/runs/blog_competitors/`
(`deep.jsonl`, 72 records) but is unused; using it would change the article's
"August archive" wording.

## (c) Six-state pair

The October driver (`run_cell.py`, `measure_plan.py`, `measurement_guard.py`,
`bootstrap.py`, `PROTOCOL.md` in the evidence bundle `falcata-six-values-20261009`)
is **not on disk**; only the data is: `/data/projects/numerai/data/falcata_six_values_20261009/{five,six}`.
Rebuild it from `docs/blog/falcata-1.1/six-state-data.json` (`configs`, `shared`,
`representations`, `endpoints` give every parameter and the draw order: five/six,
six/five, five/six). About 30 minutes. Output: a new `six-state-data.json`.

| Placeholder | Run | Expected |
| --- | --- | --- |
| `sixstate_std_stoch_2k_five_trees_s`, `sixstate_std_stoch_2k_six_trees_s`, `sixstate_std_stoch_2k_six_over_five_time` | Standard stochastic FP64, 2,000 rounds, median of 3 | five ≥ 246; six well above 180; ratio ~1.05–1.15 (cells: 3.31 vs 3.52 ms/round) |
| `sixstate_std_fixed_2k_five_trees_s`, `sixstate_std_fixed_2k_six_trees_s`, `sixstate_std_fixed_2k_six_over_five_time` | Standard fixed point FP64, 2,000 rounds | five ≥ 212; six > 175 |
| `sixstate_std_stoch_30k_five_trees_s`, `sixstate_std_stoch_30k_six_trees_s`, `sixstate_std_stoch_30k_six_over_five_time` | Standard stochastic, 30,000 rounds, one pair | five ≥ 266; six ~250–280 (interim NaN cell 284 trees/s on another split) |
| `sixstate_16k_fp32_2k_five_trees_s`, `sixstate_16k_fp32_2k_six_trees_s`, `sixstate_16k_fp32_2k_six_over_five_time` | 16,384-leaf FP32 recipe, 2,000 rounds | Unknown. #109 did not change memory admission, so six may still fall back (October: 88.21 / 10.29) |
| `sixstate_16k_view_six` | `FALCATA_DEBUG=diag` warmup, six-state 16k arm: "compact" or "full-width mask" | Possibly still mask; report as measured |
| `sixstate_std_six_code_view_trees` | Diag warmup, standard six-state: "N of 200" trees on a code view (radix or six-bit) | Most trees (#109 nsys: 55 of 60) |

## (d) Stability, FP64 CUDA half

The October sweep driver is not on disk. Rebuild it from
`docs/blog/falcata-1.1/stability-data.json`: each record's `case` and `params`
fully define a run; the gates are in `stability.md` ("Coverage and observed
quality"). Rerun the 66 guarded CUDA cases and the two CUDA `max_delta_step=0`
controls at 500 rounds (30–45 minutes). The CPU half is not rerun.

| Placeholder | Value | Expected |
| --- | --- | --- |
| `stability_cuda_guarded_pass` | Guarded cases passing all gates | 66 |
| `stability_cuda_controls_reproduced` | Controls showing runaway margins | 2 |
| `stability_cuda_fraud_auc_range` | Min–max raw-score AUC, 20 seeds on the cached split | Near 0.9708–0.9848 |
| `stability_cuda_covtype_acc_range` | Same, accuracy | Near 0.9681–0.9707 (covtype sampled recipe changes with #154) |
| `stability_cuda_fresh_fraud_auc_range` | Five fresh splits | Near 0.9633–0.9865 |
| `stability_cuda_fresh_covtype_acc_range` | Five fresh splits | Near 0.9706–0.9716 |
| `stability_rc_records` | Number of new records appended to `stability-data.json` | 68 |

## (e) Validation counts

| Placeholder | Source | Expected (latest seen) |
| --- | --- | --- |
| `canonical_gates_rc` | Nightly canonical model gates on the RC (`tests/gates/canonical.py`) | 4 |
| `lattice_cells_rc` | `tests/gates/lattice.py --check` on the RC | 441 (#153/#154: 441/441) |
| `cuda_tests_passed_rc` / `cuda_tests_skipped_rc` | `TASK=cuda` pytest, full suite, RC build | ~2,128 passed in #135's run, **plus 45 environment-related failures** on this machine; resolve or explain those before quoting a count |
| `cpu_tests_passed_rc` | CPU build (`USE_CUDA=OFF`) pytest | ~1,247 (#131/#132/#135) |
| `cpp_tests_rc` | C++ `testfalcata` | 70 (#153; #154: 70 + 1 skipped) |
| `ci_checks_rc` | `gh pr checks` / `gh run list` on the RC commit | 21 (#130) |
| `nightly_fuzz_specs_rc` | Nightly gates log (`falcata-nightly` gpuq key), fuzz section | ~180 per 10 minutes; the nightly runs 45 minutes |

`tools/release-preflight.sh` also requires `~/.cache/falcata-gates/last-status` =
`PASSED <RC>`; record that run's date next to the counts.

## Files to regenerate

| File(s) | Producer | Input |
| --- | --- | --- |
| `benchmark-matrix.json` | `kevo/runs/release_1_1/to_blog_json.py --rc RC` (writes `results/blog_<rc8>/`; copy in after review) | (a), (a′), (b) |
| `framework-comparison-data.json` | Same converter (the script's `compact_data` plus the measured draw counts and date) | Updated matrix |
| `framework-comparison.png`, `framework-comparison.svg` | `plot_framework_comparison.py --output-dir <dir with the compact JSON>` (footer draw counts and Numerai input from the JSON) | New `framework-comparison-data.json` |
| `benchmark-appendix.json` | Same converter (45-row mode appendix) | (a), (a′) |
| `six-state-data.json` | Same converter, from `six_state.py summarize` | (c) |
| `six-state-throughput.png`, `six-state-throughput.svg` | `plot_six_state.py --data six-state-data.json --commit <RC sha> --output-dir <dir>` (build, library hash and captions from the JSON; `--commit` must match its `runtime.build`) | New `six-state-data.json` |
| `stability-data.json` | Same converter, appending RC CUDA records (keep the October records) | (d) |
| `joint-histogram-codewords.*`, `gpu-coordination.*` | `plot_mechanisms.py` | No change needed (schematics) |
| `lineage-*` | `build_lineage.py` | Optional: the export is dated 9 October |

Removed in this draft: `fp64-training-time.png`, `fp64-training-time.svg`,
`plot_training_time.py`, `training-time-data.json` (PR #71 prototype).
