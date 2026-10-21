# Falcata 1.1 article: numbers to re-measure

Internal checklist for the release-candidate measurement, drafted 20 October 2026,
filled 21 October 2026 (values: [Measured values](#measured-values) at the end).
Delete this file before the article is published. Every `{{NUM:name}}` in
`docs/blog/falcata-1.1.md` and `docs/blog/falcata-1.1/*.md` is listed here with the
run that fills it, the file it is read from and the magnitude the interim data
predicts. A filled value far from its expectation needs an explanation before it is
published. `{{NUM:…}}` in the draft notes is a reference to this convention, not a
placeholder.

Paths: `kevo/` is the kernel-evo repository; `src/` is the Falcata checkout. Run
Python from a run directory, never from a Falcata repository root. All GPU jobs go
through strict `gpuq` with `--timeout`.

## Filling, 21 October 2026

- **Measured commit.** Every run (a)–(d) and `counts.sh` ran on master `1408ee79`
  (`kevo/work/falcata/wt/m1408`, library SHA256 in `rc_library_sha256`), so
  `rc_commit` is `1408ee79`. The release candidate is master `49c5622a`; the delta is
  #141 (docs) and #155 (an `extra_trees` filter in the categorical kernel for more than
  256 bins), on none of the measured shapes. The documents say "measured on
  `1408ee79`" and name `49c5622a` as the candidate.
- **Filler.** `kevo/runs/release_1_1/fill_placeholders.py` reads the `placeholders`
  objects of `results/{ab,quality,six_state,stability}_1408ee79.json` and
  `stability_1408ee79_cpu.json`, refuses conflicting duplicates, substitutes in the six
  blog files and `docs/release-notes/1.1.0.md`, and writes the table at the end.
  Display formats: seconds, speedups and six/five ratios 3 decimals, six-state trees/s
  2 decimals, quality per the (a′) table (without the leading zero in
  `framework-table.md`, as its competitor cells).
- **Validation counts mapping** (`results/counts_1408ee79.json` `placeholders`):
  `lattice_cells_rc` ← `counts.lattice_cells` (`lattice.py --list` at `1408ee79`),
  `cpu_tests_passed_rc` ← `counts.cpu_tests_passed` (CI "python-tests (cpu)" on
  `1408ee79`), `cpp_tests_rc` ← `counts.cpp_tests_passed` (CI C++ gtests on `1408ee79`),
  `ci_checks_rc` ← `counts.ci_checks` (check runs on `1408ee79`). Not used:
  `counts.cuda_tests_passed` (2,410, the `eef89a67` nightly). From the nightly gates log on `49c5622a` (gpuq 6166,
  `~/.cache/falcata-gates/nightly-2026-10-21.log`, used only with `last-status`
  `PASSED 49c5622a`): `cuda_tests_passed_rc` / `cuda_tests_skipped_rc` (python suite
  (CUDA) summary line), `nightly_fuzz_specs_rc` ("fuzz: N specs tried, 0 failure(s)"),
  `canonical_gates_rc` (PASS lines of the "canonical md5 locks" step).
- **Six-state diag.** This build prints no `[pack-codec] tree codec=K` line, so the
  driver's first heuristic found no code-view trees. Each tree that fills a code view
  prints one `compact fill: pair_code<N> on (...)` line; `six_state.py` now counts those
  (and reads `colmajor_direct: mask regime` as "full-width mask"), and `summarize`
  re-derives the diag counts from the kept logs. Standard six-state warmup: 200
  `pair_code5` fills in 200 trees (five-state arm likewise). 16,384-leaf six-state arm:
  "the compact regime does not fit (store 9184 + compact view 1392 + split view 0 +
  reserve 15921 MiB > 24455 MiB available): mask regime", `row_batch` construct.
- **Extras** used by the new prose (also in the table): `q_*_a65` (quality JSON
  `old_arm`), `ab_*_{a65,rc}_trees` (median `num_trees` of each arm's runs in the row
  JSON), `stability_cuda_fingerprints_same`, `stability_cpu_*` (CPU half on `1408ee79`),
  `sixstate_std_stoch_30k_{five,six}_holdout_corr` (`holdout.numerai_corr_mean` of the
  30,000-tree records).
- **Against expectations.** `ab_quant_md5_identical` is 28 of 30, not 30: fraud
  shallow/deep fixed point change (#146; the deep run stops at 357 of 500 trees, see
  the article's model-change notice). `ab_geomean_none` 1.006, deep non-quantized shapes
  0.889–0.944. Six-state six/five 1.054–1.064 (October 1.21–1.38) and 5.98 on the
  16,384-leaf recipe, which still takes the mask view. `ci_checks_rc` 20 (expected 21)
  and `cpp_tests_rc` 59 are CI on `1408ee79`; the nightly's CUDA C++ run has its own
  count.

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

<!-- BEGIN measured values (fill_placeholders.py) -->

## Measured values

Generated by `kevo/runs/release_1_1/fill_placeholders.py` from `kevo/runs/release_1_1/results/*_1408ee79*.json` (measured on `1408ee79`). Values as printed in the documents (framework-table.md drops the leading zero of quality values, as its competitor cells do). Sources name the summary file and, where it is not the file's `placeholders` object, the field.

| Placeholder | Value | Source |
| --- | --- | --- |
| `ab_airline_deep_fixed_a65_s` | 24.094 | ab_1408ee79.json |
| `ab_airline_deep_fixed_md5` | yes | ab_1408ee79.json |
| `ab_airline_deep_fixed_rc_s` | 22.443 | ab_1408ee79.json |
| `ab_airline_deep_fixed_x` | 1.074 | ab_1408ee79.json |
| `ab_airline_deep_none_a65_s` | 132.977 | ab_1408ee79.json |
| `ab_airline_deep_none_md5` | yes | ab_1408ee79.json |
| `ab_airline_deep_none_rc_s` | 140.881 | ab_1408ee79.json |
| `ab_airline_deep_none_x` | 0.944 | ab_1408ee79.json |
| `ab_airline_deep_stoch_a65_s` | 23.157 | ab_1408ee79.json |
| `ab_airline_deep_stoch_md5` | yes | ab_1408ee79.json |
| `ab_airline_deep_stoch_rc_s` | 22.098 | ab_1408ee79.json |
| `ab_airline_deep_stoch_x` | 1.048 | ab_1408ee79.json |
| `ab_airline_shallow_fixed_a65_s` | 14.831 | ab_1408ee79.json |
| `ab_airline_shallow_fixed_md5` | yes | ab_1408ee79.json |
| `ab_airline_shallow_fixed_rc_s` | 13.859 | ab_1408ee79.json |
| `ab_airline_shallow_fixed_x` | 1.070 | ab_1408ee79.json |
| `ab_airline_shallow_none_a65_s` | 27.225 | ab_1408ee79.json |
| `ab_airline_shallow_none_md5` | yes | ab_1408ee79.json |
| `ab_airline_shallow_none_rc_s` | 27.439 | ab_1408ee79.json |
| `ab_airline_shallow_none_x` | 0.992 | ab_1408ee79.json |
| `ab_airline_shallow_stoch_a65_s` | 14.436 | ab_1408ee79.json |
| `ab_airline_shallow_stoch_md5` | yes | ab_1408ee79.json |
| `ab_airline_shallow_stoch_rc_s` | 13.317 | ab_1408ee79.json |
| `ab_airline_shallow_stoch_x` | 1.084 | ab_1408ee79.json |
| `ab_covtype_deep_fixed_a65_s` | 2.665 | ab_1408ee79.json |
| `ab_covtype_deep_fixed_md5` | yes | ab_1408ee79.json |
| `ab_covtype_deep_fixed_rc_s` | 2.060 | ab_1408ee79.json |
| `ab_covtype_deep_fixed_x` | 1.294 | ab_1408ee79.json |
| `ab_covtype_deep_none_a65_s` | 7.799 | ab_1408ee79.json |
| `ab_covtype_deep_none_md5` | yes | ab_1408ee79.json |
| `ab_covtype_deep_none_rc_s` | 7.496 | ab_1408ee79.json |
| `ab_covtype_deep_none_x` | 1.040 | ab_1408ee79.json |
| `ab_covtype_deep_stoch_a65_s` | 2.450 | ab_1408ee79.json |
| `ab_covtype_deep_stoch_md5` | yes | ab_1408ee79.json |
| `ab_covtype_deep_stoch_rc_s` | 1.893 | ab_1408ee79.json |
| `ab_covtype_deep_stoch_x` | 1.294 | ab_1408ee79.json |
| `ab_covtype_shallow_fixed_a65_s` | 1.532 | ab_1408ee79.json |
| `ab_covtype_shallow_fixed_md5` | yes | ab_1408ee79.json |
| `ab_covtype_shallow_fixed_rc_s` | 1.169 | ab_1408ee79.json |
| `ab_covtype_shallow_fixed_x` | 1.310 | ab_1408ee79.json |
| `ab_covtype_shallow_none_a65_s` | 3.057 | ab_1408ee79.json |
| `ab_covtype_shallow_none_md5` | yes | ab_1408ee79.json |
| `ab_covtype_shallow_none_rc_s` | 2.696 | ab_1408ee79.json |
| `ab_covtype_shallow_none_x` | 1.134 | ab_1408ee79.json |
| `ab_covtype_shallow_stoch_a65_s` | 1.493 | ab_1408ee79.json |
| `ab_covtype_shallow_stoch_md5` | yes | ab_1408ee79.json |
| `ab_covtype_shallow_stoch_rc_s` | 1.133 | ab_1408ee79.json |
| `ab_covtype_shallow_stoch_x` | 1.319 | ab_1408ee79.json |
| `ab_epsilon_deep_fixed_a65_s` | 12.583 | ab_1408ee79.json |
| `ab_epsilon_deep_fixed_md5` | yes | ab_1408ee79.json |
| `ab_epsilon_deep_fixed_rc_s` | 10.373 | ab_1408ee79.json |
| `ab_epsilon_deep_fixed_x` | 1.213 | ab_1408ee79.json |
| `ab_epsilon_deep_none_a65_s` | 64.739 | ab_1408ee79.json |
| `ab_epsilon_deep_none_md5` | yes | ab_1408ee79.json |
| `ab_epsilon_deep_none_rc_s` | 72.703 | ab_1408ee79.json |
| `ab_epsilon_deep_none_x` | 0.890 | ab_1408ee79.json |
| `ab_epsilon_deep_stoch_a65_s` | 11.460 | ab_1408ee79.json |
| `ab_epsilon_deep_stoch_md5` | yes | ab_1408ee79.json |
| `ab_epsilon_deep_stoch_rc_s` | 9.822 | ab_1408ee79.json |
| `ab_epsilon_deep_stoch_x` | 1.167 | ab_1408ee79.json |
| `ab_epsilon_shallow_fixed_a65_s` | 2.925 | ab_1408ee79.json |
| `ab_epsilon_shallow_fixed_md5` | yes | ab_1408ee79.json |
| `ab_epsilon_shallow_fixed_rc_s` | 2.558 | ab_1408ee79.json |
| `ab_epsilon_shallow_fixed_x` | 1.144 | ab_1408ee79.json |
| `ab_epsilon_shallow_none_a65_s` | 9.274 | ab_1408ee79.json |
| `ab_epsilon_shallow_none_md5` | yes | ab_1408ee79.json |
| `ab_epsilon_shallow_none_rc_s` | 9.401 | ab_1408ee79.json |
| `ab_epsilon_shallow_none_x` | 0.987 | ab_1408ee79.json |
| `ab_epsilon_shallow_stoch_a65_s` | 2.539 | ab_1408ee79.json |
| `ab_epsilon_shallow_stoch_md5` | yes | ab_1408ee79.json |
| `ab_epsilon_shallow_stoch_rc_s` | 2.330 | ab_1408ee79.json |
| `ab_epsilon_shallow_stoch_x` | 1.090 | ab_1408ee79.json |
| `ab_fraud_deep_fixed_a65_s` | 0.463 | ab_1408ee79.json |
| `ab_fraud_deep_fixed_a65_trees` | 500 | ab_1408ee79/fraud-deep-fixed.json runs[].num_trees |
| `ab_fraud_deep_fixed_md5` | no | ab_1408ee79.json |
| `ab_fraud_deep_fixed_rc_s` | 0.323 | ab_1408ee79.json |
| `ab_fraud_deep_fixed_rc_trees` | 357 | ab_1408ee79/fraud-deep-fixed.json runs[].num_trees |
| `ab_fraud_deep_fixed_x` | 1.433 | ab_1408ee79.json |
| `ab_fraud_deep_none_a65_s` | 0.525 | ab_1408ee79.json |
| `ab_fraud_deep_none_md5` | yes | ab_1408ee79.json |
| `ab_fraud_deep_none_rc_s` | 0.477 | ab_1408ee79.json |
| `ab_fraud_deep_none_x` | 1.099 | ab_1408ee79.json |
| `ab_fraud_deep_stoch_a65_s` | 0.309 | ab_1408ee79.json |
| `ab_fraud_deep_stoch_md5` | yes | ab_1408ee79.json |
| `ab_fraud_deep_stoch_rc_s` | 0.250 | ab_1408ee79.json |
| `ab_fraud_deep_stoch_x` | 1.237 | ab_1408ee79.json |
| `ab_fraud_shallow_fixed_a65_s` | 0.353 | ab_1408ee79.json |
| `ab_fraud_shallow_fixed_md5` | no | ab_1408ee79.json |
| `ab_fraud_shallow_fixed_rc_s` | 0.274 | ab_1408ee79.json |
| `ab_fraud_shallow_fixed_x` | 1.291 | ab_1408ee79.json |
| `ab_fraud_shallow_none_a65_s` | 0.430 | ab_1408ee79.json |
| `ab_fraud_shallow_none_md5` | yes | ab_1408ee79.json |
| `ab_fraud_shallow_none_rc_s` | 0.379 | ab_1408ee79.json |
| `ab_fraud_shallow_none_x` | 1.135 | ab_1408ee79.json |
| `ab_fraud_shallow_stoch_a65_s` | 0.289 | ab_1408ee79.json |
| `ab_fraud_shallow_stoch_md5` | yes | ab_1408ee79.json |
| `ab_fraud_shallow_stoch_rc_s` | 0.203 | ab_1408ee79.json |
| `ab_fraud_shallow_stoch_x` | 1.424 | ab_1408ee79.json |
| `ab_geomean_fixed` | 1.216 | ab_1408ee79.json |
| `ab_geomean_none` | 1.006 | ab_1408ee79.json |
| `ab_geomean_stoch` | 1.189 | ab_1408ee79.json |
| `ab_higgs_deep_fixed_a65_s` | 3.107 | ab_1408ee79.json |
| `ab_higgs_deep_fixed_md5` | yes | ab_1408ee79.json |
| `ab_higgs_deep_fixed_rc_s` | 2.236 | ab_1408ee79.json |
| `ab_higgs_deep_fixed_x` | 1.390 | ab_1408ee79.json |
| `ab_higgs_deep_none_a65_s` | 15.566 | ab_1408ee79.json |
| `ab_higgs_deep_none_md5` | yes | ab_1408ee79.json |
| `ab_higgs_deep_none_rc_s` | 17.509 | ab_1408ee79.json |
| `ab_higgs_deep_none_x` | 0.889 | ab_1408ee79.json |
| `ab_higgs_deep_stoch_a65_s` | 2.488 | ab_1408ee79.json |
| `ab_higgs_deep_stoch_md5` | yes | ab_1408ee79.json |
| `ab_higgs_deep_stoch_rc_s` | 2.101 | ab_1408ee79.json |
| `ab_higgs_deep_stoch_x` | 1.184 | ab_1408ee79.json |
| `ab_higgs_shallow_fixed_a65_s` | 1.749 | ab_1408ee79.json |
| `ab_higgs_shallow_fixed_md5` | yes | ab_1408ee79.json |
| `ab_higgs_shallow_fixed_rc_s` | 1.457 | ab_1408ee79.json |
| `ab_higgs_shallow_fixed_x` | 1.200 | ab_1408ee79.json |
| `ab_higgs_shallow_none_a65_s` | 4.042 | ab_1408ee79.json |
| `ab_higgs_shallow_none_md5` | yes | ab_1408ee79.json |
| `ab_higgs_shallow_none_rc_s` | 3.936 | ab_1408ee79.json |
| `ab_higgs_shallow_none_x` | 1.027 | ab_1408ee79.json |
| `ab_higgs_shallow_stoch_a65_s` | 1.516 | ab_1408ee79.json |
| `ab_higgs_shallow_stoch_md5` | yes | ab_1408ee79.json |
| `ab_higgs_shallow_stoch_rc_s` | 1.305 | ab_1408ee79.json |
| `ab_higgs_shallow_stoch_x` | 1.162 | ab_1408ee79.json |
| `ab_none_rc_deterministic` | 13 of 13 | ab_1408ee79.json |
| `ab_numerai_numerai_deep_fixed_a65_s` | 133.469 | ab_1408ee79.json |
| `ab_numerai_numerai_deep_fixed_md5` | yes | ab_1408ee79.json |
| `ab_numerai_numerai_deep_fixed_rc_s` | 127.390 | ab_1408ee79.json |
| `ab_numerai_numerai_deep_fixed_x` | 1.048 | ab_1408ee79.json |
| `ab_numerai_numerai_deep_stoch_a65_s` | 112.969 | ab_1408ee79.json |
| `ab_numerai_numerai_deep_stoch_md5` | yes | ab_1408ee79.json |
| `ab_numerai_numerai_deep_stoch_rc_s` | 105.823 | ab_1408ee79.json |
| `ab_numerai_numerai_deep_stoch_x` | 1.068 | ab_1408ee79.json |
| `ab_numerai_numerai_fixed_a65_s` | 6.324 | ab_1408ee79.json |
| `ab_numerai_numerai_fixed_md5` | yes | ab_1408ee79.json |
| `ab_numerai_numerai_fixed_rc_s` | 6.127 | ab_1408ee79.json |
| `ab_numerai_numerai_fixed_x` | 1.032 | ab_1408ee79.json |
| `ab_numerai_numerai_leaf_fixed_a65_s` | 282.766 | ab_1408ee79.json |
| `ab_numerai_numerai_leaf_fixed_md5` | yes | ab_1408ee79.json |
| `ab_numerai_numerai_leaf_fixed_rc_s` | 259.415 | ab_1408ee79.json |
| `ab_numerai_numerai_leaf_fixed_x` | 1.090 | ab_1408ee79.json |
| `ab_numerai_numerai_leaf_stoch_a65_s` | 265.021 | ab_1408ee79.json |
| `ab_numerai_numerai_leaf_stoch_md5` | yes | ab_1408ee79.json |
| `ab_numerai_numerai_leaf_stoch_rc_s` | 239.700 | ab_1408ee79.json |
| `ab_numerai_numerai_leaf_stoch_x` | 1.106 | ab_1408ee79.json |
| `ab_numerai_numerai_none_a65_s` | 54.837 | ab_1408ee79.json |
| `ab_numerai_numerai_none_md5` | yes | ab_1408ee79.json |
| `ab_numerai_numerai_none_rc_s` | 55.743 | ab_1408ee79.json |
| `ab_numerai_numerai_none_x` | 0.984 | ab_1408ee79.json |
| `ab_numerai_numerai_stoch_a65_s` | 5.620 | ab_1408ee79.json |
| `ab_numerai_numerai_stoch_md5` | yes | ab_1408ee79.json |
| `ab_numerai_numerai_stoch_rc_s` | 5.370 | ab_1408ee79.json |
| `ab_numerai_numerai_stoch_x` | 1.046 | ab_1408ee79.json |
| `ab_quant_md5_identical` | 28 of 30 | ab_1408ee79.json |
| `ab_year_deep_fixed_a65_s` | 1.012 | ab_1408ee79.json |
| `ab_year_deep_fixed_md5` | yes | ab_1408ee79.json |
| `ab_year_deep_fixed_rc_s` | 0.706 | ab_1408ee79.json |
| `ab_year_deep_fixed_x` | 1.434 | ab_1408ee79.json |
| `ab_year_deep_none_a65_s` | 3.428 | ab_1408ee79.json |
| `ab_year_deep_none_md5` | yes | ab_1408ee79.json |
| `ab_year_deep_none_rc_s` | 3.690 | ab_1408ee79.json |
| `ab_year_deep_none_x` | 0.929 | ab_1408ee79.json |
| `ab_year_deep_stoch_a65_s` | 0.918 | ab_1408ee79.json |
| `ab_year_deep_stoch_md5` | yes | ab_1408ee79.json |
| `ab_year_deep_stoch_rc_s` | 0.690 | ab_1408ee79.json |
| `ab_year_deep_stoch_x` | 1.331 | ab_1408ee79.json |
| `ab_year_shallow_fixed_a65_s` | 0.408 | ab_1408ee79.json |
| `ab_year_shallow_fixed_md5` | yes | ab_1408ee79.json |
| `ab_year_shallow_fixed_rc_s` | 0.306 | ab_1408ee79.json |
| `ab_year_shallow_fixed_x` | 1.333 | ab_1408ee79.json |
| `ab_year_shallow_none_a65_s` | 0.794 | ab_1408ee79.json |
| `ab_year_shallow_none_md5` | yes | ab_1408ee79.json |
| `ab_year_shallow_none_rc_s` | 0.740 | ab_1408ee79.json |
| `ab_year_shallow_none_x` | 1.073 | ab_1408ee79.json |
| `ab_year_shallow_stoch_a65_s` | 0.391 | ab_1408ee79.json |
| `ab_year_shallow_stoch_md5` | yes | ab_1408ee79.json |
| `ab_year_shallow_stoch_rc_s` | 0.285 | ab_1408ee79.json |
| `ab_year_shallow_stoch_x` | 1.370 | ab_1408ee79.json |
| `aug_catboost_over_rc_epsilon_deep` | 10.3 | ab_1408ee79.json |
| `aug_catboost_over_rc_numerai_deep` | 49.5 | ab_1408ee79.json |
| `aug_catboost_over_rc_numerai_example` | 20.9 | ab_1408ee79.json |
| `aug_lgbm_opencl_over_rc_numerai_deep` | 97.9 | ab_1408ee79.json |
| `aug_xgb_over_rc_epsilon_deep` | 5.3 | ab_1408ee79.json |
| `aug_xgb_over_rc_numerai_deep` | 66.2 | ab_1408ee79.json |
| `aug_xgb_over_rc_numerai_example` | 53.3 | ab_1408ee79.json |
| `canonical_gates_rc` | 4 | nightly gates nightly-2026-10-21.log on 49c5622a |
| `ci_checks_rc` | 20 | counts_1408ee79.json counts.ci_checks |
| `cpp_tests_rc` | 59 | counts_1408ee79.json counts.cpp_tests_passed |
| `cpu_tests_passed_rc` | 1,396 | counts_1408ee79.json counts.cpu_tests_passed |
| `cuda_tests_passed_rc` | 2,422 | nightly gates nightly-2026-10-21.log on 49c5622a |
| `cuda_tests_skipped_rc` | 189 | nightly gates nightly-2026-10-21.log on 49c5622a |
| `lattice_cells_rc` | 441 | counts_1408ee79.json counts.lattice_cells |
| `nightly_fuzz_specs_rc` | 310 | nightly gates nightly-2026-10-21.log on 49c5622a |
| `numerai_deep_30k_a65_s` | 112.97 | ab_1408ee79.json |
| `numerai_deep_30k_launch_over_rc` | 7.02 | ab_1408ee79.json |
| `numerai_deep_30k_rc_min` | 1.8 | ab_1408ee79.json |
| `numerai_deep_30k_rc_over_a65` | 1.068 | ab_1408ee79.json |
| `numerai_deep_30k_rc_s` | 105.82 | ab_1408ee79.json |
| `numerai_deep_md5_rc_vs_a65` | identical | ab_1408ee79.json |
| `q_airline_deep_fixed_rc` | 0.8639 | quality_1408ee79.json |
| `q_airline_deep_none_rc` | 0.8637 | quality_1408ee79.json |
| `q_airline_deep_stoch_rc` | 0.8641 | quality_1408ee79.json |
| `q_airline_shallow_fixed_rc` | 0.8262 | quality_1408ee79.json |
| `q_airline_shallow_none_rc` | 0.8261 | quality_1408ee79.json |
| `q_airline_shallow_stoch_rc` | 0.8263 | quality_1408ee79.json |
| `q_covtype_deep_fixed_rc` | 0.9687 | quality_1408ee79.json |
| `q_covtype_deep_none_rc` | 0.9704 | quality_1408ee79.json |
| `q_covtype_deep_stoch_rc` | 0.9581 | quality_1408ee79.json |
| `q_covtype_shallow_fixed_rc` | 0.9221 | quality_1408ee79.json |
| `q_covtype_shallow_none_rc` | 0.9243 | quality_1408ee79.json |
| `q_covtype_shallow_stoch_rc` | 0.9204 | quality_1408ee79.json |
| `q_epsilon_deep_fixed_rc` | 0.9432 | quality_1408ee79.json |
| `q_epsilon_deep_none_rc` | 0.9430 | quality_1408ee79.json |
| `q_epsilon_deep_stoch_rc` | 0.9424 | quality_1408ee79.json |
| `q_epsilon_shallow_fixed_rc` | 0.9487 | quality_1408ee79.json |
| `q_epsilon_shallow_none_rc` | 0.9485 | quality_1408ee79.json |
| `q_epsilon_shallow_stoch_rc` | 0.9489 | quality_1408ee79.json |
| `q_fraud_deep_fixed_a65` | 0.9780 | quality_1408ee79.json old_arm |
| `q_fraud_deep_fixed_rc` | 0.9774 | quality_1408ee79.json |
| `q_fraud_deep_none_rc` | 0.9695 | quality_1408ee79.json |
| `q_fraud_deep_stoch_rc` | 0.9630 | quality_1408ee79.json |
| `q_fraud_shallow_fixed_a65` | 0.9790 | quality_1408ee79.json old_arm |
| `q_fraud_shallow_fixed_rc` | 0.9808 | quality_1408ee79.json |
| `q_fraud_shallow_none_rc` | 0.9771 | quality_1408ee79.json |
| `q_fraud_shallow_stoch_rc` | 0.9365 | quality_1408ee79.json |
| `q_higgs_deep_fixed_rc` | 0.8482 | quality_1408ee79.json |
| `q_higgs_deep_none_rc` | 0.8484 | quality_1408ee79.json |
| `q_higgs_deep_stoch_rc` | 0.8486 | quality_1408ee79.json |
| `q_higgs_shallow_fixed_rc` | 0.8303 | quality_1408ee79.json |
| `q_higgs_shallow_none_rc` | 0.8306 | quality_1408ee79.json |
| `q_higgs_shallow_stoch_rc` | 0.8310 | quality_1408ee79.json |
| `q_numerai_numerai_deep_fixed_rc` | 0.02332 | quality_1408ee79.json |
| `q_numerai_numerai_deep_stoch_rc` | 0.02385 | quality_1408ee79.json |
| `q_numerai_numerai_fixed_rc` | 0.01896 | quality_1408ee79.json |
| `q_numerai_numerai_leaf_fixed_rc` | 0.01979 | quality_1408ee79.json |
| `q_numerai_numerai_leaf_stoch_rc` | 0.01999 | quality_1408ee79.json |
| `q_numerai_numerai_none_rc` | 0.01931 | quality_1408ee79.json |
| `q_numerai_numerai_stoch_rc` | 0.01943 | quality_1408ee79.json |
| `q_year_deep_fixed_rc` | 8.971 | quality_1408ee79.json |
| `q_year_deep_none_rc` | 8.973 | quality_1408ee79.json |
| `q_year_deep_stoch_rc` | 8.972 | quality_1408ee79.json |
| `q_year_shallow_fixed_rc` | 8.971 | quality_1408ee79.json |
| `q_year_shallow_none_rc` | 8.961 | quality_1408ee79.json |
| `q_year_shallow_stoch_rc` | 9.000 | quality_1408ee79.json |
| `rc_commit` | 1408ee79 | ab_1408ee79.json |
| `rc_library_sha256` | c1902dce58ac356c9964a708e700e438062c596104617f4d91d243bcce26839a | ab_1408ee79.json |
| `sixstate_16k_fp32_2k_five_trees_s` | 64.35 | six_state_1408ee79.json |
| `sixstate_16k_fp32_2k_six_over_five_time` | 5.983 | six_state_1408ee79.json |
| `sixstate_16k_fp32_2k_six_trees_s` | 10.76 | six_state_1408ee79.json |
| `sixstate_16k_view_six` | full-width mask | six_state_1408ee79.json |
| `sixstate_std_fixed_2k_five_trees_s` | 220.70 | six_state_1408ee79.json |
| `sixstate_std_fixed_2k_six_over_five_time` | 1.054 | six_state_1408ee79.json |
| `sixstate_std_fixed_2k_six_trees_s` | 209.33 | six_state_1408ee79.json |
| `sixstate_std_six_code_view_trees` | 200 of 200 | six_state_1408ee79.json |
| `sixstate_std_stoch_2k_five_trees_s` | 261.21 | six_state_1408ee79.json |
| `sixstate_std_stoch_2k_six_over_five_time` | 1.056 | six_state_1408ee79.json |
| `sixstate_std_stoch_2k_six_trees_s` | 247.42 | six_state_1408ee79.json |
| `sixstate_std_stoch_30k_five_holdout_corr` | 0.0348 | six_state_1408ee79/full-standard-stoch-five.json holdout.numerai_corr_mean |
| `sixstate_std_stoch_30k_five_trees_s` | 285.05 | six_state_1408ee79.json |
| `sixstate_std_stoch_30k_six_holdout_corr` | 0.0352 | six_state_1408ee79/full-standard-stoch-six.json holdout.numerai_corr_mean |
| `sixstate_std_stoch_30k_six_over_five_time` | 1.064 | six_state_1408ee79.json |
| `sixstate_std_stoch_30k_six_trees_s` | 267.79 | six_state_1408ee79.json |
| `stability_cpu_controls_reproduced` | 2 | stability_1408ee79_cpu.json |
| `stability_cpu_covtype_acc_range` | 0.9692–0.9702 | stability_1408ee79_cpu.json |
| `stability_cpu_fraud_auc_range` | 0.9722–0.9840 | stability_1408ee79_cpu.json |
| `stability_cpu_guarded_pass` | 74 | stability_1408ee79_cpu.json |
| `stability_cuda_controls_reproduced` | 2 | stability_1408ee79.json |
| `stability_cuda_covtype_acc_range` | 0.9694–0.9704 | stability_1408ee79.json |
| `stability_cuda_fingerprints_same` | 2 of 68 | stability_1408ee79.json tree_fingerprints_same_as_record |
| `stability_cuda_fraud_auc_range` | 0.9697–0.9850 | stability_1408ee79.json |
| `stability_cuda_fresh_covtype_acc_range` | 0.9702–0.9710 | stability_1408ee79.json |
| `stability_cuda_fresh_fraud_auc_range` | 0.9647–0.9849 | stability_1408ee79.json |
| `stability_cuda_guarded_pass` | 66 | stability_1408ee79.json |
| `stability_rc_records` | 68 | stability_1408ee79.json |

<!-- END measured values -->
