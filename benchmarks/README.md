# Falcata GPU benchmark suite

Reproducible comparison of **Falcata** (this repo, `device_type=cuda`) against
**upstream LightGBM** (CUDA build), **XGBoost** (`device=cuda`,
`tree_method=hist`) and **CatBoost** (`task_type=GPU`) — training speed *and*
model quality on a single GPU.

## Datasets

The classic GBDT speed suite (NVIDIA gbm-bench lineage) plus Numerai:

| dataset | shape | task | split |
|---|---|---|---|
| fraud (OpenML creditcard) | 285K × 29 | binary, imbalanced | stratified 20% |
| covtype (UCI) | 581K × 54 | 7-class | stratified 20% |
| year (UCI YearPredictionMSD) | 515K × 90 | regression | canonical last 51,630 |
| higgs (UCI) | 11M × 28 | binary | canonical last 500K |
| epsilon (LIBSVM) | 500K × 2000 | binary | official train/test |
| airline (Ikonomovska) | 115M × 13 | binary (ArrDelay>0) | random 20%, seed 42 |
| numerai (v5 "all data", optional) | ~6.7M × ~2.7K | regression + era metrics | last 200 eras, 10-era embargo |

All datasets are cached as float32 arrays so every library trains from
identical bits. Numerai requires the v5 training parquet (get it with
`numerapi`) via the `NUMERAI_PARQUET` env var; its quality metrics are the
official Numerai correlation (rank → gaussianize → power 1.5 → Pearson),
per-era mean/std/Sharpe and max drawdown.

## Run matrix

Six library configs — `exaboost`, `exaboost-quant` (`use_quantized_grad`),
`lightgbm`, `lightgbm-quant`, `xgboost`, `catboost` — across two aligned
hyperparameter regimes (gbm-bench convention: 500 rounds, lr 0.1, 255 bins,
L2 leaf regularization 1.0):

- **shallow**: depth 6 / 63 leaves
- **deep**: depth 10 / 1023 leaves

L2 is aligned explicitly because engine defaults differ (XGBoost `lambda=1`,
LightGBM `lambda_l2=0`, CatBoost `l2_leaf_reg=3`) and at lr 0.1 an
unregularized leaf-wise model degenerates on imbalanced data (fraud drops to
~0.5 AUC), which would measure default choices rather than the engines.

plus the official Numerai example-model config (2000 trees, lr 0.01, depth 5,
32 leaves, colsample 0.1). Structural caveat: LightGBM-family grows leaf-wise
(`num_leaves` + `max_depth` cap), XGBoost depth-wise (`max_depth`), CatBoost
symmetric (`depth`) — the regimes align the tree budget, not the tree shape.

Each cell runs 1 discarded warmup + 3 timed repeats (median reported) + 1
`curve` run that evaluates the held-out set periodically for time-to-quality
plots. Every run is an isolated subprocess; peak GPU memory and host RSS are
polled throughout. See the docstring in `bench.py` for exact timing semantics.

## Reproducing

```bash
# 1. build both environments (needs CUDA toolkit + driver, cmake, compiler)
./benchmarks/setup_envs.sh

# 2. download + preprocess datasets (largest: airline ~6GB download, ~30GB RAM)
./benchmarks/workspace/env-competitors/bin/python benchmarks/datasets.py all
NUMERAI_PARQUET=/path/to/v5_all_data.parquet \
  ./benchmarks/workspace/env-competitors/bin/python benchmarks/datasets.py numerai   # optional

# 3. run the matrix (resumable — interrupt and relaunch freely)
python3 benchmarks/orchestrate.py

# 4. aggregate results into workspace/report/REPORT.md + charts
./benchmarks/workspace/env-competitors/bin/python benchmarks/report.py
```

Everything lands under `benchmarks/workspace/` (override with
`FALCATA_BENCH_ROOT`). `FALCATA_CUDA_ARCHS` overrides
`CMAKE_CUDA_ARCHITECTURES` (default `native`; e.g. `120-real;120-virtual` for
Blackwell). Expect ~25GB of downloads and, with all datasets, a day-plus of
GPU time for the full matrix; `--only fraud,covtype,year` gives a quick
signal in under an hour.

## Native int8 ingestion (optional, Falcata-only)

The matrix feeds every library identical float32 bits, so Falcata's native
small-int path (int8/int16 matrices pass zero-copy through the C API and are
binned via per-column LUTs — bins byte-identical to the float path) is
measured separately:

```bash
./benchmarks/workspace/env-competitors/bin/python benchmarks/datasets.py numerai-int8
./benchmarks/workspace/env-exaboost/bin/python benchmarks/ingest_bench.py
```

Reference (RTX 5090, commit 9c0f5ffa, medians of 3): construct 38.9s (f32-fed)
vs **15.8s** (int8-fed), peak host RSS 86.4GB vs **43.9GB**; Booster create and
per-tree times identical, models md5-identical.
