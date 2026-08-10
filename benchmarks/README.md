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
| airline-cat | as airline | binary | as airline |
| numerai (v5 "all data", optional) | 6.8M × 3,555 | regression + era metrics | last 200 eras, 10-era embargo |

`airline-cat` is the same data as `airline`, but `UniqueCarrier`/`Origin`/`Dest`
are declared **categorical** instead of being read as ordered integers, so the
engines take their categorical split paths. Origin/Dest carry ~300 airports
each, past the suite-wide 255-bin budget, so they are recoded by descending
train-set frequency with the tail beyond the top 254 collapsed into one rare
bucket (ranking uses the train split only).

All datasets are cached as float32 arrays so every library trains from
identical bits. Numerai requires the v5 training parquet (get it with
`numerapi`) via the `NUMERAI_PARQUET` env var; its quality metrics are the
official Numerai correlation (rank → gaussianize → power 1.5 → Pearson),
per-era mean/std/Sharpe and max drawdown.

## Run matrix

Ten library configs:

| config | what it is |
|---|---|
| `falcata-stoch` | Falcata, `quant_mode=stochastic` (the default) |
| `falcata-fixed` | Falcata, `quant_mode=fixedpoint` |
| `falcata-noquant` | Falcata, `quant_mode=none` (full-precision gradients) |
| `falcata-stoch64` | stochastic rounding at fixedpoint's 64-bin budget; isolates the rounding scheme from the bin count (`numerai-deep` only) |
| `lightgbm` | upstream LightGBM, CUDA backend |
| `lightgbm-quant` | upstream + `use_quantized_grad` |
| `lightgbm-ocl` | upstream, legacy OpenCL backend — the reference wherever its CUDA backend crashes |
| `xgboost` | `device=cuda`, `tree_method=hist`, depth-wise |
| `xgboost-lossguide` | same, `grow_policy=lossguide` with the LightGBM family's `(num_leaves, max_depth)` pair — the leaf-wise apples-to-apples cell |
| `catboost` | `task_type=GPU`, symmetric trees |

across the hyperparameter regimes (gbm-bench convention: 500 rounds, lr 0.1,
255 bins):

- **shallow**: depth 6 / 63 leaves
- **deep**: depth 10 / 1023 leaves
- **numerai**: official example-model config (2000 trees, lr 0.01, depth 5, 32 leaves, colsample 0.1)
- **numerai-deep**: official v5 benchmark-model deep params (30k trees, lr 0.001, depth 10, 1024 leaves, colsample 0.1, min_data 10k)
- **numerai-leaf**: as numerai-deep but depth-unbounded, so the 1024-leaf budget is the binding constraint (min_data 1000)

Structural caveat: the LightGBM family grows leaf-wise (`num_leaves` +
`max_depth` cap), XGBoost depth-wise (`max_depth`), CatBoost symmetric
(`depth`) — the regimes align the tree budget, not the tree shape.

**On L2 regularization.** The engines ship different default L2 leaf penalties
(XGBoost `lambda=1`, LightGBM `lambda_l2=0`, CatBoost `l2_leaf_reg=3`) and the
gbm-bench convention does not align them, so every published number is measured
on each engine's own default. `bench.py --align-l2` sets all three to 1.0 for
the stricter comparison; results from it will **not** match `docs/performance.md`.

Each cell runs 1 discarded warmup + 3 timed repeats (median reported) + 1
`curve` run that evaluates the held-out set periodically for time-to-quality
plots. The two 30k-round Numerai regimes run a single timed cell instead —
repeats there cost days. Every run is an isolated subprocess; peak GPU memory
and host RSS are polled throughout. See the docstring in `bench.py` for exact
timing semantics.

A run whose held-out quality fails a sanity floor is recorded with status
`insane` rather than `ok`, so a model that trained fast by learning nothing can
never enter the speed tables.

**Timings are only as quiet as the machine.** Contention on the host moved
medians by 18–55% in our own measurements, which is larger than most of the
effects being measured. Run the suite on an otherwise idle box.

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

To reproduce one cell rather than the matrix, call `bench.py` directly with the
venv that owns the library:

```bash
export FALCATA_BENCH_ROOT=/big/disk/falcata-bench
$FALCATA_BENCH_ROOT/env-falcata/bin/python benchmarks/bench.py \
    --library falcata-stoch --dataset covtype --regime deep --kind timed1
$FALCATA_BENCH_ROOT/env-competitors/bin/python benchmarks/bench.py \
    --library xgboost --dataset covtype --regime deep --kind timed1
```

Each call appends one JSON record to `<workspace>/results/runs.jsonl` and
prints it. `--regime smoke` (10 trees) is the fastest way to confirm all four
engines really are on the GPU before committing to a long run.

Everything lands under `benchmarks/workspace/` (override with
`FALCATA_BENCH_ROOT`; point it at a big disk — the full cache is ~200GB).
`FALCATA_CUDA_ARCHS` overrides
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
./benchmarks/workspace/env-falcata/bin/python benchmarks/ingest_bench.py
```

Reference (RTX 5090, commit 9c0f5ffa, medians of 3): construct 38.9s (f32-fed)
vs **15.8s** (int8-fed), peak host RSS 86.4GB vs **43.9GB**; Booster create and
per-tree times identical, models md5-identical.

## Feature attribution (leave-one-out ablation)

`ablation.py` measures what each `cuda_plan` key buys: for each (dataset,
regime) cell it runs the full auto plan, then flips every key off (or on, for
default-off keys) one at a time, reporting the throughput delta per key and
verifying the bit-identity contract — identity-key flips must reproduce the
baseline tree md5 exactly (a mismatch is flagged as a correctness bug);
growth/tie-break keys are judged on quality delta instead.

```bash
benchmarks/ablation.py --list                 # show the cells
benchmarks/ablation.py                        # full sweep (JSONL + tables)
benchmarks/ablation.py --cells covtype-deep-quant numerai-deep-quant
```
