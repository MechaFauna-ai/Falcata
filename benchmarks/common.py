"""Shared configuration for the Falcata GPU benchmark suite.

All artifacts (raw downloads, preprocessed caches, results, report) live under
a single workspace directory, set with ``$FALCATA_BENCH_ROOT``. It can be
anywhere -- the full suite runs to hundreds of GB and rarely belongs inside a
checkout -- and defaults to ``benchmarks/workspace`` in the checkout itself.
"""

import os

ROOT = os.environ.get(
    "FALCATA_BENCH_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace"),
)
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
REPORT_DIR = os.path.join(ROOT, "report")
RUNS_JSONL = os.path.join(RESULTS_DIR, "runs.jsonl")

SEED = 42

#: benchmark datasets; ``task`` drives objective/metric selection in bench.py
DATASETS = {
    "higgs": {"task": "binary"},
    "epsilon": {"task": "binary"},
    "airline": {"task": "binary"},
    "covtype": {"task": "multiclass", "num_class": 7},
    "year": {"task": "regression"},
    "fraud": {"task": "binary"},
    "numerai": {"task": "numerai"},
    # same rows as airline, but UniqueCarrier/Origin/Dest are declared
    # CATEGORICAL instead of being read as ordered numbers, so the engines take
    # their categorical split paths; Origin/Dest are recoded to the top-254
    # train-frequency categories plus a rare bucket (see datasets.py)
    "airline-cat": {"task": "binary", "cat_cols": [6, 9, 10]},
}

#: hyperparameter regimes, aligned across libraries (gbm-bench convention);
#: LightGBM-family maps depth/leaves to (max_depth, num_leaves), XGBoost uses
#: max_depth (native depth-wise growth), CatBoost uses symmetric depth.
REGIMES = {
    # NOTE ON L2. Each engine ships a different default L2 leaf penalty
    # (xgboost lambda=1, lightgbm lambda_l2=0, catboost l2_leaf_reg=3), and the
    # gbm-bench convention these regimes follow does not align them -- so the
    # published numbers in docs/performance.md were measured on each engine's
    # own default. Setting "l2" here would align them and change every quality
    # figure, so it is deliberately absent: what ships must reproduce what is
    # published. Pass --align-l2 to bench.py for the stricter comparison.
    "shallow": {
        "rounds": 500,
        "lr": 0.1,
        "depth": 6,
        "leaves": 63,
        "eval_every": 25,
    },
    "deep": {
        "rounds": 500,
        "lr": 0.1,
        "depth": 10,
        "leaves": 1023,
        "eval_every": 25,
    },
    # Numerai's published deep_lgbm_params, the parameters behind their v5
    # benchmark models (docs.numer.ai/numerai-tournament/models#deep-lgbm-params);
    # verified field-for-field against that page. The config most Numerai users
    # care about; the example config stays for completeness
    "numerai-deep": {
        "rounds": 30000,
        "lr": 0.001,
        "depth": 10,
        "leaves": 1024,
        "colsample": 0.1,
        "min_data": 10000,
        "eval_every": 3000,
    },
    # the model in Numerai's hello_numerai notebook
    # (github.com/numerai/example-scripts, numerai/hello_numerai.ipynb).
    # DEVIATION: that notebook sets num_leaves=2**5-1 = 31; every published
    # number was measured at 32 and the value is kept so this config still
    # reproduces them. Every engine gets the same 32, so the comparison is
    # unaffected -- but it is not a verbatim copy of the notebook.
    "numerai": {
        "rounds": 2000,
        "lr": 0.01,
        "depth": 5,
        "leaves": 32,
        "colsample": 0.1,
        "eval_every": 250,
    },
    # leaf-limited growth: depth unbounded, so the 1024-leaf budget is the
    # binding constraint (this is the shape selective grow-then-prune serves).
    # min_data drops to 1000 -- at 10000 the budget could never bind, since
    # 5.43M rows / 10k caps the tree at ~543 leaves. Maps to lossguide growth on
    # xgboost/catboost; catboost caps depth at its hard limit of 16.
    "numerai-leaf": {
        "rounds": 30000,
        "lr": 0.001,
        "depth": -1,
        "leaves": 1024,
        "colsample": 0.1,
        "min_data": 1000,
        "eval_every": 2000,
    },
    # tiny config for smoke-testing the library/GPU integration
    "smoke": {"rounds": 10, "lr": 0.1, "depth": 6, "leaves": 63, "eval_every": 5},
}

#: falcata/lightgbm both install as package "lightgbm", hence separate venvs
LIBRARIES = [
    # the three quant modes are separate cells: they are different models, not
    # a flag on one (falcata-stoch64 only appears in the mode-comparison regime)
    "falcata-stoch",
    "falcata-fixed",
    "falcata-noquant",
    "lightgbm",
    "lightgbm-quant",
    "lightgbm-ocl",
    "xgboost",
    # same (num_leaves, max_depth) pair as the LightGBM family instead of
    # depth-wise max_depth alone -- the leaf-wise apples-to-apples cell
    "xgboost-lossguide",
    "catboost",
]

#: extra libraries per regime, beyond LIBRARIES
LIBRARIES_BY_REGIME = {
    "numerai-deep": LIBRARIES + ["falcata-stoch64"],
    # depth == -1 already makes plain xgboost lossguide, so the variant would
    # duplicate the exact same cell
    "numerai-leaf": [x for x in LIBRARIES if x != "xgboost-lossguide"],
}

#: cells a library participates in ((dataset, regime) pairs); libraries not
#: listed here run the full matrix. lightgbm-ocl is upstream's legacy OpenCL
#: backend, run everywhere: it is the reference for cells where upstream's CUDA
#: backend crashes, and report.py renders it in their place (FALLBACK_LIBS).
LIBRARY_CELLS = {
    "falcata-stoch64": {("numerai", "numerai-deep")},
}


#: every library that appears anywhere in the matrix -- iterate this and let
#: library_runs_cell() filter, so regime-only libraries are never invisible
ALL_LIBRARIES = LIBRARIES + [
    x
    for x in dict.fromkeys(sum(LIBRARIES_BY_REGIME.values(), []))
    if x not in LIBRARIES
]


def libraries_for(regime: str):
    """Libraries benchmarked in ``regime``."""
    return LIBRARIES_BY_REGIME.get(regime, LIBRARIES)


def library_runs_cell(library: str, dataset: str, regime: str) -> bool:
    """Whether ``library`` participates in the (dataset, regime) cell."""
    if library not in libraries_for(regime):
        return False
    cells = LIBRARY_CELLS.get(library)
    return cells is None or (dataset, regime) in cells


def venv_python(library: str) -> str:
    """Path of the venv python that owns ``library`` (see setup_envs.sh)."""
    if library.startswith("falcata"):
        env = "env-falcata"
    elif library == "lightgbm-ocl":
        env = "env-lightgbm-ocl"
    else:
        env = "env-competitors"
    override = os.environ.get(f"FALCATA_BENCH_PY_{env.replace('-', '_').upper()}")
    return override or os.path.join(ROOT, env, "bin", "python")


def regimes_for(dataset: str):
    """Regimes to benchmark for a dataset."""
    if dataset == "numerai":
        return ["numerai", "numerai-deep", "numerai-leaf"]
    return ["shallow", "deep"]


def dataset_ready(dataset: str) -> bool:
    """Whether the preprocessed cache for ``dataset`` exists."""
    marker = "meta.json" if dataset == "numerai" else "y_test.npy"
    return os.path.exists(os.path.join(CACHE_DIR, dataset, marker))
