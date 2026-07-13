"""Shared configuration for the ExaBoost GPU benchmark suite.

All artifacts (raw downloads, preprocessed caches, results, report) live under
a single workspace directory, resolved from the ``EXABOOST_BENCH_ROOT``
environment variable and defaulting to ``benchmarks/workspace`` inside the
repository checkout.
"""

import os

ROOT = os.environ.get(
    "EXABOOST_BENCH_ROOT",
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
}

#: hyperparameter regimes, aligned across libraries (gbm-bench convention);
#: LightGBM-family maps depth/leaves to (max_depth, num_leaves), XGBoost uses
#: max_depth (native depth-wise growth), CatBoost uses symmetric depth.
REGIMES = {
    # l2=1 aligns L2 leaf regularization across engines (defaults differ:
    # xgboost lambda=1, lightgbm lambda_l2=0, catboost l2_leaf_reg=3); at
    # lr 0.1 an unregularized leaf-wise model degenerates on imbalanced data
    "shallow": {
        "rounds": 500,
        "lr": 0.1,
        "depth": 6,
        "leaves": 63,
        "l2": 1.0,
        "eval_every": 25,
    },
    "deep": {
        "rounds": 500,
        "lr": 0.1,
        "depth": 10,
        "leaves": 1023,
        "l2": 1.0,
        "eval_every": 25,
    },
    # official Numerai example-model parameters
    "numerai": {
        "rounds": 2000,
        "lr": 0.01,
        "depth": 5,
        "leaves": 32,
        "colsample": 0.1,
        "eval_every": 250,
    },
    # tiny config for smoke-testing the library/GPU integration
    "smoke": {"rounds": 10, "lr": 0.1, "depth": 6, "leaves": 63, "eval_every": 5},
}

#: exaboost/lightgbm both install as package "lightgbm", hence two venvs
LIBRARIES = [
    "exaboost",
    "exaboost-quant",
    "lightgbm",
    "lightgbm-quant",
    "xgboost",
    "catboost",
]


def venv_python(library: str) -> str:
    """Path of the venv python that owns ``library`` (see setup_envs.sh)."""
    env = "env-exaboost" if library.startswith("exaboost") else "env-competitors"
    override = os.environ.get(f"EXABOOST_BENCH_PY_{env.replace('-', '_').upper()}")
    return override or os.path.join(ROOT, env, "bin", "python")


def regimes_for(dataset: str):
    """Regimes to benchmark for a dataset."""
    return ["numerai"] if dataset == "numerai" else ["shallow", "deep"]


def dataset_ready(dataset: str) -> bool:
    """Whether the preprocessed cache for ``dataset`` exists."""
    marker = "meta.json" if dataset == "numerai" else "y_test.npy"
    return os.path.exists(os.path.join(CACHE_DIR, dataset, marker))
