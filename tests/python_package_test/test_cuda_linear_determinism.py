# coding: utf-8
"""Linear-tree leaf fits on CUDA repeat bit for bit.

The per-leaf normal equations (X^T H X, X^T g) are a fixed-order reduction over each leaf's window of the partition's
index list (CalcLinearGramChunkKernel / CalcLinearGramFinalKernel): chunk sums in a fixed thread order, then the chunk
sums in chunk order. The former kernel added every row's terms with double atomicAdd, so leaf_const / leaf_coeff
moved at the 1e-12 level from run to run (runs/device_default/LINEAR_TREE_DETERMINISM.md). Two trainings of the same
shape must give the same model text, coefficients included, on every flow that grows linear trees: the selective
one-sync flow (budget-limited), the complete device tree (depth-limited), bagging, categorical splits, L1, path
smoothing, stumps, non-quantized training and leaves wide enough for the multi-pass entry loop. CPU parity stays inexact (another summation order).
"""

import hashlib
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _dense(rows=60000, cols=20, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.02] = np.nan
    Z = np.nan_to_num(X)
    y = Z[:, 0] * 1.5 + np.where(Z[:, 1] > 0, Z[:, 2], -Z[:, 3]) + 0.3 * rng.standard_normal(rows)
    return X, y.astype(np.float32), []


def _categorical(rows=60000, cols=12, seed=6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[:, 0] = rng.integers(0, 24, rows)
    X[:, 1] = rng.integers(0, 6, rows)
    Z = np.nan_to_num(X)
    y = np.sin(X[:, 0]) * Z[:, 2] + (X[:, 1] == 3) * Z[:, 3] + 0.2 * rng.standard_normal(rows)
    return X, y.astype(np.float32), [0, 1]


def _chain(rows=120000, cols=80, seed=7):
    # a decision list (the first feature above its 90th percentile decides): leaf-wise growth peels one feature per
    # level off the rest leaf, so the deep leaves carry 60+ distinct branch features (more than 2048 gram entries)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    hit = X > 1.2816
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), cols)
    y = first.astype(np.float32) + 0.5 * X[:, 0] + 0.05 * rng.standard_normal(rows).astype(np.float32)
    return X, y.astype(np.float32), []


_DATA = {"dense": _dense, "categorical": _categorical, "chain": _chain}

# (data, params, rounds)
_SHAPES = {
    "selective-31": ("dense", {"num_leaves": 31, "max_depth": -1}, 12),
    "device-63-6": ("dense", {"num_leaves": 63, "max_depth": 6}, 12),
    "bagged-63-6": ("dense", {"num_leaves": 63, "max_depth": 6, "bagging_fraction": 0.7, "bagging_freq": 1}, 12),
    "bagged-31": ("dense", {"num_leaves": 31, "max_depth": -1, "bagging_fraction": 0.7, "bagging_freq": 1}, 12),
    "nonquant-63-6": ("dense", {"num_leaves": 63, "max_depth": 6, "quant_mode": "none"}, 12),
    "categorical-31": ("categorical", {"num_leaves": 31, "max_depth": -1}, 12),
    "categorical-63-6": ("categorical", {"num_leaves": 63, "max_depth": 6}, 12),
    "l1-63-6": ("dense", {"num_leaves": 63, "max_depth": 6, "lambda_l1": 1.0}, 12),
    "smooth-63-6": ("dense", {"num_leaves": 63, "max_depth": 6, "path_smooth": 5.0}, 12),
    "stump": ("dense", {"num_leaves": 2, "max_depth": -1}, 12),
    "wide-chain": ("chain", {"num_leaves": 90, "max_depth": -1, "min_data_in_leaf": 20}, 4),
}


def _train(name):
    data, extra, rounds = _SHAPES[name]
    X, y, cat = _DATA[data]()
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "linear_tree": True,
        "learning_rate": 0.1,
        "seed": 3,
        "verbose": -1,
        **extra,
    }
    ds = flc.Dataset(X, label=y, categorical_feature=cat, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    return booster.model_to_string().split("\nparameters:")[0], booster, X


@_REQUIRES_CUDA
@pytest.mark.parametrize("name", list(_SHAPES))
def test_linear_tree_repeats_cuda(name):
    text_a, booster, X = _train(name)
    text_b, _, _ = _train(name)
    assert "leaf_coeff=" in text_a  # the shape fits linear leaves at all
    if text_a != text_b:
        diff = [(a, b) for a, b in zip(text_a.splitlines(), text_b.splitlines(), strict=False) if a != b]
        pytest.fail(f"{len(diff)} model lines differ, first: {diff[0][0][:120]} vs {diff[0][1][:120]}")
    assert hashlib.md5(text_a.encode()).hexdigest() == hashlib.md5(text_b.encode()).hexdigest()
    # the device scores the training saw are the model's own predictions
    pred = booster.predict(X)
    assert np.all(np.isfinite(pred))


@_REQUIRES_CUDA
def test_linear_tree_wide_leaves_reach_multipass_cuda():
    # the chain shape must really put more than 2048 gram entries (nf >= 63) on some leaf, or the multi-pass entry
    # loop of the reduction kernel goes untested
    text, _, _ = _train("wide-chain")
    widest = 0
    for line in text.splitlines():
        if line.startswith("num_features="):
            widest = max(widest, *(int(v) for v in line.split("=")[1].split()))
    assert widest >= 63, widest
