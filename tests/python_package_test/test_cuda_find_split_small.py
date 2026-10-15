# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key find_split_small.

The key runs a wide split-finder task (more than 8 scan positions) on the four warps of one block when the level's
launch is small, each warp scanning 64 positions with the earlier positions' integer sum as its carry-in, and merges
the four partial winners in the block. The sweep covers 255-bin columns (wide tasks only), a mix of binary and
numeric columns (narrow units padded next to the split wide tasks), missing values (the shifted forward scan whose
bin 0 is rebuilt from the leaf total), both quantization modes, bagging, and the multiclass objective.
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


def _model_md5(params, X, y, rounds):
    ds = flc.Dataset(X, label=y, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "leaves", "depth", "binary_cols", "missing", "bagging", "quant"),
    [
        ("regression", 255, 8, 0, 0.0, 1.0, "stochastic"),
        ("binary", 511, 9, 20, 0.05, 1.0, "fixedpoint"),
        ("multiclass", 127, 7, 40, 0.0, 1.0, "fixedpoint"),
        ("regression", 255, 8, 8, 0.1, 0.5, "stochastic"),
        ("binary", 63, 6, 0, 0.0, 0.7, "stochastic"),
    ],
)
def test_find_split_small_is_bit_identical_cuda(objective, leaves, depth, binary_cols, missing, bagging, quant):
    rows = 40009
    rng = np.random.default_rng(7)
    X = rng.standard_normal((rows, 12)).astype(np.float32)
    if binary_cols > 0:
        X = np.hstack([X, (rng.random((rows, binary_cols)) < 0.3).astype(np.float32)])
    if missing > 0.0:
        X[rng.random(X.shape) < missing] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > 0).astype(np.float32)
    else:
        y = z.astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "max_bin": 255,
        "quant_mode": quant,
        "verbose": -1,
        "seed": 3,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 12)
    off = _model_md5({**params, "cuda_plan": "auto,find_split_small:off"}, X, y, 12)
    assert on == off
