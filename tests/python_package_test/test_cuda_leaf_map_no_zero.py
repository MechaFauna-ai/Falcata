# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key leaf_map_no_zero.

The key skips the tree start's zero store of the row -> leaf map (and the bagged subset's scatter of it) where the
tree's flow writes every in-bag row's leaf before the map is read (the level prefix with the batched apply; a stump
writes leaf 0's window at the tree end; quantized training only). The sweep covers both quantized modes, bagging,
missing values, multiclass, a leaf budget below 2^max_depth (selective growth, ungated), stumps (a root no split can
satisfy), trees that alternate between stumps and splits, and GOSS sampling (the bagged subset path).
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
    ("objective", "leaves", "depth", "quant", "bagging", "min_data", "min_gain"),
    [
        ("regression", 255, 8, "stochastic", 1.0, 10, 0.0),
        ("binary", 511, 9, "fixedpoint", 0.6, 10, 0.0),
        ("multiclass", 127, 7, "fixedpoint", 1.0, 10, 0.0),
        ("regression", 255, 8, "fixedpoint", 0.5, 10, 0.0),
        ("regression", 100, 8, "stochastic", 1.0, 10, 0.0),
        ("regression", 63, 10, "stochastic", 0.7, 10, 0.0),
        ("binary", 255, 8, "stochastic", 1.0, 40000, 0.0),
        ("regression", 255, 8, "stochastic", 0.5, 20, 30.0),
    ],
)
def test_leaf_map_no_zero_is_bit_identical_cuda(objective, leaves, depth, quant, bagging, min_data, min_gain):
    rows = 30011
    rng = np.random.default_rng(29)
    X = rng.standard_normal((rows, 20)).astype(np.float32)
    X[:, 16:] = rng.integers(0, 6, size=(rows, 4))
    X[rng.random(X.shape) < 0.05] = np.nan
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
        "min_data_in_leaf": min_data,
        "min_gain_to_split": min_gain,
        "learning_rate": 0.1,
        "verbose": -1,
        "quant_mode": quant,
        "seed": 3,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 15)
    off = _model_md5({**params, "cuda_plan": "auto,leaf_map_no_zero:off"}, X, y, 15)
    assert on == off


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
def test_leaf_map_no_zero_goss_is_bit_identical_cuda(quant):
    rows = 40009
    rng = np.random.default_rng(23)
    X = rng.standard_normal((rows, 16)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    y = (z > 0).astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": "binary",
        "num_leaves": 255,
        "max_depth": 8,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "quant_mode": quant,
        "data_sample_strategy": "goss",
        "top_rate": 0.2,
        "other_rate": 0.1,
        "seed": 4,
    }
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 20)
    off = _model_md5({**params, "cuda_plan": "auto,leaf_map_no_zero:off"}, X, y, 20)
    assert on == off
