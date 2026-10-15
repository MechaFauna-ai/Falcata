# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key find_group_direction.

The key orders the level warp finder's few-position tasks by scan direction (a numerical column with missing values
has a reverse and a forward task), so each warp of eight 4-lane groups runs one direction. Every item still writes at
its own task index, so the model is byte-identical with the key off. The sweep covers columns with missing values only
among few-bin columns, a mix with wide (many-bin) columns, feature sampling (the tree's used tasks), and the
multiclass objective.
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
    ("objective", "wide_cols", "feature_fraction", "quant"),
    [
        ("regression", 0, 1.0, "stochastic"),
        ("regression", 0, 0.3, "stochastic"),
        ("binary", 4, 1.0, "fixedpoint"),
        ("multiclass", 2, 0.5, "fixedpoint"),
    ],
)
def test_find_group_direction_is_bit_identical_cuda(objective, wide_cols, feature_fraction, quant):
    rows = 20011
    rng = np.random.default_rng(3)
    X = rng.integers(0, 5, size=(rows, 40)).astype(np.float32)
    X[:, ::2][rng.random((rows, 20)) < 0.08] = np.nan  # half of the few-value columns have missing values
    if wide_cols:
        X = np.hstack([X, rng.standard_normal((rows, wide_cols)).astype(np.float32)])
    z = np.nan_to_num(X[:, 0], nan=2.0) + 0.5 * np.nan_to_num(X[:, 1]) - np.nan_to_num(X[:, 2], nan=-1.0)
    z = z + 0.5 * rng.normal(size=rows)
    if objective == "multiclass":
        y = np.digitize(z, np.quantile(z, [0.33, 0.66])).astype(np.float32)
    elif objective == "binary":
        y = (z > np.median(z)).astype(np.float32)
    else:
        y = z.astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 255,
        "max_depth": 8,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "feature_fraction": feature_fraction,
        "quant_mode": quant,
        "verbose": -1,
        "seed": 5,
    }
    if objective == "multiclass":
        params["num_class"] = 3
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 12)
    off = _model_md5({**params, "cuda_plan": "auto,find_group_direction:off"}, X, y, 12)
    assert on == off
