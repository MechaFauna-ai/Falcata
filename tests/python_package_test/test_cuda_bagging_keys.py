# coding: utf-8
"""Bagged CUDA training must train the same model with or without the bagging fast paths.

cuda_plan keys bagged_code_view (a bagged tree's 4-bit fill still writes the pair-joint code view), oob_node_table
(the out-of-bag score pass reads one packed node table) and bagging_fused_scan (the device bag in three kernels without
a flag array) change only how the same values are computed, so the trained model must be byte-identical with each
key off. The sweep covers 4-bit columns with and without NaN, 8-bit columns, a column count that leaves an odd column
per byte slot, feature_fraction (the per-tree views), the regression, binary and multiclass objectives, small and
large bags, and a row count that is not a multiple of the scan block. These small datasets do not take the code view
(it engages with the column-major 4-bit store of wide data, as on the Numerai benchmark, where key on/off was compared
by md5), so bagged_code_view is a no-op here.
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

KEYS = ("bagged_code_view", "oob_node_table", "bagging_fused_scan")


def _model_md5(params, X, y, rounds):
    ds = flc.Dataset(X, label=y, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(kind, rows, cols, seed):
    rng = np.random.default_rng(seed)
    if kind == "fewbin":
        X = rng.integers(0, 5, size=(rows, cols)).astype(np.float32)
    elif kind == "fewbin-nan":
        X = rng.integers(0, 5, size=(rows, cols)).astype(np.float32)
        X[rng.random(X.shape) < 0.08] = np.nan
    else:
        X = rng.standard_normal((rows, cols)).astype(np.float32)
    return X


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "kind, cols, objective, frac, feature_fraction, leaves, depth",
    [
        ("fewbin", 7, "regression", 0.3, 1.0, 31, 5),
        ("fewbin", 120, "regression", 0.6, 0.2, 255, 8),
        ("fewbin-nan", 41, "binary", 0.8, 0.5, 63, 6),
        ("dense", 9, "binary", 0.4, 0.7, 63, 6),
        ("fewbin", 5, "multiclass", 0.5, 1.0, 15, 4),
    ],
)
def test_bagging_keys_are_bit_identical_cuda(kind, cols, objective, frac, feature_fraction, leaves, depth):
    rows = 20011
    X = _data(kind, rows, cols, seed=3)
    rng = np.random.default_rng(4)
    if objective == "multiclass":
        y = np.nan_to_num(X[:, 0]) % 3
    elif objective == "binary":
        y = (np.nan_to_num(X[:, 0]) + rng.normal(size=rows) > np.nanmean(X[:, 0])).astype(float)
    else:
        y = np.nan_to_num(X).sum(axis=1) + rng.normal(size=rows)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 5,
        "bagging_fraction": frac,
        "bagging_freq": 1,
        "feature_fraction": feature_fraction,
        "use_quantized_grad": True,
        "verbosity": -1,
        "seed": 7,
    }
    if objective == "multiclass":
        params["num_class"] = 3
    reference = _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=20)
    for key in KEYS:
        assert _model_md5({**params, "cuda_plan": f"auto,{key}:off"}, X, y, rounds=20) == reference, key
    off = ",".join(f"{key}:off" for key in KEYS)
    assert _model_md5({**params, "cuda_plan": f"auto,{off}"}, X, y, rounds=20) == reference
