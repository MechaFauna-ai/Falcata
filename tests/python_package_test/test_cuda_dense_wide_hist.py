# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key dense_wide_hist.

The key divides the feature partitions of quantized dense 8/16-bit row data against a 2x shared histogram table
(12288 bins instead of 6144) when that gives fewer partitions, and instantiates the quantized dense construct kernels
with the 2x table. The histogram's global layout and its integer sums are the same, so the model is byte-identical
with the key off. The sweep covers shapes where the key engages (28 x 255 bins: one partition instead of two; 60 x
255: two instead of three; 100 x 63: one instead of two), shapes where the gate declines (20 x 255: one partition
already; 4-bit packable columns), feature sampling (the compact view derives its partitions from the source partitions), bagging (the same
kernels over the bag), the multiclass objective, deep trees and both quantized modes. (Quantized training on CUDA
rejects columns of more than 256 bins, so a large-bin partition never exists under the key's gate.)
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
    ("cols", "max_bin", "objective", "leaves", "depth", "feature_fraction", "bagging", "quant"),
    [
        (28, 255, "binary", 1023, 10, 1.0, 1.0, "stochastic"),
        (60, 255, "regression", 255, 8, 1.0, 1.0, "fixedpoint"),
        (100, 63, "regression", 127, 7, 1.0, 1.0, "stochastic"),
        (28, 255, "binary", 255, 8, 0.5, 0.7, "stochastic"),
        (40, 255, "multiclass", 63, 6, 1.0, 1.0, "fixedpoint"),
        (20, 255, "regression", 255, 8, 1.0, 1.0, "stochastic"),
        (300, 31, "regression", 255, 8, 1.0, 1.0, "stochastic"),
        (240, 31, "regression", 255, 8, 1.0, 1.0, "fixedpoint"),
        (150, 63, "regression", 255, 8, 1.0, 0.6, "stochastic"),
        (300, 15, "regression", 255, 8, 1.0, 0.6, "stochastic"),
    ],
)
def test_dense_wide_hist_is_bit_identical_cuda(
    cols, max_bin, objective, leaves, depth, feature_fraction, bagging, quant
):
    rows = 40009
    rng = np.random.default_rng(3)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    if max_bin <= 15:
        X = np.floor(X * 2.0).astype(np.float32)  # few distinct values: 4-bit packable columns
    X[rng.random(X.shape) < 0.02] = np.nan
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
        "max_bin": max_bin,
        "feature_fraction": feature_fraction,
        "quant_mode": quant,
        "verbose": -1,
        "seed": 5,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 10)
    off = _model_md5({**params, "cuda_plan": "auto,dense_wide_hist:off"}, X, y, 10)
    assert on == off
