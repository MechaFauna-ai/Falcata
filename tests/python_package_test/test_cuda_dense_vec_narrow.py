# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key dense_vec_narrow.

The key pads single-partition narrow 8-bit quantized dense row data (<= 32 columns, <= 12288 bins, not 4-bit packed)
to a power-of-two 4-byte multiple (4, 8, 16, or 32 bytes) and runs a warp-cooperative sub-warp construct kernel
(1/2/4/8 threads per row, aligned uint32 loads, dynamically sized shared histogram). Integer sums over the same rows
are order-invariant, so the model is byte-identical with the key off.
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
    "case",
    [
        (1, "binary", 63, 6, 1.0, 1.0, "stochastic", ""),
        (4, "binary", 63, 6, 1.0, 1.0, "stochastic", ""),
        (5, "regression", 63, 6, 1.0, 1.0, "fixedpoint", ""),
        (13, "binary", 255, 8, 1.0, 1.0, "stochastic", ""),
        (16, "regression", 127, 7, 1.0, 0.7, "fixedpoint", ""),
        (17, "binary", 127, 7, 1.0, 1.0, "stochastic", ""),
        (28, "binary", 255, 8, 1.0, 1.0, "stochastic", ""),
        (29, "binary", 255, 8, 1.0, 1.0, "fixedpoint", ""),
        (32, "multiclass", 63, 6, 1.0, 1.0, "fixedpoint", ""),
        (29, "binary", 63, 6, 1.0, 1.0, "stochastic", ",device_tree:off"),
        (29, "binary", 63, 6, 1.0, 1.0, "fixedpoint", ",device_levels:off"),
        (29, "binary", 63, 6, 0.3, 1.0, "stochastic", ""),
        (29, "binary", 63, 6, 0.7, 0.7, "stochastic", ""),
        (29, "binary", 63, -1, 1.0, 1.0, "stochastic", ""),
    ],
)
def test_dense_vec_narrow_is_bit_identical_cuda(monkeypatch, case):
    """Verify dense_vec_narrow is bit-identical to key-off across narrow widths and flows."""
    cols, objective, leaves, depth, feature_fraction, bagging, quant, extra_plan = case
    monkeypatch.setenv("FALCATA_TEST_FORCE_DENSE_VEC_NARROW", "1")
    rows = 20011
    rng = np.random.default_rng(7)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.02] = np.nan
    z = (
        np.nan_to_num(X[:, 0])
        + (0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, -1]) if cols > 1 else 0.0)
        + 0.3 * rng.normal(size=rows)
    )
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
        "feature_fraction": feature_fraction,
        "quant_mode": quant,
        "verbose": -1,
        "seed": 11,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": f"auto{extra_plan},dense_vec_narrow:on"}, X, y, 10)
    off = _model_md5({**params, "cuda_plan": f"auto{extra_plan},dense_vec_narrow:off"}, X, y, 10)
    assert on == off
