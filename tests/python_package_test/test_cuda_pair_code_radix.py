# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key pair_code_radix.

A tree whose pair-joint code view needs more than 5 bits per joint cell (pair_code_wide) packs six cells per 32-bit
word in mixed radix (three digits per 16-bit half, each slot's own joint cell count as its radix) instead of five
6-bit codes, where every half's three cell counts multiply to at most 2^16 and the row takes fewer words. The fill
and the code-word construct must agree on every cell: the histograms, and so the model, are byte-identical with the
key off. The sweep covers 6 x 6 cells (5 values + the missing bin, and 6 values), mixed widths (3 to 7 values, so some
trees' halves exceed 2^16 and keep the 6-bit code), shallow and deep trees, both quantization modes, bagging, and a
row of exactly one word group past a multiple of six slots.
"""

import hashlib
import os
import subprocess
import sys

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
    ("values", "nan_frac", "num_features", "feature_fraction", "leaves", "depth", "quant", "bagging"),
    [
        ((5, 6), 0.08, 1200, 0.3, 63, 6, "stochastic", 1.0),  # 5 values + NaN: 6 x 6 cells
        ((6, 7), 0.0, 1200, 0.3, 63, 6, "stochastic", 1.0),  # 6 values, 6 x 6 cells
        ((3, 8), 0.05, 1200, 0.3, 255, 10, "fixedpoint", 1.0),  # 3..7 values (+ NaN): mixed radices
        ((5, 6), 0.08, 1210, 0.3, 255, 10, "stochastic", 0.6),  # bagged, deep
        ((5, 6), 0.08, 1180, 0.25, 127, 8, "fixedpoint", 1.0),
    ],
)
def test_pair_code_radix_is_bit_identical_cuda(
    values, nan_frac, num_features, feature_fraction, leaves, depth, quant, bagging
):
    rows = 8000
    rng = np.random.default_rng(23)
    levels = rng.integers(values[0], values[1], size=num_features)
    X = (rng.random((rows, num_features)) * levels).astype(np.int64).astype(np.float64)
    if nan_frac > 0:
        X[rng.random(X.shape) < nan_frac] = np.nan
    w = rng.standard_normal(40)
    y = np.nan_to_num(X[:, :40]) @ w + rng.standard_normal(rows)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": leaves,
        "max_depth": depth,
        "max_bin": 15,
        "feature_fraction": feature_fraction,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "quant_mode": quant,
        "verbose": -1,
        "seed": 42,
        "metric": "None",
    }
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 15)
    off = _model_md5({**params, "cuda_plan": "auto,pair_code_radix:off"}, X, y, 15)
    assert on == off


def _probe_pair_code5_flat_table_diag(plan):
    rows, num_features = 40009, 1200
    rng = np.random.default_rng(29)
    levels = rng.integers(5, 6, size=num_features)
    X = (rng.random((rows, num_features)) * levels).astype(np.int64).astype(np.float64)
    w = rng.standard_normal(40)
    y = X[:, :40] @ w + rng.standard_normal(rows)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 127,
        "max_depth": 8,
        "max_bin": 15,
        "feature_fraction": 0.25,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbosity": 1,
        "seed": 42,
        "metric": "None",
        "cuda_plan": plan,
    }
    ds = flc.Dataset(X, label=y, params=params)
    flc.train(params, ds, num_boost_round=3)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("values", "nan_frac", "quant", "bagging"),
    [
        ((5, 6), 0.0, "stochastic", 1.0),  # 5-bit code5
        ((5, 6), 0.08, "fixedpoint", 1.0),  # mixed-radix code
        ((5, 6), 0.08, "stochastic", 0.6),  # bagged mixed-radix code
    ],
)
def test_pair_code5_flat_table_is_bit_identical_cuda(values, nan_frac, quant, bagging):
    """pair_code5_flat_table must match the baseline epilogue bit-for-bit."""
    rows, num_features = 40009, 1200
    rng = np.random.default_rng(29)
    levels = rng.integers(values[0], values[1], size=num_features)
    X = (rng.random((rows, num_features)) * levels).astype(np.int64).astype(np.float64)
    if nan_frac > 0:
        X[rng.random(X.shape) < nan_frac] = np.nan
    w = rng.standard_normal(40)
    y = np.nan_to_num(X[:, :40]) @ w + rng.standard_normal(rows)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 127,
        "max_depth": 8,
        "max_bin": 15,
        "feature_fraction": 0.25,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "quant_mode": quant,
        "verbose": -1,
        "seed": 42,
        "metric": "None",
    }
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": "auto,pair_code5_flat_table:on"}, X, y, 10)
    off = _model_md5({**params, "cuda_plan": "auto,pair_code5_flat_table:off"}, X, y, 10)
    assert on == off


@_REQUIRES_CUDA
def test_pair_code5_flat_table_engages_diag_cuda():
    """FALCATA_DEBUG=diag must confirm pair_code5_flat_table engages when on and not when off."""
    child = "import runpy,sys; runpy.run_path(sys.argv[1])['_probe_pair_code5_flat_table_diag'](sys.argv[2])"
    for plan, expected in [("auto,pair_code5_flat_table:on", True), ("auto,pair_code5_flat_table:off", False)]:
        res = subprocess.run(
            [sys.executable, "-c", child, __file__, plan],
            env=dict(os.environ, FALCATA_DEBUG="diag"),
            capture_output=True,
            text=True,
            check=False,
        )
        out = res.stdout + "\n" + res.stderr
        assert res.returncode == 0, out
        assert ("[flat_table]" in out) is expected, out
