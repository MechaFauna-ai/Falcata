# coding: utf-8
"""Bit-identity regression tests for cuda_plan key nonquant_fixed_prescale.

Verifies exact model MD5 parity between nonquant_fixed_prescale:on (default) and
nonquant_fixed_prescale:off (baseline per-row scalbnf + emulated carry atomics)
across dense 8-bit device_tree, 4-bit USE_REG_BINS (numerai-like), full-view
4-bit, bagging + feature_fraction, multiclass, host level loop (device_tree:off),
selective (max_depth=-1), and nonquant_hist_cost auto-selection.
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
    text = booster.model_to_string().split("\nparameters:")[0]
    lines = [line for line in text.splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _make_data(objective, rows, cols, seed=17, discrete_5bin=False):
    rng = np.random.default_rng(seed)
    if discrete_5bin:
        X = rng.integers(0, 5, size=(rows, cols), dtype=np.int8).astype(np.float32)
        z = 0.4 * (X[:, 0] - 2.0) - 0.3 * (X[:, 1] - 2.0) + 0.2 * (X[:, 2] - 2.0) * (X[:, 3] - 2.0)
        z = z + 0.5 * rng.standard_normal(rows)
    else:
        X = rng.standard_normal((rows, cols)).astype(np.float32)
        X[rng.random(X.shape) < 0.03] = np.nan
        Xn = np.nan_to_num(X)
        z = Xn[:, 0] + 0.5 * Xn[:, 1] * Xn[:, 2] - 0.3 * Xn[:, 3] + 0.3 * rng.standard_normal(rows)
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > np.median(z)).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("name", "objective", "shape", "base_plan", "extra"),
    [
        (
            "dense-8bit-device-tree",
            "regression",
            (40000, 28, False),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": 6, "max_bin": 255},
        ),
        (
            "4bit-reg-bins-numerai",
            "regression",
            (40000, 120, True),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 127, "max_depth": 7, "max_bin": 15, "feature_fraction": 0.3},
        ),
        (
            "4bit-full-view",
            "binary",
            (40000, 20, False),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": 6, "max_bin": 15},
        ),
        (
            "bagging-feature-fraction",
            "binary",
            (40000, 80, False),
            "auto,nonquant_deterministic:on",
            {
                "num_leaves": 127,
                "max_depth": 7,
                "max_bin": 255,
                "bagging_fraction": 0.7,
                "bagging_freq": 1,
                "feature_fraction": 0.4,
            },
        ),
        (
            "multiclass-device-tree",
            "multiclass",
            (30000, 24, False),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": 6, "num_class": 4, "max_bin": 255},
        ),
        (
            "host-level-device-tree-off",
            "regression",
            (30000, 100, False),
            "auto,device_tree:off,nonquant_deterministic:on",
            {"num_leaves": 127, "max_depth": 7, "max_bin": 255, "feature_fraction": 0.4},
        ),
        (
            "selective-unbounded-depth",
            "regression",
            (30000, 30, False),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": -1, "max_bin": 255},
        ),
    ],
)
def test_nonquant_fixed_prescale_matches_baseline_cuda(name, objective, shape, base_plan, extra):
    rows, cols, discrete_5bin = shape
    X, y = _make_data(objective, rows, cols, seed=42, discrete_5bin=discrete_5bin)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": "none",
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "seed": 7,
        **extra,
    }
    on_md5 = _model_md5({**params, "cuda_plan": f"{base_plan},nonquant_fixed_prescale:on"}, X, y, 12)
    off_md5 = _model_md5({**params, "cuda_plan": f"{base_plan},nonquant_fixed_prescale:off"}, X, y, 12)
    assert on_md5 == off_md5, f"{name}: prescale:on ({on_md5}) != prescale:off ({off_md5})"


@_REQUIRES_CUDA
def test_nonquant_fixed_prescale_graph_and_hist_cost_cuda():
    # Shape where nonquant_hist_cost auto-selects fixed-point without explicit nonquant_deterministic:on
    X, y = _make_data("regression", 80000, 12, seed=31)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "none",
        "num_leaves": 63,
        "max_depth": 6,
        "max_bin": 15,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 7,
    }
    auto_on = _model_md5({**params, "cuda_plan": "auto,nonquant_fixed_prescale:on"}, X, y, 10)
    auto_off = _model_md5({**params, "cuda_plan": "auto,nonquant_fixed_prescale:off"}, X, y, 10)
    no_graph = _model_md5({**params, "cuda_plan": "auto,device_level_graph:off,nonquant_fixed_prescale:on"}, X, y, 10)
    assert auto_on == auto_off == no_graph, (auto_on, auto_off, no_graph)
