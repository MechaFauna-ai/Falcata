"""Monotone constraints must hold with path_smooth.

A split whose children break the monotone order gets a sentinel gain. That sentinel was 0, but with path_smooth the
parent's gain at its smoothed output can be negative, so a gain of 0 still beat min_gain_shift and the violating split
was kept (CPU and CUDA learners alike).
"""

import os

import numpy as np
import pytest

import falcata as flc

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata (TASK=cuda)"
        ),
    ),
]


def _data(seed):
    r = np.random.default_rng(seed)
    n = 600
    X = r.standard_normal((n, 3))
    p = 1.0 / np.arange(1, 301) ** 1.1
    X[:, 0] = r.choice(300, size=n, p=p / p.sum())
    X[:, 1:][r.random((n, 2)) < 0.7] = 0.0
    y = np.sin(X[:, 0] * 1.7) + X[:, 1] - 0.5 * X[:, 2] + 0.3 * r.standard_normal(n)
    return X, y


def _worst_violation(booster, X, feature, direction):
    grid = np.linspace(-3, 3, 41)
    base = X[:60]
    Xg = np.repeat(base, len(grid), axis=0)
    Xg[:, feature] = np.tile(grid, len(base))
    pred = booster.predict(Xg, raw_score=True).reshape(len(base), len(grid))
    return float((np.diff(pred, axis=1) * direction).min())


@pytest.mark.parametrize("device", DEVICES)
def test_monotone_constraints_hold_with_path_smooth(device):
    # seed 22: a leaf whose smoothed output gives it a negative gain, then a violating split on f2 (found by fuzzing)
    X, y = _data(22)
    params = {
        "objective": "regression",
        "device_type": device,
        "monotone_constraints": [0, 1, -1],
        "path_smooth": 1.0,
        "lambda_l1": 1.0,
        "lambda_l2": 10.0,
        "num_leaves": 63,
        "max_bin": 3,
        "min_data_in_leaf": 20,
        "deterministic": True,
        "num_threads": 2,
        "verbose": -1,
    }
    if device == "cuda":
        params["quant_mode"] = "none"
    booster = flc.train(params, flc.Dataset(X, y, categorical_feature=[0], params=params), 20)
    assert _worst_violation(booster, X, 1, 1) >= 0.0
    assert _worst_violation(booster, X, 2, -1) >= 0.0
