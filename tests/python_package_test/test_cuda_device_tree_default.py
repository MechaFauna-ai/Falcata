# coding: utf-8
"""CUDA training must train the same model with the default plan, device_tree:off and device_levels:off.

The default plan runs the complete device tree (cuda_plan key device_tree) where its depth bound holds (positive
max_depth <= 12 and 2^max_depth <= num_leaves + 1) and leaves every other shape to the device_levels prefix and the
host flow; device_tree:off restores the prefix flow, device_levels:off the host flow. The sweep covers eligible shapes
(a full and a one-short leaf budget, regression, binary and multiclass, bagging, feature_fraction, missing values) and
shapes outside the bound, whose models must not change either.
"""

import hashlib
import os
import re
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
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(objective, rows=6000, cols=12, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    if objective == "multiclass":
        return X, np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    if objective == "binary":
        return X, (z > 0).astype(np.float32)
    return X, z.astype(np.float32)


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize(
    ("leaves", "depth", "extra"),
    [
        (64, 6, {}),  # eligible: 2^6 <= 64
        (63, 6, {"bagging_fraction": 0.6, "bagging_freq": 1}),  # eligible: 2^6 <= 63 + 1
        (31, 5, {"feature_fraction": 0.5}),  # eligible, feature-sampled
        (40, 6, {}),  # outside the bound (a binding leaf budget): selective growth on the host
        (31, -1, {}),  # unlimited depth: the host flow
    ],
)
def test_device_tree_default_same_model_cuda(objective, leaves, depth, extra):
    X, y = _data(objective)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 5,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 3,
        **extra,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    host = _model_md5({**params, "cuda_plan": "auto,device_levels:off"}, X, y, 12)
    assert _model_md5({**params, "cuda_plan": "auto,device_tree:off"}, X, y, 12) == host
    assert _model_md5({**params, "cuda_plan": "auto"}, X, y, 12) == host


_ENGAGE = (
    "import sys, numpy as np, falcata as flc\n"
    "rng = np.random.default_rng(1)\n"
    "X = rng.standard_normal((20000, 10)).astype(np.float32)\n"
    "y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)\n"
    "p = {'device_type': 'cuda', 'objective': 'binary', 'verbose': -1, 'quant_mode': 'stochastic',\n"
    "     'use_quantized_grad': True, 'cuda_plan': sys.argv[1],\n"
    "     'num_leaves': int(sys.argv[2]), 'max_depth': int(sys.argv[3])}\n"
    "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=3)\n"
)


def _diag(plan, leaves=64, depth=6):
    env = dict(os.environ, FALCATA_DEBUG="diag")
    args = [sys.executable, "-c", _ENGAGE, plan, str(leaves), str(depth)]
    return subprocess.run(args, env=env, capture_output=True, text=True, check=True).stderr


@_REQUIRES_CUDA
def test_device_tree_default_engages_cuda():
    # the default plan runs the whole depth-bounded tree on the device (diag names its 6 device-driven levels and one
    # complete tree per round); device_tree:off restores the prefix's 5 levels, device_levels:off the host flow
    on = _diag("auto")
    assert "device_levels: 6 device-driven levels per tree" in on
    assert on.count("device_tree: complete depth=6 ") == 3
    assert not re.search(r"device_tree: fallback \((?!plan_off\))", on)
    assert "host fallback" not in on
    prefix = _diag("auto,device_tree:off")
    assert "device_levels: 5 device-driven levels per tree" in prefix
    assert "device_tree: complete" not in prefix
    assert "device_tree: fallback (plan_off) route=prefix" in prefix
    host = _diag("auto,device_levels:off")
    assert "device-driven levels per tree" not in host
    assert "device_tree: complete" not in host
    assert "device_tree: fallback (plan_off) route=host_two_sync" in host


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("leaves", "depth", "flow", "reason"),
    [
        (8192, 13, "prefix", "depth_gt_12"),  # deeper than 12: the device_levels prefix, then host levels
        # a binding leaf budget: selective growth, its first levels as the device prefix
        (40, 6, "selective", "selective_budget"),
        (255, -1, "selective", "selective_unbounded"),
    ],
)
def test_device_tree_default_reports_fallback_cuda(leaves, depth, flow, reason):
    # outside the depth bound the default reports the fallback under diag and keeps the flow master ran
    diag = _diag("auto", leaves, depth)
    assert "device_tree: complete" not in diag
    # once per route change (one binary learner on one route here), not once per tree
    route = "prefix" if flow == "prefix" else "selective_prefix"
    assert diag.count("device_tree: fallback (") == 1, diag[-3000:]
    assert diag.count(f"device_tree: fallback ({reason}) route={route}\n") == 1, diag[-3000:]
    if flow == "prefix":
        assert "device-driven levels per tree" in diag
    else:
        # cuda_plan key selective_device_prefix (default on); off, the host's selective loop runs every level
        assert "selective_device_prefix: levels=" in diag
        off = _diag("auto,selective_device_prefix:off", leaves, depth)
        assert "device_levels: host fallback (plan_off)" in off
        assert f"device_tree: fallback ({reason}) route=selective_two_sync" in off
