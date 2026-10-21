# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key device_tree_lean_end.

The key gives the complete device tree's lean tree end (no host leaf cache, the inline final leaf map) to every
objective boosted on the device (the device renewal of L1/quantile/MAPE reads only device leaf windows), also under row
bagging (the bag's map inline) and with weights, not only unweighted, unbagged constant-hessian L2. The sweep covers those
objectives, bagged and weighted ones.
"""

import hashlib
import json
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


def _model_md5(params, X, y, rounds, weight=None):
    booster = flc.train(params, flc.Dataset(X, label=y, weight=weight, params=params), num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(objective, rows=6000, cols=12, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    if objective in ("multiclass", "multiclassova"):
        return X, np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    if objective in ("binary", "cross_entropy"):
        return X, (z > 0).astype(np.float32)
    if objective in ("poisson", "gamma", "tweedie"):
        return X, np.exp(0.5 * z).astype(np.float32)
    return X, z.astype(np.float32)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "extra", "weighted"),
    [
        ("binary", {}, False),
        ("multiclass", {"num_class": 4}, False),
        ("multiclassova", {"num_class": 4}, False),
        ("cross_entropy", {}, False),
        ("poisson", {}, False),
        ("huber", {}, False),
        ("regression", {}, True),  # weighted
        ("binary", {"bagging_fraction": 0.6, "bagging_freq": 1}, True),
        ("binary", {"bagging_fraction": 0.6, "bagging_freq": 1}, False),  # bagged: the bag's map written inline
        ("binary", {"bagging_fraction": 0.6, "bagging_freq": 3}, False),  # a bag kept for three trees
        ("binary", {"pos_bagging_fraction": 0.7, "neg_bagging_fraction": 0.4, "bagging_freq": 1}, False),
        ("multiclass", {"num_class": 4, "bagging_fraction": 0.7, "bagging_freq": 1}, False),
        ("regression", {"bagging_fraction": 0.5, "bagging_freq": 1}, False),
        ("regression_l1", {}, False),  # renews its outputs on the device
        ("quantile", {"alpha": 0.7, "bagging_fraction": 0.6, "bagging_freq": 1}, False),
    ],
)
@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint"])
def test_device_tree_lean_end_same_model_cuda(objective, extra, weighted, quant_mode):
    X, y = _data(objective)
    weight = np.random.default_rng(3).uniform(0.5, 2.0, size=len(y)) if weighted else None
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 63,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "use_quantized_grad": True,
        "quant_mode": quant_mode,
        "verbose": -1,
        "seed": 3,
        **extra,
    }
    off = _model_md5({**params, "cuda_plan": "auto,device_tree_lean_end:off"}, X, y, 12, weight)
    assert _model_md5({**params, "cuda_plan": "auto"}, X, y, 12, weight) == off


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["lambdarank", "rank_xendcg"])
@pytest.mark.parametrize("extra", [{}, {"bagging_fraction": 0.6, "bagging_freq": 1}])
def test_device_tree_lean_end_ranking_same_model_cuda(objective, extra):
    X, z = _data("regression")
    y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    group = [30] * (len(y) // 30)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 63,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "use_quantized_grad": True,
        "verbose": -1,
        "seed": 3,
        **extra,
    }
    md5s = []
    for plan in ("auto,device_tree_lean_end:off", "auto"):
        p = {**params, "cuda_plan": plan}
        booster = flc.train(p, flc.Dataset(X, label=y, group=group, params=p), num_boost_round=12)
        lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
        md5s.append(hashlib.md5("\n".join(lines).encode()).hexdigest())
    assert md5s[0] == md5s[1]


_ENGAGE = (
    "import sys, json, numpy as np, falcata as flc\n"
    "rng = np.random.default_rng(1)\n"
    "X = rng.standard_normal((20000, 10)).astype(np.float32)\n"
    "z = X[:, 0] + X[:, 1]\n"
    "p = dict(json.loads(sys.argv[2]), cuda_plan=sys.argv[1])\n"
    "y = np.digitize(z, [-1.0, 0.0, 1.0]) if p['objective'] == 'multiclass' else (z > 0) if p['objective'] == 'binary' else z\n"
    "flc.train(p, flc.Dataset(X, label=y.astype(np.float32), params=p), num_boost_round=3)\n"
)


def _host_cache_lines(plan, extra):
    params = {
        "device_type": "cuda",
        "objective": "binary",
        "num_leaves": 64,
        "max_depth": 6,
        "verbose": -1,
        "quant_mode": "stochastic",
        "use_quantized_grad": True,
        **extra,
    }
    out = subprocess.run(
        [sys.executable, "-c", _ENGAGE, plan, json.dumps(params)],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    return re.findall(r"device_tree_host_cache: skipped=(\d)", out.stderr)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "extra",
    [
        {},  # binary
        {"bagging_fraction": 0.6, "bagging_freq": 1},  # a row bag: the tree's root holds fewer rows than the data
        {"objective": "regression_l1"},  # renewed on the device
        {"objective": "quantile", "alpha": 0.7},
        {"objective": "multiclass", "num_class": 4},
    ],
)
def test_device_tree_lean_end_engages_cuda(extra):
    # every complete tree of these shapes skips the host leaf cache under the default plan; the key off restores it.
    # md5 equality alone would also pass if the predicate silently excluded them.
    on = _host_cache_lines("auto", extra)
    assert on, "no complete device tree"
    assert set(on) == {"1"}, on
    off = _host_cache_lines("auto,device_tree_lean_end:off", extra)
    assert off, "no complete device tree"
    assert set(off) == {"0"}, off


@_REQUIRES_CUDA
def test_device_tree_lean_end_not_for_one_level_cuda():
    # a one-level binary tree (max_depth 1) keeps the full tree end: measured slower on the lean end
    code = (
        "import numpy as np, falcata as flc\n"
        "rng = np.random.default_rng(1)\n"
        "X = rng.standard_normal((20000, 10)).astype(np.float32)\n"
        "y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)\n"
        "p = {'device_type': 'cuda', 'objective': 'binary', 'num_leaves': 8, 'max_depth': 1, 'verbose': -1,\n"
        "     'quant_mode': 'stochastic', 'use_quantized_grad': True, 'cuda_plan': 'auto'}\n"
        "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=3)\n"
    )
    env = dict(os.environ, FALCATA_DEBUG="diag")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert "device_tree: complete depth=1 " in out.stderr
    assert "device_tree_host_cache: skipped=1" not in out.stderr
    assert "device_tree_host_cache: skipped=0" in out.stderr
