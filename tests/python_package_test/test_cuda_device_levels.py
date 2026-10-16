# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key device_levels.

The key runs the first levels of the quantized two-sync level prefix (those where the leaf budget cannot bind and the
children stay below max_depth) device-driven: the level sync's select kernel and the apply's pairs kernel replay the
host loop's decisions (the splittable leaves, the apply descriptors, the tree record, the children's search
descriptors and the construct's block map), the launches are sized for the level's worst case and read the live
counts from the device, and the host reads the levels back once. The sweep covers both quantized modes, regression,
binary and multiclass, leaf budgets that are and are not powers of two (the host loop takes over where the budget can
bind), max_depth below and above log2(num_leaves), min_data_in_leaf large enough to stop a tree inside the device
levels, min_gain_to_split stumps, bagging, GOSS, feature_fraction (the compact column view), few-bin and wide data,
and missing values; the configurations the key leaves to the host loop (per-node feature masks, categorical
features, interaction constraints) must still train the same model.
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


def _model_md5(params, X, y, rounds, categorical=None):
    ds = flc.Dataset(X, label=y, params=params, categorical_feature=categorical or "auto")
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(rows, cols, objective, seed, few_bin_cols=4, nan_frac=0.05):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[:, cols - few_bin_cols :] = rng.integers(0, 6, size=(rows, few_bin_cols))
    if nan_frac > 0:
        X[rng.random(X.shape) < nan_frac] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > 0).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


def _params(objective, leaves, depth, quant, seed=3, **extra):
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "verbose": -1,
        "quant_mode": quant,
        "seed": seed,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    params.update(extra)
    return params


def _same(params, X, y, rounds, categorical=None):
    off = _model_md5({**params, "cuda_plan": "auto,device_levels:off"}, X, y, rounds, categorical)
    on = _model_md5({**params, "cuda_plan": "auto,device_levels:on,device_level_graph:off"}, X, y, rounds, categorical)
    graph = _model_md5(
        {**params, "cuda_plan": "auto,device_levels:on,device_level_graph:on"}, X, y, rounds, categorical
    )
    assert on == off
    assert graph == off


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "leaves", "depth", "quant", "extra"),
    [
        ("regression", 255, 8, "stochastic", {}),
        ("binary", 511, 9, "fixedpoint", {}),
        ("multiclass", 127, 7, "fixedpoint", {}),
        ("regression", 1023, 10, "stochastic", {"bagging_fraction": 0.5, "bagging_freq": 1}),
        ("regression", 1024, 10, "fixedpoint", {}),
        ("regression", 100, 8, "stochastic", {}),
        ("binary", 300, 9, "stochastic", {"bagging_fraction": 0.7, "bagging_freq": 1}),
        ("regression", 255, 4, "stochastic", {}),
        ("regression", 255, -1, "fixedpoint", {}),
        ("binary", 255, 8, "stochastic", {"min_data_in_leaf": 3000}),
        ("regression", 255, 8, "stochastic", {"min_gain_to_split": 30.0}),
        ("regression", 255, 8, "fixedpoint", {"feature_fraction": 0.5}),
        ("binary", 255, 8, "stochastic", {"feature_fraction": 0.3, "bagging_fraction": 0.5, "bagging_freq": 1}),
        ("regression", 511, 9, "stochastic", {"min_sum_hessian_in_leaf": 5.0}),
        ("regression", 255, 8, "stochastic", {"quant_bins": 16}),
        ("regression", 255, 8, "stochastic", {"max_bin": 15}),
    ],
)
def test_device_levels_is_bit_identical_cuda(objective, leaves, depth, quant, extra):
    X, y = _data(30011, 20, objective, 29)
    _same(_params(objective, leaves, depth, quant, **extra), X, y, 12)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
def test_device_levels_wide_and_goss_cuda(quant):
    # more columns than one construct partition holds, and GOSS's bagged subset
    X, y = _data(20011, 700, "binary", 31, few_bin_cols=300, nan_frac=0.0)
    _same(_params("binary", 255, 8, quant, data_sample_strategy="goss", top_rate=0.2, other_rate=0.1), X, y, 8)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
def test_device_levels_few_values_sampled_cuda(quant):
    # Numerai-like: many five-value columns, a tenth of them sampled per tree (the compact view, the pair-joint
    # constructs, per-tree launch layouts that a cached level graph must key on)
    rng = np.random.default_rng(41)
    rows, cols = 30011, 400
    X = rng.integers(0, 5, size=(rows, cols)).astype(np.float32)
    y = (X[:, :8].sum(axis=1) + rng.normal(size=rows)).astype(np.float32)
    _same(_params("regression", 1023, 10, quant, feature_fraction=0.1, min_data_in_leaf=20), X, y, 20)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "extra",
    [
        {"feature_fraction_bynode": 0.5},
        {"interaction_constraints": [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]},
        {"categorical": True},
    ],
)
def test_device_levels_host_fallbacks_cuda(extra):
    # flows the key leaves to the host loop: the same model either way
    extra = dict(extra)
    categorical = None
    X, y = _data(20011, 12, "regression", 37)
    if extra.pop("categorical", False):
        X[:, 0] = np.random.default_rng(5).integers(0, 30, size=X.shape[0])
        categorical = [0]
    _same(_params("regression", 255, 8, "stochastic", **extra), X, y, 8, categorical)


@_REQUIRES_CUDA
def test_device_levels_engage_cuda():
    # the key runs where its gate holds: the hybrid diag line names the device levels (a child process: the line is
    # printed once per process)
    code = (
        "import numpy as np, falcata as flc\n"
        "rng = np.random.default_rng(1)\n"
        "X = rng.standard_normal((20000, 10)).astype(np.float32)\n"
        "y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)\n"
        "p = {'device_type': 'cuda', 'objective': 'binary', 'num_leaves': 255, 'max_depth': 8, 'verbose': -1,\n"
        "     'quant_mode': 'stochastic', 'cuda_plan': 'auto,device_levels:on'}\n"
        "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=3)\n"
    )
    env = dict(os.environ, FALCATA_DEBUG="diag")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert "device_levels: 7 device-driven levels per tree" in out.stderr
