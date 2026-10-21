# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key device_level_graph_update.

device_level_graph_update updates a cached complete-device-tree graph exec in place (cudaGraphExecUpdate) when the
whole-tree graph keys keep changing (a sampled feature layout every tree) instead of instantiating a new one.
"""

import hashlib
import json
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
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _churn_data():
    # 600 columns of 2 to 63 distinct values at feature_fraction 0.1: the sampled columns' bin total (part of the
    # graph key) differs from tree to tree (28 of 30 trees miss; 300 columns of 2 to 31 values repeat their layouts)
    rng = np.random.default_rng(2)
    X = (rng.integers(0, 1 << 20, size=(6000, 600)) % (2 + np.arange(600) % 62)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (X[:, :10].sum(axis=1) + rng.normal(size=len(X))).astype(np.float32)
    return X, y


_PARAMS = {
    "device_type": "cuda",
    "objective": "regression",
    "num_leaves": 32,
    "max_depth": 5,
    "feature_fraction": 0.1,
    "min_data_in_leaf": 20,
    "use_quantized_grad": True,
    "quant_mode": "stochastic",
    "verbose": -1,
    "seed": 5,
}


@_REQUIRES_CUDA
@pytest.mark.parametrize("extra", [{}, {"objective": "binary"}, {"bagging_fraction": 0.6, "bagging_freq": 1}])
def test_device_level_graph_update_same_model_cuda(extra):
    # many features at feature_fraction 0.1: after the ninth miss the cached execs are updated in place
    X, y = _churn_data()
    if extra.get("objective") == "binary":
        y = (y > np.median(y)).astype(np.float32)
    params = {**_PARAMS, **extra}
    off = _model_md5({**params, "cuda_plan": "auto,device_level_graph_update:off"}, X, y, 30)
    assert _model_md5({**params, "cuda_plan": "auto"}, X, y, 30) == off


@_REQUIRES_CUDA
def test_device_level_graph_update_engages_cuda():
    code = (
        "import sys, json, numpy as np, falcata as flc\n"
        "rng = np.random.default_rng(2)\n"
        "X = (rng.integers(0, 1 << 20, size=(6000, 600)) % (2 + np.arange(600) % 62)).astype(np.float32)\n"
        "y = (X[:, :10].sum(axis=1) + rng.normal(size=len(X))).astype(np.float32)\n"
        "p = dict(json.loads(sys.argv[2]), cuda_plan=sys.argv[1])\n"
        "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=30)\n"
    )
    env = dict(os.environ, FALCATA_DEBUG="diag")

    def last_update_line(plan):
        out = subprocess.run(
            [sys.executable, "-c", code, plan, json.dumps(_PARAMS)], env=env, capture_output=True, text=True, check=True
        )
        lines = [line for line in out.stderr.splitlines() if "device_level_graph_update: updates=" in line]
        assert lines, out.stderr[-2000:]
        return lines[-1]

    assert "updates=0 " not in last_update_line("auto")
    assert "updates=0 refused=0" in last_update_line("auto,device_level_graph_update:off")
