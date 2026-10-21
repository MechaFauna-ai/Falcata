# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key index_parity_canonical.

Every level swaps the data partition's two index buffers, so a tree of odd depth hands the next tree the buffers in
the other role and the graph key (which holds both pointers) alternates from tree to tree: each layout is captured
twice. index_parity_canonical starts every tree with the lower-address buffer as the main array (unbagged trees before
the identity fill, bagged trees when a new bag is copied in). The sweep covers odd and even depths, the prefix and
host flows, bagging and a non-quantized tree; the diag test checks that an odd-depth shape captures one graph, not two.
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

_OFF = ",index_parity_canonical:off"


def _model_md5(params, X, y, rounds):
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(rows=8000, cols=12, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    return X, z.astype(np.float32)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("depth", "leaves", "plan", "extra"),
    [
        (1, 2, "auto", {}),
        (5, 32, "auto", {}),  # odd depth: the parity flips every tree without the key
        (6, 64, "auto", {}),
        (7, 128, "auto", {"objective": "binary"}),
        (5, 32, "auto", {"bagging_fraction": 0.6, "bagging_freq": 1}),  # a new bag every tree
        (5, 32, "auto", {"bagging_fraction": 0.6, "bagging_freq": 3}),  # a bag kept for three trees
        (5, 32, "auto", {"bagging_fraction": 0.6, "bagging_freq": 2}),  # RestoreBagOrder's copy canonicalizes
        (5, 32, "auto", {"feature_fraction": 0.5}),
        (5, 32, "auto,device_tree:off", {}),  # the device_levels prefix
        (5, 32, "auto,device_levels:off", {}),  # the host flow
        (7, 100, "auto", {}),  # outside the device tree's bound: prefix / host flow
        (5, 32, "auto", {"use_quantized_grad": False}),  # non-quantized
    ],
)
def test_index_parity_canonical_same_model_cuda(depth, leaves, plan, extra):
    X, y = _data()
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 5,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 3,
        **extra,
    }
    if params["objective"] == "binary":
        y = (y > 0).astype(np.float32)
    off = _model_md5({**params, "cuda_plan": plan + _OFF}, X, y, 20)
    assert _model_md5({**params, "cuda_plan": plan}, X, y, 20) == off


_DIAG_CODE = (
    "import sys, json, numpy as np, falcata as flc\n"
    "rng = np.random.default_rng(1)\n"
    "X = rng.standard_normal((20000, 10)).astype(np.float32)\n"
    "y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.3 * rng.standard_normal(20000)).astype(np.float32)\n"
    "p = dict(json.loads(sys.argv[2]), cuda_plan=sys.argv[1])\n"
    "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=12)\n"
)


def _last_misses(plan, depth, extra=None):
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 1 << depth,
        "max_depth": depth,
        "min_data_in_leaf": 5,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 3,
        **(extra or {}),
    }
    out = subprocess.run(
        [sys.executable, "-c", _DIAG_CODE, plan, json.dumps(params)],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    found = re.findall(r"device_tree: complete .* hits=(\d+) misses=(\d+)", out.stderr)
    assert len(found) == 12, out.stderr[-2000:]
    return int(found[-1][1])


@_REQUIRES_CUDA
@pytest.mark.parametrize("depth", [5, 7])
def test_index_parity_canonical_one_capture_cuda(depth):
    # an odd-depth tree whose layout never changes: one capture with the key, one per parity (two) without it
    assert _last_misses("auto" + _OFF, depth) == 2
    assert _last_misses("auto", depth) == 1


@_REQUIRES_CUDA
@pytest.mark.parametrize("freq", [1, 2])
def test_index_parity_canonical_one_capture_bagged_cuda(freq):
    # bagged odd-depth trees: a new bag every tree (SetUsedDataIndices) or a bag kept for two trees (RestoreBagOrder's
    # copy at the score update) start canonical too
    bag = {"bagging_fraction": 0.6, "bagging_freq": freq}
    assert _last_misses("auto" + _OFF, 5, bag) == 2
    assert _last_misses("auto", 5, bag) == 1


@_REQUIRES_CUDA
def test_index_parity_canonical_even_depth_unchanged_cuda():
    assert _last_misses("auto" + _OFF, 6) == _last_misses("auto", 6) == 1
