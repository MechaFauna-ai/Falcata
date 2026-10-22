# coding: utf-8
"""The complete device tree under per-node feature masks (cuda_plan key device_bynode_masks).

With feature_fraction_bynode < 1 (no interaction constraints) each node searches a keyed sample of its own
(ColSampler::GetByNodeKeyed: a generator seeded by the seed, the tree count and the node's path). The host level flows
draw those samples on the host and upload one mask row per searched leaf every level, which kept such trees off the
complete device tree. Under device_bynode_masks (default on) a device kernel draws the same samples (Random::Sample's
branches, the same generator), the same mask rows and the same task lists inside the device levels, so the tree grows
on the device. The models must be md5-identical to the key off (the host two-sync level loop), to device_tree:off and
to device_levels:off, for every branch of Random::Sample (the selection scan, Floyd's set draw, all candidates),
with and without feature_fraction and bagging, for regression, binary and multiclass.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

NUM_FEATURES = 40


def _model_md5(params, X, y, rounds=12):
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(objective, rows=8000, cols=NUM_FEATURES, seed=11):
    # every feature carries signal, so a node's split feature depends on its sample
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.03] = np.nan
    z = np.nan_to_num(X).sum(axis=1) / np.sqrt(cols) + 0.3 * rng.standard_normal(rows)
    if objective == "multiclass":
        return X, np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    if objective == "binary":
        return X, (z > 0).astype(np.float32)
    return X, z.astype(np.float32)


def _params(objective, bynode, **extra):
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 64,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "feature_fraction_bynode": bynode,
        "verbose": -1,
        "seed": 3,
        **extra,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    return params


# feature_fraction_bynode -> Random::Sample's branch over 40 candidates: 0.3 / 0.5 / 0.8 take the selection scan
# (K > N / log2(K)), 0.1 (K = 4) and 0.02 (K = 1) Floyd's set draw, 0.99 (K = N) every candidate
@pytest.mark.parametrize("objective", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("bynode", [0.3, 0.5, 0.8])
@pytest.mark.parametrize(
    "extra",
    [{}, {"bagging_fraction": 0.7, "bagging_freq": 1}],
    ids=["nobag", "bag"],
)
def test_device_bynode_same_model_cuda(objective, bynode, extra):
    X, y = _data(objective)
    params = _params(objective, bynode, **extra)
    host = _model_md5({**params, "cuda_plan": "auto,device_levels:off"}, X, y)
    assert _model_md5({**params, "cuda_plan": "auto,device_tree:off"}, X, y) == host
    assert _model_md5({**params, "cuda_plan": "auto,device_bynode_masks:off"}, X, y) == host
    assert _model_md5({**params, "cuda_plan": "auto"}, X, y) == host


@pytest.mark.parametrize(
    ("bynode", "extra"),
    [
        (0.1, {}),  # Floyd, K = 4
        (0.02, {}),  # Floyd, K = 1
        (0.99, {}),  # K == N: every candidate
        (0.5, {"feature_fraction": 0.5}),  # the candidates are the tree's sample (they change every tree)
        (0.1, {"feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1}),
        (0.5, {"num_leaves": 63}),  # a binding final level (the device arbitration)
        (0.5, {"num_leaves": 1023, "max_depth": 10}),
        (0.5, {"num_leaves": 3, "max_depth": 1}),  # one level: only the root's mask
    ],
)
def test_device_bynode_branches_same_model_cuda(bynode, extra):
    X, y = _data("binary")
    params = _params("binary", bynode, **extra)
    host = _model_md5({**params, "cuda_plan": "auto,device_bynode_masks:off"}, X, y)
    assert _model_md5({**params, "cuda_plan": "auto"}, X, y) == host


@pytest.mark.parametrize(
    "plan",
    [
        "auto,device_level_graph:off",  # the direct enqueue (no capture)
        "auto,device_tree_lean_end:off",  # the full tree end
        "auto,pdl_level_chain:off",
    ],
)
def test_device_bynode_plan_variants_same_model_cuda(plan):
    X, y = _data("multiclass")
    params = _params("multiclass", 0.5, bagging_fraction=0.7, bagging_freq=1)
    host = _model_md5({**params, "cuda_plan": "auto,device_bynode_masks:off"}, X, y)
    assert _model_md5({**params, "cuda_plan": plan}, X, y) == host


def test_device_bynode_outside_bound_keeps_host_draws_cuda():
    # 2^6 > 31 + 1: outside the complete tree's bound; the device_levels prefix does not take per-node masks, so the
    # host level loop grows these trees with host draws, key on or off
    X, y = _data("binary")
    params = _params("binary", 0.5, num_leaves=31, max_depth=6)
    off = _model_md5({**params, "cuda_plan": "auto,device_bynode_masks:off"}, X, y)
    assert _model_md5({**params, "cuda_plan": "auto"}, X, y) == off
    err = _diag("auto", num_leaves=31, max_depth=6)
    assert "device masks" not in err
    assert "device_tree: complete" not in err


def test_device_bynode_rf_unchanged_cuda():
    # boosting=rf: forest slots keep the host level loop and the host draws; a group's only tree
    # (forest_single_job_device) draws on the device. The same masks either way, key on or off
    X, y = _data("binary")
    params = _params("binary", 0.5, boosting="rf", bagging_fraction=0.632, bagging_freq=1)
    off = _model_md5({**params, "cuda_plan": "auto,device_bynode_masks:off"}, X, y, rounds=6)
    assert _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=6) == off


_ENGAGE = (
    "import sys, json, numpy as np, falcata as flc\n"
    "rng = np.random.default_rng(1)\n"
    "X = rng.standard_normal((20000, 40)).astype(np.float32)\n"
    "y = (X[:, :20].sum(axis=1) > 0).astype(np.float32)\n"
    "p = {'device_type': 'cuda', 'objective': 'binary', 'verbose': -1, 'quant_mode': 'stochastic',\n"
    "     'use_quantized_grad': True, 'num_leaves': 64, 'max_depth': 6, 'feature_fraction_bynode': 0.5}\n"
    "p.update(json.loads(sys.argv[2]))\n"
    "p['cuda_plan'] = sys.argv[1]\n"
    "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=4)\n"
)


def _diag(plan, debug="diag", **extra):
    env = dict(os.environ, FALCATA_DEBUG=debug)
    args = [sys.executable, "-c", _ENGAGE, plan, json.dumps(extra)]
    return subprocess.run(args, env=env, capture_output=True, text=True, check=True).stderr


def test_device_bynode_engages_cuda():
    # the default plan grows every bynode tree as a complete device tree with device-drawn masks
    on = _diag("auto")
    assert on.count("device_tree: complete depth=6 ") == 4, on[-3000:]
    assert "[bynode-diag] device masks: candidates=40 sampled=20 mode=2 " in on
    assert "host fallback" not in on
    # the key off keeps the host two-sync level loop, as before
    off = _diag("auto,device_bynode_masks:off")
    assert "device_tree: complete" not in off
    assert "device masks" not in off
    # the device_levels prefix declines too (the masks are drawn on the host per level)
    assert "device_tree: fallback (plan_off) route=host_two_sync prefix=bynode_host_masks" in off


def test_device_bynode_debug_bynode_keeps_host_draws_cuda():
    # FALCATA_DEBUG=bynode prints each split's host-drawn mask: those trees keep the host draws
    err = _diag("auto", debug="diag,bynode")
    assert "device_tree: complete" not in err
    assert "BYNODE " in err


def test_device_bynode_interaction_constraints_keep_host_cuda():
    # interaction constraints draw from the shared stream on the per-split loop: no device tree
    err = _diag("auto", interaction_constraints=[list(range(20)), list(range(20, 40))])
    assert "device_tree: complete" not in err
    assert "device masks" not in err
