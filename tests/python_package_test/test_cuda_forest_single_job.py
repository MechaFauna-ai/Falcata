# coding: utf-8
"""A random forest group of one tree trains off slot mode (cuda_plan key forest_single_job_device).

Under forest_batched a group of one tree (forest_group 1, a binary or regression forest whose auto size is one round,
the last group of a forest_group N training) trains on the booster's learner with no other slot thread running, so it
takes the flows a forest slot never takes: the complete device tree where its depth bound holds, the device_levels
prefix and the quantized graph prefix elsewhere. Groups of two or more trees stay in slot mode. The trees are the ones
sequential training grows, so every case compares the model md5 (parameter block excepted) with the key off and with
forest_batched off, across objectives, quantization, per-node and per-tree feature sampling, depth-bounded and
unbounded shapes and group sizes that switch between one and more trees within a training (auto).
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

KEY_OFF = ",forest_single_job_device:off"
PLANS = ["auto", "auto,forest_group:1", "auto,forest_group:2", "auto,forest_group:3"]


def _strip_parameters(model_str):
    out, skip = [], False
    for line in model_str.splitlines():
        if line.startswith("parameters:"):
            skip = True
        if not skip:
            out.append(line)
        if line.startswith("end of parameters"):
            skip = False
    return "\n".join(out)


def _data(kind, rows=20011, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, 16)).astype(np.float32)
    X[rng.random(X.shape) < 0.03] = np.nan
    z = np.nan_to_num(X[:, 0]) - 0.5 * np.nan_to_num(X[:, 1]) + np.sin(np.nan_to_num(X[:, 2]))
    z = z + 0.2 * np.nan_to_num(X[:, 3:]).sum(1) + 0.3 * rng.standard_normal(rows)
    if kind == "binary":
        y = (z > 0).astype(np.float32)
    elif kind == "multiclass":
        y = np.digitize(z, [-1.5, -0.5, 0.5, 1.5]).astype(np.float32)
    else:
        y = (z + rng.standard_exponential(rows)).astype(np.float32)
    return X, y


def _params(objective, quant_mode, **extra):
    params = {
        "boosting": "rf",
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": quant_mode,
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "min_data_in_leaf": 5,
        "max_bin": 63,
        "num_leaves": 63,
        "max_depth": 6,
        "metric": "None",
        "seed": 7,
        "verbose": -1,
    }
    if objective == "multiclass":
        params["num_class"] = 5
    params.update(extra)
    return params


def _md5(params, plan, X, y, rounds):
    if params["quant_mode"] == "none":
        # sequential training (forest_batched off) may take the complete non-quantized device tree (fixed-point
        # levels), forest groups never take it: both arms keep the one-sync host loop
        plan += ",device_tree:off"
    ds = flc.Dataset(X, label=y, params={"max_bin": 63, "verbose": -1})
    booster = flc.train({**params, "cuda_plan": plan}, ds, num_boost_round=rounds)
    return hashlib.md5(_strip_parameters(booster.model_to_string()).encode()).hexdigest()


@pytest.mark.parametrize(
    ("objective", "kind", "quant_mode", "extra"),
    [
        ("binary", "binary", "stochastic", {}),
        ("binary", "binary", "fixedpoint", {}),
        ("binary", "binary", "none", {}),
        ("regression", "regression", "stochastic", {}),
        ("multiclass", "multiclass", "stochastic", {}),
        # per-node masks: device-drawn in a solo tree, host-drawn in a slot (keyed: the same masks)
        ("binary", "binary", "stochastic", {"feature_fraction_bynode": 0.5}),
        # per-tree samples drawn in tree order by the forest trainer
        ("binary", "binary", "stochastic", {"feature_fraction": 0.6}),
        # outside the device tree's bound: the device_levels prefix / quantized graph prefix and the selective tail
        ("binary", "binary", "stochastic", {"num_leaves": 127, "max_depth": -1}),
        ("regression", "regression", "stochastic", {"num_leaves": 255, "max_depth": 8, "min_data_in_leaf": 1}),
        # no row sampling (rf needs bagging or a per-tree feature sample): every tree on every row
        ("binary", "binary", "stochastic", {"bagging_fraction": 1.0, "bagging_freq": 0, "feature_fraction": 0.8}),
        # a binding leaf budget (2^7 > 40 + 1): selective growth with the selective device prefix in a solo tree
        ("binary", "binary", "stochastic", {"num_leaves": 40, "max_depth": 7}),
    ],
)
def test_forest_single_job_same_forest_cuda(objective, kind, quant_mode, extra):
    X, y = _data(kind)
    params = _params(objective, quant_mode, **extra)
    # 25 rounds: auto grows one round per group, then two (a second slot), then settles; forest_group 3 ends with a
    # group of one round
    want = _md5(params, "auto,forest_batched:off", X, y, 25)
    for plan in PLANS:
        for key in ("", KEY_OFF):
            assert _md5(params, plan + key, X, y, 25) == want, (plan + key, objective, quant_mode, extra)


_CHILD = r"""
import json, sys
import numpy as np
import falcata as flc
cfg = json.loads(sys.argv[1])
rounds = cfg.pop("rounds")
rng = np.random.default_rng(3)
X = rng.standard_normal((20000, 16)).astype(np.float32)
z = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(20000)
y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32) if cfg.get("num_class") else (z > 0).astype(np.float32)
ds = flc.Dataset(X, label=y, params={"max_bin": 63, "verbose": -1})
flc.train(cfg, ds, num_boost_round=rounds)
"""


def _diag_lines(params, rounds):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["FALCATA_DEBUG"] = "diag"
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps({**params, "rounds": rounds})],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stderr.splitlines()


def _diag(params, rounds):
    lines = _diag_lines(params, rounds)
    groups = [line for line in lines if line.startswith("[forest-diag] rounds")]
    solo = [line for line in groups if line.endswith("mode single job (non-slot)")]
    slots = [line for line in groups if line.endswith("mode slots")]
    assert len(solo) + len(slots) == len(groups), groups
    complete = sum("device_tree: complete" in line for line in lines)
    return solo, slots, complete


def _engage_params(**extra):
    params = {
        "boosting": "rf",
        "device_type": "cuda",
        "objective": "binary",
        "quant_mode": "stochastic",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "num_leaves": 63,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "seed": 7,
        "verbose": -1,
    }
    params.update(extra)
    return params


def test_forest_single_job_takes_the_device_tree_cuda():
    # forest_group 1: every group is one tree, each a complete device tree
    solo, slots, complete = _diag(_engage_params(cuda_plan="auto,forest_group:1"), 12)
    assert len(solo) == 12, solo + slots
    assert not slots, solo + slots
    assert complete == 12
    # key off: every group a slot, no device tree
    solo, slots, complete = _diag(_engage_params(cuda_plan="auto,forest_group:1" + KEY_OFF), 12)
    assert not solo, solo + slots
    assert len(slots) == 12, solo + slots
    assert complete == 0
    # forest_group 3 over 7 rounds: 3, 3, 1; only the last group's tree is alone
    solo, slots, complete = _diag(_engage_params(cuda_plan="auto,forest_group:3"), 7)
    assert len(solo) == 1, solo + slots
    assert len(slots) == 2, solo + slots
    assert complete == 1


def test_forest_single_job_auto_switches_modes_cuda():
    # auto: one round per group first (alone), then two (slots), within one training; the device tree only on the
    # trees that were alone
    solo, slots, complete = _diag(_engage_params(), 25)
    assert solo, slots
    assert slots, solo
    solo_trees = sum(int(line.split(": ")[1].split()[0]) for line in solo)
    assert complete == solo_trees


@pytest.mark.parametrize(
    ("extra", "mode"),
    [
        # a round of several class trees is one group of several jobs: slots
        ({"objective": "multiclass", "num_class": 4}, "slots"),
        # per-node masks: alone, a complete device tree with device-drawn masks
        ({"feature_fraction_bynode": 0.5}, "solo"),
    ],
)
def test_forest_single_job_modes_cuda(extra, mode):
    solo, slots, complete = _diag(_engage_params(cuda_plan="auto,forest_group:1", **extra), 6)
    if mode == "slots":
        assert not solo, solo + slots
        assert len(slots) == 6, solo + slots
        assert complete == 0
    else:
        assert len(solo) == 6, solo + slots
        assert not slots, solo + slots
        assert complete == 6


@pytest.mark.parametrize("shape", [{"num_leaves": 127, "max_depth": -1}, {"num_leaves": 40, "max_depth": 7}])
def test_forest_single_job_takes_the_selective_device_prefix_cuda(shape):
    # unbounded depth / a binding leaf budget: selective growth, whose levels up to the first binding level run on the
    # device in a solo tree
    for key, want in (("", 6), (KEY_OFF, 0)):
        lines = _diag_lines(_engage_params(cuda_plan="auto,forest_group:1" + key, **shape), 6)
        assert sum(line.endswith("mode single job (non-slot)") for line in lines) == want
        assert sum("device_tree: selective hand-off at level" in line for line in lines) == want
        assert not any("device_tree: complete" in line for line in lines)


def test_forest_single_job_no_nonquantized_device_tree_cuda():
    # a solo non-quantized tree keeps the slots' level loop: the complete non-quantized device tree (the graph loop's
    # model before P3) constructs in fixed point where the slots' loop takes the double-slot construct, so a forest's
    # model would depend on its group sizes. Sequential training (forest_batched off) takes it on this shape
    graph_line = "[hybrid-diag] device_tree: complete"
    sequential = _diag_lines(_engage_params(quant_mode="none", cuda_plan="auto,forest_batched:off"), 6)
    assert any(graph_line in line for line in sequential), sequential[-20:]
    lines = _diag_lines(_engage_params(quant_mode="none", cuda_plan="auto,forest_group:1"), 6)
    assert sum(line.endswith("mode single job (non-slot)") for line in lines) == 6
    assert not any(graph_line in line for line in lines), [line for line in lines if graph_line in line]
