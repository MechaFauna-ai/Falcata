# coding: utf-8
"""CUDA selective growth must train the same model with and without its device prefix.

A binding leaf budget (2^max_depth > num_leaves + 1) or unlimited depth trains with selective grow-then-prune growth.
cuda_plan key selective_device_prefix (default on) runs its first K = min(floor(log2(num_leaves)), max_depth - 1, 10)
levels as the device_levels prefix: on those levels the host's greedy simulation selects every valid candidate, which
is the prefix's "split every valid leaf". The host's selective loop then continues from the prefix's journal. The
models must be byte-identical to selective_device_prefix:off and to the host flow (device_levels:off) across
objectives, leaf budgets, gain and data gates, bagging and graph capture; FALCATA_DEBUG=diag reports one line per
tree with the levels taken, so the tests also check that the prefix ran.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

_OFF = ",selective_device_prefix:off"


def _data(objective, rows=8000, cols=12, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    z = (
        np.nan_to_num(X[:, 0])
        + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2])
        + 0.4 * np.sin(3 * np.nan_to_num(X[:, 3]))
        + 0.3 * rng.normal(size=rows)
    )
    if objective == "multiclass":
        return X, np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    if objective == "binary":
        return X, (z > 0).astype(np.float32)
    return X, z.astype(np.float32)


def _params(objective, leaves, depth, **extra):
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 5,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 11,
        **extra,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    return params


_TRAIN = (
    "import sys, json, hashlib, numpy as np, falcata as flc\n"
    "d = np.load(sys.argv[1])\n"
    "p = json.loads(sys.argv[2])\n"
    "b = flc.train(p, flc.Dataset(d['X'], label=d['y'], params=p), num_boost_round=int(sys.argv[3]))\n"
    "lines = [x for x in b.model_to_string().splitlines() if 'cuda_plan' not in x]\n"
    "print(json.dumps({'md5': hashlib.md5('\\n'.join(lines).encode()).hexdigest(), 'trees': b.num_trees()}))\n"
)


def _train_diag(params, X, y, rounds):
    # one training in a fresh process under FALCATA_DEBUG=diag: the model md5, the tree count and the prefix lines
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data.npz")
        np.savez(data, X=X, y=y)
        env = dict(os.environ, FALCATA_DEBUG="diag")
        args = [sys.executable, "-c", _TRAIN, data, json.dumps(params), str(rounds)]
        out = subprocess.run(args, env=env, capture_output=True, text=True, check=True)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    return res["md5"], res["trees"], [tuple(int(v) for v in m.groups()) for m in _LINE.finditer(out.stderr)]


def _same(params, X, y, rounds=8, plans=("auto", "auto" + _OFF, "auto,device_levels:off"), engaged=True, trained=None):
    # the same model under every plan; the first plan (the default) took the prefix on every trained tree iff engaged
    # (trained: trees grown when the model keeps fewer, e.g. stumps), the others never did. Returns its lines.
    md5 = {}
    first = []
    for i, plan in enumerate(plans):
        md5[plan], trees, lines = _train_diag({**params, "cuda_plan": plan}, X, y, rounds)
        expected = (trained if trained is not None else trees) if engaged and i == 0 else 0
        assert len(lines) == expected, (plan, trees, len(lines))
        first = first or lines
    assert len(set(md5.values())) == 1, md5
    return first


_SHAPES = [
    (31, -1, {}),
    (32, -1, {}),  # exact fit: 2^K = num_leaves, the last prefix level can fill the budget
    (64, -1, {}),  # exact fit
    (63, -1, {}),
    (127, -1, {}),
    (63, 10, {}),  # binding budget at a finite depth: 2^10 > 64
    (63, -1, {"min_gain_to_split": 2.0}),  # some candidates fail the gain gate
    (63, -1, {"min_data_in_leaf": 150}),  # leaves of the last prefix levels cannot split
    (63, -1, {"bagging_fraction": 0.6, "bagging_freq": 1}),
    (40, 7, {"feature_fraction": 0.7}),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize(("leaves", "depth", "extra"), _SHAPES)
def test_selective_device_prefix_same_model_cuda(objective, leaves, depth, extra):
    X, y = _data(objective)
    _same(_params(objective, leaves, depth, **extra), X, y)


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["on", "off"])
def test_selective_device_prefix_graph_same_model_cuda(graph):
    # the captured graph and the plain device enqueue of the prefix
    X, y = _data("binary")
    plans = tuple(f"{p},device_level_graph:{graph}" for p in ("auto", "auto" + _OFF, "auto,device_levels:off"))
    _same(_params("binary", 63, -1), X, y, rounds=12, plans=plans)


@_REQUIRES_CUDA
def test_selective_device_prefix_stump_same_model_cuda():
    # the root has no valid split: the prefix applies no level and the host loop reports a stump, as before
    # (a gain gate no split passes: the root is searched, so the prefix runs and applies no level)
    X, y = _data("regression", rows=2000)
    # (the booster keeps one constant tree; each of the 3 rounds grew a stump)
    lines = _same(_params("regression", 31, -1, min_gain_to_split=1e9), X, y, rounds=3, trained=3)
    assert all(applied == 0 and splits == 0 for _, applied, splits, _, _ in lines), lines


_ENGAGE = (
    "import sys, json, numpy as np, falcata as flc\n"
    "rng = np.random.default_rng(3)\n"
    "X = rng.standard_normal((20000, 10)).astype(np.float32)\n"
    "z = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + np.sin(3 * X[:, 3]) + 0.3 * rng.standard_normal(20000)\n"
    "p = json.loads(sys.argv[1])\n"
    "p.update({'device_type': 'cuda', 'verbose': -1, 'quant_mode': 'stochastic', 'use_quantized_grad': True})\n"
    "y = (z > 0).astype(np.float32) if p['objective'] == 'binary' else z.astype(np.float32)\n"
    "b = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=int(sys.argv[2]))\n"
    "print(json.dumps([t['num_leaves'] for t in b.dump_model()['tree_info']]))\n"
)


def _diag(params, rounds=4):
    env = dict(os.environ, FALCATA_DEBUG="diag")
    args = [sys.executable, "-c", _ENGAGE, json.dumps(params), str(rounds)]
    out = subprocess.run(args, env=env, capture_output=True, text=True, check=True)
    return out.stderr, json.loads(out.stdout.strip().splitlines()[-1])


_LINE = re.compile(
    r"selective_device_prefix: levels=(\d+) applied_levels=(\d+) splits=(\d+) leaves=(\d+) frontier=(\d+)"
)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("leaves", "depth", "levels"),
    [
        (31, -1, 4),
        (63, -1, 5),
        (255, -1, 7),
        (1024, -1, 10),  # the prefix's 512-pair bound and an exact fit: 2^10 = num_leaves
        (151, 10, 7),  # a binding budget at a finite depth
        (40, 6, 5),  # depth caps K at max_depth - 1
    ],
)
def test_selective_device_prefix_engages_cuda(leaves, depth, levels):
    params = {"objective": "regression", "num_leaves": leaves, "max_depth": depth, "min_data_in_leaf": 5}
    rounds = 4
    on, tree_leaves = _diag({**params, "cuda_plan": "auto"}, rounds)
    assert f"selective_device_prefix: {levels} device-driven levels per tree" in on
    lines = [tuple(int(v) for v in m.groups()) for m in _LINE.finditer(on)]
    assert len(lines) == rounds, on
    for (lv, applied, splits, prefix_leaves, frontier), final_leaves in zip(lines, tree_leaves, strict=True):
        assert lv == levels
        assert applied == levels  # this data splits leaves on every prefix level
        # a leaf of the prefix levels may find no valid split (it stays a dormant leaf)
        assert prefix_leaves == splits + 1 <= 2**levels
        # the host's selective loop continued from the prefix's frontier to the leaf budget
        assert frontier > 0
        assert final_leaves == leaves
    assert "device_levels: host fallback" not in on
    off, off_leaves = _diag({**params, "cuda_plan": "auto" + _OFF}, rounds)
    assert "selective_device_prefix" not in off
    assert "device_levels: host fallback (plan_off)" in off
    assert off_leaves == tree_leaves


@_REQUIRES_CUDA
def test_selective_device_prefix_partial_levels_cuda():
    # gates that leave some prefix leaves without a valid split: fewer splits than a full level, still one line per
    # tree, and the same model as the host flow
    params = {"objective": "regression", "num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 900}
    on, _ = _diag({**params, "cuda_plan": "auto"}, 3)
    lines = [tuple(int(v) for v in m.groups()) for m in _LINE.finditer(on)]
    assert len(lines) == 3
    assert any(splits + 1 < 2**applied for _, applied, splits, _, _ in lines), lines
    X, y = _data("regression")
    _same(_params("regression", 63, -1, min_data_in_leaf=900), X, y)


@_REQUIRES_CUDA
def test_selective_device_prefix_declines_forced_splits_cuda(tmp_path):
    # forced splits take the classic per-split loop (no selective growth): the key declines and the model is unchanged
    forced = tmp_path / "forced.json"
    forced.write_text(json.dumps({"feature": 0, "threshold": 0.0, "left": {"feature": 1, "threshold": 0.5}}))
    params = {
        "objective": "regression",
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 5,
        "forcedsplits_filename": str(forced),
    }
    on, _ = _diag({**params, "cuda_plan": "auto"}, 2)
    assert "selective_device_prefix" not in on
    X, y = _data("regression")
    _same(_params("regression", 63, -1, forcedsplits_filename=str(forced)), X, y, rounds=4, engaged=False)


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["binary", "multiclass"])
def test_selective_device_prefix_bynode_same_model_cuda(objective):
    # per-node feature masks (feature_fraction_bynode < 1) with unlimited depth: the device masks
    # (device_bynode_masks) serve only the complete device tree, so the prefix declines (the masks are drawn on the
    # host per level) and selective growth runs on the host with its keyed masks; the model must not change
    X, y = _data(objective)
    _same(_params(objective, 63, -1, feature_fraction_bynode=0.5), X, y, engaged=False)
    params = {
        "objective": "regression",
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 5,
        "feature_fraction_bynode": 0.5,
    }
    on, _ = _diag({**params, "cuda_plan": "auto"}, 2)
    assert "selective_device_prefix" not in on
    assert "device_levels: host fallback (bynode_outside_bound)" in on
    assert "device_tree: fallback (bynode_outside_bound) route=selective_two_sync\n" in on
