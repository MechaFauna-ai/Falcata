# coding: utf-8
"""CUDA selective growth must train the same model with and without its device levels.

A binding leaf budget (2^max_depth > num_leaves + 1) or unlimited depth trains with selective grow-then-prune growth.
With device_tree and device_levels on (the default), its levels run on the device while A + S <= num_leaves - 1 (A
applied splits, S valid leaves of the level: dynamic R1, tested on the device per level), where the host's greedy
simulation selects every valid candidate, which is the device level's "split every valid leaf". The first level where
it fails (the first binding level) and everything after stays with the host's selective loop, which continues from the
levels' journal. Levels below floor(log2(num_leaves)) (where R1 cannot fail) are captured unrolled; the host tests R1
for the next level from their readback, and later levels run as a relaunched two-level body. The models must be byte-identical to device_tree:off (the host's selective loop for every level) and to the
host flow (device_levels:off) across objectives, leaf budgets, gain and data gates, bagging, graph capture and pair
chunks; FALCATA_DEBUG=diag reports one `device_tree: selective hand-off at level k (binding|complete)` line per tree,
so the tests also check where the device handed the tree over.
"""

import collections
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

_OFF = ",device_tree:off"


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


def _train_diag(params, X, y, rounds, debug="diag"):
    # one training in a fresh process under FALCATA_DEBUG=diag: the model md5, the tree count and the hand-off lines
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data.npz")
        np.savez(data, X=X, y=y)
        env = dict(os.environ, FALCATA_DEBUG=debug)
        args = [sys.executable, "-c", _TRAIN, data, json.dumps(params), str(rounds)]
        out = subprocess.run(args, env=env, capture_output=True, text=True, check=True)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    return res["md5"], res["trees"], [_handoff(m) for m in _LINE.finditer(out.stderr)]


def _same(
    params,
    X,
    y,
    rounds=8,
    plans=("auto", "auto" + _OFF, "auto,device_levels:off"),
    engaged=True,
    trained=None,
    debug="diag",
):
    # the same model under every plan; the first plan (the default) ran the device levels on every trained tree iff
    # engaged (trained: trees grown when the model keeps fewer, e.g. stumps), the others never did. Returns its lines.
    md5 = {}
    first = []
    for i, plan in enumerate(plans):
        md5[plan], trees, lines = _train_diag({**params, "cuda_plan": plan}, X, y, rounds, debug)
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
    # the root has no valid split: the device levels apply no level and the host loop reports a stump, as before
    # (a gain gate no split passes: the root is searched, so the levels run and apply none)
    X, y = _data("regression", rows=2000)
    # (the booster keeps one constant tree; each of the 3 rounds grew a stump)
    lines = _same(_params("regression", 31, -1, min_gain_to_split=1e9), X, y, rounds=3, trained=3)
    assert all(h.level == 0 and h.kind == "complete" and h.splits == 0 for h in lines), lines


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
    r"device_tree: selective hand-off at level (\d+) \((binding|complete)\) splits=(\d+) leaves=(\d+) "
    r"frontier=(\d+) unrolled=(\d+) bodies=(\d+)"
)
Handoff = collections.namedtuple("Handoff", "level kind splits leaves frontier unrolled bodies")


def _handoff(match):
    level, kind, *rest = match.groups()
    return Handoff(int(level), kind, *(int(v) for v in rest))


def _floor_log2(n):
    return n.bit_length() - 1


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("leaves", "depth", "static_k"),
    [
        (31, -1, 4),
        (63, -1, 5),
        (255, -1, 7),
        (1024, -1, 10),  # an exact fit: 2^10 = num_leaves
        (151, 10, 7),  # a binding budget at a finite depth
        (40, 6, 5),  # depth 6 bounds the levels
    ],
)
def test_selective_device_prefix_engages_cuda(leaves, depth, static_k):
    # static_k: the removed static prefix's levels, min(floor(log2(num_leaves)), max_depth - 1, 10). The unrolled levels
    # are floor(log2(num_leaves)), on which R1 always holds: the dynamic hand-off is never earlier on this data, where
    # every one of those levels splits leaves
    params = {"objective": "regression", "num_leaves": leaves, "max_depth": depth, "min_data_in_leaf": 5}
    rounds = 4
    on, tree_leaves = _diag({**params, "cuda_plan": "auto"}, rounds)
    unrolled = _floor_log2(leaves)
    assert f"selective device levels to the first binding level (dynamic R1): {unrolled} unrolled" in on
    lines = [_handoff(m) for m in _LINE.finditer(on)]
    assert len(lines) == rounds, on
    for h, final_leaves in zip(lines, tree_leaves, strict=True):
        assert h.level >= static_k
        assert h.unrolled == unrolled
        # every level before the hand-off split leaves; a leaf without a valid split stays a dormant leaf
        assert h.leaves == h.splits + 1 <= min(2**h.level, leaves)
        # R1 held on every applied level and failed at the hand-off: the host's selective loop grew the rest
        assert h.kind == "binding"
        assert h.frontier > 0
        assert h.leaves + h.frontier > leaves
        assert final_leaves == leaves
    assert "device_levels: host fallback" not in on
    assert "device_tree: fallback (" not in on
    off, off_leaves = _diag({**params, "cuda_plan": "auto" + _OFF}, rounds)
    assert "selective hand-off" not in off
    assert "device_levels: host fallback (plan_off)" in off
    assert off_leaves == tree_leaves


@_REQUIRES_CUDA
def test_selective_device_prefix_partial_levels_cuda():
    # gates that leave some leaves without a valid split: fewer splits than a full level, one line per tree, the same
    # model as the host flow
    params = {"objective": "regression", "num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 900}
    on, _ = _diag({**params, "cuda_plan": "auto"}, 3)
    lines = [_handoff(m) for m in _LINE.finditer(on)]
    assert len(lines) == 3
    assert any(h.splits + 1 < 2**h.level for h in lines), lines
    X, y = _data("regression")
    _same(_params("regression", 63, -1, min_data_in_leaf=900), X, y)


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["on", "off"])
def test_selective_device_prefix_deeper_than_static_levels_cuda(graph):
    # num_leaves 300 (the removed static prefix: 8 levels) with leaves that run out of rows early: levels past 8 still
    # satisfy R1 (few valid leaves), so the device hands over deeper, or finishes the tree (complete), with the host
    # flow's model. Levels past floor(log2(300)) = 8 run in the relaunched body
    X, y = _data("regression", rows=20000)
    plans = tuple(f"{p},device_level_graph:{graph}" for p in ("auto", "auto" + _OFF, "auto,device_levels:off"))
    lines = _same(_params("regression", 300, -1, min_data_in_leaf=60), X, y, rounds=6, plans=plans)
    assert all(h.unrolled == 8 for h in lines), lines
    assert any(h.level > 8 for h in lines), lines
    assert any(h.bodies > 0 for h in lines), lines


@_REQUIRES_CUDA
def test_selective_device_prefix_body_stops_mid_body_cuda():
    # unbounded depth with few valid leaves per level: the trees grow past the unrolled levels (5 for 63 leaves) in
    # relaunched two-level bodies and stop at a level with no valid leaf, at the select after a body's first level
    # (its second level then has nothing to apply) on some trees and after its second on others, without a binding
    # level (complete)
    X, y = _data("regression", rows=8000)
    lines = _same(_params("regression", 63, -1, min_data_in_leaf=250), X, y, rounds=12)
    deep = [h for h in lines if h.bodies > 0]
    assert deep, lines
    assert all(h.kind == "complete" for h in deep), lines
    # the stop's position in the last body: (level - unrolled) is odd for a stop at the select after a body's first
    # level, even after its second
    assert {(h.level - h.unrolled) % 2 for h in deep} == {0, 1}, lines


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["on", "off"])
def test_selective_device_prefix_pair_chunks_cuda(graph):
    # vramfree caps the free memory the pair chunk plan sees: the selective levels' fix/subtract and find run in chunks
    # of a few pairs (PlanDeviceTreeChunks), the relaunched body too; the same model
    X, y = _data("regression", rows=20000)
    params = _params("regression", 300, -1, min_data_in_leaf=60)
    plans = tuple(f"{p},device_level_graph:{graph}" for p in ("auto", "auto" + _OFF, "auto,device_levels:off"))
    lines = _same(params, X, y, rounds=4, plans=plans, debug="diag,vramfree=4")
    assert any(h.bodies > 0 for h in lines), lines
    env = dict(os.environ, FALCATA_DEBUG="diag,vramfree=4")
    args = [sys.executable, "-c", _ENGAGE, json.dumps({**params, "cuda_plan": "auto"}), "1"]
    err = subprocess.run(args, env=env, capture_output=True, text=True, check=True).stderr
    chunks = re.search(r"selective device levels .*?; (\d+) pair chunks", err)
    assert chunks is not None, err[-3000:]
    assert int(chunks.group(1)) > 1, err[-3000:]


@_REQUIRES_CUDA
def test_selective_device_prefix_declines_forced_splits_cuda(tmp_path):
    # forced splits take the classic per-split loop (no selective growth): the device levels decline and the model is
    # unchanged
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
    assert "selective hand-off" not in on
    X, y = _data("regression")
    _same(_params("regression", 63, -1, forcedsplits_filename=str(forced)), X, y, rounds=4, engaged=False)


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["binary", "multiclass"])
def test_selective_device_prefix_bynode_same_model_cuda(objective):
    # per-node feature masks (feature_fraction_bynode < 1) with unlimited depth: the device masks
    # (device_bynode_masks) serve only the complete device tree, so the device levels decline (the masks are drawn on
    # the host per level) and selective growth runs on the host with its keyed masks; the model must not change
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
    assert "selective hand-off" not in on
    assert "device_levels: host fallback (bynode_outside_bound)" in on
    assert "device_tree: fallback (bynode_outside_bound) route=selective_two_sync\n" in on


# the selective levels record no device tree, select in the fused sync tail from registers, rebuild the host records
# in place and bound their partition launches; the default must still give the host flow's model where the levels
# reach every select path (with _SHAPES: leaves without a valid split, bagging, exact fits)
@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize(
    ("leaves", "depth", "extra"),
    [
        (63, 10, {"min_data_in_leaf": 150}),  # binding budget, leaves without a valid split on the early levels
        (63, 10, {"bagging_fraction": 0.6, "bagging_freq": 1}),  # binding budget under row bagging
    ],
)
def test_selective_device_prefix_trimmed_levels_same_model_cuda(objective, leaves, depth, extra):
    X, y = _data(objective)
    _same(_params(objective, leaves, depth, **extra), X, y)


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["on", "off"])
def test_selective_device_prefix_wide_same_model_cuda(graph):
    # level 9 with over 256 splits: the fused select's two splits per thread of its 256-thread block. Levels 0-8 hold
    # at most 511 splits, so more than 767 splits by level 10 put more than 256 on level 9 (this data: ~810-830)
    X, y = _data("regression", rows=100000)
    plans = tuple(f"{p},device_level_graph:{graph}" for p in ("auto", "auto" + _OFF, "auto,device_levels:off"))
    lines = _same(_params("regression", 1024, -1), X, y, rounds=3, plans=plans)
    assert all(h.level >= 10 and h.splits > 767 for h in lines), lines


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["on", "off"])
def test_selective_device_prefix_beyond_register_select_cuda(graph):
    # num_leaves 4095: levels past 10 hold more than 1024 leaves, so their select runs as the standalone shared-memory
    # block (the register-batched one fused into the sync covers 1024 leaves); the same model
    X, y = _data("regression", rows=200000)
    plans = tuple(f"{p},device_level_graph:{graph}" for p in ("auto", "auto" + _OFF))
    lines = _same(_params("regression", 4095, -1, min_data_in_leaf=20), X, y, rounds=2, plans=plans)
    assert all(h.unrolled == 11 for h in lines), lines
    assert any(h.level >= 11 for h in lines), lines
