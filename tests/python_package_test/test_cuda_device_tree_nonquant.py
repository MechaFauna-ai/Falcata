# coding: utf-8
"""Non-quantized training on the complete device tree (FULL_DEVICE_PLAN P3).

Depth-limited non-quantized trees inside the bound the removed non-quantized graph loop served (max_depth <= 11,
num_leaves <= 2047; single GPU, scalar leaves, no categorical features, no per-node masks) grow as the complete device
tree: its levels construct in 64-bit fixed point and convert in place (cuda_plan key nonquant_deterministic), then run
the non-quantized fix / subtract and level find on the level's live pairs, the graph loop's kernels. The models are the
graph loop's (the lattice's route/r04a and nonquant cells pin their md5s). Shapes outside the bound keep the one-sync
host flow, budget-limited ones the selective one-sync flow; device_tree:off restores the one-sync flow everywhere.

The device tree must be run-to-run deterministic (#122), train the same model with its level graph captured or
enqueued directly, and report every tree as `device_tree: complete`.
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

# (name, objective, rows, cols, params, rounds)
_SHAPES = {
    "reg-63-6": ("regression", 20000, 30, {"num_leaves": 63, "max_depth": 6}, 20),
    "binary-1023-10": ("binary", 30000, 20, {"num_leaves": 1023, "max_depth": 10, "min_data_in_leaf": 2}, 6),
    "multiclass-31-5-bag": (
        "multiclass",
        20000,
        20,
        {"num_leaves": 31, "max_depth": 5, "num_class": 3, "bagging_fraction": 0.7, "bagging_freq": 1},
        8,
    ),
    "compact-255-8": ("regression", 12000, 200, {"num_leaves": 255, "max_depth": 8, "feature_fraction": 0.3}, 10),
    "deep-2047-11": ("regression", 40000, 20, {"num_leaves": 2047, "max_depth": 11, "min_data_in_leaf": 1}, 3),
    "fourbit-l1": ("regression", 20000, 20, {"num_leaves": 127, "max_depth": 7, "max_bin": 15, "objective": "l1"}, 10),
    # no split passes the gain floor: the root's classic search confirms the stump, as in the host flows
    "stump": ("regression", 20000, 30, {"num_leaves": 63, "max_depth": 6, "min_gain_to_split": 1e12}, 2),
}


def _data(objective, rows, cols, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.03] = np.nan
    t = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.standard_normal(rows)
    if objective == "binary":
        return X, (t > 0).astype(np.float32)
    if objective == "multiclass":
        return X, np.digitize(t, [-0.5, 0.5]).astype(np.float32)
    return X, t.astype(np.float32)


def _params(name, plan="auto"):
    objective, _, _, extra, _ = _SHAPES[name]
    return {
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": "none",
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "seed": 7,
        "verbose": -1,
        "cuda_plan": plan,
        **extra,
    }


def _md5(name, plan="auto"):
    objective, rows, cols, _, rounds = _SHAPES[name]
    X, y = _data(objective, rows, cols)
    params = _params(name, plan)
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _subprocess(name, plan):
    """md5 and the FALCATA_DEBUG=diag lines of one training in a fresh process"""
    code = "import runpy,sys; print(runpy.run_path(sys.argv[1])['_md5'](sys.argv[2], sys.argv[3]))"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, name, plan],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()[-1], result.stderr


@_REQUIRES_CUDA
@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_nonquant_device_tree_complete_and_deterministic_cuda(name):
    # three trainings in fresh processes: every tree is a complete device tree, and the model is the same each time
    runs = [_subprocess(name, "auto") for _ in range(3)]
    md5s = {md5 for md5, _ in runs}
    assert len(md5s) == 1, f"{name}: {len(md5s)} different models in 3 trainings"
    _, diag = runs[0]
    rounds = _SHAPES[name][4] * (_SHAPES[name][3].get("num_class", 1))
    assert len(re.findall(r"device_tree: complete depth=", diag)) == rounds, diag[-3000:]
    assert not re.search(r"device_tree: fallback", diag), diag[-3000:]


@_REQUIRES_CUDA
@pytest.mark.parametrize("name", ["reg-63-6", "compact-255-8", "multiclass-31-5-bag"])
def test_nonquant_device_tree_graph_matches_enqueue_cuda(name):
    # the cached level graph (captured on a tree's first key, replayed after) and the direct enqueue: the same model
    assert _md5(name, "auto,device_level_graph:off") == _md5(name, "auto")


@_REQUIRES_CUDA
def test_nonquant_device_tree_off_takes_one_sync_cuda():
    # device_tree:off keeps the one-sync host flow; the removed graph_loop / graph_det keys are accepted and ignored
    # (a warning, silent under verbose=-1): the default plan's complete device tree
    _, diag = _subprocess("reg-63-6", "auto,device_tree:off")
    assert "device_tree: fallback (plan_off) route=one_sync" in diag, diag[-3000:]
    assert "device_tree: complete" not in diag
    removed, diag = _subprocess("reg-63-6", "auto,graph_loop:off,graph_det:on")
    assert "device_tree: complete" in diag
    assert not re.search(r"device_tree: fallback", diag), diag[-3000:]
    assert removed == _subprocess("reg-63-6", "auto")[0]


def _train_diag(params, rounds, X, y):
    """FALCATA_DEBUG=diag stderr and the training-set predictions of one training in a fresh process"""
    code = (
        "import json,sys,numpy as np,falcata as flc; p=json.loads(sys.argv[1]); d=np.load(sys.argv[2]);"
        "b=flc.train(p, flc.Dataset(d['X'], label=d['y'], params=p), num_boost_round=int(sys.argv[3]));"
        "np.save(sys.argv[4], b.predict(d['X']))"
    )
    base = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"nq_{os.getpid()}")
    np.savez(base + ".npz", X=X, y=y)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, json.dumps(params), base + ".npz", str(rounds), base + ".pred.npy"],
            env=dict(os.environ, FALCATA_DEBUG="diag"),
            capture_output=True,
            text=True,
            check=True,
        )
        pred = np.load(base + ".pred.npy")
    finally:
        for path in (base + ".npz", base + ".pred.npy"):
            if os.path.exists(path):
                os.remove(path)
    return result.stderr, pred


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("extra", "line"),
    [
        # one level past the bound: the one-sync host flow
        ({"num_leaves": 4095, "max_depth": 12, "min_data_in_leaf": 1}, "(nonquant_onesync) route=one_sync"),
        # the bound's edge: depth 11 holds, but 2048 leaves are one past num_leaves <= 2047
        ({"num_leaves": 2048, "max_depth": 11, "min_data_in_leaf": 1}, "(nonquant_onesync) route=one_sync"),
        # a binding leaf budget: the selective one-sync flow
        ({"num_leaves": 40, "max_depth": 6}, "(nonquant_selective) route=selective_one_sync"),
        # per-node masks are drawn on the host
        ({"feature_fraction_bynode": 0.5}, "(bynode_host_masks) route=one_sync"),
    ],
)
def test_nonquant_device_tree_fallback_cuda(extra, line):
    X, y = _data("regression", 20000, 30)
    diag, _ = _train_diag({**_params("reg-63-6"), **extra}, 3, X, y)
    assert f"device_tree: fallback {line}\n" in diag, diag[-3000:]
    assert "device_tree: complete" not in diag


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "extra",
    [
        {"data_sample_strategy": "goss"},
        # sequential random-forest trees (forest_batched:off; slot groups keep their own loop, forest_slots)
        {"boosting": "rf", "bagging_fraction": 0.632, "bagging_freq": 1, "cuda_plan": "auto,forest_batched:off"},
        {"linear_tree": True},
    ],
)
def test_nonquant_device_tree_engages_cuda(extra):
    # GOSS, random forests and depth-limited linear trees grow as complete device trees too
    X, y = _data("regression", 20000, 30)
    params = {**_params("reg-63-6"), **extra}
    rounds = 6
    diag, pred = _train_diag(params, rounds, X, y)
    assert len(re.findall(r"device_tree: complete depth=", diag)) == rounds, diag[-3000:]
    assert not re.search(r"device_tree: fallback", diag), diag[-3000:]
    if params.get("linear_tree"):
        # the linear leaf fit is a fixed-order reduction over each leaf's rows in partition order, and both flows
        # leave every leaf's rows in the same order: the one-sync flow's model is the same, coefficients included
        off_diag, off_pred = _train_diag({**params, "cuda_plan": "auto,device_tree:off"}, rounds, X, y)
        assert "device_tree: complete" not in off_diag
        np.testing.assert_array_equal(pred, off_pred)
