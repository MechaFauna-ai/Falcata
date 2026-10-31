# coding: utf-8
"""The device greedy simulation R3 (cuda_selective_r3) must reproduce the host's selective levels in shadow mode.

Under FALCATA_DEBUG=r3shadow every level of the host's selective loop (the first binding level the selective device
levels hand over, and every level after it) also runs R3 on the device from the same state, and the learner compares
the selection (pop order), the classic leaf numbers, the collapse order, the freed leaves, the free list and the new
records (links, dead flags, gains as bit patterns) and the frontier (leaves, gains) with the host's. A mismatch prints a `r3_shadow: MISMATCH` line; each tree prints one `r3_shadow:` line per
level and a summary with the cumulative mismatch count. The model must not change (the host decides; R3 only runs
beside it). The shapes follow the P8a measurement cells (binding-budget 151/10, unbounded-fewbin 1024/-1, fraud
255/-1, 4095/-1) on synthetic data, plus a shape whose leaves tie exactly (duplicated rows), where the host's
simulation must resolve ties (the per-level ties= count). The test hook r3fault corrupts the device result, so a
broken comparison cannot pass silently; plain diag does not run the shadow.
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

_TRAIN = (
    "import sys, json, hashlib, numpy as np, falcata as flc\n"
    "d = np.load(sys.argv[1])\n"
    "p = json.loads(sys.argv[2])\n"
    "b = flc.train(p, flc.Dataset(d['X'], label=d['y'], params=p), num_boost_round=int(sys.argv[3]))\n"
    "lines = [x for x in b.model_to_string().splitlines() if 'cuda_plan' not in x]\n"
    "print(json.dumps({'md5': hashlib.md5('\\n'.join(lines).encode()).hexdigest(), 'trees': b.num_trees()}))\n"
)

_LEVEL = re.compile(r"r3_shadow: tree=\d+ level=\d+ .*")
_FIELD = re.compile(r"(\w+)=(-?[\d.]+)")
_TREE = re.compile(r"r3_shadow: tree=(\d+) levels=(\d+) mismatches=(\d+) cumulative levels=(\d+) mismatches=(\d+)")


def _levels(err):
    # the per-level lines as dicts (the tree summary lines have no level= field)
    return [{k: float(v) for k, v in _FIELD.findall(m.group(0))} for m in _LEVEL.finditer(err)]


def _train(params, X, y, rounds, debug):
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data.npz")
        np.savez(data, X=X, y=y)
        env = {k: v for k, v in os.environ.items() if k != "FALCATA_DEBUG"}
        if debug:
            env["FALCATA_DEBUG"] = debug
        args = [sys.executable, "-c", _TRAIN, data, json.dumps(params), str(rounds)]
        out = subprocess.run(args, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


def _data(kind, rows, cols, seed=3):
    rng = np.random.default_rng(seed)
    if kind == "fewbin":
        X = rng.integers(0, 6, size=(rows, cols)).astype(np.float32)
        z = X[:, 0] - 0.7 * X[:, 1] * (X[:, 2] > 2) + 0.3 * X[:, 3] + rng.normal(scale=0.8, size=rows)
        return X, z.astype(np.float32)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.03] = np.nan
    z = (
        np.nan_to_num(X[:, 0])
        + 0.6 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2])
        + 0.5 * np.sin(2 * np.nan_to_num(X[:, 3]))
    )
    if kind == "imbalanced":
        return X, (z + rng.normal(scale=0.5, size=rows) > 2.2).astype(np.float32)
    if kind == "binary":
        return X, (z + rng.normal(scale=0.5, size=rows) > 0).astype(np.float32)
    if kind == "tied":
        # every row twice, a first feature that splits the copies apart: the two halves grow identical subtrees,
        # so leaves of equal gain meet in the same simulation (the tie-break decides their order)
        half = rows // 2
        Xh = rng.integers(0, 8, size=(half, cols)).astype(np.float32)
        yh = (Xh[:, 1] + Xh[:, 2] * (Xh[:, 3] > 3) + rng.normal(scale=0.5, size=half)).astype(np.float32)
        Xh[:, 0] = 0.0
        X2 = Xh.copy()
        X2[:, 0] = 1.0
        return np.vstack([Xh, X2]), np.concatenate([yh, yh])
    return X, (z + rng.normal(scale=0.3, size=rows)).astype(np.float32)


_CELLS = {
    # binding-budget 151/10: binary, bagging 0.65, feature_fraction 0.55
    "budget": (
        "binary",
        {
            "objective": "binary",
            "num_leaves": 151,
            "max_depth": 10,
            "max_bin": 31,
            "min_data_in_leaf": 35,
            "feature_fraction": 0.55,
            "bagging_fraction": 0.65,
            "bagging_freq": 1,
        },
        30000,
        20,
        12,
    ),
    # unbounded-fewbin 1024/-1: regression on few-valued columns, bagging 0.75
    "fewbin": (
        "fewbin",
        {
            "objective": "regression",
            "num_leaves": 1024,
            "max_depth": -1,
            "max_bin": 31,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.75,
            "bagging_freq": 1,
        },
        60000,
        16,
        6,
    ),
    # fraud 255/-1: imbalanced binary, unlimited depth
    "fraud": (
        "imbalanced",
        {"objective": "binary", "num_leaves": 255, "max_depth": -1, "max_bin": 127, "min_data_in_leaf": 40},
        40000,
        20,
        12,
    ),
    # 4095/-1: a level wider than 1024 leaves, the heap in opted-in shared memory
    "wide": (
        "regression",
        {
            "objective": "regression",
            "num_leaves": 4095,
            "max_depth": -1,
            "max_bin": 63,
            "min_data_in_leaf": 3,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        },
        60000,
        10,
        3,
    ),
    # exact ties across leaves
    "tied": (
        "tied",
        {"objective": "regression", "num_leaves": 63, "max_depth": -1, "max_bin": 15, "min_data_in_leaf": 5},
        8000,
        6,
        6,
    ),
}


def _params(extra):
    return {
        "device_type": "cuda",
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 5,
        **extra,
    }


@_REQUIRES_CUDA
@pytest.mark.parametrize("cell", sorted(_CELLS))
def test_r3_shadow_matches_the_host_simulation_cuda(cell):
    kind, extra, rows, cols, rounds = _CELLS[cell]
    X, y = _data(kind, rows, cols)
    params = _params(extra)
    shadow, err = _train(params, X, y, rounds, "r3shadow")
    plain, plain_err = _train(params, X, y, rounds, "")
    # the shadow never changes the model and runs only under r3shadow
    assert shadow["md5"] == plain["md5"]
    assert "r3_shadow" not in plain_err
    assert "MISMATCH" not in err, [x for x in err.splitlines() if "MISMATCH" in x][:5]
    levels = _levels(err)
    trees = [tuple(int(g) for g in m.groups()) for m in _TREE.finditer(err)]
    assert levels, err[-2000:]
    assert len(trees) == shadow["trees"]
    assert all(t[2] == 0 for t in trees)
    assert trees[-1][3] == len(levels)
    assert trees[-1][4] == 0
    # every tree hands over at a binding level (or completes) and is then shadowed to its final simulation
    binding = [lv for lv in levels if lv["binding"] == 1]
    assert binding
    budget = extra["num_leaves"] - 1
    assert all(lv["reuse"] <= lv["pops"] <= budget for lv in levels)
    if cell == "tied":
        # equal-gain leaves met in a binding simulation: the tie-break decided pops the device reproduced
        assert any(lv["ties"] > 0 for lv in binding), [lv["ties"] for lv in levels]
    if cell in ("fraud", "fewbin", "wide"):
        # displaced splits collapsed
        assert any(lv["collapsed"] > 0 for lv in levels)


@_REQUIRES_CUDA
def test_r3_shadow_reports_a_corrupted_device_result_cuda():
    kind, extra, rows, cols, rounds = _CELLS["fraud"]
    X, y = _data(kind, rows, cols)
    params = _params(extra)
    faulty, err = _train(params, X, y, 2, "r3shadow,r3fault")
    mismatches = [x for x in err.splitlines() if "r3_shadow: MISMATCH" in x]
    levels = _levels(err)
    assert levels
    # every compared level reports its first difference: the corrupted first pop
    assert len(mismatches) == len(levels)
    assert all("field=order pos=0" in x for x in mismatches)
    last = [tuple(int(g) for g in m.groups()) for m in _TREE.finditer(err)][-1]
    assert last[4] == len(levels)
    # the host still decided: the model is the plain one; plain diag runs no shadow
    plain, diag_err = _train(params, X, y, 2, "diag")
    assert faulty["md5"] == plain["md5"]
    assert "r3_shadow" not in diag_err
