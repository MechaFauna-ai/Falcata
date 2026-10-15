# coding: utf-8
"""Per-node feature sampling in the level-batched flows on CUDA (cuda_plan key level_bynode_masks).

feature_fraction_bynode < 1 used to keep every level-batched flow off (hybrid prefix, selective grow-then-prune,
one-sync) and train on the per-split loop, 5-40x slower per round on deep trees. Under level_bynode_masks (default on,
without interaction constraints) a node's mask is drawn on the host from a generator of its own, seeded by
(feature_fraction_seed, the learner's tree count, the node's path from the root), each level uploads one mask row per
searched leaf, and the level finders read their leaf's row. The masks then depend on neither the flow nor the order or
number of draws before them, so the models are run-to-run deterministic under quantized gradients, but they are not
the models of the shared-stream draws (key off: the per-split loop and ColSampler::GetByNode's stream, as before):
same sample size per node, other masks, a holdout metric within noise. bynode = 1 draws nothing: md5-identical.
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

NUM_FEATURES = 30
BYNODE = 0.3
# tree shape -> the level flow it runs with per-node masks, per quantization (FALCATA_DEBUG=diag's [bynode-diag])
SHAPES = {
    # budget-limited (num_leaves << 2^depth): selective grow-then-prune; quantized two-sync, non-quantized one-sync
    "selective": ({"num_leaves": 31}, {"stochastic": "selective-two-sync", "none": "selective-one-sync"}),
    # depth-limited (2^max_depth <= num_leaves + 1): the exact-fit level prefix, one-sync without quantization
    "prefix": ({"num_leaves": 31, "max_depth": 5}, {"stochastic": "prefix-two-sync", "none": "prefix-one-sync"}),
}
ON = "auto"
OFF = "auto,level_bynode_masks:off"


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


def _data(rows=20000, seed=7):
    # every feature informative with the same weight: which feature a node splits on depends on the node's sample
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, NUM_FEATURES)).astype(np.float32)
    y = X.sum(axis=1) + 0.5 * rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _params(quant_mode, bynode, shape, plan=ON, seed=3, boosting="gbdt", **extra):
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "boosting": boosting,
        "quant_mode": quant_mode,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "feature_fraction_bynode": bynode,
        "seed": seed,
        "cuda_plan": plan,
        "verbose": -1,
        **SHAPES[shape][0],
    }
    if boosting == "rf":
        params.update(bagging_fraction=0.632, bagging_freq=1)
    params.update(extra)
    return params


def _train(params, X, y, rounds=10):
    ds = flc.Dataset(X, label=y, params={"max_bin": 63, "verbose": -1})
    return flc.train(params, ds, num_boost_round=rounds)


def _md5(booster):
    return hashlib.md5(_strip_parameters(booster.model_to_string()).encode()).hexdigest()


_CHILD = r"""
import json, sys
import numpy as np
import falcata as flc
cfg = json.loads(sys.argv[1])
rng = np.random.default_rng(7)
X = rng.standard_normal((cfg["rows"], cfg["features"])).astype(np.float32)
y = (X.sum(axis=1) + 0.5 * rng.standard_normal(cfg["rows"])).astype(np.float32)
ds = flc.Dataset(X, label=y, params={"max_bin": 63, "verbose": -1})
flc.train(cfg["params"], ds, num_boost_round=cfg["rounds"])
"""


def _child_stderr(params, rounds=5):
    cfg = {"rows": 20000, "features": NUM_FEATURES, "rounds": rounds, "params": params}
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["FALCATA_DEBUG"] = "diag,bynode"
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(cfg)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stderr


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("quant_mode", ["stochastic", "none"])
def test_level_flow_takes_node_masks(quant_mode, shape):
    # the level flow runs (not the per-split loop), and every applied split's feature is in its node's sample
    rounds = 5
    err = _child_stderr(_params(quant_mode, BYNODE, shape), rounds=rounds)
    assert f"[bynode-diag] masks=keyed flow={SHAPES[shape][1][quant_mode]}\n" in err, err[-3000:]
    lines = [ln for ln in err.splitlines() if ln.startswith("BYNODE ")]
    assert len(lines) >= rounds * 20, err[-3000:]
    expected = round(NUM_FEATURES * BYNODE)
    masks = set()
    for ln in lines:
        fields = dict(kv.split("=", 1) for kv in ln.split()[1:])
        feat, mask = int(fields["feat"]), fields["mask"]
        assert len(mask) == NUM_FEATURES
        assert mask.count("1") == expected, ln
        assert mask[feat] == "1", ln
        masks.add(mask)
    assert len(masks) > len(lines) // 2


@pytest.mark.parametrize("shape", list(SHAPES))
def test_key_off_keeps_the_per_split_loop(shape):
    err = _child_stderr(_params("stochastic", BYNODE, shape, plan=OFF), rounds=2)
    assert "[bynode-diag] masks=shared-stream flow=per-split\n" in err, err[-3000:]


def test_interaction_constraints_keep_the_per_split_loop():
    groups = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
    params = _params("stochastic", 1.0, "selective", interaction_constraints=groups)
    err = _child_stderr(params, rounds=2)
    assert "[bynode-diag] masks=shared-stream flow=per-split\n" in err, err[-3000:]


@pytest.mark.parametrize("boosting", ["gbdt", "rf"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint"])
def test_level_bynode_deterministic(quant_mode, shape, boosting):
    X, y = _data()
    a = _train(_params(quant_mode, BYNODE, shape, boosting=boosting), X, y)
    b = _train(_params(quant_mode, BYNODE, shape, boosting=boosting), X, y)
    assert _md5(a) == _md5(b)


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint"])
def test_masks_do_not_depend_on_the_flow(quant_mode, shape):
    # a node's mask is a function of (seed, tree, node path): the per-split loop (hybrid:off) draws the same masks as
    # the level flow, and under quantized gradients (integer histograms) grows the same trees. Selective growth
    # renumbers its leaves in the per-split loop's order (the same model text); the depth-limited prefix numbers them
    # by level (another text, the same trees: identical predictions)
    X, y = _data()
    level = _train(_params(quant_mode, BYNODE, shape), X, y)
    per_split = _train(_params(quant_mode, BYNODE, shape, plan=ON + ",hybrid:off"), X, y)
    if shape == "selective":
        assert _md5(level) == _md5(per_split)
    Xte, _ = _data(rows=5000, seed=11)
    np.testing.assert_array_equal(level.predict(Xte), per_split.predict(Xte))


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("quant_mode", ["stochastic", "none"])
def test_key_on_and_off_same_metric_other_model(quant_mode, shape):
    # other masks (keyed vs the shared stream), the same sampling: a different model with the same holdout error
    X, y = _data()
    Xte, yte = _data(rows=5000, seed=11)
    on = _train(_params(quant_mode, BYNODE, shape), X, y, rounds=30)
    off = _train(_params(quant_mode, BYNODE, shape, plan=OFF), X, y, rounds=30)
    assert _md5(on) != _md5(off)
    mse_on = float(np.mean((on.predict(Xte) - yte) ** 2))
    mse_off = float(np.mean((off.predict(Xte) - yte) ** 2))
    assert abs(mse_on - mse_off) <= 0.05 * mse_off, (mse_on, mse_off)


@pytest.mark.parametrize("boosting", ["gbdt", "rf"])
@pytest.mark.parametrize(
    ("quant_mode", "shape"),
    [
        ("stochastic", "selective"),
        ("stochastic", "prefix"),
        ("fixedpoint", "selective"),
        ("fixedpoint", "prefix"),
        # the non-quantized depth-limited prefix (one-sync) is not run-to-run md5-stable at bynode = 1 with the key on
        # or off alike (float atomics; the lattice's dense/nonquant cells), so only its selective shape is compared
        ("none", "selective"),
    ],
)
def test_bynode_one_identical_key_on_and_off(quant_mode, shape, boosting):
    # bynode = 1 draws no node mask: the key changes nothing (quant none: deterministic kernels for run-to-run md5)
    X, y = _data()
    extra = {"deterministic": True} if quant_mode == "none" else {}
    on = _train(_params(quant_mode, 1.0, shape, boosting=boosting, **extra), X, y)
    off = _train(_params(quant_mode, 1.0, shape, plan=OFF, boosting=boosting, **extra), X, y)
    assert _md5(on) == _md5(off)


def test_keyed_masks_follow_the_seed():
    # the feature_fraction_seed enters the key: other seeds, other roots
    X, y = _data()
    roots = {
        s: _train(_params("stochastic", BYNODE, "selective", seed=s), X, y, rounds=1).dump_model()["tree_info"][0][
            "tree_structure"
        ]["split_feature"]
        for s in range(8)
    }
    assert len(set(roots.values())) > 1, roots
