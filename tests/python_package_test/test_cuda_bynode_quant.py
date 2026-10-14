# coding: utf-8
"""feature_fraction_bynode and interaction constraints must reach the quantized split finder on CUDA.

With quantized gradients (quant_mode stochastic or fixedpoint) the discretized leaf finder used to read the per-TREE
feature mask for every leaf, so feature_fraction_bynode < 1 (and interaction constraints) trained the model of
feature_fraction_bynode = 1. Per-node masks select the classic per-split loop (the batched level flows are gated off
by select_features_by_node), whose discretized finder now reads each leaf's own mask, as the non-quantized finder
always did. FALCATA_DEBUG=bynode prints, per split, the split leaf, its inner feature and the mask the leaf was
searched with; the subprocess cases read it to check that every chosen feature belongs to its node's sample.
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
QUANT_MODES = ["stochastic", "fixedpoint"]


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
    # every feature informative with the same weight: the best split is close to a tie, so which feature a node
    # splits on depends on the node's sample
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, NUM_FEATURES)).astype(np.float32)
    y = X.sum(axis=1) + 0.5 * rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _params(quant_mode, bynode, seed=1, **extra):
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "quant_mode": quant_mode,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "feature_fraction_bynode": bynode,
        "seed": seed,
        "deterministic": True,
        "verbose": -1,
    }
    params.update(extra)
    return params


def _train(params, X, y, rounds=10):
    ds = flc.Dataset(X, label=y, params={"max_bin": 63, "verbose": -1})
    return flc.train(params, ds, num_boost_round=rounds)


def _md5(booster):
    return hashlib.md5(_strip_parameters(booster.model_to_string()).encode()).hexdigest()


def _root_feature(booster, tree=0):
    return booster.dump_model()["tree_info"][tree]["tree_structure"]["split_feature"]


@pytest.mark.parametrize("quant_mode", QUANT_MODES)
def test_bynode_changes_quantized_model(quant_mode):
    X, y = _data()
    full = _train(_params(quant_mode, 1.0), X, y)
    sampled = _train(_params(quant_mode, 0.3), X, y)
    assert _md5(full) != _md5(sampled)
    pred_full = full.predict(X[:2000])
    pred_sampled = sampled.predict(X[:2000])
    assert np.max(np.abs(pred_full - pred_sampled)) > 1e-6


@pytest.mark.parametrize("quant_mode", QUANT_MODES)
def test_bynode_root_feature_varies_with_seed(quant_mode):
    # Under stochastic rounding the seed alone moves the root (8 seeds gave 5 roots before the fix, bynode 0.3 and
    # 1.0 alike), so a root that varies with the seed does not show the node sample at work. Compare each seed's root
    # with the bynode 1.0 root of the same seed too: before the fix bynode 0.3 trained the bynode 1.0 model, and
    # every seed gave the same root both ways.
    X, y = _data()
    roots = {
        s: (
            _root_feature(_train(_params(quant_mode, 0.3, seed=s), X, y, rounds=1)),
            _root_feature(_train(_params(quant_mode, 1.0, seed=s), X, y, rounds=1)),
        )
        for s in range(8)
    }
    assert len({sampled for sampled, _ in roots.values()}) > 1, roots
    assert any(sampled != full for sampled, full in roots.values()), roots


@pytest.mark.parametrize("quant_mode", QUANT_MODES)
@pytest.mark.parametrize("bynode", [1.0, 0.3])
def test_bynode_quantized_deterministic(quant_mode, bynode):
    X, y = _data()
    a = _train(_params(quant_mode, bynode, seed=3), X, y)
    b = _train(_params(quant_mode, bynode, seed=3), X, y)
    assert _md5(a) == _md5(b)


@pytest.mark.parametrize("quant_mode", QUANT_MODES)
def test_interaction_constraints_quantized(quant_mode):
    # interaction constraints use the same per-node masks: every root-to-leaf path stays inside one group
    X, y = _data()
    groups = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
    booster = _train(_params(quant_mode, 1.0, interaction_constraints=groups), X, y)

    def paths(node, prefix):
        if "split_feature" not in node:
            yield prefix
            return
        here = [*prefix, node["split_feature"]]
        yield from paths(node["left_child"], here)
        yield from paths(node["right_child"], here)

    group_of = {f: g for g, feats in enumerate(groups) for f in feats}
    checked = 0
    for tree in booster.dump_model()["tree_info"]:
        for path in paths(tree["tree_structure"], []):
            assert len({group_of[f] for f in path}) <= 1, path
            checked += 1
    assert checked > 0


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


@pytest.mark.parametrize("quant_mode", [*QUANT_MODES, "none"])
def test_bynode_split_feature_in_node_sample(quant_mode):
    rounds = 5
    cfg = {"rows": 20000, "features": NUM_FEATURES, "rounds": rounds, "params": _params(quant_mode, 0.3)}
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["FALCATA_DEBUG"] = "bynode"
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(cfg)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    lines = [ln for ln in res.stderr.splitlines() if ln.startswith("BYNODE ")]
    assert len(lines) >= rounds * 20, res.stderr[-3000:]
    expected = round(NUM_FEATURES * 0.3)
    masks = set()
    for ln in lines:
        fields = dict(kv.split("=", 1) for kv in ln.split()[1:])
        feat, mask = int(fields["feat"]), fields["mask"]
        assert len(mask) == NUM_FEATURES
        assert mask.count("1") == expected, ln
        assert mask[feat] == "1", ln
        masks.add(mask)
    assert len(masks) > len(lines) // 2
