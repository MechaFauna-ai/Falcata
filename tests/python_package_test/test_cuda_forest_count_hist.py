# coding: utf-8
"""A random forest's histograms from counts (cuda_plan key forest_count_hist).

A forest's gradients are constant for the whole training (computed once from the init score) and, per class tree,
take few values with one hessian value. Where the non-quantized learner would run the deterministic fp64 construct,
forest_count_hist (default on) runs a count construct: one byte per row codes the row's gradient value, integer
atomics count per bin the rows and their code sum, and a convert kernel stores the fp64 (gradient, hessian) pair. A
check when the codes are built (constant hessian, at most 256 values, the exactness bound of
docs/design/forest-count-histograms.md) makes the stored pair the exact sum the deterministic construct's fixed-order
double merge produces, so the histograms, the splits and the model are the same: md5-identical to the key-off model
wherever the key-off learner runs the deterministic construct. Where it runs the float atomic construct instead (the
compact column view of feature_fraction < 1; levels with more pairs than the deterministic slot slab holds, which the
shapes here do not reach) its sums round and its model need not be run-to-run stable; the count construct's are exact
and deterministic, and its predictions agree with the key-off ones to rounding.
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

OFF = "auto,forest_count_hist:off,graph_loop:off"
ON = "auto,graph_loop:off"


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


def _md5(booster):
    return hashlib.md5(_strip_parameters(booster.model_to_string()).encode()).hexdigest()


def _data(kind, rows=30011, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, 16)).astype(np.float32)
    X[:, :4] = np.round(X[:, :4] * 2)  # few-bin columns
    X[rng.random(X.shape) < 0.03] = np.nan
    z = np.nan_to_num(X[:, 0]) - 0.5 * np.nan_to_num(X[:, 1]) + np.sin(np.nan_to_num(X[:, 5])) + 0.2 * X[:, 6:].sum(1)
    z = np.nan_to_num(z) + 0.3 * rng.standard_normal(rows)
    if kind == "binary":
        y = (z > 0).astype(np.float32)
    elif kind == "imbalanced":
        y = (z > 2.2).astype(np.float32)
    elif kind == "multiclass":
        y = np.digitize(z, [-1.5, -0.5, 0.5, 1.5]).astype(np.float32)
    elif kind == "lattice":
        # five label values (Numerai-like targets 0, 0.25, ..., 1)
        y = (np.digitize(z, [-1.0, -0.3, 0.3, 1.0]) * 0.25).astype(np.float32)
    else:
        y = (z + rng.standard_exponential(rows)).astype(np.float32)
    return X, y


def _params(objective, shape, **extra):
    params = {
        "boosting": "rf",
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": "none",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "min_data_in_leaf": 1,
        "feature_fraction_bynode": 0.25,
        "max_bin": 63,
        "metric": "None",
        "seed": 7,
        "verbose": -1,
        # selective grow-then-prune (budget-limited), the depth-limited level prefix, or a standard shape
        **{
            "selective": {"num_leaves": 255},
            "depth": {"num_leaves": 4095, "max_depth": 8},
            "std": {"num_leaves": 31, "min_data_in_leaf": 20},
        }[shape],
    }
    if objective in ("multiclass", "multiclassova"):
        params["num_class"] = 5
    params.update(extra)
    return params


def _train(params, plan, X, y, rounds, weight=None):
    ds = flc.Dataset(X, label=y, weight=weight, params={"max_bin": 63, "verbose": -1}, free_raw_data=False)
    return flc.train({**params, "cuda_plan": plan}, ds, num_boost_round=rounds)


QUALIFYING = [
    ("binary", "binary"),
    ("binary", "imbalanced"),
    ("multiclass", "multiclass"),
    ("multiclassova", "multiclass"),
    ("regression", "lattice"),
    ("regression", "binary"),
]


@pytest.mark.parametrize(("objective", "kind"), QUALIFYING)
@pytest.mark.parametrize("shape", ["selective", "depth", "std"])
def test_count_hist_same_model(objective, kind, shape):
    # every qualifying objective: the key-on model is the key-off model (same histogram bits), run to run
    X, y = _data(kind)
    params = _params(objective, shape)
    rounds = 6
    ref = _train(params, OFF, X, y, rounds)
    got = _train(params, ON, X, y, rounds)
    again = _train(params, ON, X, y, rounds)
    name = f"{objective}/{kind}/{shape}"
    assert _md5(got) == _md5(ref), name
    assert _md5(again) == _md5(got), name
    np.testing.assert_array_equal(got.predict(X[:3000]), ref.predict(X[:3000]), err_msg=name)


@pytest.mark.parametrize(
    "extra",
    [
        {"feature_fraction_bynode": 1.0},  # no per-node sampling
        {"cuda_plan_extra": "forest_batched:off"},  # one tree after another (stage-1 learner)
        {"cuda_plan_extra": "forest_group:3"},
        {"max_bin": 255},
    ],
    ids=["bynode1", "sequential", "group3", "bin255"],
)
def test_count_hist_same_model_variants(extra):
    X, y = _data("multiclass")
    extra = dict(extra)
    plan_extra = extra.pop("cuda_plan_extra", "")
    params = _params("multiclass", "selective", **extra)
    plan_off = OFF + ("," + plan_extra if plan_extra else "")
    plan_on = ON + ("," + plan_extra if plan_extra else "")
    ref = _train(params, plan_off, X, y, 4)
    got = _train(params, plan_on, X, y, 4)
    assert _md5(got) == _md5(ref), extra
    np.testing.assert_array_equal(got.predict(X[:3000]), ref.predict(X[:3000]))


@pytest.mark.parametrize(
    ("objective", "extra"),
    [
        ("binary", {}),
        ("multiclass", {}),
        # no row sampling (rf then needs feature_fraction < 1)
        ("binary", {"bagging_fraction": 1.0, "bagging_freq": 0, "feature_fraction": 0.8}),
    ],
    ids=["binary", "multiclass", "nobag"],
)
def test_count_hist_compact_view_deterministic(objective, extra):
    # feature_fraction < 1 puts the tree on the compact column view, whose key-off batched construct is the float
    # atomic kernel (not run-to-run stable); the count construct serves it with exact integer sums: the key-on model
    # is run-to-run identical, and its predictions are the key-off ones to rounding (most rows), up to the occasional
    # split the key-off rounding moves
    X, y = _data("binary" if objective == "binary" else "multiclass")
    params = _params(objective, "selective", **{"feature_fraction": 0.5, "feature_fraction_bynode": 1.0, **extra})
    got = _train(params, ON, X, y, 6)
    again = _train(params, ON, X, y, 6)
    ref = _train(params, OFF, X, y, 6)
    assert _md5(again) == _md5(got)
    np.testing.assert_array_equal(again.predict(X[:3000]), got.predict(X[:3000]))
    diff = np.abs(got.predict(X[:3000]) - ref.predict(X[:3000]))
    assert float(np.median(diff)) < 1e-5, float(np.median(diff))
    assert float(np.mean(diff)) < 1e-2, float(np.mean(diff))


_CHILD = """
import json, sys
import numpy as np
import falcata as flc
cfg = json.loads(sys.argv[1])
rng = np.random.default_rng(7)
X = rng.standard_normal((20000, 12)).astype(np.float32)
X[:, :3] = np.round(X[:, :3])
z = X[:, 0] + 0.3 * rng.standard_normal(20000)
kind = cfg.pop("_labels")
if kind == "binary":
    y = (z > 0).astype(np.float32)
elif kind == "lattice":
    y = (np.digitize(z, [-1.0, 0.0, 1.0]) * 0.25).astype(np.float32)
elif kind == "multiclass":
    y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
else:
    y = z.astype(np.float32)
w = rng.uniform(0.5, 2.0, 20000).astype(np.float32) if cfg.pop("_weights", False) else None
ds = flc.Dataset(X, label=y, weight=w, params={"max_bin": 63, "verbose": -1})
flc.train(cfg, ds, num_boost_round=4)
"""


def _diag(params, labels="binary", weights=False):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["FALCATA_DEBUG"] = "diag"
    cfg = {**params, "_labels": labels, "_weights": weights}
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(cfg)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return [line for line in res.stderr.splitlines() if line.startswith("[count-hist-diag]")]


def _small(objective, **extra):
    params = {
        "boosting": "rf",
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 63,
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "feature_fraction_bynode": 0.5,
        "quant_mode": "none",
        "metric": "None",
        "seed": 3,
        "verbose": -1,
        "cuda_plan": "auto,graph_loop:off",
    }
    params.update(extra)
    return params


@pytest.mark.parametrize(
    ("params", "labels", "weights", "expected"),
    [
        (_small("binary"), "binary", False, "count construct"),
        (_small("multiclassova", num_class=4), "multiclass", False, "count construct"),
        (_small("multiclass", num_class=4), "multiclass", False, "count construct"),
        (_small("regression"), "lattice", False, "count construct"),
        # not a forest objective of the key: the deterministic construct
        (_small("huber"), "lattice", False, "deterministic construct: objective huber"),
        # weights: many gradient values, no constant hessian
        (_small("binary"), "binary", True, "deterministic construct: weights"),
        # continuous labels: thousands of gradient values
        (_small("regression"), "continuous", False, "deterministic construct: more than 256 gradient values"),
    ],
    ids=["binary", "multiclassova", "multiclass", "regression", "huber", "weights", "continuous"],
)
def test_count_hist_engagement(params, labels, weights, expected):
    lines = _diag(params, labels=labels, weights=weights)
    assert any(line.startswith("[count-hist-diag] " + expected) for line in lines), lines


@pytest.mark.parametrize(
    "params",
    [
        _small("binary", cuda_plan="auto,graph_loop:off,forest_count_hist:off"),
        _small("binary", quant_mode="stochastic"),
        _small("binary", boosting="gbdt"),
    ],
    ids=["key-off", "quantized", "gbdt"],
)
def test_count_hist_not_engaged(params):
    # key off, quantized gradients (their own integer pipeline) and gbdt never take the count construct
    assert _diag(params) == []


@pytest.mark.parametrize(
    ("objective", "labels", "weights"),
    [("huber", "lattice", False), ("binary", "binary", True), ("regression", "continuous", False)],
    ids=["huber", "weights", "continuous"],
)
def test_count_hist_non_qualifying_keeps_the_old_path(objective, labels, weights):
    # a tree that does not qualify runs the deterministic construct: the same model with the key on and off
    rng = np.random.default_rng(11)
    X, y = _data("lattice" if labels == "lattice" else ("binary" if labels == "binary" else "regression"))
    w = rng.uniform(0.5, 2.0, len(y)).astype(np.float32) if weights else None
    params = _params(objective, "selective")
    ref = _train(params, OFF, X, y, 4, weight=w)
    got = _train(params, ON, X, y, 4, weight=w)
    assert _md5(got) == _md5(ref)
