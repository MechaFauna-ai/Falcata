# coding: utf-8
"""A random forest on CUDA grows the trees of several rounds concurrently (cuda_plan key forest_batched).

Under forest_batched (default on for boosting=rf on one CUDA device with the device objective) the booster grows the
class trees of one or more rounds at once, one tree per learner ("slot") and per host thread (CUDAForestTrainer). Each
tree gets what sequential training gives it, in tree order: its round's bag, its class's gradients, the next per-tree
feature_fraction sample of the booster learner's sampler and the tree counters of the bynode node keys and of the
stochastic rounding noise. The forest is therefore the forest of the sequential learner (key off), for every group
size (forest_group), and the readers of the scores (validation and training metrics, early stopping, rollback, a
parameter reset, continued training) see what they see without the key. Every case compares the models (parameter
block excepted), the predictions and the evaluation records with the key off.

Quantized training is run-to-run deterministic, so every quantized case compares md5s; without quantization the cases
use shapes whose fp64 level loop is run-to-run stable here (the graph loop off: forest slots never take it).
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

OFF = "auto,forest_batched:off"
GROUPS = {
    "auto": "auto",
    "g1": "auto,forest_group:1",
    "g3": "auto,forest_group:3",
    "g40": "auto,forest_group:40",
}


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
    X[rng.random(X.shape) < 0.03] = np.nan
    z = np.nan_to_num(X[:, 0]) - 0.5 * np.nan_to_num(X[:, 1]) + np.sin(np.nan_to_num(X[:, 2])) + 0.2 * X[:, 3:].sum(1)
    z = np.nan_to_num(z) + 0.3 * rng.standard_normal(rows)
    if kind == "binary":
        y = (z > 0).astype(np.float32)
    elif kind == "multiclass":
        y = np.digitize(z, [-1.5, -0.5, 0.5, 1.5]).astype(np.float32)
    else:
        y = (z + rng.standard_exponential(rows)).astype(np.float32)
    return X, y


def _params(objective, quant_mode, shape, **extra):
    params = {
        "boosting": "rf",
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": quant_mode,
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "min_data_in_leaf": 5,
        "max_bin": 63,
        "metric": "None",
        "seed": 7,
        "verbose": -1,
        # selective grow-then-prune (budget-limited) or the depth-limited level prefix
        **({"num_leaves": 127} if shape == "selective" else {"num_leaves": 63, "max_depth": 6}),
    }
    if objective == "multiclass":
        params["num_class"] = 5
    params.update(extra)
    return params


def _train(params, plan, X, y, rounds, valid=None, eval_train=False, early_stopping=None, callbacks=()):
    if params["quant_mode"] == "none":
        plan += ",graph_loop:off"
    ds = flc.Dataset(X, label=y, params={"max_bin": 63, "verbose": -1}, free_raw_data=False)
    valid_sets, valid_names = [], []
    if eval_train:
        valid_sets.append(ds)
        valid_names.append("train")
    if valid is not None:
        valid_sets.append(flc.Dataset(valid[0], label=valid[1], reference=ds))
        valid_names.append("valid")
    evals = {}
    cbs = list(callbacks)
    if valid_sets:
        cbs.append(flc.record_evaluation(evals))
    if early_stopping is not None:
        cbs.append(flc.early_stopping(early_stopping, verbose=False))
    booster = flc.train(
        {**params, "cuda_plan": plan},
        ds,
        num_boost_round=rounds,
        valid_sets=valid_sets or None,
        valid_names=valid_names or None,
        callbacks=cbs,
        keep_training_booster=True,
    )
    return booster, evals


def _assert_same(got, want, X, name):
    (b, evals), (ref, ref_evals) = got, want
    assert _md5(b) == _md5(ref), name
    np.testing.assert_array_equal(b.predict(X[:3000]), ref.predict(X[:3000]), err_msg=name)
    assert b.current_iteration() == ref.current_iteration(), name
    np.testing.assert_array_equal(
        np.array(b._Booster__inner_predict(data_idx=0)), np.array(ref._Booster__inner_predict(data_idx=0)), err_msg=name
    )
    # the CUDA metrics reduce with atomics: the same scores give the same metric to the last bits or so
    assert evals.keys() == ref_evals.keys(), name
    for data_name, metrics in ref_evals.items():
        for metric, values in metrics.items():
            np.testing.assert_allclose(evals[data_name][metric], values, rtol=1e-12, err_msg=f"{name} {data_name}")


@pytest.mark.parametrize(
    ("objective", "kind", "quant_mode", "shape", "extra"),
    [
        ("multiclass", "multiclass", "stochastic", "selective", {"feature_fraction_bynode": 0.4}),
        ("multiclass", "multiclass", "stochastic", "prefix", {"feature_fraction": 0.6}),
        # error feedback: one round per group (the class trees of a round), the residual slot per class
        ("multiclass", "multiclass", "fixedpoint", "selective", {}),
        ("multiclass", "multiclass", "none", "selective", {"feature_fraction_bynode": 0.4}),
        ("binary", "binary", "stochastic", "selective", {"feature_fraction": 0.6, "feature_fraction_bynode": 0.5}),
        ("binary", "binary", "fixedpoint", "prefix", {"feature_fraction_bynode": 0.4}),
        ("binary", "binary", "none", "prefix", {}),
        ("regression", "reg", "stochastic", "prefix", {"feature_fraction_bynode": 0.4}),
        (
            "regression",
            "reg",
            "stochastic",
            "selective",
            {"bagging_fraction": 1.0, "bagging_freq": 0, "feature_fraction": 0.5},
        ),
        # leaf renewal: the tree's own slot's partition renews it
        ("regression_l1", "reg", "stochastic", "selective", {}),
        ("quantile", "reg", "fixedpoint", "prefix", {"alpha": 0.7}),
    ],
)
def test_forest_batched_same_forest_cuda(objective, kind, quant_mode, shape, extra):
    X, y = _data(kind)
    params = _params(objective, quant_mode, shape, **extra)
    want = _train(params, OFF, X, y, 13)
    for name, plan in GROUPS.items():
        _assert_same(_train(params, plan, X, y, 13), want, X, name)


def test_forest_batched_readers_cuda():
    # validation and training metrics every round (the per-tree train score update with the tree's own slot and bag),
    # early stopping (the rounds grown ahead are dropped)
    X, y = _data("multiclass")
    Xv, yv = _data("multiclass", rows=7001, seed=11)
    params = _params("multiclass", "stochastic", "selective", metric="multi_logloss", feature_fraction=0.7)
    want = _train(params, OFF, X, y, 12, valid=(Xv, yv), eval_train=True)
    for name, plan in GROUPS.items():
        _assert_same(_train(params, plan, X, y, 12, valid=(Xv, yv), eval_train=True), want, X, name)
    want = _train(params, OFF, X, y, 60, valid=(Xv, yv), early_stopping=3)
    assert want[0].current_iteration() < 60  # the case stops early
    for name, plan in GROUPS.items():
        _assert_same(_train(params, plan, X, y, 60, valid=(Xv, yv), early_stopping=3), want, X, name)


@pytest.mark.parametrize(
    ("objective", "kind", "quant_mode"),
    [
        ("binary", "binary", "stochastic"),
        # error feedback: the class residuals live in the slots, which a rollback keeps
        ("multiclass", "multiclass", "fixedpoint"),
    ],
)
def test_forest_batched_reset_and_rollback_cuda(objective, kind, quant_mode):
    # a parameter reset mid-training (the rounds grown ahead were grown under the old parameters: dropped, and the
    # booster learner's tree counters and feature sampler go back to the first dropped tree's), then rollbacks
    X, y = _data(kind)
    params = _params(objective, quant_mode, "selective", feature_fraction=0.6, feature_fraction_bynode=0.5)
    reset = flc.reset_parameter(min_data_in_leaf=[5] * 4 + [40] * 8)
    results = {}
    for name, plan in {"off": OFF, **GROUPS}.items():
        booster, _ = _train(params, plan, X, y, 12, callbacks=[reset])
        booster.rollback_one_iter()
        booster.rollback_one_iter()
        for _ in range(3):
            booster.update()
        results[name] = (
            _md5(booster),
            booster.predict(X[:3000]),
            np.array(booster._Booster__inner_predict(data_idx=0)),
        )
    for name, (model, pred, train) in results.items():
        assert model == results["off"][0], name
        np.testing.assert_array_equal(pred, results["off"][1], err_msg=name)
        np.testing.assert_array_equal(train, results["off"][2], err_msg=name)


_CHILD = r"""
import json, sys
import numpy as np
import falcata as flc
cfg = json.loads(sys.argv[1])
rng = np.random.default_rng(7)
X = rng.standard_normal((20000, 12)).astype(np.float32)
y = (X[:, 0] + 0.3 * rng.standard_normal(20000) > 0).astype(np.float32)
ds = flc.Dataset(X, label=y, params={"max_bin": 63, "verbose": -1})
flc.train(cfg, ds, num_boost_round=12)
"""


def _diag(params):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["FALCATA_DEBUG"] = "diag"
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(params)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return [line for line in res.stderr.splitlines() if line.startswith("[forest-diag] rounds")]


@pytest.mark.parametrize(
    ("extra", "engaged"),
    [
        ({}, True),
        ({"cuda_plan": "auto,forest_group:4"}, True),
        ({"cuda_plan": "auto,forest_batched:off"}, False),
        # the sampler carries a round's bag into the next ones / the host sampler's stream
        ({"bagging_freq": 2}, False),
        ({"pos_bagging_fraction": 0.8}, False),
        ({"data_sample_strategy": "goss", "bagging_freq": 0, "bagging_fraction": 1.0}, False),
        # per-node draws from the shared stream
        ({"feature_fraction_bynode": 0.5, "interaction_constraints": [[0, 1, 2], [3, 4, 5]]}, False),
        ({"linear_tree": True}, False),
    ],
)
def test_forest_batched_engages_cuda(extra, engaged):
    params = {
        "boosting": "rf",
        "device_type": "cuda",
        "objective": "binary",
        "quant_mode": "stochastic",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "num_leaves": 31,
        "verbose": -1,
        **extra,
    }
    lines = _diag(params)
    assert bool(lines) == engaged, lines
    if "forest_group:4" in params.get("cuda_plan", ""):
        # 12 rounds in groups of 4 (one tree per round)
        assert len(lines) == 3, lines
        assert all(" 4 trees on 4 slots" in line for line in lines), lines
