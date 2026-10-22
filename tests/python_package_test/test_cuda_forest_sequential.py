# coding: utf-8
"""A random forest on CUDA trains its trees one after another on the booster's learner.

Until 1.2 a forest group of two or more trees (every multiclass round, forest_group N, the auto size) trained
concurrently, one tree per learner ("slot") and per host thread (CUDAForestTrainer, cuda_plan keys forest_batched,
forest_group and forest_single_job_device). Each tree got what sequential training gives it, so the forests were the
sequential ones; the slots are removed and every tree trains on the booster's learner in tree order, taking what a
gbdt tree takes: the complete device tree where its bound holds (device per-node masks included), the device_levels
prefix, the selective device prefix, and the host level loop with the forest count construct for non-quantized
forests (which never take the graph loop, as the slots never did).

The models must not change: every case compares the model md5 (parameter block excepted) with the md5 the same case
gave on master 5bec2574 (slot mode, its default plan; RTX 5090, CUDA 13.3) across objectives, quantization modes,
per-node and per-tree sampling, bounded, unbounded and leaf-budget shapes, leaf renewal, metrics every round, early
stopping, a parameter reset with rollbacks. The removed keys are accepted with a warning and change nothing.
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

ROUNDS = 25


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


def _kind(objective):
    return {"binary": "binary", "multiclass": "multiclass"}.get(objective, "regression")


def _train(params, X, y, rounds=ROUNDS, valid=None, eval_train=False, early_stopping=None, callbacks=()):
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
        params,
        ds,
        num_boost_round=rounds,
        valid_sets=valid_sets or None,
        valid_names=valid_names or None,
        callbacks=cbs,
        keep_training_booster=True,
    )
    return booster, evals


# case -> (objective, quant_mode, extra params, md5 on master 5bec2574 with forest slots)
CASES = {
    "binary-stochastic": ("binary", "stochastic", {}, "fb25048b4150277ef610421af3396178"),
    "binary-fixedpoint": ("binary", "fixedpoint", {}, "0f484812e198cb3ad6798c3925f3088e"),
    "binary-none": ("binary", "none", {}, "31946388e8b4c25064d15ea3a7db035a"),
    "regression-stochastic": ("regression", "stochastic", {}, "2ceb15e50d0b91f1b06711ec41d21e13"),
    "multiclass-stochastic": ("multiclass", "stochastic", {}, "441cc09aca50d42756c16f1cad617d79"),
    # error feedback: one residual per class, carried from tree to tree (each class had its own slot)
    "multiclass-fixedpoint": ("multiclass", "fixedpoint", {}, "400c35eea215bdd4af5506f0eaa5faa5"),
    "multiclass-fixedpoint-unbounded": (
        "multiclass",
        "fixedpoint",
        {"num_leaves": 127, "max_depth": -1},
        "c769269f5edfb18e5775ed7a11305547",
    ),
    # non-quantized: the host level loop and the forest count construct, per-node masks drawn on the host
    "multiclass-none-bynode": (
        "multiclass",
        "none",
        {"num_leaves": 255, "max_depth": 8, "min_data_in_leaf": 20, "feature_fraction_bynode": 0.4},
        "7647db7d664813e2f423d24a952896b1",
    ),
    # per-node masks: device-drawn in the complete device tree (the slots drew them on the host: the same keys)
    "binary-bynode": ("binary", "stochastic", {"feature_fraction_bynode": 0.5}, "fbfe58399d63839be44b0eb6dc8ebd4f"),
    "multiclass-bynode": (
        "multiclass",
        "stochastic",
        {"feature_fraction_bynode": 0.4},
        "6564fd51f1367e0a2d89a467ab34d933",
    ),
    # per-tree samples, drawn in tree order (the slots' trainer drew them for every tree of a group)
    "binary-feature-fraction": ("binary", "stochastic", {"feature_fraction": 0.6}, "bd51647ab7109ea033cf9b77d6b0dd29"),
    "multiclass-feature-fraction": (
        "multiclass",
        "stochastic",
        {"feature_fraction": 0.6},
        "85f727c1901c15d8a7c9858587a7579c",
    ),
    # outside the device tree's bound: the device_levels prefix and the selective tail; a binding leaf budget
    "binary-unbounded": (
        "binary",
        "stochastic",
        {"num_leaves": 127, "max_depth": -1},
        "8db5c6c5f55d599d56fc782000a3926c",
    ),
    "multiclass-unbounded-bynode": (
        "multiclass",
        "stochastic",
        {"num_leaves": 255, "max_depth": -1, "min_data_in_leaf": 1, "feature_fraction_bynode": 0.4},
        "ea463aa2326c5d9f451691c26f964af0",
    ),
    "binary-budget": ("binary", "stochastic", {"num_leaves": 40, "max_depth": 7}, "ec0cae2eb18c6c678d9976bbccc1b153"),
    "regression-depth8": (
        "regression",
        "stochastic",
        {"num_leaves": 255, "max_depth": 8, "min_data_in_leaf": 1},
        "58232642866bcf92c755402b3473fb17",
    ),
    # no row sampling (rf needs bagging or a per-tree feature sample)
    "binary-no-bagging": (
        "binary",
        "stochastic",
        {"bagging_fraction": 1.0, "bagging_freq": 0, "feature_fraction": 0.8},
        "33314612a497099fb25f0a0c9a3dd4a3",
    ),
    # leaf renewal against the init score column
    "regression-l1": ("regression_l1", "stochastic", {}, "1e8272aa01c746a7eba5ad38721bfa4a"),
    "quantile-fixedpoint": ("quantile", "fixedpoint", {"alpha": 0.7}, "7337ef15108759d5b45c590f5a0dd85e"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_forest_sequential_same_model_cuda(case):
    objective, quant_mode, extra, want = CASES[case]
    X, y = _data(_kind(objective))
    booster, _ = _train(_params(objective, quant_mode, **extra), X, y)
    assert _md5(booster) == want, case


# metrics every round on the training and a validation set, then early stopping: md5 on master 5bec2574, and the
# iteration early stopping ended at
READERS = {
    "every-round": "a17289db912fcdfda59ba0017dcc6998",
    "early-stopping": ("7c288d15448c3cc651fd18aae81a8437", 23),
}


def test_forest_sequential_readers_cuda():
    X, y = _data("multiclass", rows=30011)
    Xv, yv = _data("multiclass", rows=7001, seed=11)
    params = _params("multiclass", "stochastic", num_leaves=127, max_depth=-1, metric="multi_logloss")
    params["feature_fraction"] = 0.7
    booster, evals = _train(params, X, y, 12, valid=(Xv, yv), eval_train=True)
    assert _md5(booster) == READERS["every-round"]
    assert len(evals["train"]["multi_logloss"]) == 12
    # the train score the metric read is the forest's average over the trees (the deferred score synced)
    np.testing.assert_allclose(
        np.array(booster._Booster__inner_predict(data_idx=0)).reshape(len(X), -1),
        booster.predict(X),
        rtol=1e-9,
        atol=1e-12,
    )
    booster, _ = _train(params, X, y, 60, valid=(Xv, yv), early_stopping=3)
    assert (_md5(booster), booster.current_iteration()) == READERS["early-stopping"]
    assert booster.current_iteration() < 60  # the case stops early


# a parameter reset mid-training, then two rollbacks and three more rounds: md5 on master 5bec2574
RESET_ROLLBACK = {
    "binary-stochastic": "182de67019241a72276e29abd485cf7e",
    "multiclass-fixedpoint": "6f7e94c7cf6da372df3b922e8010f7bf",
}


@pytest.mark.parametrize("case", list(RESET_ROLLBACK))
def test_forest_sequential_reset_and_rollback_cuda(case):
    objective, quant_mode = case.split("-")
    X, y = _data(_kind(objective), rows=30011)
    params = _params(objective, quant_mode, num_leaves=127, max_depth=-1, feature_fraction=0.6)
    params["feature_fraction_bynode"] = 0.5
    reset = flc.reset_parameter(min_data_in_leaf=[5] * 4 + [40] * 8)
    booster, _ = _train(params, X, y, 12, callbacks=[reset])
    booster.rollback_one_iter()
    booster.rollback_one_iter()
    for _ in range(3):
        booster.update()
    assert _md5(booster) == RESET_ROLLBACK[case]
    # a rollback takes its trees back out of the running average (rescale, subtract, rescale): the train score keeps
    # the rounding of those steps, a few float32 ulps from the prediction of the final model
    np.testing.assert_allclose(
        np.array(booster._Booster__inner_predict(data_idx=0)).reshape(len(X), -1).squeeze(),
        booster.predict(X, raw_score=False),
        rtol=1e-6,
        atol=1e-7,
    )


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
b = flc.train(cfg, ds, num_boost_round=rounds)
print("TREES", b.num_trees(), file=sys.stderr)
"""


def _run(params, rounds, diag=True):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FALCATA_DEBUG")}
    if diag:
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
    return res.stdout, res.stderr.splitlines()


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


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"feature_fraction_bynode": 0.5},
        # a multiclass round's class trees (always slots before): each a complete device tree
        {"objective": "multiclass", "num_class": 4},
        {"objective": "multiclass", "num_class": 4, "feature_fraction_bynode": 0.5},
        {"quant_mode": "fixedpoint", "objective": "multiclass", "num_class": 4},
    ],
)
def test_forest_every_tree_takes_the_device_tree_cuda(extra):
    _, lines = _run(_engage_params(**extra), 6)
    trees = int(next(line for line in lines if line.startswith("TREES ")).split()[1])
    assert trees == 6 * extra.get("num_class", 1)
    assert sum("device_tree: complete" in line for line in lines) == trees
    assert not any(line.startswith("[forest-diag]") for line in lines)


@pytest.mark.parametrize("shape", [{"num_leaves": 127, "max_depth": -1}, {"num_leaves": 40, "max_depth": 7}])
def test_forest_takes_the_selective_device_prefix_cuda(shape):
    # unbounded depth / a binding leaf budget: selective growth, whose first levels run on the device, every tree
    _, lines = _run(_engage_params(**shape), 6)
    assert sum("selective_device_prefix: levels=" in line for line in lines) == 6
    assert not any("device_tree: complete" in line for line in lines)


def test_forest_nonquantized_count_construct_no_graph_loop_cuda():
    # a non-quantized forest keeps the host level loop and the count construct; the non-quantized graph loop (which
    # the same shape takes under gbdt) is not taken, so the forest's model is the one the slots grew
    _, lines = _run(_engage_params(quant_mode="none"), 6)
    assert not any("graphs L1.5: instantiated" in line for line in lines)
    assert "[count-hist-diag] count construct" in lines
    _, gbdt = _run(_engage_params(quant_mode="none", boosting="gbdt"), 6)
    assert any("graphs L1.5: instantiated" in line for line in gbdt), gbdt[-20:]


@pytest.mark.parametrize(
    "plan",
    ["auto,forest_batched:off", "auto,forest_group:4", "auto,forest_single_job_device:off,forest_group:0"],
)
def test_forest_removed_keys_ignored_cuda(plan):
    params = _engage_params(objective="multiclass", num_class=4, verbose=0)
    stdout, lines = _run({**params, "cuda_plan": plan}, 4, diag=False)
    for key in [kv.split(":")[0] for kv in plan.split(",")[1:]]:
        assert f"cuda_plan key {key} was removed (2026-10-23): ignored" in stdout + "\n".join(lines), key
    X, y = _data("multiclass")
    p = _params("multiclass", "stochastic")
    assert _md5(_train({**p, "cuda_plan": plan}, X, y, 6)[0]) == _md5(_train(p, X, y, 6)[0])
