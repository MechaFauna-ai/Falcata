# coding: utf-8
"""A random forest on CUDA must train the same forest with the device objective as with the host objective.

cuda_plan key rf_device_objective boosts a forest (boosting=rf) on the device: the CUDA objective computes the
constant gradients once, they stay in device memory, and the scores have no host mirror. rf_lazy_train_score defers
the train score's per-tree update until something reads it (training metrics, predictions on the training data,
rollback) and then replays the trees in order. Neither may change the forest, the validation scores or the train score
a reader sees: every case trains with both keys on, with rf_lazy_train_score off, and with rf_device_objective off
(the host objective path), and compares the models (parameter block excepted), the predictions, the evaluation
records and the train scores read during training.
"""

import hashlib
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

PLANS = {
    "on": "auto",
    "eager": "auto,rf_lazy_train_score:off",
    "off": "auto,rf_device_objective:off",
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


def _data(kind, rows, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, 12)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    z = np.nan_to_num(X[:, 0]) - 0.5 * np.nan_to_num(X[:, 1]) + np.sin(np.nan_to_num(X[:, 2]))
    z = z + 0.3 * rng.standard_normal(rows)
    if kind == "binary":
        y = (z > 0).astype(np.float32)
    elif kind == "multiclass":
        y = np.digitize(z, [-1.5, -0.5, 0.5, 1.5]).astype(np.float32)
    else:
        y = (z + rng.standard_exponential(rows)).astype(np.float32)
    return X, y


def _train(params, X, y, rounds, valid=None, eval_train=False, read_train_at=(), rollback_at=None):
    ds = flc.Dataset(X, label=y, params={"device_type": "cuda", "max_bin": 63, "verbose": -1}, free_raw_data=False)
    valid_sets, valid_names = [], []
    if eval_train:
        valid_sets.append(ds)
        valid_names.append("train")
    if valid is not None:
        valid_sets.append(flc.Dataset(valid[0], label=valid[1], reference=ds))
        valid_names.append("valid")
    evals = {}
    train_reads = []
    callbacks = [flc.record_evaluation(evals)] if valid_sets else []
    if read_train_at:

        def read_train(env):
            if env.iteration + 1 in read_train_at:
                train_reads.append(np.array(env.model._Booster__inner_predict(data_idx=0), dtype=np.float64))

        callbacks.append(read_train)
    booster = flc.train(
        params,
        ds,
        num_boost_round=rounds,
        valid_sets=valid_sets or None,
        valid_names=valid_names or None,
        callbacks=callbacks,
        keep_training_booster=True,
    )
    if rollback_at is not None:
        while booster.current_iteration() > rollback_at:
            booster.rollback_one_iter()
        booster.update()
        train_reads.append(np.array(booster._Booster__inner_predict(data_idx=0), dtype=np.float64))
    pred = np.asarray(booster.predict(X[:5000]), dtype=np.float64)
    return {
        "model": hashlib.md5(_strip_parameters(booster.model_to_string()).encode()).hexdigest(),
        "pred": pred,
        "evals": evals,
        "train_reads": train_reads,
    }


def _params(objective, quant_mode, extra=None):
    params = {
        "boosting": "rf",
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": quant_mode,
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "num_leaves": 63,
        "max_depth": 8,
        "min_data_in_leaf": 20,
        "max_bin": 63,
        "metric": "None",
        "seed": 7,
        "verbose": -1,
    }
    if objective == "multiclass":
        params["num_class"] = 5
    if quant_mode == "none":
        # the hybrid level loop with fp64 histograms is not run-to-run deterministic; the host loop is
        params["cuda_plan_extra"] = ",graph_loop:off"
    params.update(extra or {})
    return params


def _run_plans(params, *args, **kwargs):
    extra = params.pop("cuda_plan_extra", "")
    return {name: _train({**params, "cuda_plan": plan + extra}, *args, **kwargs) for name, plan in PLANS.items()}


def _assert_same(runs):
    ref = runs["off"]
    for name, run in runs.items():
        assert run["model"] == ref["model"], name
        np.testing.assert_array_equal(run["pred"], ref["pred"], err_msg=name)
        assert len(run["train_reads"]) == len(ref["train_reads"]), name
        for got, want in zip(run["train_reads"], ref["train_reads"], strict=True):
            np.testing.assert_array_equal(got, want, err_msg=name)
    # the device paths evaluate with the CUDA metrics where there is one, the host path with the host metrics: same
    # scores, sums in another order (the host metrics' OpenMP reductions are not run-to-run fixed either)
    for name, run in runs.items():
        assert run["evals"].keys() == ref["evals"].keys(), name
        for data_name, metrics in ref["evals"].items():
            for metric, values in metrics.items():
                got = run["evals"][data_name][metric]
                np.testing.assert_allclose(got, values, rtol=1e-9, atol=1e-12, err_msg=f"{name} {data_name} {metric}")


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "kind", "quant_mode", "extra"),
    [
        ("binary", "binary", "stochastic", None),
        ("binary", "binary", "fixedpoint", {"feature_fraction": 0.7}),
        ("binary", "binary", "none", None),
        ("multiclass", "multiclass", "fixedpoint", None),
        ("multiclass", "multiclass", "stochastic", {"bagging_fraction": 0.5}),
        # one bag for the five class trees of a round and for three rounds: each tree starts its partition from the
        # bag order the previous one's in-bag score update restored, deferred or not
        ("multiclass", "multiclass", "none", {"bagging_freq": 3}),
        ("binary", "binary", "stochastic", {"bagging_freq": 3}),
        ("regression", "reg", "stochastic", None),
        ("regression", "reg", "stochastic", {"boost_from_average": False}),
        ("regression_l1", "reg", "stochastic", None),
        ("regression_l1", "reg", "fixedpoint", {"boost_from_average": False}),
        ("quantile", "reg", "stochastic", {"alpha": 0.7}),
    ],
)
def test_rf_device_objective_same_forest_cuda(objective, kind, quant_mode, extra):
    X, y = _data(kind, 30011, seed=3)
    _assert_same(_run_plans(_params(objective, quant_mode, extra), X, y, 12))


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "kind", "metric"),
    [
        ("binary", "binary", "binary_logloss,auc"),
        ("multiclass", "multiclass", "multi_logloss,multi_error"),
        ("regression_l1", "reg", "l1,l2"),
    ],
)
def test_rf_device_objective_metric_and_valid_cuda(objective, kind, metric):
    X, y = _data(kind, 30011, seed=3)
    valid = _data(kind, 7001, seed=5)
    params = _params(objective, "stochastic", {"metric": metric})
    _assert_same(_run_plans(params, X, y, 12, valid=valid))
    # the training metrics read the (deferred) train score every round
    _assert_same(
        _run_plans(_params(objective, "stochastic", {"metric": metric}), X, y, 12, valid=valid, eval_train=True)
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "kind", "extra"),
    [
        ("binary", "binary", {"feature_fraction": 0.6}),
        ("multiclass", "multiclass", {"quant_mode": "fixedpoint"}),
        ("regression", "reg", {"bagging_fraction": 0.3}),
    ],
)
def test_rf_lazy_train_score_readers_cuda(objective, kind, extra):
    X, y = _data(kind, 30011, seed=3)
    params = {**_params(objective, "stochastic"), **extra}
    # the train score read mid-training (the deferred trees replayed, then updated per tree) and after a rollback
    _assert_same(_run_plans(dict(params), X, y, 14, read_train_at=(6, 7, 14), rollback_at=10))
    # read once at the end: every tree replayed
    _assert_same(_run_plans(dict(params), X, y, 14, read_train_at=(14,)))


def _numpy_metric(metric, y, pred):
    if metric in ("binary_logloss", "cross_entropy"):
        pred = np.clip(pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))
    return float(np.mean((y - pred) ** 2))


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "label", "metric", "extra"),
    [
        ("binary", "binary", "binary_logloss", {"data_sample_strategy": "goss"}),
        ("tweedie", "count", "l2", {"bagging_fraction": 0.632, "bagging_freq": 1}),
        # (binary_logloss binarizes the label; the cross_entropy metric takes it as a probability)
        ("cross_entropy", "prob", "cross_entropy", {"bagging_fraction": 0.632, "bagging_freq": 1}),
        ("binary", "binary", "binary_logloss", {"bagging_fraction": 0.632, "bagging_freq": 1}),
    ],
)
def test_rf_metrics_follow_the_objective_cuda(objective, label, metric, extra):
    # GOSS and objectives without a CUDA kernel keep the host objective, and with it the host metrics: a CUDA metric
    # beside a host objective would evaluate the raw scores. Train and valid metrics must match the key-off values
    # and the metric computed from predict.
    rng = np.random.default_rng(3)
    rows = 30011
    X = rng.standard_normal((rows, 12))
    z = X[:, 0] - 0.5 * X[:, 1] + np.sin(X[:, 2]) + 0.3 * rng.standard_normal(rows)
    y = {
        "binary": (z > 0).astype(float),
        "prob": 1 / (1 + np.exp(-z)),
        "count": rng.poisson(np.exp(0.3 * z)).astype(float),
    }[label]
    results = {}
    for name, plan in PLANS.items():
        params = {
            "boosting": "rf",
            "device_type": "cuda",
            "objective": objective,
            "metric": metric,
            "num_leaves": 31,
            "verbose": -1,
            "seed": 7,
            "cuda_plan": plan,
            **extra,
        }
        ds = flc.Dataset(X, label=y, free_raw_data=False)
        valid = flc.Dataset(X[:5000], label=y[:5000], reference=ds)
        booster = flc.Booster(params, ds)
        booster.add_valid(valid, "valid")
        for _ in range(10):
            booster.update()
        pred = booster.predict(X)
        results[name] = (booster.eval_train()[0][2], booster.eval_valid()[0][2])
        np.testing.assert_allclose(results[name][0], _numpy_metric(metric, y, pred), rtol=1e-6, err_msg=name)
        np.testing.assert_allclose(
            results[name][1], _numpy_metric(metric, y[:5000], pred[:5000]), rtol=1e-6, err_msg=name
        )
    for name, values in results.items():
        np.testing.assert_allclose(values, results["off"], rtol=1e-9, err_msg=name)


@_REQUIRES_CUDA
def test_rf_shuffle_models_with_deferred_trees_cuda():
    # shuffle_models replaces the trees (copies): the deferred ones are added to the train score first
    X, y = _data("binary", 30011, seed=3)
    runs = {}
    for name, plan in PLANS.items():
        params = {**_params("binary", "stochastic"), "cuda_plan": plan}
        ds = flc.Dataset(X, label=y, params={"device_type": "cuda", "max_bin": 63, "verbose": -1}, free_raw_data=False)
        booster = flc.Booster(params, ds)
        for _ in range(6):
            booster.update()
        booster.shuffle_models()
        for _ in range(2):
            booster.update()
        runs[name] = (
            np.array(booster._Booster__inner_predict(data_idx=0), dtype=np.float64),
            np.asarray(booster.predict(X[:5000]), dtype=np.float64),
            hashlib.md5(_strip_parameters(booster.model_to_string()).encode()).hexdigest(),
        )
    for name, (train, pred, model) in runs.items():
        np.testing.assert_array_equal(train, runs["off"][0], err_msg=name)
        np.testing.assert_array_equal(pred, runs["off"][1], err_msg=name)
        assert model == runs["off"][2], name
