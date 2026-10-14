# coding: utf-8
"""gbdt on CUDA must report the metric of its predictions, whether its objective runs on the device or the host.

The CUDA metrics convert the raw score only through a CUDA objective (ConvertOutputCUDA). GOSS, objectives without a
CUDA kernel (cross_entropy, tweedie, ...) and custom objectives keep the host objective, and with it the host
metrics; a CUDA metric beside a host objective evaluated the raw scores (binary_logloss 0.999 for 0.324). Each case
trains on CUDA and on the CPU, and the train and valid metrics of both must match the metric computed with numpy
from predict(). The reset cases change the objective of a live booster (reset_parameter, and update(fobj=...), which
resets the objective to "none") between a device and a host objective: the metrics must follow the new objective.
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

_ROWS = 30011
_VALID = 5000
_GROUP = 11


def _labels(kind, z, rng):
    if kind == "binary":
        return (z > 0).astype(float)
    if kind == "prob":
        return 1 / (1 + np.exp(-z))
    if kind == "count":
        return rng.poisson(np.exp(0.3 * z)).astype(float)
    if kind == "gamma":
        return rng.gamma(2.0, np.exp(0.3 * z) / 2.0)
    if kind in ("multi", "rank"):
        return np.digitize(z, [-1.0, 0.0, 1.0]).astype(float)
    return z + 0.5 * rng.standard_normal(len(z))


def _groups(rows):
    group = np.full(rows // _GROUP, _GROUP)
    group[-1] += rows - group.sum()
    return group


def _ndcg_at(y, score, groups, k):
    values, start = [], 0
    for size in groups:
        gain = 2.0 ** y[start : start + size] - 1
        order = np.argsort(-score[start : start + size], kind="stable")
        start += size
        discount = 1 / np.log2(np.arange(2, size + 2))
        ideal = np.sort(gain)[::-1][:k] @ discount[:k]
        values.append(1.0 if ideal <= 0 else gain[order][:k] @ discount[:k] / ideal)
    return float(np.mean(values))


def _logloss(y, pred):
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))


def _huber(y, pred):
    diff = np.abs(y - pred)
    return np.mean(np.where(diff <= 0.9, 0.5 * diff * diff, 0.9 * (diff - 0.45)))


def _poisson(y, pred):
    pred = np.maximum(pred, 1e-10)
    return np.mean(pred - y * np.log(pred))


def _gamma(y, pred):
    # GammaMetric with psi 1: label / pred + log(pred)
    return np.mean(y / pred + np.log(pred))


def _xentlambda(y, pred):
    # CrossEntropyLambdaMetric: predict() gives hhat = log1p(exp(score)), the loss is the cross entropy of 1 - exp(-hhat)
    prob = 1 - np.exp(-pred)
    return -np.mean(y * np.log(np.maximum(prob, 1e-12)) + (1 - y) * np.log(np.maximum(1 - prob, 1e-12)))


def _l2_fobj(preds, train_data):
    return preds - train_data.get_label(), np.ones_like(preds)


def _tweedie(y, pred):
    # variance power 1.5
    pred = np.maximum(pred, 1e-10)
    return np.mean(y * pred**-0.5 / 0.5 + pred**0.5 / 0.5)


_NUMPY_METRICS = {
    "binary_logloss": _logloss,
    "cross_entropy": _logloss,
    "binary_error": lambda y, pred: np.mean((pred > 0.5) != (y > 0.5)),
    "multi_logloss": lambda y, pred: -np.mean(np.log(np.clip(pred[np.arange(len(y)), y.astype(int)], 1e-15, None))),
    "l2": lambda y, pred: np.mean((y - pred) ** 2),
    "l1": lambda y, pred: np.mean(np.abs(y - pred)),
    "huber": _huber,
    "poisson": _poisson,
    "tweedie": _tweedie,
    "gamma": _gamma,
    "cross_entropy_lambda": _xentlambda,
}


def _numpy_metric(metric, y, pred, groups):
    if metric == "ndcg":
        return _ndcg_at(y, pred, groups, 5)
    return float(_NUMPY_METRICS[metric](y, pred))


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("sample", "objective", "label", "metric"),
    [
        # GOSS keeps the host objective
        ("goss", "binary", "binary", "binary_logloss"),
        ("goss", "binary", "binary", "binary_error"),
        ("goss", "multiclass", "multi", "multi_logloss"),
        ("goss", "regression", "reg", "l2"),
        ("goss", "regression_l1", "reg", "l1"),
        ("goss", "huber", "reg", "huber"),
        ("goss", "poisson", "count", "poisson"),
        ("goss", "cross_entropy", "prob", "cross_entropy"),
        ("goss", "cross_entropy", "binary", "binary_logloss"),
        ("goss", "lambdarank", "rank", "ndcg"),
        # objectives without a CUDA kernel keep the host objective without GOSS too
        ("none", "cross_entropy", "binary", "binary_logloss"),
        ("none", "tweedie", "count", "tweedie"),
        ("none", "tweedie", "count", "l2"),
        ("none", "gamma", "gamma", "gamma"),
        ("none", "gamma", "gamma", "l2"),
        ("none", "cross_entropy_lambda", "prob", "cross_entropy_lambda"),
        # a Python objective (fobj): no objective in the library, the host metrics on the raw score
        ("none", "custom", "reg", "l2"),
        ("goss", "custom", "reg", "l2"),
        # the device objective and the CUDA metrics, unchanged
        ("none", "binary", "binary", "binary_logloss"),
        ("none", "regression", "reg", "l2"),
        ("none", "poisson", "count", "poisson"),
    ],
)
def test_gbdt_metrics_match_numpy_cuda(sample, objective, label, metric):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((_ROWS, 12))
    z = X[:, 0] - 0.5 * X[:, 1] + np.sin(X[:, 2]) + 0.3 * rng.standard_normal(_ROWS)
    y = _labels(label, z, rng)
    groups = valid_groups = None
    if label == "rank":
        groups, valid_groups = _groups(_ROWS), _groups(_VALID)
    params = {"boosting": "gbdt", "objective": objective, "metric": metric, "num_leaves": 31, "seed": 7, "verbose": -1}
    if sample == "goss":
        params["data_sample_strategy"] = "goss"
    if objective == "multiclass":
        params["num_class"] = 4
    if metric == "ndcg":
        params["eval_at"] = [5]
    fobj = None
    if objective == "custom":
        params["objective"], fobj = "none", _l2_fobj
    for device in ("cuda", "cpu"):
        ds = flc.Dataset(X, label=y, group=groups, free_raw_data=False)
        valid = flc.Dataset(X[:_VALID], label=y[:_VALID], group=valid_groups, reference=ds)
        booster = flc.Booster({**params, "device_type": device}, ds)
        booster.add_valid(valid, "valid")
        for _ in range(10):
            booster.update(fobj=fobj)
        pred = booster.predict(X)
        np.testing.assert_allclose(
            booster.eval_train()[0][2], _numpy_metric(metric, y, pred, groups), rtol=1e-6, err_msg=f"{device} train"
        )
        np.testing.assert_allclose(
            booster.eval_valid()[0][2],
            _numpy_metric(metric, y[:_VALID], pred[:_VALID], valid_groups),
            rtol=1e-6,
            err_msg=f"{device} valid",
        )


def _check_metrics(booster, X, y, metric, tag):
    pred = booster.predict(X)
    np.testing.assert_allclose(
        booster.eval_train()[0][2], _numpy_metric(metric, y, pred, None), rtol=1e-6, err_msg=f"{tag} train"
    )
    np.testing.assert_allclose(
        booster.eval_valid()[0][2],
        _numpy_metric(metric, y[:_VALID], pred[:_VALID], None),
        rtol=1e-6,
        err_msg=f"{tag} valid",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("first", "second", "label", "metric"),
    [
        # device objective -> host objective, and back
        ("binary", "cross_entropy", "binary", "binary_logloss"),
        ("cross_entropy", "binary", "binary", "binary_logloss"),
        # update(fobj=...) resets the objective to "none" (host): the device regression objective before it
        ("regression", "custom", "reg", "l2"),
    ],
)
def test_reset_objective_rebuilds_metrics_cuda(first, second, label, metric):
    rng = np.random.default_rng(5)
    X = rng.standard_normal((_ROWS, 12))
    z = X[:, 0] - 0.5 * X[:, 1] + np.sin(X[:, 2]) + 0.3 * rng.standard_normal(_ROWS)
    y = _labels(label, z, rng)
    params = {"boosting": "gbdt", "objective": first, "metric": metric, "num_leaves": 31, "seed": 7, "verbose": -1}
    for device in ("cuda", "cpu"):
        ds = flc.Dataset(X, label=y, free_raw_data=False)
        valid = flc.Dataset(X[:_VALID], label=y[:_VALID], reference=ds)
        booster = flc.Booster({**params, "device_type": device}, ds)
        booster.add_valid(valid, "valid")
        for _ in range(5):
            booster.update()
        _check_metrics(booster, X, y, metric, f"{device} {first}")
        fobj = None
        if second == "custom":
            fobj = _l2_fobj
        else:
            booster.reset_parameter({"objective": second})
        for _ in range(5):
            booster.update(fobj=fobj)
        _check_metrics(booster, X, y, metric, f"{device} {first} -> {second}")


def _model_md5(booster):
    return hashlib.md5(booster.model_to_string().split("\nparameters:")[0].encode()).hexdigest()


@_REQUIRES_CUDA
def test_reset_learning_rate_keeps_cpu_booster_on_cpu(capfd):
    # A reset that does not name the objective leaves the objective and the metrics alone. The reset's config gets
    # only the reset keys, and with device_type unset the auto rule reads it as cuda on a machine with a GPU, so a
    # rebuild there would give a device_type=cpu booster a CUDA objective and CUDA metrics (with a cpu Dataset: a
    # FalcataError). The model must be the one of the same reset with device_type stated.
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4000, 10))
    y = (X[:, 0] - X[:, 1] + 0.5 * rng.standard_normal(4000) > 0).astype(float)
    params = {"device_type": "cpu", "objective": "binary", "metric": "auc", "seed": 1, "verbose": 0}
    md5s = []
    for reset in ({"learning_rate": 0.05}, {"learning_rate": 0.05, "device_type": "cpu"}):
        ds = flc.Dataset(X, label=y, params={"device_type": "cpu", "verbose": 0}, free_raw_data=False)
        booster = flc.Booster(params, ds)
        booster.add_valid(flc.Dataset(X[:500], label=y[:500], reference=ds), "valid")
        for _ in range(3):
            booster.update()
        booster.reset_parameter(reset)
        for _ in range(3):
            booster.update()
        booster.eval_train()
        booster.eval_valid()
        md5s.append(_model_md5(booster))
    out = capfd.readouterr()
    assert "not implemented in cuda version" not in out.out + out.err
    assert md5s[0] == md5s[1]


@_REQUIRES_CUDA
def test_reset_data_sample_strategy_is_refused():
    # the sample strategy object is built once; the objective and metric factories read the config's strategy
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4000, 10))
    y = (X[:, 0] > 0).astype(float)
    booster = flc.Booster(
        {"device_type": "cuda", "objective": "binary", "data_sample_strategy": "goss", "verbose": -1},
        flc.Dataset(X, label=y),
    )
    booster.update()
    with pytest.raises(flc.basic.FalcataError, match="Cannot change data_sample_strategy during training"):
        booster.reset_parameter({"data_sample_strategy": "bagging"})
    booster.reset_parameter({"data_sample_strategy": "goss", "learning_rate": 0.05})  # unchanged: accepted
    booster.update()
