# coding: utf-8
"""gbdt on CUDA must report the metric of its predictions, whether its objective runs on the device or the host.

The CUDA metrics convert the raw score only through a CUDA objective (ConvertOutputCUDA). GOSS, objectives without a
CUDA kernel (cross_entropy, tweedie, ...) and custom objectives keep the host objective, and with it the host
metrics; a CUDA metric beside a host objective evaluated the raw scores (binary_logloss 0.999 for 0.324). Each case
trains on CUDA and on the CPU, and the train and valid metrics of both must match the metric computed with numpy
from predict().
"""

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
    for device in ("cuda", "cpu"):
        ds = flc.Dataset(X, label=y, group=groups, free_raw_data=False)
        valid = flc.Dataset(X[:_VALID], label=y[:_VALID], group=valid_groups, reference=ds)
        booster = flc.Booster({**params, "device_type": device}, ds)
        booster.add_valid(valid, "valid")
        for _ in range(10):
            booster.update()
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
