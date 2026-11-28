# coding: utf-8
"""Single-thread multi-stream concurrent multiclass GBDT rounds on CUDA (cuda_plan key multiclass_concurrent).

Verifies exact bit-identity (tree_md5) between ``multiclass_concurrent:on`` and ``multiclass_concurrent:off``
across feature subsampling, multi-wave class counts (C > 8 with quant_ef), selective unbounded growth,
bynode feature subsampling, validation scoring + multi_logloss, reset_parameter LR schedules, and categorical
features, plus an observable ``FALCATA_DEBUG=diag`` engagement check.
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


def _make_multiclass_data(rows=24000, features=32, seed=12345):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, features), dtype=np.float32)
    logits = X[:, :12] * 0.7 + rng.normal(0, 0.3, size=(rows, 12)).astype(np.float32)
    y7 = np.argmax(logits[:, :7], axis=1).astype(np.float32)
    y12 = np.argmax(logits, axis=1).astype(np.float32)
    return X, y7, y12


def _train_arm(X, y, base_params, plan, *, rounds=10, valid=None, cat_features="auto", reset_lr=False):
    params = {
        "device_type": "cuda",
        "verbose": -1,
        "num_threads": 8,
        **base_params,
        "cuda_plan": plan,
    }
    ds = flc.Dataset(
        np.ascontiguousarray(X),
        label=y,
        params=params,
        categorical_feature=cat_features,
    )
    valid_sets = []
    evals_result = {}
    callbacks = []
    if valid is not None:
        Xv, yv = valid
        dv = flc.Dataset(
            np.ascontiguousarray(Xv),
            label=yv,
            reference=ds,
            params=params,
            categorical_feature=cat_features,
        )
        valid_sets = [dv]
        callbacks.append(flc.record_evaluation(evals_result))
    if reset_lr:
        callbacks.append(flc.reset_parameter(learning_rate=lambda i: 0.1 if i < rounds // 2 else 0.05))
    bst = flc.train(params, ds, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    return _md5(bst), evals_result


def _plan(on, nonquant_det=False):
    parts = ["auto"]
    if nonquant_det:
        parts.append("nonquant_deterministic:on")
    parts.append(f"multiclass_concurrent:{'on' if on else 'off'}")
    return ",".join(parts)


BASE7 = {
    "objective": "multiclass",
    "num_class": 7,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "max_depth": 6,
    "max_bin": 255,
    "lambda_l2": 1.0,
    "seed": 42,
}


@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint", "none"])
def test_feature_fraction_depth6_bit_identical(quant_mode):
    X, y7, _ = _make_multiclass_data()
    params = {**BASE7, "feature_fraction": 0.7, "quant_mode": quant_mode}
    nonquant_det = quant_mode == "none"
    md5_off, _ = _train_arm(X, y7, params, _plan(False, nonquant_det=nonquant_det))
    md5_on, _ = _train_arm(X, y7, params, _plan(True, nonquant_det=nonquant_det))
    assert md5_on == md5_off


@pytest.mark.parametrize("quant_mode", ["fixedpoint", "stochastic", "none"])
def test_multiwave_12class_bit_identical(quant_mode):
    X, _, y12 = _make_multiclass_data()
    params = {**BASE7, "num_class": 12, "quant_mode": quant_mode}
    nonquant_det = quant_mode == "none"
    md5_off, _ = _train_arm(X, y12, params, _plan(False, nonquant_det=nonquant_det))
    md5_on, _ = _train_arm(X, y12, params, _plan(True, nonquant_det=nonquant_det))
    assert md5_on == md5_off


@pytest.mark.parametrize("feature_fraction", [1.0, 0.7])
def test_selective_unbounded_bit_identical(feature_fraction):
    X, y7, _ = _make_multiclass_data()
    params = {
        **BASE7,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": feature_fraction,
        "quant_mode": "stochastic",
    }
    md5_off, _ = _train_arm(X, y7, params, _plan(False))
    md5_on, _ = _train_arm(X, y7, params, _plan(True))
    assert md5_on == md5_off


def test_bynode_feature_fraction_bit_identical():
    X, y7, _ = _make_multiclass_data()
    params = {**BASE7, "feature_fraction_bynode": 0.7, "quant_mode": "fixedpoint"}
    md5_off, _ = _train_arm(X, y7, params, _plan(False))
    md5_on, _ = _train_arm(X, y7, params, _plan(True))
    assert md5_on == md5_off


def test_valid_sets_multi_logloss_bit_identical():
    X, y7, _ = _make_multiclass_data()
    params = {**BASE7, "metric": "multi_logloss", "quant_mode": "fixedpoint"}
    md5_off, ev_off = _train_arm(X[:18000], y7[:18000], params, _plan(False), valid=(X[18000:], y7[18000:]))
    md5_on, ev_on = _train_arm(X[:18000], y7[:18000], params, _plan(True), valid=(X[18000:], y7[18000:]))
    assert md5_on == md5_off
    assert ev_on.keys() == ev_off.keys()
    for ds_k in ev_off:
        assert ev_on[ds_k].keys() == ev_off[ds_k].keys()
        for m_k in ev_off[ds_k]:
            np.testing.assert_allclose(ev_on[ds_k][m_k], ev_off[ds_k][m_k], rtol=1e-12, atol=1e-12)


def test_reset_parameter_lr_schedule_bit_identical():
    X, y7, _ = _make_multiclass_data()
    params = {**BASE7, "quant_mode": "fixedpoint"}
    md5_off, _ = _train_arm(X, y7, params, _plan(False), reset_lr=True)
    md5_on, _ = _train_arm(X, y7, params, _plan(True), reset_lr=True)
    assert md5_on == md5_off


def test_categorical_multiclass_bit_identical():
    rng = np.random.default_rng(777)
    X, y7, _ = _make_multiclass_data()
    X_cat = X.copy()
    X_cat[:, 0] = rng.integers(0, 12, size=X.shape[0]).astype(np.float32)
    X_cat[:, 1] = rng.integers(0, 8, size=X.shape[0]).astype(np.float32)
    params = {**BASE7, "quant_mode": "fixedpoint"}
    md5_off, _ = _train_arm(X_cat, y7, params, _plan(False), cat_features=[0, 1])
    md5_on, _ = _train_arm(X_cat, y7, params, _plan(True), cat_features=[0, 1])
    assert md5_on == md5_off


_DIAG_CHILD = r"""
import json, sys
import numpy as np
import falcata as flc
cfg = json.loads(sys.argv[1])
rng = np.random.default_rng(12345)
X = rng.standard_normal((12000, 24), dtype=np.float32)
logits = X[:, :7] * 0.7 + rng.normal(0, 0.3, size=(12000, 7)).astype(np.float32)
y = np.argmax(logits, axis=1).astype(np.float32)
ds = flc.Dataset(X, label=y, params=cfg["params"])
flc.train(cfg["params"], ds, num_boost_round=4)
"""


def _run_diag_child(plan):
    params = {
        "device_type": "cuda",
        "verbose": -1,
        "num_threads": 8,
        **BASE7,
        "quant_mode": "fixedpoint",
        "cuda_plan": plan,
    }
    env = os.environ.copy()
    env["FALCATA_DEBUG"] = "diag"
    res = subprocess.run(
        [sys.executable, "-c", _DIAG_CHILD, json.dumps({"params": params})],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stderr


def test_diag_observable_engagement():
    stderr_on = _run_diag_child(_plan(True))
    stderr_off = _run_diag_child(_plan(False))
    assert "[multiclass-diag] multiclass_concurrent: active=7 slots=7" in stderr_on, stderr_on[-2000:]
    assert "[multiclass-diag]" not in stderr_off, stderr_off[-2000:]
