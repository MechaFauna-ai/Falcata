# coding: utf-8
"""Native CUDA multiclass metrics, cached sum_weights_, and fused ConvertOutput parity tests."""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _softmax(raw):
    shifted = raw - np.max(raw, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _numpy_multi_logloss(y, probs, weights=None):
    p = np.clip(probs[np.arange(len(y)), y.astype(int)], 1e-15, None)
    losses = -np.log(p)
    return float(np.average(losses, weights=weights))


def _numpy_multi_error(y, scores, top_k=1, weights=None):
    true_scores = scores[np.arange(len(y)), y.astype(int)][:, None]
    num_larger = np.sum(scores >= true_scores, axis=1)
    errs = (num_larger > top_k).astype(np.float64)
    return float(np.average(errs, weights=weights))


def _numpy_multi_rmse(y, preds, weights=None):
    sq = (preds - y) ** 2
    if weights is None:
        return float(np.sqrt(np.mean(sq)))
    w = weights[:, None]
    return float(np.sqrt(np.sum(sq * w) / (np.sum(weights) * y.shape[1])))


def _make_multiclass_data(n_train, n_val, num_class, n_features=12, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_train + n_val, n_features))
    W = rng.standard_normal((n_features, num_class)) * 0.6
    logits = X @ W + 0.5 * rng.standard_normal((n_train + n_val, num_class))
    y = np.argmax(logits, axis=1).astype(np.float32)
    # Ensure every class appears in train
    for c in range(num_class):
        y[c] = float(c)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_class", [3, 7, 8, 9, 16, 17, 32, 33, 40])
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_multiclass_logloss_and_error_parity(num_class, weighted, capfd):
    X_tr, y_tr, X_va, y_va = _make_multiclass_data(2000, 800, num_class, seed=100 + num_class)
    rng = np.random.default_rng(200 + num_class)
    w_tr = rng.uniform(0.2, 3.0, len(y_tr)).astype(np.float32) if weighted else None
    w_va = rng.uniform(0.2, 3.0, len(y_va)).astype(np.float32) if weighted else None

    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": num_class,
        "metric": ["multi_logloss", "multi_error"],
        "num_leaves": 15,
        "learning_rate": 0.1,
        "seed": 7,
        "verbose": 0,
    }
    tr = flc.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)
    va = flc.Dataset(X_va, label=y_va, weight=w_va, reference=tr)
    evals_result = {}
    booster = flc.train(
        params,
        tr,
        num_boost_round=6,
        valid_sets=[tr, va],
        valid_names=["train", "valid"],
        callbacks=[flc.record_evaluation(evals_result)],
    )
    out = capfd.readouterr()
    assert "not implemented in cuda version" not in out.out + out.err

    # Compare final train & valid metrics against exact numpy calculation from predict()
    for split, X_s, y_s, w_s in [("train", X_tr, y_tr, w_tr), ("valid", X_va, y_va, w_va)]:
        raw = booster.predict(X_s, raw_score=True)
        probs = _softmax(raw)
        w64 = w_s.astype(np.float64) if w_s is not None else None
        expected_ll = _numpy_multi_logloss(y_s, probs, weights=w64)
        expected_err = _numpy_multi_error(y_s, probs, top_k=1, weights=w64)
        np.testing.assert_allclose(evals_result[split]["multi_logloss"][-1], expected_ll, rtol=1e-12, atol=1e-12)
        if weighted:
            np.testing.assert_allclose(evals_result[split]["multi_error"][-1], expected_err, rtol=1e-12, atol=1e-12)
        else:
            assert evals_result[split]["multi_error"][-1] == expected_err


@_REQUIRES_CUDA
def test_cuda_multiclass_single_block():
    # rows=128 <= MULTICLASS_EVAL_BLOCK_SIZE (256) exercises the single-block reduction fast path.
    X_tr, y_tr, X_va, y_va = _make_multiclass_data(128, 128, 7, seed=111)
    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": 7,
        "metric": ["multi_logloss", "multi_error"],
        "num_leaves": 7,
        "min_data_in_leaf": 5,
        "seed": 3,
        "verbose": -1,
    }
    tr = flc.Dataset(X_tr, label=y_tr, free_raw_data=False)
    va = flc.Dataset(X_va, label=y_va, reference=tr)
    evals = {}
    booster = flc.train(
        params,
        tr,
        num_boost_round=5,
        valid_sets=[va],
        valid_names=["valid"],
        callbacks=[flc.record_evaluation(evals)],
    )
    probs = _softmax(booster.predict(X_va, raw_score=True))
    np.testing.assert_allclose(evals["valid"]["multi_logloss"][-1], _numpy_multi_logloss(y_va, probs), rtol=1e-12)
    assert evals["valid"]["multi_error"][-1] == _numpy_multi_error(y_va, probs)

    # Single-block multi_rmse
    rng = np.random.default_rng(112)
    Y_tr = rng.standard_normal((128, 4)).astype(np.float32)
    Y_va = rng.standard_normal((128, 4)).astype(np.float32)
    tr_r = flc.Dataset(X_tr, label=Y_tr, free_raw_data=False)
    va_r = flc.Dataset(X_va, label=Y_va, reference=tr_r)
    evals_r = {}
    booster_r = flc.train(
        {
            "device_type": "cuda",
            "objective": "multi_regression",
            "num_class": 4,
            "metric": "multi_rmse",
            "num_leaves": 7,
            "min_data_in_leaf": 5,
            "seed": 3,
            "verbose": -1,
        },
        tr_r,
        num_boost_round=5,
        valid_sets=[va_r],
        valid_names=["valid"],
        callbacks=[flc.record_evaluation(evals_r)],
    )
    np.testing.assert_allclose(
        evals_r["valid"]["multi_rmse"][-1],
        _numpy_multi_rmse(Y_va.astype(np.float64), booster_r.predict(X_va)),
        rtol=1e-12,
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("objective", ["multiclass", "multiclassova"])
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_multiclass_top_k_and_ova(top_k, objective, weighted):
    num_class = 6
    X_tr, y_tr, X_va, y_va = _make_multiclass_data(3000, 1200, num_class, seed=303)
    rng = np.random.default_rng(404)
    w_tr = rng.uniform(0.5, 2.5, len(y_tr)).astype(np.float32) if weighted else None
    w_va = rng.uniform(0.5, 2.5, len(y_va)).astype(np.float32) if weighted else None

    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_class": num_class,
        "sigmoid": 1.5,
        "metric": ["multi_logloss", "multi_error"],
        "multi_error_top_k": top_k,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "seed": 11,
        "verbose": -1,
    }
    tr = flc.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)
    va = flc.Dataset(X_va, label=y_va, weight=w_va, reference=tr)
    evals_result = {}
    booster = flc.train(
        params,
        tr,
        num_boost_round=6,
        valid_sets=[tr, va],
        valid_names=["train", "valid"],
        callbacks=[flc.record_evaluation(evals_result)],
    )
    err_name = "multi_error" if top_k == 1 else f"multi_error@{top_k}"
    for split, X_s, y_s, w_s in [("train", X_tr, y_tr, w_tr), ("valid", X_va, y_va, w_va)]:
        probs = booster.predict(X_s)
        w64 = w_s.astype(np.float64) if w_s is not None else None
        expected_ll = _numpy_multi_logloss(y_s, probs, weights=w64)
        expected_err = _numpy_multi_error(y_s, probs, top_k=top_k, weights=w64)
        np.testing.assert_allclose(evals_result[split]["multi_logloss"][-1], expected_ll, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(evals_result[split][err_name][-1], expected_err, rtol=1e-12, atol=1e-12)


@_REQUIRES_CUDA
def test_cuda_multi_error_tie_semantics_and_larger_valid_set():
    # Valid set larger than train set (used to crash ConvertOutputCUDA scratch check)
    # and boost_from_average=False with init_score ties (all-zero scores at iter 0 -> tie = error).
    num_class = 5
    X_tr, y_tr, X_va, y_va = _make_multiclass_data(500, 1500, num_class, seed=505)
    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": num_class,
        "metric": ["multi_logloss", "multi_error"],
        "boost_from_average": False,
        "num_leaves": 7,
        "seed": 1,
        "verbose": -1,
    }
    tr = flc.Dataset(X_tr, label=y_tr, free_raw_data=False)
    va = flc.Dataset(X_va, label=y_va, reference=tr)
    booster = flc.Booster(params, tr)
    booster.add_valid(va, "valid")
    # Before any boosting round, all scores are 0.0 -> ties for every row -> CPU MultiErrorMetric returns 1.0
    valid_eval_0 = {name: val for _, name, val, _ in booster.eval_valid()}
    assert valid_eval_0["multi_error"] == 1.0
    np.testing.assert_allclose(valid_eval_0["multi_logloss"], np.log(num_class), rtol=1e-12, atol=1e-12)
    for _ in range(5):
        booster.update()
    probs = booster.predict(X_va)
    valid_eval_5 = {name: val for _, name, val, _ in booster.eval_valid()}
    np.testing.assert_allclose(valid_eval_5["multi_logloss"], _numpy_multi_logloss(y_va, probs), rtol=1e-12, atol=1e-12)
    assert valid_eval_5["multi_error"] == _numpy_multi_error(y_va, probs)


@_REQUIRES_CUDA
def test_cuda_multi_error_near_tie_matches_cpu_softmax():
    # Raw scores [0.0, -1e-17, -10.0] have strictly distinct raw values (0.0 > -1e-17),
    # but exp(0.0) == exp(-1e-17) == 1.0 in FP64 softmax, so CPU MultiErrorMetric sees a tie -> error 1.0.
    n = 64
    rng = np.random.default_rng(515)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y_tr = (np.arange(n) % 3).astype(np.float32)
    y_va = np.zeros(n, dtype=np.float32)
    init_va = np.tile(np.array([0.0, -1e-17, -10.0], dtype=np.float64), (n, 1))

    base_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": ["multi_logloss", "multi_error"],
        "boost_from_average": False,
        "verbose": -1,
    }
    results = {}
    for device in ("cpu", "cuda"):
        tr = flc.Dataset(X, label=y_tr, params={"device_type": device, "verbose": -1}, free_raw_data=False)
        va = flc.Dataset(X, label=y_va, init_score=init_va, reference=tr)
        booster = flc.Booster({**base_params, "device_type": device}, tr)
        booster.add_valid(va, "valid")
        results[device] = {name: val for _, name, val, _ in booster.eval_valid()}

    assert results["cpu"]["multi_error"] == 1.0
    assert results["cuda"]["multi_error"] == results["cpu"]["multi_error"]
    np.testing.assert_allclose(results["cuda"]["multi_logloss"], results["cpu"]["multi_logloss"], rtol=1e-12)


@_REQUIRES_CUDA
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_multi_rmse_parity(weighted, capfd):
    rng = np.random.default_rng(606)
    n_train, n_val, num_target = 3000, 1200, 5
    X = rng.standard_normal((n_train + n_val, 10))
    W = rng.standard_normal((10, num_target))
    Y = (X @ W + 0.3 * rng.standard_normal((n_train + n_val, num_target))).astype(np.float32)
    w_tr = rng.uniform(0.3, 2.5, n_train).astype(np.float32) if weighted else None
    w_va = rng.uniform(0.3, 2.5, n_val).astype(np.float32) if weighted else None

    params = {
        "device_type": "cuda",
        "objective": "multi_regression",
        "num_class": num_target,
        "metric": "multi_rmse",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "seed": 9,
        "verbose": 0,
    }
    tr = flc.Dataset(X[:n_train], label=Y[:n_train], weight=w_tr, free_raw_data=False)
    va = flc.Dataset(X[n_train:], label=Y[n_train:], weight=w_va, reference=tr)
    evals_result = {}
    booster = flc.train(
        params,
        tr,
        num_boost_round=8,
        valid_sets=[tr, va],
        valid_names=["train", "valid"],
        callbacks=[flc.record_evaluation(evals_result)],
    )
    out = capfd.readouterr()
    assert "not implemented in cuda version" not in out.out + out.err
    for split, X_s, Y_s, w_s in [
        ("train", X[:n_train], Y[:n_train], w_tr),
        ("valid", X[n_train:], Y[n_train:], w_va),
    ]:
        preds = booster.predict(X_s)
        w64 = w_s.astype(np.float64) if w_s is not None else None
        expected = _numpy_multi_rmse(Y_s.astype(np.float64), preds, weights=w64)
        np.testing.assert_allclose(evals_result[split]["multi_rmse"][-1], expected, rtol=1e-12, atol=1e-12)


@_REQUIRES_CUDA
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_fused_binary_and_poisson_parity(weighted):
    rng = np.random.default_rng(707)
    n_train, n_val = 4000, 1500
    X = rng.standard_normal((n_train + n_val, 10))
    z = X[:, 0] - 0.7 * X[:, 1] + 0.5 * X[:, 2]
    y_bin = (z + 0.4 * rng.standard_normal(len(z)) > 0).astype(np.float32)
    y_poi = rng.poisson(np.exp(0.3 * z)).astype(np.float32)
    w_tr = rng.uniform(0.2, 4.0, n_train).astype(np.float32) if weighted else None
    w_va = rng.uniform(0.2, 4.0, n_val).astype(np.float32) if weighted else None

    # Binary with non-default sigmoid (tests fused sigmoid param propagation and reset_parameter)
    tr_b = flc.Dataset(X[:n_train], label=y_bin[:n_train], weight=w_tr, free_raw_data=False)
    va_b = flc.Dataset(X[n_train:], label=y_bin[n_train:], weight=w_va, reference=tr_b)
    booster_b = flc.Booster(
        {
            "device_type": "cuda",
            "objective": "binary",
            "sigmoid": 2.0,
            "metric": ["binary_logloss", "binary_error"],
            "num_leaves": 15,
            "seed": 5,
            "verbose": -1,
        },
        tr_b,
    )
    booster_b.add_valid(va_b, "valid")
    for _ in range(5):
        booster_b.update()
    booster_b.reset_parameter({"sigmoid": 0.7})
    for _ in range(5):
        booster_b.update()
    for eval_fn, X_s, y_s, w_s in [
        (booster_b.eval_train, X[:n_train], y_bin[:n_train], w_tr),
        (booster_b.eval_valid, X[n_train:], y_bin[n_train:], w_va),
    ]:
        probs = booster_b.predict(X_s)
        w64 = w_s.astype(np.float64) if w_s is not None else None
        p_clip = np.clip(probs, 1e-15, 1.0 - 1e-15)
        exp_ll = float(np.average(-(y_s * np.log(p_clip) + (1.0 - y_s) * np.log(1.0 - p_clip)), weights=w64))
        exp_err = float(np.average(((probs > 0.5) != (y_s > 0.5)).astype(np.float64), weights=w64))
        got = {name: val for _, name, val, _ in eval_fn()}
        np.testing.assert_allclose(got["binary_logloss"], exp_ll, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got["binary_error"], exp_err, rtol=1e-12, atol=1e-12)

    # Poisson (tests fused exp)
    tr_p = flc.Dataset(X[:n_train], label=y_poi[:n_train], weight=w_tr, free_raw_data=False)
    va_p = flc.Dataset(X[n_train:], label=y_poi[n_train:], weight=w_va, reference=tr_p)
    evals_p = {}
    booster_p = flc.train(
        {
            "device_type": "cuda",
            "objective": "poisson",
            "metric": "poisson",
            "num_leaves": 15,
            "seed": 5,
            "verbose": -1,
        },
        tr_p,
        num_boost_round=8,
        valid_sets=[tr_p, va_p],
        valid_names=["train", "valid"],
        callbacks=[flc.record_evaluation(evals_p)],
    )
    for split, X_s, y_s, w_s in [
        ("train", X[:n_train], y_poi[:n_train], w_tr),
        ("valid", X[n_train:], y_poi[n_train:], w_va),
    ]:
        preds = np.maximum(booster_p.predict(X_s), np.float32(1e-10))
        w64 = w_s.astype(np.float64) if w_s is not None else None
        exp_poi = float(np.average(preds - y_s * np.log(preds), weights=w64))
        np.testing.assert_allclose(evals_p[split]["poisson"][-1], exp_poi, rtol=1e-12, atol=1e-12)


@_REQUIRES_CUDA
def test_cuda_multiclass_early_stopping():
    X_tr, y_tr, X_va, y_va = _make_multiclass_data(3000, 1000, 7, seed=808)
    tr = flc.Dataset(X_tr, label=y_tr, free_raw_data=False)
    va = flc.Dataset(X_va, label=y_va, reference=tr)
    evals = {}
    booster = flc.train(
        {
            "device_type": "cuda",
            "objective": "multiclass",
            "num_class": 7,
            "metric": "multi_logloss",
            "num_leaves": 63,
            "learning_rate": 0.3,
            "seed": 42,
            "verbose": -1,
        },
        tr,
        num_boost_round=100,
        valid_sets=[va],
        valid_names=["valid"],
        callbacks=[flc.early_stopping(10, verbose=False), flc.record_evaluation(evals)],
    )
    assert booster.best_iteration > 0
    assert booster.best_iteration < 100
    assert np.isfinite(evals["valid"]["multi_logloss"][booster.best_iteration - 1])
