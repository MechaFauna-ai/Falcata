"""Bit-identity and state-invalidation tests for cuda_plan key score_grad_fuse.

Fusing AddPredictionToScore (end of round t) with GetGradients (start of round
t+1) for single-tree-per-iteration unbagged CUDA GBDT (regression L2, binary
logloss, L1, huber, quantile) eliminates a full DRAM round-trip of the score
array every round. The fused kernel runs the exact same arithmetic in the same
order, so the trained model and evaluation metrics must be bit-identical to
score_grad_fuse:off across objectives, weights, odd row counts, non-quantized
deterministic training, early stopping, RollbackOneIter, ResetConfig,
ResetTrainingData, dataset field mutation, RefitTree, and non-fused modes
(bagging, DART, RF, multiclass, linear trees, custom objective).
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


def _model_md5(booster):
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _train_md5(
    params,
    X,
    y,
    *,
    rounds=25,
    weight=None,
    init_score=None,
    valid_sets=None,
    callbacks=None,
):
    ds = flc.Dataset(X, label=y, weight=weight, init_score=init_score, params=params)
    evals_result = {}
    cbs = list(callbacks) if callbacks else []
    if valid_sets is not None:
        cbs.append(flc.record_evaluation(evals_result))
    booster = flc.train(params, ds, num_boost_round=rounds, valid_sets=valid_sets, callbacks=cbs)
    return _model_md5(booster), booster.best_iteration, evals_result


def _make_data(objective, rows=10011, cols=12, seed=13):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] - 0.3 * X[:, 3] + 0.25 * rng.standard_normal(rows)
    if objective == "binary":
        y = (signal > 0.2).astype(np.float32)
    elif objective == "multiclass":
        y = np.digitize(signal, np.quantile(signal, [0.25, 0.5, 0.75])).astype(np.float32)
    else:
        y = (signal + 2.0).astype(np.float32)
    return X, y, rng


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "extra", "quantized", "weighted"),
    [
        ("regression", {}, True, False),
        ("regression", {}, True, True),
        ("regression", {"quant_mode": "fixedpoint"}, True, False),
        ("regression", {"reg_sqrt": True}, True, False),
        ("regression", {}, False, False),
        ("regression", {}, False, True),
        ("binary", {}, True, False),
        ("binary", {}, True, True),
        ("binary", {"is_unbalance": True}, True, False),
        ("binary", {"scale_pos_weight": 2.5, "sigmoid": 0.4}, True, True),
        ("binary", {"sigmoid": 3.5}, False, False),
        ("binary", {}, False, True),
        ("regression_l1", {}, True, False),
        ("regression_l1", {}, True, True),
        ("regression_l1", {}, False, False),
        ("huber", {"alpha": 0.7}, True, False),
        ("huber", {"alpha": 1.3}, True, True),
        ("huber", {"alpha": 0.9}, False, False),
        ("quantile", {"alpha": 0.3}, True, False),
        ("quantile", {"alpha": 0.75}, True, True),
        ("quantile", {"alpha": 0.5}, False, False),
    ],
)
def test_score_grad_fuse_objectives_bit_identical_cuda(objective, extra, quantized, weighted):
    # Odd row count (10011) exercises both vectorized pairs and the trailing scalar element.
    X, y, rng = _make_data(objective, rows=10011, cols=12, seed=17)
    if extra.get("reg_sqrt"):
        y = np.abs(y) + 0.1
    weight = rng.uniform(0.3, 2.5, size=len(y)).astype(np.float32) if weighted else None
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 10,
        "learning_rate": 0.15,
        "use_quantized_grad": quantized,
        "verbosity": -1,
        "seed": 11,
        **extra,
    }
    base = "auto" if quantized else "auto,nonquant_deterministic:on"
    md5_on, _, _ = _train_md5(
        {**params, "cuda_plan": f"{base},score_grad_fuse:on"},
        X,
        y,
        rounds=20,
        weight=weight,
    )
    md5_off, _, _ = _train_md5(
        {**params, "cuda_plan": f"{base},score_grad_fuse:off"},
        X,
        y,
        rounds=20,
        weight=weight,
    )
    assert md5_on == md5_off


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "plan_prefix",
    [
        "auto",
        "auto,device_tree:off",
        "auto,async_tree_start:off",
        "auto,gradients_no_sync:off",
        "auto,gradient_fp64_lean:off",
    ],
)
def test_score_grad_fuse_plan_combinations_cuda(plan_prefix):
    for objective in ("regression", "binary"):
        X, y, _ = _make_data(objective, rows=4097, cols=10, seed=23)
        params = {
            "device_type": "cuda",
            "objective": objective,
            "num_leaves": 31,
            "max_depth": 5,
            "min_data_in_leaf": 10,
            "learning_rate": 0.2,
            "use_quantized_grad": True,
            "verbosity": -1,
            "seed": 5,
        }
        md5_on, _, _ = _train_md5(
            {**params, "cuda_plan": f"{plan_prefix},score_grad_fuse:on"},
            X,
            y,
            rounds=15,
        )
        md5_off, _, _ = _train_md5(
            {**params, "cuda_plan": f"{plan_prefix},score_grad_fuse:off"},
            X,
            y,
            rounds=15,
        )
        assert md5_on == md5_off, (objective, plan_prefix)


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_score_grad_fuse_rollback_and_reset_config_cuda(objective):
    X, y, _ = _make_data(objective, rows=4099, cols=10, seed=29)

    def run(fuse_val):
        params = {
            "device_type": "cuda",
            "objective": objective,
            "num_leaves": 31,
            "max_depth": 5,
            "min_data_in_leaf": 10,
            "learning_rate": 0.2,
            "use_quantized_grad": True,
            "cuda_plan": f"auto,score_grad_fuse:{fuse_val}",
            "verbosity": -1,
            "seed": 9,
        }
        ds = flc.Dataset(X, label=y, params=params)
        booster = flc.Booster(params, ds)
        for _ in range(5):
            booster.update()
        # Rollback one iteration, then train more rounds.
        booster.rollback_one_iter()
        for _ in range(4):
            booster.update()
        # Change learning rate via reset_parameter.
        booster.reset_parameter({"learning_rate": 0.05})
        for _ in range(4):
            booster.update()
        # Change objective parameter for binary (rebuilds objective).
        if objective == "binary":
            booster.reset_parameter({"sigmoid": 1.8})
            for _ in range(3):
                booster.update()
        # Toggle bagging on and back off via reset_parameter.
        booster.reset_parameter({"bagging_fraction": 0.7, "bagging_freq": 1})
        for _ in range(3):
            booster.update()
        booster.reset_parameter({"bagging_fraction": 1.0, "bagging_freq": 0})
        for _ in range(3):
            booster.update()
        return _model_md5(booster)

    assert run("on") == run("off")


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_score_grad_fuse_valid_sets_early_stopping_cuda(objective):
    X, y, _ = _make_data(objective, rows=6005, cols=10, seed=31)
    X_tr, y_tr = X[:4503], y[:4503]
    X_va, y_va = X[4503:], y[4503:]
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 10,
        "learning_rate": 0.35,
        "use_quantized_grad": True,
        "verbosity": -1,
        "seed": 3,
    }

    def run(fuse_val):
        p = {**params, "cuda_plan": f"auto,score_grad_fuse:{fuse_val}"}
        ds_tr = flc.Dataset(X_tr, label=y_tr, params=p)
        ds_va = flc.Dataset(X_va, label=y_va, reference=ds_tr, params=p)
        evals = {}
        booster = flc.train(
            p,
            ds_tr,
            num_boost_round=60,
            valid_sets=[ds_tr, ds_va],
            valid_names=["train", "valid"],
            callbacks=[
                flc.early_stopping(5, verbose=False),
                flc.record_evaluation(evals),
            ],
        )
        return _model_md5(booster), booster.best_iteration, evals

    md5_on, best_on, evals_on = run("on")
    md5_off, best_off, evals_off = run("off")
    assert md5_on == md5_off
    assert best_on == best_off
    assert evals_on == evals_off


@_REQUIRES_CUDA
def test_score_grad_fuse_dataset_mutation_and_refit_cuda():
    X, y, rng = _make_data("regression", rows=3003, cols=8, seed=41)
    y2 = y + 0.5 * rng.standard_normal(len(y)).astype(np.float32)
    w2 = rng.uniform(0.5, 2.0, size=len(y)).astype(np.float32)

    def run(fuse_val):
        params = {
            "device_type": "cuda",
            "objective": "regression",
            "num_leaves": 15,
            "max_depth": 4,
            "min_data_in_leaf": 10,
            "learning_rate": 0.1,
            "use_quantized_grad": True,
            "cuda_plan": f"auto,score_grad_fuse:{fuse_val}",
            "verbosity": -1,
            "seed": 19,
        }
        ds = flc.Dataset(X, label=y.copy(), params=params)
        booster = flc.Booster(params, ds)
        for _ in range(5):
            booster.update()
        # Mutate labels and weights on the live training Dataset.
        ds.set_label(y2)
        ds.set_weight(w2)
        for _ in range(5):
            booster.update()
        refit_booster = booster.refit(X, y2)
        return _model_md5(booster), _model_md5(refit_booster)

    assert run("on") == run("off")


@_REQUIRES_CUDA
def test_score_grad_fuse_init_score_and_no_bfa_cuda():
    X, y, rng = _make_data("regression", rows=2049, cols=8, seed=47)
    init_score = rng.normal(0.0, 0.5, size=len(y)).astype(np.float64)
    for bfa, init in [(False, None), (True, init_score)]:
        params = {
            "device_type": "cuda",
            "objective": "regression",
            "boost_from_average": bfa,
            "num_leaves": 15,
            "max_depth": 4,
            "min_data_in_leaf": 10,
            "learning_rate": 0.15,
            "use_quantized_grad": True,
            "verbosity": -1,
            "seed": 13,
        }
        md5_on, _, _ = _train_md5(
            {**params, "cuda_plan": "auto,score_grad_fuse:on"},
            X,
            y,
            rounds=12,
            init_score=init,
        )
        md5_off, _, _ = _train_md5(
            {**params, "cuda_plan": "auto,score_grad_fuse:off"},
            X,
            y,
            rounds=12,
            init_score=init,
        )
        assert md5_on == md5_off


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "extra"),
    [
        ("regression", {"bagging_fraction": 0.7, "bagging_freq": 1}),
        ("regression", {"boosting_type": "dart", "drop_rate": 0.2}),
        (
            "regression",
            {"boosting_type": "rf", "bagging_fraction": 0.7, "bagging_freq": 1},
        ),
        ("regression", {"linear_tree": True, "use_quantized_grad": False}),
        ("multiclass", {"num_class": 4}),
    ],
)
def test_score_grad_fuse_non_fused_modes_bit_identical_cuda(objective, extra):
    X, y, _ = _make_data(objective, rows=3005, cols=10, seed=53)
    quantized = extra.get("use_quantized_grad", True)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 10,
        "learning_rate": 0.15,
        "use_quantized_grad": quantized,
        "verbosity": -1,
        "seed": 7,
        **extra,
    }
    base = "auto" if quantized else "auto,nonquant_deterministic:on"
    md5_on, _, _ = _train_md5({**params, "cuda_plan": f"{base},score_grad_fuse:on"}, X, y, rounds=12)
    md5_off, _, _ = _train_md5({**params, "cuda_plan": f"{base},score_grad_fuse:off"}, X, y, rounds=12)
    assert md5_on == md5_off
