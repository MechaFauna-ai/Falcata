"""End-to-end coverage for automatic classification Newton-step stabilization."""

import json
from os import getenv

import numpy as np
import pytest

import falcata as lgb


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only validation")),
    ]
)
def device_params(request):
    return {"device_type": request.param, "cuda_precision": "fp64", "quant_mode": "none"}


def _leaf_values(node):
    if "leaf_value" in node:
        return [node["leaf_value"]]
    return _leaf_values(node["left_child"]) + _leaf_values(node["right_child"])


def _binary_problem():
    # Both leaves are confident in their majority class, but each contains a
    # small misclassified counter-group that drives an oversized Newton step.
    X = np.repeat([0.0, 1.0], 32).reshape(-1, 1)
    y = np.array([1] * 30 + [0] * 2 + [1] * 2 + [0] * 30)
    init_score = np.r_[np.full(32, 8.0), np.full(32, -8.0)]
    dataset = lgb.Dataset(X, label=y, init_score=init_score, params={"max_bin": 2})
    params = {
        "objective": "binary",
        "device_type": "cpu",
        "learning_rate": 0.1,
        "num_leaves": 2,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l2": 0.0,
        "max_bin": 2,
        "feature_pre_filter": False,
        "boost_from_average": False,
        "verbosity": -1,
    }
    return X, y, init_score, dataset, params


def _binary_logloss(y, raw_score):
    return np.mean(np.logaddexp(0.0, (1.0 - 2.0 * y) * raw_score))


def test_binary_auto_guard_prevents_saturated_newton_overshoot(device_params):
    X, y, init_score, dataset, params = _binary_problem()
    params.update(device_params)

    guarded = lgb.train(params, dataset, num_boost_round=1)
    guarded_delta = guarded.predict(X, raw_score=True)
    assert np.max(np.abs(guarded_delta)) <= 1.0
    assert _binary_logloss(y, init_score + guarded_delta) < _binary_logloss(y, init_score)

    # Explicit zero preserves the historical unbounded update for callers
    # that need to reproduce older models or benchmarks.
    unbounded = lgb.train({**params, "max_delta_step": 0.0}, dataset, num_boost_round=1)
    unbounded_delta = unbounded.predict(X, raw_score=True)
    assert np.max(np.abs(unbounded_delta)) > 1.0
    assert _binary_logloss(y, init_score + unbounded_delta) > _binary_logloss(y, init_score)

    # Positive values remain manual raw-output caps and take precedence over
    # the automatic classification policy.
    manual = lgb.train({**params, "max_delta_step": 0.25}, dataset, num_boost_round=1)
    assert np.max(np.abs(manual.predict(X, raw_score=True))) <= 0.1 * 0.25


@pytest.mark.parametrize(("sigmoid", "learning_rate"), [(1.0, 0.1), (2.0, 0.2)])
def test_binary_auto_guard_scales_with_sigmoid_and_learning_rate(sigmoid, learning_rate, device_params):
    X, y, init_score, _, params = _binary_problem()
    dataset = lgb.Dataset(X, label=y, init_score=init_score / sigmoid, params={"max_bin": 2})
    params.update({**device_params, "sigmoid": sigmoid, "learning_rate": learning_rate})
    model = lgb.train(params, dataset, num_boost_round=1)
    tree = model.dump_model()["tree_info"][0]
    assert tree["num_leaves"] == 2
    assert max(map(abs, _leaf_values(tree["tree_structure"]))) == pytest.approx(1.0 / sigmoid)


def test_multiclass_automatic_policy_is_serialized_and_can_continue(tmp_path, device_params):
    X = np.repeat(np.arange(3, dtype=float), 32).reshape(-1, 1)
    group = np.repeat(np.arange(3), 32)
    y = np.concatenate([np.r_[np.full(30, k), (k + 1) % 3, (k + 2) % 3] for k in range(3)])
    init_score = np.zeros((len(y), 3))
    init_score[np.arange(len(y)), group] = 8.0
    dataset = lgb.Dataset(X, label=y, init_score=init_score, params={"max_bin": 3}, free_raw_data=False)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "device_type": "cpu",
        "learning_rate": 0.1,
        "num_leaves": 3,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l2": 0.0,
        "max_bin": 3,
        "feature_pre_filter": False,
        "boost_from_average": False,
        "verbosity": -1,
    }
    params.update(device_params)
    model = lgb.train(params, dataset, num_boost_round=1, keep_training_booster=True)
    for tree in model.dump_model()["tree_info"]:
        assert tree["num_leaves"] > 1
        assert max(map(abs, _leaf_values(tree["tree_structure"]))) <= 1.0

    def loss(scores):
        maximum = scores.max(axis=1)
        logsum = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
        return np.mean(logsum - scores[np.arange(len(y)), y])

    assert loss(init_score + model.predict(X, raw_score=True)) < loss(init_score)
    legacy = lgb.train({**params, "max_delta_step": 0}, dataset, num_boost_round=1)
    assert loss(init_score + legacy.predict(X, raw_score=True)) > loss(init_score)

    model_file = tmp_path / "guarded-multiclass.txt"
    model.save_model(str(model_file))
    saved_text = model_file.read_text(encoding="utf-8")
    assert "[max_delta_step: -1]" in saved_text

    continued = lgb.train(
        {**params, "learning_rate": 0.2},
        dataset,
        num_boost_round=1,
        init_model=str(model_file),
    )
    continued_tree = continued.dump_model()["tree_info"][-3:]
    for tree in continued_tree:
        assert max(map(abs, _leaf_values(tree["tree_structure"]))) <= 1.0


def test_multiclassova_auto_guard_caps_saturated_classwise_updates(device_params):
    X = np.repeat(np.arange(3, dtype=float), 32).reshape(-1, 1)
    y = np.repeat(np.arange(3), 32)
    init_score = np.full((len(y), 3), 8.0)
    init_score[np.arange(len(y)), y] = -8.0
    dataset = lgb.Dataset(X, label=y, init_score=init_score, params={"max_bin": 3})
    params = {
        "objective": "multiclassova",
        "num_class": 3,
        "device_type": "cpu",
        "learning_rate": 0.1,
        "num_leaves": 3,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l2": 0.0,
        "max_bin": 3,
        "feature_pre_filter": False,
        "boost_from_average": False,
        "verbosity": -1,
    }
    params.update(device_params)
    guarded = lgb.train(params, dataset, num_boost_round=1)
    assert max(max(map(abs, _leaf_values(tree["tree_structure"]))) for tree in guarded.dump_model()["tree_info"]) <= 1.0

    unbounded = lgb.train({**params, "max_delta_step": 0.0}, dataset, num_boost_round=1)
    assert (
        max(max(map(abs, _leaf_values(tree["tree_structure"]))) for tree in unbounded.dump_model()["tree_info"]) > 1.0
    )


def test_auto_guard_is_not_applied_to_regression_custom_objective_or_rf():
    X = np.repeat([0.0, 1.0], 32).reshape(-1, 1)
    labels = np.r_[np.full(32, 30.0), np.full(32, -30.0)]
    dataset = lgb.Dataset(X, label=labels, params={"max_bin": 2})
    common = {
        "device_type": "cpu",
        "learning_rate": 0.1,
        "num_leaves": 2,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l2": 0.0,
        "max_bin": 2,
        "feature_pre_filter": False,
        "boost_from_average": False,
        "verbosity": -1,
    }

    regression = lgb.train({**common, "objective": "regression"}, dataset, num_boost_round=1)
    assert max(map(abs, _leaf_values(regression.dump_model()["tree_info"][0]["tree_structure"]))) > 1.0

    def large_step_objective(preds, train_data):
        del preds
        return np.sign(train_data.get_label()) * -100.0, np.ones_like(train_data.get_label())

    custom = lgb.train({**common, "objective": large_step_objective}, dataset, num_boost_round=1)
    assert max(map(abs, _leaf_values(custom.dump_model()["tree_info"][0]["tree_structure"]))) > 1.0

    rf = lgb.train(
        {
            **common,
            "objective": "binary",
            "boosting_type": "rf",
            "learning_rate": 1.0,
            "bagging_freq": 1,
            "bagging_fraction": 0.8,
            "feature_fraction": 1.0,
        },
        lgb.Dataset(X, label=(labels > 0).astype(int), params={"max_bin": 2}),
        num_boost_round=1,
    )
    assert max(map(abs, _leaf_values(rf.dump_model()["tree_info"][0]["tree_structure"]))) > 1.0


def test_max_tree_output_alias_overrides_automatic_guard(device_params):
    X, _, _, dataset, params = _binary_problem()
    params.update(device_params)
    model = lgb.train({**params, "max_tree_output": 0.2}, dataset, num_boost_round=1)
    assert max(map(abs, _leaf_values(model.dump_model()["tree_info"][0]["tree_structure"]))) <= 0.1 * 0.2


def test_partial_parameter_resets_preserve_auto_policy_and_manual_override(device_params):
    _, _, _, dataset, params = _binary_problem()
    params.update(device_params)
    model = lgb.train(params, dataset, num_boost_round=1, keep_training_booster=True)
    model.reset_parameter({"learning_rate": 0.5})
    model.update()
    tree = model.dump_model()["tree_info"][-1]
    assert tree["num_leaves"] == 2
    assert max(map(abs, _leaf_values(tree["tree_structure"]))) == pytest.approx(1.0)
    assert "[max_delta_step: -1]" in model.model_to_string()

    model.reset_parameter({"max_delta_step": 0})
    model.update()
    assert max(map(abs, _leaf_values(model.dump_model()["tree_info"][-1]["tree_structure"]))) > 1.0
    assert "[max_delta_step: 0]" in model.model_to_string()


def test_auto_guard_preserves_small_classification_updates(device_params):
    X, y, _, _, params = _binary_problem()
    dataset = lgb.Dataset(X, label=y, params={"max_bin": 2})
    params.update({**device_params, "learning_rate": 0.05, "lambda_l2": 1.0})
    guarded = lgb.train(params, dataset, num_boost_round=5)
    legacy = lgb.train({**params, "max_delta_step": 0}, dataset, num_boost_round=5)
    assert all(max(map(abs, _leaf_values(tree["tree_structure"]))) < 1.0 for tree in legacy.dump_model()["tree_info"])
    np.testing.assert_allclose(
        guarded.predict(X, raw_score=True), legacy.predict(X, raw_score=True), rtol=1e-6, atol=1e-8
    )


def test_legacy_goss_alias_retains_automatic_guard(device_params):
    X, _, _, dataset, params = _binary_problem()
    params.update(device_params)
    model = lgb.train({**params, "boosting": "goss"}, dataset, num_boost_round=1)
    assert model.dump_model()["tree_info"][0]["num_leaves"] == 2
    assert np.max(np.abs(model.predict(X, raw_score=True))) == pytest.approx(1.0)


@pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only validation")
@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint"])
def test_auto_guard_preserves_quantized_training_math(quant_mode):
    X, _, _, dataset, params = _binary_problem()
    params.update({"device_type": "cuda", "cuda_precision": "fp64", "quant_mode": quant_mode})
    automatic = lgb.train(params, dataset, num_boost_round=1)
    legacy = lgb.train({**params, "max_delta_step": 0}, dataset, num_boost_round=1)
    # Policy serialization changes, but the trained tree must be identical.
    assert automatic.dump_model()["tree_info"] == legacy.dump_model()["tree_info"]
    np.testing.assert_array_equal(automatic.predict(X, raw_score=True), legacy.predict(X, raw_score=True))
    assert "[max_delta_step: -1]" in automatic.model_to_string()
    manual = lgb.train({**params, "max_delta_step": 0.25}, dataset, num_boost_round=1)
    assert np.max(np.abs(manual.predict(X, raw_score=True))) <= 0.1 * 0.25


@pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only validation")
def test_deterministic_cuda_promotion_keeps_quantized_steps_unconstrained():
    _, _, _, dataset, params = _binary_problem()
    params.update({"device_type": "cuda", "cuda_precision": "fp64", "deterministic": True})
    automatic = lgb.train(params, dataset, num_boost_round=1)
    legacy = lgb.train({**params, "max_delta_step": 0}, dataset, num_boost_round=1)
    assert "[quant_mode: fixedpoint]" in automatic.model_to_string()
    assert automatic.dump_model()["tree_info"] == legacy.dump_model()["tree_info"]


def _empty_cpu_split_problem():
    rng = np.random.default_rng(4002)
    y = np.zeros(40, dtype=np.int32)
    y[rng.choice(len(y), 2, replace=False)] = 1
    columns = []
    for column in range(4):
        values = rng.normal(size=len(y))
        if column == 0:
            values[rng.random(len(y)) < 0.5] = 0.0
        elif column == 1:
            values = rng.integers(0, 8, size=len(y)).astype(float)
        elif column == 2:
            values = (rng.random(len(y)) < 0.1).astype(float)
        else:
            values += 2 * y
        columns.append(values)
    return np.column_stack(columns), y


@pytest.mark.parametrize("histogram_layout", ["force_row_wise", "force_col_wise"])
@pytest.mark.parametrize("minimum_hessian", [0.0, 1e-20])
@pytest.mark.parametrize(
    "extra_params",
    [
        {},
        {"linear_tree": True},
        {"cegb_tradeoff": 0.05, "cegb_penalty_feature_lazy": [0.01, 0.02, 0.03, 0.04]},
        {"cegb_tradeoff": 0.05, "cegb_penalty_feature_coupled": [0.01, 0.02, 0.03, 0.04]},
    ],
    ids=["constant", "linear", "cegb_lazy", "cegb_coupled"],
)
def test_tiny_hessian_minimum_does_not_create_empty_cpu_children(histogram_layout, minimum_hessian, extra_params):
    # Hessian-derived counts can admit empty bins because of roundoff.
    # The actual partition must be checked before changing tree or cost state.
    X, y = _empty_cpu_split_problem()
    params = {
        "objective": "binary",
        "device_type": "cpu",
        "quant_mode": "none",
        "learning_rate": 0.1,
        "num_leaves": 64,
        "max_depth": 10,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": minimum_hessian,
        "num_threads": 1,
        histogram_layout: True,
        "verbosity": -1,
        **extra_params,
    }
    model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=30)
    raw = model.predict(X, raw_score=True)
    assert np.isfinite(raw).all()
    prior = np.full(len(y), np.log(y.mean() / (1 - y.mean())))
    assert _binary_logloss(y, raw) < _binary_logloss(y, prior)

    def check_counts(node):
        if "leaf_count" in node:
            assert node["leaf_count"] > 0
            return node["leaf_count"]
        return check_counts(node["left_child"]) + check_counts(node["right_child"])

    for tree in model.dump_model()["tree_info"]:
        assert check_counts(tree["tree_structure"]) == len(y)


def test_forced_empty_cpu_split_preserves_all_rows(tmp_path):
    X, y = _empty_cpu_split_problem()
    forced = tmp_path / "forced.json"
    forced.write_text(json.dumps({"feature": 0, "threshold": 1e100}))
    params = {
        "objective": "binary",
        "device_type": "cpu",
        "quant_mode": "none",
        "num_leaves": 64,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.0,
        "forcedsplits_filename": forced,
        "num_threads": 1,
        "verbosity": -1,
    }
    model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=10)
    assert np.isfinite(model.predict(X, raw_score=True)).all()
    tree = model.dump_model()["tree_info"][0]
    assert tree["num_leaves"] == 1
    assert tree["tree_structure"]["leaf_count"] == len(y)
