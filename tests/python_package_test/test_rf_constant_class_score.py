import numpy as np
import pytest

import falcata as flc

# With 100+ feature groups RF bagging does not use a subset, so the training
# score of a class with nothing to learn (constant labels) went through the tree
# learner: the serial learner added nothing for the 1-leaf tree, and the linear
# learner read linear metadata the constant tree does not have (segfault).
N_FEATURES = 120


def _data(objective):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, N_FEATURES))
    if objective == "binary":
        y = np.zeros(400)
    else:
        y = rng.integers(0, 2, 400).astype(float)  # class 2 never occurs
    return X, y


def _params(objective, linear_tree):
    params = {
        "objective": objective,
        "boosting": "rf",
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "linear_tree": linear_tree,
        "num_threads": 1,
        "verbose": -1,
    }
    if objective != "binary":
        params["num_class"] = 3
        params["metric"] = "multi_logloss"
    else:
        params["metric"] = "binary_logloss"
    return params


@pytest.mark.parametrize("objective", ["binary", "multiclassova"])
@pytest.mark.parametrize("linear_tree", [False, True])
def test_rf_constant_class_training_score_matches_predictions(objective, linear_tree):
    X, y = _data(objective)
    params = _params(objective, linear_tree)
    train = flc.Dataset(X, y)
    bst = flc.train(params, train, num_boost_round=3, keep_training_booster=True)
    reference = flc.Dataset(X, y, reference=train)
    bst.add_valid(reference, "copy")
    ((_, _, train_metric, _),) = bst.eval_train()
    ((_, _, valid_metric, _),) = bst.eval_valid()
    np.testing.assert_allclose(train_metric, valid_metric, rtol=1e-6, atol=1e-12)
    assert np.all(np.isfinite(bst.predict(X)))


@pytest.mark.parametrize("n_features", [5, N_FEATURES])
def test_rf_absent_class_prediction_does_not_drift_with_iterations(n_features):
    # Only the first round's tree of a class with nothing to learn carried the
    # class's constant; later rounds added 0-output trees, and RF averages its
    # trees, so the absent class's probability rose from ~0 toward 0.5.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, n_features))
    y = rng.integers(0, 2, 400).astype(float)  # class 2 never occurs
    params = {**_params("multiclassova", False), "metric": "None"}
    probs = [
        flc.train(params, flc.Dataset(X, y), num_boost_round=rounds).predict(X[:20])[:, 2].max()
        for rounds in (1, 10, 50)
    ]
    assert max(probs) < 1e-6, probs
    bin_params = {**_params("binary", False), "metric": "None"}
    bst = flc.train(bin_params, flc.Dataset(X, np.zeros(400)), num_boost_round=50)
    assert bst.predict(X[:20]).max() < 1e-6


@pytest.mark.parametrize("objective", ["cross_entropy", "regression", "regression_l1", "binary"])
def test_rf_empty_bag_keeps_predictions_finite(objective):
    # Bagging draws each row independently, so 5 rows at bagging_fraction=0.3
    # often get an empty bag. The tree grown on it had a 0/0 = NaN root output
    # (handoff from fz-model: rf + cross_entropy, 5 rows, a -nan constant tree).
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3))
    y = {"binary": np.array([0.0, 1.0, 0.0, 1.0, 1.0]), "cross_entropy": rng.random(5)}.get(
        objective, rng.standard_normal(5)
    )
    for seed in range(20):
        params = {
            "objective": objective,
            "boosting": "rf",
            "bagging_fraction": 0.3,
            "bagging_freq": 1,
            "bagging_seed": seed,
            "min_data_in_leaf": 1,
            "verbose": -1,
        }
        bst = flc.train(params, flc.Dataset(X, y), num_boost_round=10)
        assert np.all(np.isfinite(bst.predict(X))), seed
