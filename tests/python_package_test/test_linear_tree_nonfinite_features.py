import numpy as np
import pytest

import falcata as flc

# A linear tree fitted its leaf regressions on the raw feature values, skipping
# only NaN. A +-inf value, or a finite one beyond the float range (1e300 becomes
# inf in the raw float copy), entered X^T H X as inf, the coefficients became
# NaN, and the model predicted NaN for every row. Non-finite values are now
# handled like NaN: the row takes the leaf's constant, at fit and score time.


@pytest.mark.parametrize("bad", [np.inf, -np.inf, 1e300])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_linear_tree_with_non_finite_feature_values(bad, objective):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 4))
    y = 2 * X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(2000)
    if objective == "binary":
        y = (y > 0).astype(float)
    X[rng.random(2000) < 0.05, 1] = bad
    params = {"objective": objective, "linear_tree": True, "deterministic": True, "num_threads": 2, "verbose": -1}
    bst = flc.train(params, flc.Dataset(X, y), num_boost_round=10)
    raw = bst.predict(X, raw_score=True)
    assert np.all(np.isfinite(raw))
    # the fit is still a fit: linear trees beat a constant comfortably
    clean = np.isfinite(X).all(axis=1) & (np.abs(X) < 1e10).all(axis=1)
    if objective == "regression":
        assert np.mean((raw[clean] - y[clean]) ** 2) < 0.5 * np.var(y[clean])


def test_linear_tree_prediction_with_inf_uses_the_leaf_constant():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1000, 3))
    y = 3 * X[:, 0] + 0.1 * rng.standard_normal(1000)
    bst = flc.train({"objective": "regression", "linear_tree": True, "verbose": -1}, flc.Dataset(X, y), 3)
    row = X[:1].copy()
    row[0, 0] = np.inf
    pred = bst.predict(row)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("bad", [np.inf, 1e300])
def test_linear_tree_prediction_matches_training_scores(bad):
    # predict() must score a row the way training did: the training scores read the
    # features from a float copy, where 1e300 is inf and takes the leaf's constant
    rng = np.random.default_rng(2)
    X = rng.standard_normal((1500, 3))
    y = 2 * X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(1500)
    X[rng.random(1500) < 0.05, 1] = bad
    params = {"objective": "regression", "linear_tree": True, "metric": "l2", "verbose": -1}
    ds = flc.Dataset(X, y)
    evals = {}
    bst = flc.train(
        params, ds, num_boost_round=5, valid_sets=[ds], valid_names=["train"], callbacks=[flc.record_evaluation(evals)]
    )
    assert np.isfinite(evals["train"]["l2"][-1])
    np.testing.assert_allclose(np.mean((bst.predict(X) - y) ** 2), evals["train"]["l2"][-1], rtol=1e-5)
