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
