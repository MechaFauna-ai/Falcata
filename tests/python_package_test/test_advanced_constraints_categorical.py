# coding: utf-8
"""monotone_constraints_method=advanced on data with a categorical feature and no monotone constraint.

The categorical split finder read the split's constraints from the feature constraint whether or not monotone
constraints were in use; only their use initialises the advanced constraints' cumulative arrays, so without
them it read empty arrays (segfault in AdvancedFeatureConstraints::LeftToBasicConstraint).
"""

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("monotone_constraints", [None, [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
def test_advanced_method_with_categorical_feature(monotone_constraints):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6))
    X[:, 2] = rng.integers(0, 6, 400)
    y = X[:, 0] + X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=400)
    params = {"objective": "regression", "verbose": -1, "device_type": "cpu"}
    if monotone_constraints is not None:
        params["monotone_constraints"] = monotone_constraints
    advanced = flc.train(
        {**params, "monotone_constraints_method": "advanced"}, flc.Dataset(X, y, categorical_feature=[2]), 5
    )
    assert np.isfinite(advanced.predict(X)).all()
    if monotone_constraints is None or not any(monotone_constraints):
        # without a monotone constraint the method changes nothing
        basic = flc.train(params, flc.Dataset(X, y, categorical_feature=[2]), 5)
        np.testing.assert_allclose(advanced.predict(X), basic.predict(X))
