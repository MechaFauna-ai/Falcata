# coding: utf-8
"""monotone_constraints takes -1, 0 or 1; any other value is an error.

Values such as 2 or -5 were accepted, and the learners disagree on them: most code tests
the sign, but the advanced method's constraint propagation tests == 1 / == -1, so a
constraint of 2 trained a model that is not monotone in that feature.
"""

import numpy as np
import pytest

import falcata as flc


def _data():
    rng = np.random.default_rng(0)
    X = rng.random((500, 3))
    return X, 3 * X[:, 0] + np.sin(8 * X[:, 1]) + 0.3 * rng.normal(size=500)


@pytest.mark.parametrize("value", [2, -5, 127])
def test_monotone_constraint_outside_minus_one_to_one_is_rejected(value):
    X, y = _data()
    params = {"verbose": -1, "monotone_constraints": [0, value, 0]}
    with pytest.raises(
        flc.basic.FalcataError, match=f"monotone_constraints must be -1, 0 or 1 .* got {value} for feature 1"
    ):
        flc.train(params, flc.Dataset(X, y), 1)


@pytest.mark.parametrize("method", ["basic", "intermediate", "advanced"])
def test_valid_monotone_constraints_still_train(method):
    X, y = _data()
    params = {"verbose": -1, "monotone_constraints": [1, -1, 0], "monotone_constraints_method": method}
    flc.train(params, flc.Dataset(X, y), 2)


def test_unknown_monotone_constraints_method_is_rejected():
    # any other name silently trained with the basic method
    X, y = _data()
    params = {"verbose": -1, "monotone_constraints": [1, 0, 0], "monotone_constraints_method": "advnaced"}
    with pytest.raises(flc.basic.FalcataError, match='Unknown monotone_constraints_method "advnaced"'):
        flc.train(params, flc.Dataset(X, y), 1)


def test_monotone_constraints_method_is_case_insensitive():
    # "Advanced" used to fall back to basic: it must train exactly like "advanced"
    X, y = _data()

    def model(method):
        params = {"verbose": -1, "monotone_constraints": [1, -1, 0], "monotone_constraints_method": method}
        return flc.train(params, flc.Dataset(X, y), 3).model_to_string().split("parameters:")[0]

    assert model("Advanced") == model("advanced")
