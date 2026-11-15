# coding: utf-8
"""Malformed interaction_constraints are rejected.

The value was split on brackets, so "x", "[[0,1]" or "[]x" were read as fewer constraints (or none) without
an error, and the raw string went into the model's parameters, which then failed to load in Python
(JSONDecodeError while reading the loaded parameters).
"""

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError

_PARAMS = {"objective": "regression", "verbose": -1, "device_type": "cpu"}


def _data():
    X = np.random.default_rng(0).normal(size=(200, 4))
    return X, X[:, 0] * X[:, 1] + X[:, 2]


@pytest.mark.parametrize("value", ["x", "0,1", "[[0,1]", "[0,1]]", "[[0,1],[2,3]]x", "[]x", "[[[0,1]]]", "[0;1]"])
def test_malformed_interaction_constraints_are_rejected(value):
    X, y = _data()
    with pytest.raises(FalcataError, match="Invalid interaction_constraints"):
        flc.train({**_PARAMS, "interaction_constraints": value}, flc.Dataset(X, y), 2)
    booster = flc.train(_PARAMS, flc.Dataset(X, y), 1)
    with pytest.raises(FalcataError, match="Invalid interaction_constraints"):
        booster.reset_parameter({"interaction_constraints": value})


@pytest.mark.parametrize(
    "value", ["[0,1],[2,3]", "[[0,1],[2,3]]", " [ 0, 1 ] , [2,3] ", [[0, 1], [2, 3]], "[[0,1],[]]"]
)
def test_wellformed_interaction_constraints_round_trip(value):
    X, y = _data()
    booster = flc.train({**_PARAMS, "interaction_constraints": value}, flc.Dataset(X, y), 3)
    reloaded = flc.Booster(model_str=booster.model_to_string())
    np.testing.assert_allclose(reloaded.predict(X), booster.predict(X))
