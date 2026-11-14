"""Linear-tree models must pickle under the default FALB pickle format.

FALB v1 cannot store linear trees, so pickling one used to raise
"FALB v1 cannot store linear trees" -- for the Booster and for every sklearn
estimator holding one. The text payload carries them, and unpickling accepts
both payloads.
"""

import copy
import pickle

import numpy as np
import pytest

import falcata as flc
from falcata import basic


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    x[rng.random(x.shape) < 0.1] = np.nan
    y = 2.0 * np.nan_to_num(x[:, 0]) - np.nan_to_num(x[:, 1])
    return x, y


@pytest.mark.parametrize("fmt", ["falb", "text"])
def test_linear_booster_pickles(linear_data, fmt):
    x, y = linear_data
    booster = flc.train(
        {"linear_tree": True, "verbose": -1, "num_threads": 1, "min_data_in_leaf": 5},
        flc.Dataset(x, label=y),
        num_boost_round=4,
    )
    old = basic.get_pickle_format()
    basic.set_pickle_format(fmt)
    try:
        restored = pickle.loads(pickle.dumps(booster))
    finally:
        basic.set_pickle_format(old)
    np.testing.assert_array_equal(restored.predict(x), booster.predict(x))
    np.testing.assert_array_equal(copy.deepcopy(booster).predict(x), booster.predict(x))


def test_linear_sklearn_estimator_pickles(linear_data):
    x, y = linear_data
    model = flc.FalcataRegressor(linear_tree=True, n_estimators=4, min_child_samples=5, verbose=-1, n_jobs=1).fit(x, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


def test_plain_booster_still_pickles_as_falb(linear_data):
    x, y = linear_data
    booster = flc.train({"verbose": -1, "num_threads": 1}, flc.Dataset(x, label=y), num_boost_round=2)
    old = basic.get_pickle_format()
    basic.set_pickle_format("falb")
    try:
        state = booster.__getstate__()
    finally:
        basic.set_pickle_format(old)
    assert isinstance(state["_handle"], bytes)
    assert state["_handle"][:4] == b"FALB"
