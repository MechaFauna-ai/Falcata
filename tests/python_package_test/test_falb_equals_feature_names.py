"""Feature names containing '=' must survive FALB, pickle and zero-tree text models.

The text loader parsed every line before the first "Tree=" as a header key. A
model with no trees has no such line, so it also parsed the feature importance
lines after "end of trees", and "a=b=3" (feature "a=b", 3 splits) was rejected
as "Wrong line at model file". Every FALB metadata block is a zero-tree model,
so FALB loads -- and with them the default pickle -- failed for such names.
"""

import pickle

import numpy as np
import pytest

import falcata as flc
from falcata import basic


@pytest.fixture
def booster_and_data():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 3))
    x[:, 1] = rng.integers(0, 5, size=300)
    y = x[:, 0] + x[:, 1]
    names = ["a=b", "c=1=2", "d"]
    ds = flc.Dataset(x, label=y, feature_name=names, categorical_feature=[1])
    booster = flc.train({"verbose": -1, "num_threads": 1}, ds, num_boost_round=3)
    return booster, x, names


def test_falb_round_trip_with_equals_in_feature_names(booster_and_data):
    booster, x, names = booster_and_data
    loaded = flc.Booster(model_bin=booster.model_to_binary(with_stats=True, with_diagnostics=True))
    assert loaded.feature_name() == names
    np.testing.assert_array_equal(loaded.predict(x), booster.predict(x))


def test_falb_pickle_with_equals_in_feature_names(booster_and_data):
    booster, x, _ = booster_and_data
    old = basic.get_pickle_format()
    basic.set_pickle_format("falb")
    try:
        restored = pickle.loads(pickle.dumps(booster))
    finally:
        basic.set_pickle_format(old)
    np.testing.assert_array_equal(restored.predict(x), booster.predict(x))


def test_zero_tree_text_model_with_equals_in_feature_names(booster_and_data):
    booster, _, names = booster_and_data
    empty = flc.Booster(model_str=booster.model_to_string(start_iteration=booster.current_iteration()))
    assert empty.num_trees() == 0
    assert empty.feature_name() == names
