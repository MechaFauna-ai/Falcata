# coding: utf-8
"""List-valued numeric parameters reject tokens that are not numbers, with a clear error.

Common::StringToArray parsed integers with Atoi, which read "x" as 0 and "20x" as 20:
monotone_constraints="1,x" silently meant "no constraint" for the second feature, and
max_bin_by_feature="10,20x" trained with 20. Floating-point lists went through
std::stod, which failed with the bare message "stod" -- and also rejected subnormal
values, so a linear-tree model text holding one could not be loaded.
"""

import re

import numpy as np
import pytest

import falcata as flc


def _data():
    rng = np.random.default_rng(0)
    X = rng.random((200, 2))
    return X, (X[:, 0] > 0.5).astype(float)


@pytest.mark.parametrize(
    ("key", "value", "token"),
    [
        ("monotone_constraints", "1,x", "x"),
        ("max_bin_by_feature", "10,20x", "20x"),
        ("interaction_constraints", "[0,x]", "x"),
        ("feature_contri", "1,abc", "abc"),
        ("feature_contri", "1,2junk", "2junk"),
        ("label_gain", "0,1,x", "x"),
        ("auc_mu_weights", "1e999,0", "1e999"),
    ],
)
def test_bad_list_token_is_rejected(key, value, token):
    X, y = _data()
    params = {"verbose": -1, key: value}
    expected = f'Cannot parse "{re.escape(token)}" as an? (integer|number)'
    if key == "interaction_constraints":
        # the interaction_constraints syntax check (#585) may reject the list before its tokens are parsed
        expected += "|Invalid interaction_constraints"
    with pytest.raises(flc.basic.FalcataError, match=expected):
        flc.train(params, flc.Dataset(X, y, params=params), 1)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("monotone_constraints", "+1,-1"),
        ("max_bin_by_feature", "10,20"),
        ("interaction_constraints", "[0],[1]"),
        ("feature_contri", "1.5,2e-310"),
        ("label_gain", "0,1,3"),
    ],
)
def test_good_list_values_are_accepted(key, value):
    X, y = _data()
    params = {"verbose": -1, key: value}
    flc.train(params, flc.Dataset(X, y, params=params), 1)


def test_linear_tree_model_with_subnormal_coefficient_loads():
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = 2 * X[:, 0] + X[:, 1]
    booster = flc.train({"verbose": -1, "linear_tree": True, "num_leaves": 4}, flc.Dataset(X, y), 2)
    text = booster.model_to_string()
    first = re.search(r"leaf_coeff=(\S+)", text).group(1)
    edited = text.replace(f"leaf_coeff={first}", "leaf_coeff=4.9406564584124654e-324", 1)
    loaded = flc.Booster(model_str=edited)
    # the coefficient is ~0: predictions move by at most that term's former contribution
    assert np.all(np.isfinite(loaded.predict(X[:5])))


@pytest.mark.parametrize(
    ("key", "value", "kind"),
    [
        # Atof failed with "Unknown token abc in data file" for a parameter
        ("learning_rate", "abc", "double"),
        # Atoi consumed the sign and returned 0
        ("max_depth", "-", "int"),
    ],
)
def test_bad_scalar_value_is_a_type_error(key, value, kind):
    X, y = _data()
    with pytest.raises(
        flc.basic.FalcataError, match=f'Parameter {key} should be of type {kind}, got "{re.escape(value)}"'
    ):
        flc.train({"verbose": -1, key: value}, flc.Dataset(X, y), 1)
