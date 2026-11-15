# coding: utf-8
"""Bad dataset inputs fail with a clear message, and a missing label column fails at all.

label_column one past the last column of a CSV/TSV file was accepted: the loader's range
check compared against the feature count, which then included every column, so every
label read as 0. Other bad inputs failed with an internal CHECK message such as
"Check failed: (static_cast<size_t>(num_col)) == (config_.max_bin_by_feature.size()) at <file>".
"""

import re

import numpy as np
import pytest

import falcata as flc


@pytest.fixture
def csv_4_columns(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    path = tmp_path / "train.csv"
    np.savetxt(path, np.column_stack([10 * X[:, 0], X]), delimiter=",")
    return str(path)


def test_label_column_one_past_the_last_is_rejected(csv_4_columns):
    with pytest.raises(flc.basic.FalcataError, match=r"label_column 4 is out of range: the data file .* has 4 columns"):
        flc.Dataset(csv_4_columns, params={"verbose": -1, "label_column": 4}).construct()


def test_last_column_as_label_still_works(csv_4_columns):
    ds = flc.Dataset(csv_4_columns, params={"verbose": -1, "label_column": 3}).construct()
    assert ds.num_feature() == 3
    assert np.any(ds.get_label() != 0)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"weight_column": 9}, "weight_column 9 is out of range: the data has 3 columns besides the label"),
        ({"group_column": 9}, "group_column 9 is out of range: the data has 3 columns besides the label"),
        ({"max_bin_by_feature": [10, 20]}, "max_bin_by_feature has 2 values, but the data has 3 features"),
    ],
)
def test_bad_text_file_params_have_clear_errors(csv_4_columns, params, message):
    with pytest.raises(flc.basic.FalcataError, match=re.escape(message)):
        flc.Dataset(csv_4_columns, params={"verbose": -1, **params}).construct()


@pytest.mark.parametrize(
    ("max_bin_by_feature", "message"),
    [
        ([10, 20], "max_bin_by_feature has 2 values, but the data has 3 features; give one per feature"),
        ([10, 1, 5], "max_bin_by_feature must be greater than 1 for every feature, got 1 for feature 1"),
    ],
)
def test_max_bin_by_feature_errors_are_clear(max_bin_by_feature, message):
    X = np.random.default_rng(0).random((200, 3))
    with pytest.raises(flc.basic.FalcataError, match=re.escape(message)):
        flc.Dataset(X, X[:, 0], params={"verbose": -1, "max_bin_by_feature": max_bin_by_feature}).construct()


def test_eval_at_must_be_positive():
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    y = rng.integers(0, 3, 200)
    with pytest.raises(flc.basic.FalcataError, match="eval_at positions must be positive, got 0"):
        flc.train({"verbose": -1, "objective": "lambdarank", "eval_at": [0]}, flc.Dataset(X, y, group=[100, 100]), 1)


def test_empty_subset_error_is_clear():
    X = np.random.default_rng(0).random((200, 3))
    ds = flc.Dataset(X, X[:, 0], params={"verbose": -1}).construct()
    with pytest.raises(flc.basic.FalcataError, match="Cannot create a subset with no rows"):
        ds.subset([]).construct()
