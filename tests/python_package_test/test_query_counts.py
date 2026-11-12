"""Query counts that do not describe the rows are rejected instead of indexing out of bounds.

A negative group count made the query boundaries go backwards, so the ranking objective
read outside the labels (heap-buffer-overflow under ASan, a std::length_error abort
otherwise). A ``.query`` file summing past the ``.weight`` file read past the weights
while averaging the per-query weights, before the size checks against the data ran.
"""

import numpy as np
import pytest

import falcata as flc


def test_negative_group_count_is_rejected():
    X = np.arange(8.0).reshape(4, 2)
    y = np.array([1.0, 0.0, 1.0, 0.0])
    ds = flc.Dataset(X, y, group=[-2, 6], params={"verbose": -1})
    with pytest.raises(flc.basic.FalcataError, match="Query counts must be non-negative"):
        ds.construct()


def test_negative_count_in_query_file_is_rejected(tmp_path):
    data = tmp_path / "t.csv"
    data.write_text("1,1\n0,2\n1,3\n0,4\n")
    (tmp_path / "t.csv.query").write_text("-2\n6\n")
    with pytest.raises(flc.basic.FalcataError, match="is negative"):
        flc.Dataset(str(data), params={"verbose": -1}).construct()


def test_query_file_longer_than_weight_file_is_rejected(tmp_path):
    data = tmp_path / "t.csv"
    data.write_text("1,1\n0,2\n1,3\n0,4\n")
    (tmp_path / "t.csv.weight").write_text("1\n1\n1\n")
    (tmp_path / "t.csv.query").write_text("2\n2\n")
    with pytest.raises(flc.basic.FalcataError, match="Weights size doesn't match data size"):
        flc.Dataset(str(data), params={"verbose": -1}).construct()


def test_matching_query_and_weight_files_still_train(tmp_path):
    data = tmp_path / "t.csv"
    data.write_text("".join(f"{i % 3},{i},{i * 0.5}\n" for i in range(12)))
    (tmp_path / "t.csv.weight").write_text("".join(f"{1 + i % 2}\n" for i in range(12)))
    (tmp_path / "t.csv.query").write_text("4\n8\n")
    ds = flc.Dataset(str(data), params={"verbose": -1, "min_data_in_bin": 1})
    booster = flc.train(
        {"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 1, "num_leaves": 3}, ds, num_boost_round=2
    )
    assert np.all(np.isfinite(booster.predict(np.arange(24.0).reshape(12, 2))))
