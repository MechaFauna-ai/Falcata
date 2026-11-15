# coding: utf-8
"""categorical_feature / ignore_column indices at or past the feature count are errors.

They used to be ignored silently: categorical_feature=[3] on 3-feature data (a typo for
2, say) left the intended feature numerical, and ignore_column=9 on a text file ignored
nothing, with no message.
"""

import numpy as np
import pytest

import falcata as flc


@pytest.fixture
def data():
    X = np.random.default_rng(0).random((200, 3))
    return X, 10 * X[:, 0]


@pytest.mark.parametrize("index", [3, 9])
def test_out_of_range_categorical_feature_is_rejected(data, index):
    X, y = data
    with pytest.raises(
        flc.basic.FalcataError, match=f"categorical_feature {index} is out of range: the data has 3 features"
    ):
        flc.Dataset(X, y, categorical_feature=[index], params={"verbose": -1}).construct()


@pytest.mark.parametrize(
    ("param", "value"), [("categorical_feature", "3"), ("ignore_column", "3"), ("ignore_column", "9")]
)
def test_out_of_range_text_file_index_is_rejected(tmp_path, data, param, value):
    X, y = data
    path = tmp_path / "train.csv"
    np.savetxt(path, np.column_stack([y, X]), delimiter=",")
    with pytest.raises(flc.basic.FalcataError, match=f"{param} {value} is out of range: the data has 3 features"):
        flc.Dataset(str(path), params={"verbose": -1, param: value}).construct()


def test_in_range_indices_still_work(tmp_path, data):
    X, y = data
    Xc = X.copy()
    Xc[:, 2] = np.arange(200) % 4
    ds = flc.Dataset(Xc, y, categorical_feature=[2], params={"verbose": -1}).construct()
    assert ds.num_feature() == 3
    path = tmp_path / "train.csv"
    np.savetxt(path, np.column_stack([y, X]), delimiter=",")
    assert flc.Dataset(str(path), params={"verbose": -1, "ignore_column": "2"}).construct().num_feature() == 3
