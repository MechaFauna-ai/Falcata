# coding: utf-8
"""A validation text file whose header orders the training features differently is refused.

Validation data is read by position into the training data's bins. A header naming the same
columns in another order was accepted, so the features were silently scrambled (here the
validation l2 went from ~0.04 to ~3.6) and early stopping watched a meaningless metric.
Headers naming other columns entirely are still read by position, as before.
"""

import numpy as np
import pytest

import falcata as flc


def _write(path, header, X, y, cols):
    rows = "\n".join(",".join(str(v) for v in [y[i], *X[i, cols]]) for i in range(len(y)))
    path.write_text(header + "\n" + rows + "\n")


@pytest.fixture
def train_file(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = 5 * X[:, 0] + 0.1 * rng.random(300)
    _write(tmp_path / "train.csv", "label,a,b,c", X, y, [0, 1, 2])
    return tmp_path, X, y


def _valid_l2(tmp_path, name):
    params = {"verbose": -1, "header": True}
    train = flc.Dataset(str(tmp_path / "train.csv"), params=params)
    valid = flc.Dataset(str(tmp_path / name), params=params, reference=train)
    evals = {}
    flc.train({"verbose": -1, "metric": "l2"}, train, 20, valid_sets=[valid], callbacks=[flc.record_evaluation(evals)])
    return evals["valid_0"]["l2"][-1]


def test_reordered_header_is_refused(train_file):
    tmp_path, X, y = train_file
    _write(tmp_path / "valid.csv", "label,c,b,a", X, y, [2, 1, 0])
    with pytest.raises(
        flc.basic.FalcataError, match='Column "c" is feature 0 of .* but feature 2 of the training data'
    ):
        _valid_l2(tmp_path, "valid.csv")


@pytest.mark.parametrize("header", ["label,a,b,c", "label,x,y,z"])
def test_same_or_renamed_header_still_loads(train_file, header):
    tmp_path, X, y = train_file
    _write(tmp_path / "valid.csv", header, X, y, [0, 1, 2])
    assert _valid_l2(tmp_path, "valid.csv") < 0.5


@pytest.mark.parametrize("ncol", [2, 4])
def test_csv_validation_file_column_count_must_match(tmp_path, ncol):
    # read by position: an extra column was dropped and a missing one read as 0, silently
    rng = np.random.default_rng(0)
    X = rng.random((200, 4))
    y = X[:, 0]
    for name, k in [("train.csv", 3), ("valid.csv", ncol)]:
        rows = "\n".join(",".join(str(v) for v in [y[i], *X[i, :k]]) for i in range(200))
        (tmp_path / name).write_text(rows + "\n")
    train = flc.Dataset(str(tmp_path / "train.csv"), params={"verbose": -1})
    with pytest.raises(flc.basic.FalcataError, match=f"has {ncol} feature columns, but the training data has 3"):
        flc.Dataset(str(tmp_path / "valid.csv"), params={"verbose": -1}, reference=train).construct()


def test_user_feature_names_do_not_trigger_the_order_check(train_file):
    # display names set by the user need not describe the training file's header: the check reads the file
    tmp_path, X, y = train_file
    _write(tmp_path / "valid.csv", "label,a,b,c", X, y, [0, 1, 2])
    params = {"verbose": -1, "header": True}
    train = flc.Dataset(str(tmp_path / "train.csv"), params=params, feature_name=["c", "b", "a"])
    valid = flc.Dataset(str(tmp_path / "valid.csv"), params=params, reference=train)
    flc.train({"verbose": -1, "metric": "l2"}, train, 2, valid_sets=[valid])
