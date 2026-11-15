# coding: utf-8
"""Feature names survive the text model, and validate_features accepts the training frame.

Dataset::set_feature_names replaced only ' ' with '_'. A name with a newline (or an empty
name) trained fine and then failed to load ("Wrong size of feature_names"), which
flc.train itself hit in its final model round trip. And predict(df, validate_features=True)
rejected the very DataFrame the model was trained on when a column name had a space,
because the native side compared the raw "a b" with the stored "a_b".
"""

import numpy as np
import pandas as pd
import pytest

import falcata as flc


def _data():
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    return X, X[:, 0] + X[:, 2]


@pytest.mark.parametrize("name", ["x\ny", "x\ry", "x y"])
def test_names_with_whitespace_round_trip(name):
    X, y = _data()
    booster = flc.train({"verbose": -1}, flc.Dataset(X, y, feature_name=[name, "c", "d"], params={"verbose": -1}), 3)
    assert booster.feature_name() == ["x_y", "c", "d"]
    reloaded = flc.Booster(model_str=booster.model_to_string())
    assert reloaded.feature_name() == ["x_y", "c", "d"]
    np.testing.assert_allclose(reloaded.predict(X), booster.predict(X))


def test_empty_name_is_rejected():
    X, y = _data()
    with pytest.raises(flc.basic.FalcataError, match="Feature name 0 is empty"):
        flc.Dataset(X, y, feature_name=["", "c", "d"], params={"verbose": -1}).construct()


@pytest.mark.parametrize("column", ["a b", "a\nb"])
def test_validate_features_accepts_the_training_frame(column):
    X, y = _data()
    df = pd.DataFrame(X, columns=[column, "c", "d"])
    booster = flc.train({"verbose": -1}, flc.Dataset(df, y, params={"verbose": -1}), 3)
    np.testing.assert_allclose(booster.predict(df, validate_features=True), booster.predict(df))
    with pytest.raises(flc.basic.FalcataError, match="Expected 'a_b' at position 0 but found 'zzz'"):
        booster.predict(df.rename(columns={column: "zzz"}), validate_features=True)
