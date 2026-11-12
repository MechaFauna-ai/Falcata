"""A single-target objective rejects a 2-D label instead of training on its first column.

Metadata accepts any multiple of #data labels (column-major) for objective=multi_regression,
but every other objective reads only the first #data of them, so y with shape (n, 2) passed
by mistake silently trained a model of y[:, 0].
"""

import numpy as np
import pytest

import falcata as flc


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    Y = np.column_stack([X[:, 0], -X[:, 1]])
    return X, Y


@pytest.mark.parametrize("objective", ["regression", "binary", "huber"])
def test_two_column_label_is_rejected(objective):
    X, Y = _data()
    if objective == "binary":
        Y = (Y > 0).astype(float)
    with pytest.raises(flc.basic.FalcataError, match="takes one label per row"):
        flc.train({"objective": objective, "verbose": -1}, flc.Dataset(X, Y), num_boost_round=2)


def test_two_column_validation_label_is_rejected():
    X, Y = _data()
    train = flc.Dataset(X, Y[:, 0])
    valid = flc.Dataset(X, Y, reference=train)
    with pytest.raises(flc.basic.FalcataError, match="validation data has 200 labels for 100 rows"):
        flc.train({"objective": "regression", "verbose": -1}, train, num_boost_round=2, valid_sets=[valid])


def test_multi_regression_still_takes_two_columns():
    X, Y = _data()
    params = {"objective": "multi_regression", "num_class": 2, "verbose": -1}
    booster = flc.train(params, flc.Dataset(X, Y), num_boost_round=3)
    assert booster.predict(X).shape == (100, 2)


def test_custom_objective_is_left_alone():
    X, Y = _data()

    def fobj(preds, data):
        y = data.get_label()[: len(preds)]
        return preds - y, np.ones_like(preds)

    booster = flc.train({"objective": fobj, "verbose": -1}, flc.Dataset(X, Y), num_boost_round=2)
    assert booster.predict(X).shape == (100,)
