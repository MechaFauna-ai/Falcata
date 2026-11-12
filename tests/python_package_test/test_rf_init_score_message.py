"""boosting=rf with an init_score fails with a clear message, not a bare CHECK."""

import numpy as np
import pytest

import falcata as flc


def test_rf_init_score_is_rejected_clearly():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0]
    ds = flc.Dataset(X, y, init_score=np.zeros(100))
    params = {"boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1, "verbose": -1}
    with pytest.raises(flc.basic.FalcataError, match="boosting=rf does not support init_score"):
        flc.train(params, ds, num_boost_round=2)


def test_rf_without_init_score_still_trains():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    params = {"boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1, "verbose": -1}
    booster = flc.train(params, flc.Dataset(X, X[:, 0]), num_boost_round=3)
    assert np.all(np.isfinite(booster.predict(X)))
