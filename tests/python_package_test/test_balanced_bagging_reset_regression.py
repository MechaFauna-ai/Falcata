"""Resetting balanced bagging restores ordinary row sampling."""

import numpy as np

import falcata as flc


def test_disable_balanced_bagging():
    x = np.random.default_rng(8).normal(size=(200, 3))
    data = flc.Dataset(x, label=(x[:, 0] > 0).astype(int), free_raw_data=False)
    params = {
        "objective": "binary",
        "pos_bagging_fraction": 0.8,
        "neg_bagging_fraction": 0.8,
        "bagging_freq": 1,
        "num_leaves": 3,
        "min_data_in_leaf": 5,
        "num_threads": 1,
        "verbosity": -1,
    }
    booster = flc.train(params, data, num_boost_round=1, keep_training_booster=True)
    initial = booster.model_to_string()
    reset = {"pos_bagging_fraction": 1.0, "neg_bagging_fraction": 1.0, "bagging_fraction": 0.8}
    booster.reset_parameter(reset)
    booster.update()
    expected = flc.train({**params, **reset}, data, num_boost_round=1, init_model=flc.Booster(model_str=initial))
    np.testing.assert_allclose(booster.predict(x), expected.predict(x), rtol=0, atol=0)
