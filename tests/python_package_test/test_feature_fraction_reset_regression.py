"""Restoring all features must remove the preceding tree's sampled mask."""

import numpy as np

import falcata as flc


def test_reset_feature_fraction_to_one():
    x = np.random.default_rng(8).normal(size=(200, 10))
    y = 10 * x[:, 0] + x[:, 1]
    params = {
        "objective": "regression",
        "feature_fraction": 0.1,
        "feature_fraction_seed": 4,
        "num_leaves": 3,
        "min_data_in_leaf": 5,
        "num_threads": 2,
        "verbosity": -1,
    }
    data = flc.Dataset(x, label=y, free_raw_data=False)
    booster = flc.train(params, data, num_boost_round=1, keep_training_booster=True)
    initial = booster.model_to_string()
    booster.reset_parameter({"feature_fraction": 1.0})
    booster.update()
    expected = flc.train(
        {**params, "feature_fraction": 1.0},
        data,
        num_boost_round=1,
        init_model=flc.Booster(model_str=initial),
    )
    np.testing.assert_allclose(booster.predict(x), expected.predict(x), rtol=0, atol=0)
