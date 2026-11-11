"""Removing the final forest tree must leave usable score buffers."""

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("with_valid", [False, True])
def test_rollback_first_forest_iteration(with_valid):
    x = np.random.default_rng(3).normal(size=(100, 3))
    train = flc.Dataset(x, label=x[:, 0])
    valid = flc.Dataset(x + 0.1, label=x[:, 0], reference=train)
    booster = flc.train(
        {
            "objective": "regression",
            "boosting": "rf",
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": 0.8,
            "num_threads": 2,
            "verbosity": -1,
        },
        train,
        num_boost_round=1,
        valid_sets=[valid] if with_valid else None,
        keep_training_booster=True,
    )
    booster.rollback_one_iter()
    assert booster.current_iteration() == 0
    np.testing.assert_array_equal(booster._Booster__inner_predict(data_idx=0), 0.0)
    if with_valid:
        np.testing.assert_array_equal(booster._Booster__inner_predict(data_idx=1), 0.0)
    booster.update()
    assert np.isfinite(booster._Booster__inner_predict(data_idx=0)).all()
    assert np.isfinite(booster.eval_train()[0][2])
