"""A DART iteration that adds no tree must leave the model unchanged.

DART::TrainOneIter drops trees before growing the new one: each dropped tree is negated (Shrinkage(-1)) and its
contribution taken out of the training score. Normalize gives the dropped trees their new weight afterwards. When no
leaf met the split requirements, GBDT::TrainOneIter returned "finished" and DART returned before Normalize, so the
dropped trees stayed negated in the model. CUDA hits this often (a stopped iteration ended a DART run with the first
tree, which carries the init score, negated).
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

_DEVICES = ["cpu"] + (["cuda"] if getenv("TASK", "") == "cuda" else [])


@pytest.mark.parametrize("device_type", _DEVICES)
@pytest.mark.parametrize("xgboost_dart_mode", [False, True])
def test_dart_update_without_a_split_keeps_the_model(device_type, xgboost_dart_mode):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 5))
    y = (X[:, 0] + 0.3 * rng.standard_normal(2000) > 0).astype(float)
    params = {
        "objective": "binary",
        "boosting": "dart",
        "device_type": device_type,
        "num_leaves": 7,
        "learning_rate": 0.3,
        "drop_rate": 0.5,
        "skip_drop": 0.0,
        "xgboost_dart_mode": xgboost_dart_mode,
        "verbose": -1,
        "seed": 3,
    }
    train = flc.Dataset(X, y, params=params)
    booster = flc.Booster(params, train)
    for _ in range(10):
        booster.update()
    before = booster.predict(X)
    booster.reset_parameter({"min_gain_to_split": 1e10})
    assert booster.update(), "the update must find no split"
    assert booster.num_trees() == 10
    np.testing.assert_allclose(booster.predict(X), before, rtol=1e-12, atol=1e-12)
    # the training score is the model's again
    np.testing.assert_allclose(booster._Booster__inner_predict(data_idx=0), before, rtol=1e-9, atol=1e-12)
