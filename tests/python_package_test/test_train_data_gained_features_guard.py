"""A Booster whose training Dataset gained features refuses resets (parameters and training data), refits and new validation sets.

FLC_DatasetAddFeaturesFrom on a Dataset a Booster trains on leaves the tree learner sized for
the old features. Training already refuses ("The training Dataset gained features"), but a
reset_parameter went on into the histogram pool, which re-read the replaced feature metadata
(heap-use-after-free in FeatureHistogram::ResetFunc), and refit and add_valid also worked on
the stale learner.
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB


@pytest.fixture
def booster_with_extended_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0]
    train = flc.Dataset(X, y, free_raw_data=False).construct()
    booster = flc.Booster({"objective": "regression", "verbose": -1}, train)
    booster.update()
    extra = flc.Dataset(rng.normal(size=(100, 2)), free_raw_data=False).construct()
    train.add_features_from(extra)
    return X, y, train, booster


def test_reset_parameter_is_refused(booster_with_extended_data):
    X, y, train, booster = booster_with_extended_data
    with pytest.raises(flc.basic.FalcataError, match="gained features"):
        booster.reset_parameter({"num_leaves": 7, "learning_rate": 0.2})


def test_refit_and_add_valid_are_refused(booster_with_extended_data):
    X, y, train, booster = booster_with_extended_data
    leaf_preds = np.ascontiguousarray(booster.predict(X, pred_leaf=True).astype(np.int32))
    rc = _LIB.FLC_BoosterRefit(
        booster._handle,
        leaf_preds.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(leaf_preds.shape[0]),
        ctypes.c_int32(leaf_preds.shape[1]),
    )
    assert rc == -1
    assert "gained features" in _LIB.FLC_GetLastError().decode()
    with pytest.raises(flc.basic.FalcataError, match="gained features"):
        booster.add_valid(flc.Dataset(X, y, reference=train), "v")


def test_reset_training_data_is_refused(booster_with_extended_data):
    # a subset of the extended Dataset passes GBDT's bin-mapper check against the (equally extended) training
    # Dataset, so the reset used to take the new feature count and then fail in the tree learner, leaving the
    # Booster expecting 5 features: a FastConfig made for 3 then wrote contributions past its output buffer
    X, y, train, booster = booster_with_extended_data
    subset = train.subset(list(range(50))).construct()
    rc = _LIB.FLC_BoosterResetTrainingData(booster._handle, subset._handle)
    assert rc == -1
    assert "gained features" in _LIB.FLC_GetLastError().decode()
    assert booster.predict(X, pred_contrib=True).shape == (X.shape[0], X.shape[1] + 1)
