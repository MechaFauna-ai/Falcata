"""FLC_BoosterRefit rejects leaf predictions outside a tree's leaves.

GBDT::RefitTree checked each leaf index with a CHECK inside an OpenMP loop: an index
past the tree's leaves threw inside the parallel region and terminated the process, and
a negative index was not checked at all. It also checked lazily, after refitting the
earlier trees.
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB


def _booster():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + rng.normal(size=60) * 0.1
    ds = flc.Dataset(X, y, free_raw_data=False)
    booster = flc.train(
        {"objective": "regression", "num_leaves": 4, "verbose": -1}, ds, num_boost_round=3, keep_training_booster=True
    )
    return X, booster


def _refit(booster, leaf_preds):
    leaf_preds = np.ascontiguousarray(leaf_preds, dtype=np.int32)
    return _LIB.FLC_BoosterRefit(
        booster._handle,
        leaf_preds.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(leaf_preds.shape[0]),
        ctypes.c_int32(leaf_preds.shape[1]),
    )


@pytest.mark.parametrize("bad", [99, -1])
def test_out_of_range_leaf_prediction_is_rejected(bad):
    X, booster = _booster()
    before = booster.predict(X)
    leaf_preds = booster.predict(X, pred_leaf=True)
    leaf_preds[-1, -1] = bad  # the last tree: the earlier ones must not be refit either
    assert _refit(booster, leaf_preds) == -1
    assert "is outside the tree's" in _LIB.FLC_GetLastError().decode()
    np.testing.assert_allclose(booster.predict(X), before)


def test_valid_leaf_predictions_still_refit():
    X, booster = _booster()
    assert _refit(booster, booster.predict(X, pred_leaf=True)) == 0
