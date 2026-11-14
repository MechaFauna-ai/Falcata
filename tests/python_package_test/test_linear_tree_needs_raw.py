"""A linear-tree Booster on a Dataset built without linear_tree is rejected.

The linear models are fit on the raw feature values, which a Dataset keeps only when it
was constructed with linear_tree=true. The Python package rejects the mismatch through
FLC_DatasetUpdateParamChecking, but FLC_BoosterCreate itself did not check, so the
learner read the Dataset's missing raw buffer (UBSan: reference binding to null pointer in
Dataset::raw_index from LinearTreeLearner::InitLinear).
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB


def test_booster_create_linear_tree_without_raw_data_is_rejected():
    X = np.random.default_rng(0).normal(size=(200, 3))
    ds = flc.Dataset(X, X[:, 0], params={"verbose": -1}).construct()
    booster = ctypes.c_void_p()
    rc = _LIB.FLC_BoosterCreate(
        ds._handle, ctypes.c_char_p(b"objective=regression linear_tree=true verbose=-1"), ctypes.byref(booster)
    )
    assert rc == -1
    assert "linear_tree=true needs a Dataset constructed with linear_tree=true" in _LIB.FLC_GetLastError().decode()


@pytest.mark.parametrize("free_raw_data", [True, False])
def test_linear_tree_dataset_still_trains(free_raw_data):
    X = np.random.default_rng(1).normal(size=(200, 3))
    params = {"objective": "regression", "linear_tree": True, "verbose": -1}
    booster = flc.train(params, flc.Dataset(X, X[:, 0], free_raw_data=free_raw_data), num_boost_round=3)
    assert np.all(np.isfinite(booster.predict(X)))
