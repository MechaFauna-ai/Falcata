"""CUDA training on a Dataset subset through the packed per-tree column view.

A subset's CUDAColumnData (CUDAColumnData::CopySubrow) did not carry the per-column sparse flags, which
SetCompactPackedColumnView reads: cv() on CUDA (its folds are subsets) segfaulted once the packed read served 8-bit
rows (#613, split_packed_read8); with nibble rows (max_bin <= 15) the same path was reachable before.

The full Dataset is trained on once first: its device columns are built on first use, and only a parent whose columns
are on the device makes CopySubrow build the folds from them. Without that, each fold builds its own column data
(CUDAColumnData::Init, which sets the flags) and the test passes with or without the fix.
"""

import numpy as np
import pytest

import falcata as flc

from .utils import BuildInfo


@pytest.mark.skipif(not BuildInfo.has_cuda, reason="the packed column view exists only on CUDA")
@pytest.mark.parametrize("max_bin", [15, 255])
@pytest.mark.parametrize("feature_fraction", [1.0, 0.5])
def test_cv_folds_train_through_the_packed_view(max_bin, feature_fraction):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(1500, 12))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=1500) * 0.1
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "verbose": -1,
        "max_bin": max_bin,
        "feature_fraction": feature_fraction,
        "deterministic": True,
    }
    ds = flc.Dataset(X, y, params={"max_bin": max_bin}, free_raw_data=False)
    flc.train(params, ds, num_boost_round=1)  # puts the parent's columns on the device, so the folds copy from them
    res = flc.cv(params, ds, num_boost_round=5, nfold=3, stratified=False, shuffle=False, return_cvbooster=True)
    for booster in res["cvbooster"].boosters:
        assert np.all(np.isfinite(booster.predict(X)))
