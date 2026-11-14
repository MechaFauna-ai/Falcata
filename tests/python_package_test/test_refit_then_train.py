"""A Booster keeps training after an in-place refit (FLC_BoosterRefit).

The refit re-sized the data partition to the refit tree's number of leaves and never
restored it, so the next iteration failed with "Check failed: (tree->num_leaves()) <=
(data_partition_->num_leaves())". With linear trees it went further wrong first:
GBDT::RefitTree also re-initialised the linear learner's per-leaf workspaces for the largest
refit tree only, and CalculateLinear then indexed them past that size
(heap-buffer-overflow under ASan).
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, _safe_call


@pytest.mark.parametrize("linear", [True, False])
def test_booster_trains_after_in_place_refit(linear):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(132, 4))
    y = X[:, 0] * 2 + rng.normal(size=132)
    params = {"objective": "regression", "verbose": -1, "linear_tree": linear, "min_data_in_leaf": 5}
    ds = flc.Dataset(X, y, params={"linear_tree": linear, "verbose": -1}, free_raw_data=False)
    booster = flc.Booster(params, ds)
    booster.update()
    leaf_preds = np.ascontiguousarray(booster.predict(X, pred_leaf=True).astype(np.int32))
    _safe_call(
        _LIB.FLC_BoosterRefit(
            booster._handle,
            leaf_preds.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(leaf_preds.shape[0]),
            ctypes.c_int32(leaf_preds.shape[1]),
        )
    )
    for _ in range(3):
        booster.update()
    assert booster.current_iteration() == 4
    assert np.all(np.isfinite(booster.predict(X)))
