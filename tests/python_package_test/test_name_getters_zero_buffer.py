# coding: utf-8
"""The C API name getters write nothing when called with buffer_len == 0.

FLC_DatasetGetFeatureNames, FLC_BoosterGetFeatureNames and
FLC_BoosterGetEvalNames wrote out[i][buffer_len - 1] = '\\0' for every
i < len, so a size query with buffer_len == 0 and len > 0 wrote at offset
SIZE_MAX, i.e. (with pointer wrap-around) one byte before each buffer. They
must report the required size and leave the memory alone.
"""

import ctypes

import numpy as np

import falcata as flc
from falcata.basic import _LIB, _safe_call


def _query(fn, handle, n):
    # each slot points one byte into a sentinel-filled block, so a write to
    # out[i][SIZE_MAX] (== out[i] - 1) or to out[i][0] is caught
    buffers = [ctypes.create_string_buffer(b"\x7f\x7f", 2) for _ in range(n)]
    ptrs = (ctypes.c_char_p * n)(*(ctypes.addressof(b) + 1 for b in buffers))
    out_len = ctypes.c_int(0)
    required = ctypes.c_size_t(0)
    _safe_call(fn(handle, ctypes.c_int(n), ctypes.byref(out_len), ctypes.c_size_t(0), ctypes.byref(required), ptrs))
    assert all(b.raw == b"\x7f\x7f" for b in buffers)
    return out_len.value, required.value


def test_name_getters_size_query_writes_nothing():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X[:, 0]
    names = ["alpha", "b", "gamma_long_name"]
    train = flc.Dataset(X, label=y, feature_name=names, params={"verbose": -1})
    booster = flc.Booster({"objective": "regression", "metric": "l2", "verbose": -1}, train)
    assert _query(_LIB.FLC_DatasetGetFeatureNames, train._handle, 3) == (3, len("gamma_long_name") + 1)
    assert _query(_LIB.FLC_BoosterGetFeatureNames, booster._handle, 3) == (3, len("gamma_long_name") + 1)
    assert _query(_LIB.FLC_BoosterGetEvalNames, booster._handle, 1) == (1, len("l2") + 1)
