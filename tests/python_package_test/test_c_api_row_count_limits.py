# coding: utf-8
"""C API entry points reject row counts that do not fit the int32 row index.

FLC_DatasetCreateFromMats summed per-matrix row counts in int32 without a
check, and the CSR/CSC creators, CSR pushers and CSR/CSC predictors narrowed
their 64-bit row counts with a plain cast, so oversized inputs wrapped
silently (to a negative total, or to a small row count that predicted the
wrong number of rows). They must raise instead. The data arrays passed here are
tiny: the check has to fire before any of them is read.
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, FalcataError, _safe_call

_INT32_MAX = 2**31 - 1


def test_create_from_mats_rejects_total_rows_over_int32():
    mat = np.zeros((1, 2), dtype=np.float64)
    ptr = mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    data = (ctypes.POINTER(ctypes.c_double) * 2)(ptr, ptr)
    nrow = np.array([_INT32_MAX, 2], dtype=np.int32)
    layouts = (ctypes.c_int * 2)(1, 1)
    handle = ctypes.c_void_p()
    with pytest.raises(FalcataError, match="INT32_MAX"):
        _safe_call(
            _LIB.FLC_DatasetCreateFromMats(
                ctypes.c_int32(2),
                data,
                ctypes.c_int(1),  # C_API_DTYPE_FLOAT64
                nrow.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int32(2),
                layouts,
                ctypes.c_char_p(b"verbose=-1"),
                None,
                ctypes.byref(handle),
            )
        )


def test_predict_for_csc_rejects_rows_over_int32():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    booster = flc.train({"objective": "regression", "verbose": -1}, flc.Dataset(X, label=X[:, 0]), num_boost_round=1)
    indptr = np.array([0, 0, 0], dtype=np.int32)
    indices = np.zeros(1, dtype=np.int32)
    values = np.zeros(1, dtype=np.float64)
    preds = np.zeros(8, dtype=np.float64)
    out_len = ctypes.c_int64(0)
    # 2**32 + 5 used to be cast to int 5 and predicted five rows
    with pytest.raises(FalcataError, match="INT32_MAX"):
        _safe_call(
            _LIB.FLC_BoosterPredictForCSC(
                booster._handle,
                indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int(2),  # C_API_DTYPE_INT32
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(1),  # C_API_DTYPE_FLOAT64
                ctypes.c_int64(len(indptr)),
                ctypes.c_int64(0),
                ctypes.c_int64(2**32 + 5),
                ctypes.c_int(0),
                ctypes.c_int(0),
                ctypes.c_int(-1),
                ctypes.c_char_p(b""),
                ctypes.byref(out_len),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
