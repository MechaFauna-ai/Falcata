# coding: utf-8
"""Negative CSR column indices are rejected or ignored, never used as offsets.

FLC_DatasetCreateFromCSR only upper-bounded the column index while sampling,
so a negative index wrote sample_values[-1]. Dataset::PushOneRow (the
reference / streaming path) and the predictor's dense buffer likewise only
checked the upper bound and indexed used_feature_map_[-1] / pred_buf[-1].
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, FalcataError, _safe_call

_INT32, _FLOAT64 = 2, 1  # C_API_DTYPE_INT32, C_API_DTYPE_FLOAT64


def _csr_args(indptr, indices, data):
    indptr = np.asarray(indptr, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    data = np.asarray(data, dtype=np.float64)
    return (
        indptr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(_INT32),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        data.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(_FLOAT64),
        ctypes.c_int64(len(indptr)),
        ctypes.c_int64(len(data)),
    ), (indptr, indices, data)


def test_create_from_csr_rejects_negative_column():
    args, keep = _csr_args([0, 2, 3], [0, -1, 1], [1.0, 2.0, 3.0])
    handle = ctypes.c_void_p()
    with pytest.raises(FalcataError, match="column index -1"):
        _safe_call(
            _LIB.FLC_DatasetCreateFromCSR(
                *args, ctypes.c_int64(2), ctypes.c_char_p(b"verbose=-1"), None, ctypes.byref(handle)
            )
        )


def test_predict_for_csr_ignores_negative_column():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    booster = flc.train({"objective": "regression", "verbose": -1}, flc.Dataset(X, label=X[:, 0]), num_boost_round=3)
    expected = booster.predict(np.array([[0.5, 0.0]]))
    args, keep = _csr_args([0, 2], [-1, 0], [9.0, 0.5])
    out = np.zeros(1, dtype=np.float64)
    out_len = ctypes.c_int64(0)
    _safe_call(
        _LIB.FLC_BoosterPredictForCSR(
            booster._handle,
            *args,
            ctypes.c_int64(2),
            ctypes.c_int(0),
            ctypes.c_int(0),
            ctypes.c_int(-1),
            ctypes.c_char_p(b""),
            ctypes.byref(out_len),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    np.testing.assert_allclose(out, expected)
