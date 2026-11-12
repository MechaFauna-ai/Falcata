"""The C API rejects an unknown predict_type.

FLC_BoosterPredictFor* and FLC_BoosterCalcNumPredict treated any predict_type other than
raw score, leaf index or contrib as C_API_PREDICT_NORMAL, so a wrong value (for example a
caller's own enum, off by one) silently returned normal predictions.
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB

C_API_DTYPE_FLOAT64 = 1


def _booster():
    X = np.random.default_rng(0).normal(size=(40, 3))
    return X, flc.train({"objective": "regression", "verbose": -1}, flc.Dataset(X, X[:, 0]), num_boost_round=2)


@pytest.mark.parametrize("predict_type", [-1, 4, 100])
def test_unknown_predict_type_is_rejected(predict_type):
    X, booster = _booster()
    X = np.ascontiguousarray(X[:5])
    out_len = ctypes.c_int64()
    assert _LIB.FLC_BoosterCalcNumPredict(booster._handle, 5, predict_type, 0, -1, ctypes.byref(out_len)) == -1
    assert "Unknown predict_type" in _LIB.FLC_GetLastError().decode()
    out = np.zeros(5)
    rc = _LIB.FLC_BoosterPredictForMat(
        booster._handle,
        X.ctypes.data_as(ctypes.c_void_p),
        C_API_DTYPE_FLOAT64,
        ctypes.c_int32(5),
        ctypes.c_int32(3),
        1,
        predict_type,
        0,
        -1,
        ctypes.c_char_p(b""),
        ctypes.byref(out_len),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert rc == -1
    assert "Unknown predict_type" in _LIB.FLC_GetLastError().decode()
    fast = ctypes.c_void_p()
    rc = _LIB.FLC_BoosterPredictForMatSingleRowFastInit(
        booster._handle, predict_type, 0, -1, C_API_DTYPE_FLOAT64, 3, ctypes.c_char_p(b""), ctypes.byref(fast)
    )
    assert rc == -1


def test_known_predict_types_still_work():
    X, booster = _booster()
    assert booster.predict(X).shape == (40,)
    assert booster.predict(X, raw_score=True).shape == (40,)
    assert booster.predict(X, pred_leaf=True).shape == (40, booster.num_trees())
    assert booster.predict(X, pred_contrib=True).shape == (40, 4)
