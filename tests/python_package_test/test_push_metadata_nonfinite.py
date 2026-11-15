# coding: utf-8
"""Metadata pushed with the rows is sanitised like metadata set as a field.

Dataset.set_field (SetLabel / SetWeights / SetInitScore) maps NaN to 0 (and clamps infinities, AvoidInf).
FLC_DatasetPushRowsWithMetadata copied the label, weight and init score arrays as they were, so a NaN
label made every gradient NaN and the pushed Dataset trained a different model than the same data set
through fields.
"""

import ctypes

import numpy as np
import pytest
import scipy.sparse as sp

import falcata as flc
from falcata.basic import _LIB, _c_str, _safe_call

N, M = 200, 4


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, M))
    y = X[:, 0].astype(np.float32)
    y[[3, 50]] = np.nan
    w = rng.uniform(0.5, 2.0, N).astype(np.float32)
    w[11] = np.nan
    init = 0.1 * rng.normal(size=N)
    init[13] = np.nan
    return X, y, w, init


@pytest.mark.parametrize("csr", [False, True])
def test_pushed_metadata_matches_fields(csr):
    X, y, w, init = _data()
    params = {"verbose": -1, "min_data_in_leaf": 2, "feature_pre_filter": False}
    train_params = {
        "objective": "regression",
        "num_leaves": 7,
        "min_data_in_leaf": 2,
        "verbose": -1,
        "device_type": "cpu",
    }
    reference = flc.Dataset(X, params=params, free_raw_data=False).construct()
    expected = flc.Dataset(X, label=y, weight=w, init_score=init, params=params, reference=reference)
    expected_pred = flc.train(train_params, expected, 3).predict(X)

    handle = ctypes.c_void_p()
    _safe_call(_LIB.FLC_DatasetCreateByReference(reference._handle, ctypes.c_int64(N), ctypes.byref(handle)))
    try:
        _safe_call(_LIB.FLC_DatasetInitStreaming(handle, 1, 1, 0, 1, 1, -1))
        label_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        weight_ptr = w.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        init_ptr = init.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if csr:
            mat = sp.csr_matrix(X)
            indptr = mat.indptr.astype(np.int64)
            indices = mat.indices.astype(np.int32)
            data = mat.data.astype(np.float64)
            _safe_call(
                _LIB.FLC_DatasetPushRowsByCSRWithMetadata(
                    handle,
                    indptr.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(3),
                    indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    data.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(1),
                    ctypes.c_int64(len(indptr)),
                    ctypes.c_int64(len(data)),
                    ctypes.c_int64(0),
                    label_ptr,
                    weight_ptr,
                    init_ptr,
                    None,
                    ctypes.c_int32(0),
                )
            )
        else:
            block = np.ascontiguousarray(X)
            _safe_call(
                _LIB.FLC_DatasetPushRowsWithMetadata(
                    handle,
                    block.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(1),
                    ctypes.c_int32(N),
                    ctypes.c_int32(M),
                    ctypes.c_int32(0),
                    label_ptr,
                    weight_ptr,
                    init_ptr,
                    None,
                    ctypes.c_int32(0),
                )
            )
        _safe_call(_LIB.FLC_DatasetMarkFinished(handle))
        booster = ctypes.c_void_p()
        param_str = " ".join(f"{k}={v}" for k, v in train_params.items())
        _safe_call(_LIB.FLC_BoosterCreate(handle, _c_str(param_str), ctypes.byref(booster)))
        finished = ctypes.c_int()
        for _ in range(3):
            _safe_call(_LIB.FLC_BoosterUpdateOneIter(booster, ctypes.byref(finished)))
        out = np.zeros(N)
        out_len = ctypes.c_int64()
        Xc = np.ascontiguousarray(X)
        _safe_call(
            _LIB.FLC_BoosterPredictForMat(
                booster,
                Xc.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(1),
                ctypes.c_int32(N),
                ctypes.c_int32(M),
                ctypes.c_int(1),
                ctypes.c_int(0),
                ctypes.c_int(0),
                ctypes.c_int(-1),
                _c_str(""),
                ctypes.byref(out_len),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        _safe_call(_LIB.FLC_BoosterFree(booster))
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))
    assert np.isfinite(out).all()
    assert np.ptp(expected_pred) > 0  # the model learned something
    np.testing.assert_allclose(out, expected_pred, rtol=0, atol=1e-12)
