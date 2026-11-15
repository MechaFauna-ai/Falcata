# coding: utf-8
"""Push buffers of sparse bins and the thread ids of the push API.

A sparse bin keeps one push buffer per pushing thread, sized by the OpenMP thread count when the
bin was made (or by FLC_DatasetInitStreaming: nthreads x omp_max_threads). A push ran with the
OpenMP thread count of the moment and took any tid, so a larger thread count, or a tid outside the
threads InitStreaming registered, wrote past the buffers (heap-buffer-overflow in SparseBin::Push).
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, FalcataError, _c_str, _safe_call

NROW, NCOL = 5000, 3
PARAMS = {"objective": "binary", "num_leaves": 7, "num_threads": 4, "verbose": -1, "device_type": "cpu"}


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(NROW, NCOL))
    X[rng.random(X.shape) < 0.95] = 0.0  # sparse bins
    return X, (X[:, 0] > 0).astype(np.float32)


def _reference(X, y, num_threads):
    return flc.Dataset(X, label=y, params={"verbose": -1, "num_threads": num_threads}, free_raw_data=False).construct()


def _by_reference(reference):
    handle = ctypes.c_void_p()
    _safe_call(_LIB.FLC_DatasetCreateByReference(reference._handle, ctypes.c_int64(NROW), ctypes.byref(handle)))
    return handle


def _push_with_metadata(handle, X, y, tid):
    block = np.ascontiguousarray(X)
    _safe_call(
        _LIB.FLC_DatasetPushRowsWithMetadata(
            handle,
            block.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1),
            ctypes.c_int32(NROW),
            ctypes.c_int32(NCOL),
            ctypes.c_int32(0),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            None,
            None,
            None,
            ctypes.c_int32(tid),
        )
    )


def _train_predict(handle, y, X):
    _safe_call(
        _LIB.FLC_DatasetSetField(
            handle, _c_str("label"), y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(NROW), ctypes.c_int(0)
        )
    )
    booster = ctypes.c_void_p()
    params = " ".join(f"{k}={v}" for k, v in PARAMS.items())
    _safe_call(_LIB.FLC_BoosterCreate(handle, _c_str(params), ctypes.byref(booster)))
    finished = ctypes.c_int()
    for _ in range(5):
        _safe_call(_LIB.FLC_BoosterUpdateOneIter(booster, ctypes.byref(finished)))
    out = np.zeros(NROW)
    out_len = ctypes.c_int64()
    Xc = np.ascontiguousarray(X)
    _safe_call(
        _LIB.FLC_BoosterPredictForMat(
            booster,
            Xc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1),
            ctypes.c_int32(NROW),
            ctypes.c_int32(NCOL),
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
    return out


def test_push_after_the_thread_count_rose():
    X, y = _data()
    reference = _reference(X, y, num_threads=1)
    handle = _by_reference(reference)  # made under one OpenMP thread: its sparse bins get one push buffer
    try:
        # training raises the OpenMP thread count before the push
        expected = flc.train(PARAMS, reference, 5).predict(X)
        block = np.ascontiguousarray(X)
        _safe_call(
            _LIB.FLC_DatasetPushRows(
                handle,
                block.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(1),
                ctypes.c_int32(NROW),
                ctypes.c_int32(NCOL),
                ctypes.c_int32(0),
            )
        )
        np.testing.assert_allclose(_train_predict(handle, y, X), expected)
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))


@pytest.mark.parametrize("tid", [-1, 1, 3])
def test_push_tid_outside_the_streaming_threads(tid):
    X, y = _data()
    handle = _by_reference(_reference(X, y, num_threads=2))
    try:
        _safe_call(_LIB.FLC_DatasetInitStreaming(handle, 0, 0, 0, 1, ctypes.c_int32(1), ctypes.c_int32(2)))
        with pytest.raises(FalcataError, match="outside the 1 threads registered"):
            _push_with_metadata(handle, X, y, tid)
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))


def test_push_tid_without_init_streaming():
    X, y = _data()
    handle = _by_reference(_reference(X, y, num_threads=2))
    try:
        with pytest.raises(FalcataError, match="needs FLC_DatasetInitStreaming"):
            _push_with_metadata(handle, X, y, 1)
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))


def test_init_streaming_needs_a_thread():
    X, y = _data()
    handle = _by_reference(_reference(X, y, num_threads=2))
    try:
        with pytest.raises(FalcataError, match="at least one external thread"):
            _safe_call(_LIB.FLC_DatasetInitStreaming(handle, 0, 0, 0, 1, ctypes.c_int32(0), ctypes.c_int32(2)))
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))


def test_streaming_push_with_more_openmp_threads_than_registered():
    X, y = _data()
    reference = _reference(X, y, num_threads=1)
    handle = _by_reference(reference)
    try:
        # one OpenMP thread per external thread registered, then a push under 4 OpenMP threads
        _safe_call(_LIB.FLC_DatasetInitStreaming(handle, 0, 0, 0, 1, ctypes.c_int32(2), ctypes.c_int32(1)))
        expected = flc.train(PARAMS, reference, 5).predict(X)
        _push_with_metadata(handle, X, y, 1)
        _safe_call(_LIB.FLC_DatasetMarkFinished(handle))
        np.testing.assert_allclose(_train_predict(handle, y, X), expected)
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))
