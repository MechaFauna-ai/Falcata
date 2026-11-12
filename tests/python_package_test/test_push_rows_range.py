"""The streaming push functions reject a row range outside the Dataset.

FLC_DatasetPushRows* write rows [start_row, start_row + nrow) of a Dataset created for
num_data rows (FLC_DatasetCreateByReference) without checking the range, so a push past
the end, or at a negative start_row, wrote past the feature bins
(heap-buffer-overflow under ASan, silent heap corruption otherwise).
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, _safe_call

C_API_DTYPE_FLOAT64 = 1
C_API_DTYPE_INT32 = 2


def _reference_and_target(num_rows):
    X = np.random.default_rng(0).normal(size=(50, 4))
    ref = flc.Dataset(X, X[:, 0], params={"verbose": -1}).construct()
    handle = ctypes.c_void_p()
    _safe_call(_LIB.FLC_DatasetCreateByReference(ref._handle, ctypes.c_int64(num_rows), ctypes.byref(handle)))
    _safe_call(_LIB.FLC_DatasetInitStreaming(handle, 0, 0, 0, 1, 1, -1))
    return X, ref, handle


def _push_dense(handle, X, start_row):
    X = np.ascontiguousarray(X, dtype=np.float64)
    return _LIB.FLC_DatasetPushRows(
        handle,
        X.ctypes.data_as(ctypes.c_void_p),
        C_API_DTYPE_FLOAT64,
        ctypes.c_int32(X.shape[0]),
        ctypes.c_int32(X.shape[1]),
        ctypes.c_int32(start_row),
    )


def _push_csr(handle, X, start_row):
    X = np.ascontiguousarray(X, dtype=np.float64)
    indptr = np.arange(0, X.size + 1, X.shape[1], dtype=np.int32)
    indices = np.tile(np.arange(X.shape[1], dtype=np.int32), X.shape[0])
    data = X.ravel()
    return _LIB.FLC_DatasetPushRowsByCSR(
        handle,
        indptr.ctypes.data_as(ctypes.c_void_p),
        C_API_DTYPE_INT32,
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        data.ctypes.data_as(ctypes.c_void_p),
        C_API_DTYPE_FLOAT64,
        ctypes.c_int64(len(indptr)),
        ctypes.c_int64(len(data)),
        ctypes.c_int64(X.shape[1]),
        ctypes.c_int64(start_row),
    )


@pytest.mark.parametrize("push", [_push_dense, _push_csr])
@pytest.mark.parametrize(("start_row", "nrow"), [(45, 10), (50, 1), (-1, 5), (2**31 - 1, 1)])
def test_push_outside_the_dataset_is_rejected(push, start_row, nrow):
    X, ref, handle = _reference_and_target(50)
    try:
        assert push(handle, X[:nrow], start_row) == -1
        assert "Cannot push" in _LIB.FLC_GetLastError().decode()
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))


@pytest.mark.parametrize("push", [_push_dense, _push_csr])
def test_push_inside_the_dataset_still_works(push):
    X, ref, handle = _reference_and_target(50)
    try:
        assert push(handle, X[:30], 0) == 0
        assert push(handle, X[30:], 30) == 0
        n = ctypes.c_int()
        _safe_call(_LIB.FLC_DatasetGetNumData(handle, ctypes.byref(n)))
        assert n.value == 50
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))


def test_csr_push_at_int64_max_start_row_is_rejected():
    # the CSR variant takes an int64_t start_row: the check must not overflow on it
    X, ref, handle = _reference_and_target(50)
    try:
        assert _push_csr(handle, X[:1], 2**63 - 1) == -1
        assert "Cannot push" in _LIB.FLC_GetLastError().decode()
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))
