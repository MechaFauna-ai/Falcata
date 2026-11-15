# coding: utf-8
"""Sparse matrices outside scipy's canonical format (unsorted indices, duplicate entries).

The C API walks each CSC column's row indices in order (a row it has passed reads as 0) and takes a
duplicate entry once, where scipy sums duplicates: such a matrix was trained on and predicted for as a
different matrix, without any error. The Python package now hands the library canonical matrices, and
the C API rejects CSC row indices that are not strictly increasing.
"""

import ctypes

import numpy as np
import pytest
import scipy.sparse as sp

import falcata as flc
from falcata.basic import _LIB, FalcataError, _c_str, _safe_call

PARAMS = {"objective": "regression", "num_leaves": 7, "verbose": -1, "device_type": "cpu"}


def _data(n, m, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    X[rng.random(X.shape) < 0.5] = 0.0
    return X


def _shuffle_inner_indices(matrix, seed):
    rng = np.random.default_rng(seed)
    matrix = matrix.copy()
    for i in range(len(matrix.indptr) - 1):
        a, z = matrix.indptr[i], matrix.indptr[i + 1]
        perm = rng.permutation(np.arange(a, z))
        matrix.indices[a:z] = matrix.indices[perm]
        matrix.data[a:z] = matrix.data[perm]
    matrix.has_sorted_indices = False
    return matrix


@pytest.fixture(scope="module")
def booster():
    X = _data(400, 8, 1)
    return flc.train(PARAMS, flc.Dataset(X, X[:, 0] + X[:, 1]), 12)


@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_predict_unsorted_indices(booster, fmt):
    X = _data(64, 8, 2)
    unsorted = _shuffle_inner_indices(fmt(X), 3)
    indices_before = unsorted.indices.copy()
    np.testing.assert_allclose(booster.predict(unsorted), booster.predict(X))
    np.testing.assert_array_equal(unsorted.indices, indices_before)  # the caller's matrix is left as it was


@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_predict_duplicate_entries_are_summed(booster, fmt):
    # row 0 stores column 0 twice (1.0 + 1.0): scipy's value is 2.0
    if fmt is sp.csr_matrix:
        matrix = fmt((np.array([1.0, 1.0]), np.array([0, 0]), np.array([0, 2])), shape=(1, 8))
    else:
        matrix = fmt((np.array([1.0, 1.0]), np.array([0, 0]), np.array([0, 2] + [2] * 7)), shape=(1, 8))
    assert not matrix.has_canonical_format
    np.testing.assert_allclose(booster.predict(matrix), booster.predict(matrix.toarray()))


@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_train_on_unsorted_indices(booster, fmt):
    X = _data(400, 8, 1)
    y = X[:, 0] + X[:, 1]
    trained = flc.train(PARAMS, flc.Dataset(_shuffle_inner_indices(fmt(X), 4), y), 12)
    X_test = _data(64, 8, 2)
    np.testing.assert_allclose(trained.predict(X_test), booster.predict(X_test))


def _unsorted_csc_arrays():
    csc = _shuffle_inner_indices(sp.csc_matrix(_data(64, 8, 2)), 3)
    return (
        csc.indptr.astype(np.int32),
        csc.indices.astype(np.int32),
        csc.data.astype(np.float64),
        csc.shape,
    )


def test_c_api_rejects_unsorted_csc_rows(booster):
    indptr, indices, data, (nrow, _) = _unsorted_csc_arrays()
    out = np.zeros(nrow)
    out_len = ctypes.c_int64()
    with pytest.raises(FalcataError, match="row indices of column .* are not strictly increasing"):
        _safe_call(
            _LIB.FLC_BoosterPredictForCSC(
                booster._handle,
                indptr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(2),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                data.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(1),
                ctypes.c_int64(len(indptr)),
                ctypes.c_int64(len(data)),
                ctypes.c_int64(nrow),
                ctypes.c_int(0),
                ctypes.c_int(0),
                ctypes.c_int(-1),
                _c_str(""),
                ctypes.byref(out_len),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
    handle = ctypes.c_void_p()
    with pytest.raises(FalcataError, match="row indices of column .* are not strictly increasing"):
        _safe_call(
            _LIB.FLC_DatasetCreateFromCSC(
                indptr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(2),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                data.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(1),
                ctypes.c_int64(len(indptr)),
                ctypes.c_int64(len(data)),
                ctypes.c_int64(nrow),
                _c_str("verbose=-1"),
                None,
                ctypes.byref(handle),
            )
        )
