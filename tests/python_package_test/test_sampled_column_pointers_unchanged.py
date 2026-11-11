# coding: utf-8
"""FLC_DatasetCreateFromSampledColumn leaves the caller's index pointers alone.

FastFeatureBundling rewrote entries of the caller's sample_indices pointer
array to point into a local std::vector (tmp_indices) that is destroyed when
it returns, leaving the caller holding dangling pointers. It now works on
its own copy of the pointer array.
"""

import ctypes

import numpy as np

from falcata.basic import _LIB, _safe_call


def test_sampled_column_index_pointers_unchanged():
    rng = np.random.default_rng(0)
    n = 1000
    cols = [
        # mostly 1.0: its most frequent bin is not the zero bin, so bundling
        # builds a fixed index list for it (the case that rewrote the pointer)
        np.where(rng.random(n) < 0.9, 1.0, rng.normal(size=n)),
        # sparse columns that can be bundled
        np.where(rng.random(n) < 0.05, rng.normal(size=n), 0.0),
        np.where(rng.random(n) < 0.05, rng.normal(size=n), 0.0),
    ]
    values, indices = [], []
    for c in cols:
        nz = np.flatnonzero(c).astype(np.int32)
        indices.append(nz)
        values.append(c[nz].astype(np.float64))
    ncol = len(cols)
    value_ptrs = (ctypes.POINTER(ctypes.c_double) * ncol)(
        *[v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for v in values]
    )
    index_ptrs = (ctypes.POINTER(ctypes.c_int32) * ncol)(
        *[i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) for i in indices]
    )
    before = [ctypes.cast(index_ptrs[k], ctypes.c_void_p).value for k in range(ncol)]
    num_per_col = np.array([len(i) for i in indices], dtype=np.int32)
    handle = ctypes.c_void_p()
    _safe_call(
        _LIB.FLC_DatasetCreateFromSampledColumn(
            value_ptrs,
            index_ptrs,
            ctypes.c_int32(ncol),
            num_per_col.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(n),
            ctypes.c_int32(n),
            ctypes.c_int64(n),
            ctypes.c_char_p(b"verbose=-1 min_data_in_bin=1"),
            ctypes.byref(handle),
        )
    )
    try:
        after = [ctypes.cast(index_ptrs[k], ctypes.c_void_p).value for k in range(ncol)]
        assert after == before
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))
