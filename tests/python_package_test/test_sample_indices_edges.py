"""FLC_GetSampleCount / FLC_SampleIndices edge cases.

With num_total_row = 0 FLC_SampleIndices passed the empty sample vector's null data() to
memcpy (UBSan: null pointer passed as argument 2), and a negative num_total_row came back
from FLC_GetSampleCount as a negative sample count.
"""

import ctypes

import numpy as np

from falcata.basic import _LIB


def test_zero_rows_sample_nothing():
    out = ctypes.c_int()
    assert _LIB.FLC_GetSampleCount(ctypes.c_int32(0), ctypes.c_char_p(b""), ctypes.byref(out)) == 0
    assert out.value == 0
    buf = np.full(1, -7, dtype=np.int32)
    out_len = ctypes.c_int32()
    rc = _LIB.FLC_SampleIndices(
        ctypes.c_int32(0), ctypes.c_char_p(b""), buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(out_len)
    )
    assert rc == 0
    assert out_len.value == 0
    assert buf[0] == -7


def test_negative_rows_are_rejected():
    out = ctypes.c_int()
    assert _LIB.FLC_GetSampleCount(ctypes.c_int32(-5), ctypes.c_char_p(b""), ctypes.byref(out)) == -1
    assert "is negative" in _LIB.FLC_GetLastError().decode()
    buf = np.zeros(1, dtype=np.int32)
    out_len = ctypes.c_int32()
    rc = _LIB.FLC_SampleIndices(
        ctypes.c_int32(-5), ctypes.c_char_p(b""), buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(out_len)
    )
    assert rc == -1


def test_sampling_still_works():
    out = ctypes.c_int()
    params = ctypes.c_char_p(b"bin_construct_sample_cnt=5")
    assert _LIB.FLC_GetSampleCount(ctypes.c_int32(100), params, ctypes.byref(out)) == 0
    assert out.value == 5
    buf = np.zeros(5, dtype=np.int32)
    out_len = ctypes.c_int32()
    assert (
        _LIB.FLC_SampleIndices(ctypes.c_int32(100), params, buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(out_len))
        == 0
    )
    assert out_len.value == 5
    assert np.all((buf >= 0) & (buf < 100))
