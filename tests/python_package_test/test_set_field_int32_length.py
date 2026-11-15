# coding: utf-8
"""A Dataset field longer than INT32_MAX entries is refused in Python.

FLC_DatasetSetField takes an int element count and ctypes.c_int wraps silently, so a
field of 2**28 rows x 17 classes of init_score would have arrived as exactly 2**28
entries and passed as a one-class init score. A zero-stride view keeps the test free of
the memory such an array would need.
"""

import numpy as np
import pytest
from numpy.lib.stride_tricks import as_strided

import falcata as flc


def test_field_longer_than_int32_is_refused():
    X = np.random.default_rng(0).random((100, 2))
    ds = flc.Dataset(X, X[:, 0], params={"verbose": -1}).construct()
    huge = as_strided(np.zeros(1), shape=((1 << 32) + 100,), strides=(0,))
    with pytest.raises(ValueError, match="more than the 2147483647 a Dataset field can hold"):
        ds.set_field("init_score", huge)
