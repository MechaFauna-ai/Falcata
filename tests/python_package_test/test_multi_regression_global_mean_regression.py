"""Exercise distributed initial means with two identical virtual partitions."""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, _safe_call


@pytest.mark.parametrize("weighted", [False, True])
def test_multi_regression_global_initial_mean(weighted):
    comm_size = ctypes.c_int32
    gather_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        comm_size,
        ctypes.POINTER(comm_size),
        ctypes.POINTER(comm_size),
        ctypes.c_int,
        ctypes.c_void_p,
        comm_size,
    )

    @gather_type
    def gather(data, size, starts, lengths, count, output, output_size):  # noqa: PLR0917
        # The remote partition is identical, including its labels and weights.
        ctypes.memmove(output + starts[0], data, size)
        ctypes.memmove(output + starts[1], data, size)

    scatter_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        comm_size,
        ctypes.c_int,
        ctypes.POINTER(comm_size),
        ctypes.POINTER(comm_size),
        ctypes.c_int,
        ctypes.c_void_p,
        comm_size,
        ctypes.c_void_p,
    )

    @scatter_type
    def scatter(data, size, type_size, starts, lengths, count, output, output_size, reducer):  # noqa: PLR0917
        # A valid callback pair is required, although this small-message test
        # only invokes gather. Sum two identical double-valued partitions.
        values = np.ctypeslib.as_array(
            ctypes.cast(data + starts[0], ctypes.POINTER(ctypes.c_double)),
            shape=(lengths[0] // ctypes.sizeof(ctypes.c_double),),
        )
        result = values * 2
        ctypes.memmove(output, result.ctypes.data, result.nbytes)

    x = np.random.default_rng(2).normal(size=(100, 2))
    y = np.column_stack([np.full(100, 3.0), np.full(100, 10.0)])
    weights = np.arange(1, 101, dtype=float) if weighted else None
    # All reductions here fit the small-message allgather path.
    _safe_call(
        _LIB.FLC_NetworkInitWithFunctions(
            2, 0, ctypes.cast(scatter, ctypes.c_void_p), ctypes.cast(gather, ctypes.c_void_p)
        )
    )
    try:
        booster = flc.train(
            {"objective": "multi_regression", "num_class": 2, "num_threads": 1, "verbosity": -1},
            flc.Dataset(x, label=y, weight=weights),
            num_boost_round=1,
        )
        np.testing.assert_allclose(booster.predict(x), y, rtol=0, atol=1e-12)
    finally:
        _safe_call(_LIB.FLC_NetworkFree())
