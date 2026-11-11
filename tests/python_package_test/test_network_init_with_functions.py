# coding: utf-8
"""FLC_NetworkInitWithFunctions rejects a missing collective callback.

That overload never creates the socket linkers or their bruck /
recursive-halving maps, yet Allgather and ReduceScatter fell back to the
built-in, linker-based algorithms whenever the matching external callback
was null, dereferencing a null (or stale) Linkers. It now requires both
callbacks, and a rank inside [0, num_machines).
"""

import ctypes

import pytest

from falcata.basic import _LIB, FalcataError, _safe_call

_REDUCE_SCATTER = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.c_void_p,
)
_ALLGATHER = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int32,
)


def _noop(*_args):
    return None


@pytest.mark.parametrize(
    ("with_reduce_scatter", "with_allgather", "rank"),
    [(True, False, 0), (False, True, 0), (False, False, 0), (True, True, 2), (True, True, -1)],
)
def test_network_init_with_functions_rejects_bad_setup(with_reduce_scatter, with_allgather, rank):
    reduce_scatter = _REDUCE_SCATTER(_noop) if with_reduce_scatter else None
    allgather = _ALLGATHER(_noop) if with_allgather else None
    try:
        with pytest.raises(FalcataError, match="external functions|Invalid rank"):
            _safe_call(
                _LIB.FLC_NetworkInitWithFunctions(
                    ctypes.c_int(2),
                    ctypes.c_int(rank),
                    ctypes.cast(reduce_scatter, ctypes.c_void_p) if reduce_scatter else None,
                    ctypes.cast(allgather, ctypes.c_void_p) if allgather else None,
                )
            )
    finally:
        _safe_call(_LIB.FLC_NetworkFree())
