# coding: utf-8
"""Bin mappers gathered from the other machines are read within the gathered buffer.

Distributed bin finding gathers every machine's serialized bin mappers and restored each with the
unsized BinMapper::CopyFrom, which takes the mapper's length from its own num_bin. A corrupt or
truncated block from a peer (a broken collective, a version mismatch) made it read past the gathered
buffer (heap-buffer-overflow). One process plays rank 0 of 2 through FLC_NetworkInitWithFunctions,
with an allgather that hands back a peer block whose first mapper claims 2^20 bins.
"""

import ctypes
import struct

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, FalcataError, _safe_call

_I32P = ctypes.POINTER(ctypes.c_int32)
_ALLGATHER = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int32, _I32P, _I32P, ctypes.c_int, ctypes.c_void_p, ctypes.c_int32
)
_REDUCE_SCATTER = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int,
    _I32P,
    _I32P,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_void_p,
)


def _allgather(inp, input_size, block_start, block_len, num_block, output, output_size):
    # the peer (rank 1) sends what rank 0 sent; the bin mappers (the only large gather here) are corrupted
    for b in range(num_block):
        ctypes.memmove(output + block_start[b], inp, min(block_len[b], input_size))
    if input_size > 64 and num_block > 1:
        ctypes.memmove(output + block_start[1], struct.pack("<i", 1 << 20), 4)  # num_bin of its first mapper


def _reduce_scatter(inp, input_size, type_size, block_start, block_len, num_block, output, output_size, reducer):
    ctypes.memmove(output, inp + block_start[0], block_len[0])


def test_corrupt_gathered_bin_mapper_is_an_error():
    allgather, reduce_scatter = _ALLGATHER(_allgather), _REDUCE_SCATTER(_reduce_scatter)
    X = np.random.default_rng(0).normal(size=(200, 4))
    _safe_call(
        _LIB.FLC_NetworkInitWithFunctions(
            ctypes.c_int(2),
            ctypes.c_int(0),
            ctypes.cast(reduce_scatter, ctypes.c_void_p),
            ctypes.cast(allgather, ctypes.c_void_p),
        )
    )
    try:
        params = {"objective": "regression", "tree_learner": "data", "num_machines": 2, "verbose": -1}
        with pytest.raises(FalcataError, match="Serialized bin mapper"):
            flc.train(params, flc.Dataset(X, label=X[:, 0]), 1)
    finally:
        _safe_call(_LIB.FLC_NetworkFree())
