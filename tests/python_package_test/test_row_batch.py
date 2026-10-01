# coding: utf-8
"""The row-batched quantized construct must build exactly the histograms of the unbatched loop.

cuda_plan key row_batch switches the batched quantized dense construct between issuing the loads of 8 rows
before their shared atomics and the one-row-at-a-time loop. Integer sums are order-invariant, so the trained model
must be byte-identical whichever runs. The sweep covers what the construct branches on: the per-tree compact view
(4-bit, with feature sampling), 8-bit and 4-bit row data, wide partitions (two columns per thread, which keep the
unbatched loop), per-tree feature masks, the bagging ridge, and leaves large enough for 32-bit histogram bins.
Only the deterministic (quantized) modes are compared. Every run pins construct_jit:off: the JIT construct
replaces the AOT kernel this key changes.
"""

import os
import re

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

BASE = {
    "objective": "regression",
    "device_type": "cuda",
    "num_leaves": 63,
    "max_depth": 8,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "verbosity": -1,
    "seed": 9,
}


def _data(kind, rows, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "float255":  # 8-bit row data
        X = rng.standard_normal((rows, 40)).astype(np.float32)
        extra = {"max_bin": 255}
    elif kind == "int4bit":  # <= 16 bins: 4-bit packed row data, and the compact view under feature sampling
        X = rng.integers(0, 7, size=(rows, 120)).astype(np.float32)
        extra = {"max_bin": 15}
    elif kind == "wide":  # > 504 few-bin dense columns (no zeros, so not stored sparse): wide partitions
        X = rng.integers(1, 4, size=(rows, 1300)).astype(np.float32)
        extra = {"max_bin": 15}
    else:
        raise ValueError(kind)
    y = X[:, :8] @ rng.standard_normal(8) + rng.standard_normal(rows)
    return X, y.astype(np.float32), extra


def _train(kind, rows, params, rounds=12):
    X, y, extra = _data(kind, rows)
    p = {**BASE, **extra, **params}
    model = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=rounds).model_to_string()
    # the parameter dump records the plan string itself
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


CASES = [
    pytest.param("float255", 60_001, {}, id="8bit-rows"),
    pytest.param("int4bit", 60_001, {}, id="4bit-rows"),
    pytest.param("int4bit", 60_001, {"feature_fraction": 0.3}, id="compact-view"),
    pytest.param("float255", 60_001, {"feature_fraction": 0.5, "compact_quant": False}, id="feature-masks"),
    pytest.param("wide", 20_003, {}, id="wide-partitions"),
    pytest.param("float255", 60_001, {"bagging_fraction": 0.7, "bagging_freq": 1}, id="bagging"),
    pytest.param("float255", 1_200_000, {"num_leaves": 15, "max_depth": 4}, id="32bit-leaf-bins"),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize(("kind", "rows", "config"), CASES)
def test_row_batch_is_bit_identical_cuda(kind, rows, config, quant):
    params = {"quant_mode": quant, **config}
    if params.pop("compact_quant", True) is False:
        plan = "auto,construct_jit:off,compact_quant:off"
    else:
        plan = "auto,construct_jit:off"
    rounds = 6 if rows > 1_000_000 else 12
    batched = _train(kind, rows, {**params, "cuda_plan": plan}, rounds)
    unbatched = _train(kind, rows, {**params, "cuda_plan": plan + ",row_batch:off"}, rounds)
    assert batched == unbatched


@_REQUIRES_CUDA
def test_row_batch_is_bit_identical_to_the_construct_jit_cuda():
    """Under auto, row_batch replaces the JIT construct on long runs; the two must agree byte for byte."""
    params = {"quant_mode": "stochastic", "feature_fraction": 0.3}
    batched = _train("int4bit", 60_001, {**params, "cuda_plan": "auto"}, rounds=12)
    jit = _train("int4bit", 60_001, {**params, "cuda_plan": "auto,row_batch:off,construct_jit:on"}, rounds=12)
    assert batched == jit
