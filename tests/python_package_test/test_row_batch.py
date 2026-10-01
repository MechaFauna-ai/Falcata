# coding: utf-8
"""The row-batched quantized construct must build exactly the histograms of the unbatched loop.

cuda_plan key row_batch switches the batched quantized dense construct between issuing the loads of 8 rows
before their shared atomics and the one-row-at-a-time loop. Integer sums are order-invariant, so the trained model
must be byte-identical whichever runs. The sweep covers what the construct branches on: the per-tree compact view
(4-bit, with feature sampling), 8-bit and 4-bit row data, wide partitions (two columns per thread, which keep the
unbatched loop), per-tree feature masks, the bagging ridge, and leaves large enough for 32-bit histogram bins.
Only the deterministic (quantized) modes are compared. The sweep pins construct_jit:off: the JIT construct
replaces the AOT kernel this key changes. A separate test checks how row_batch and construct_jit resolve.
"""

import os
import re
import subprocess
import sys

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


_PROBE = """
import hashlib, sys
sys.path.insert(0, {here!r})
from test_row_batch import _train
model = _train("float255", 20_001, {params!r}, rounds={rounds!r})
print("MODEL_MD5", hashlib.md5(model.encode()).hexdigest(), flush=True)
"""


def _probe(plan, rounds):
    """Train in a fresh process (the plan is process-global); return the construct kernels logged and the md5."""
    params = {"quant_mode": "stochastic", "num_leaves": 15, "max_depth": 4, "verbosity": 1, "cuda_plan": plan}
    code = _PROBE.format(here=os.path.dirname(os.path.abspath(__file__)), params=params, rounds=rounds)
    env = {**os.environ, "FALCATA_DEBUG": "diag"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    kernels = set(re.findall(r"quantized construct: (\S+) kernel", out))
    return kernels, re.search(r"MODEL_MD5 (\w+)", out).group(1), out


@_REQUIRES_CUDA
def test_row_batch_and_construct_jit_resolution_cuda():
    """Which construct runs under each plan, on runs long enough for the JIT's 300-round gate, and that all agree.

    auto keeps row_batch and never arms the JIT; row_batch:off restores the JIT on long runs (and the unbatched
    AOT loop on short ones); an explicit construct_jit:on runs the JIT in place of row_batch, with a warning.
    """
    auto_kernels, auto_md5, auto_out = _probe("auto", 300)
    assert auto_kernels == {"row_batch"}, auto_kernels
    assert "CUDAConstructJIT: compiled" not in auto_out

    off_kernels, off_md5, _ = _probe("auto,row_batch:off", 300)
    assert "jit" in off_kernels, off_kernels
    assert off_kernels <= {"jit", "unbatched"}, off_kernels

    short_kernels, _, _ = _probe("auto,row_batch:off", 12)
    assert short_kernels == {"unbatched"}, short_kernels

    forced_kernels, forced_md5, forced_out = _probe("auto,construct_jit:on", 300)
    assert "jit" in forced_kernels, forced_kernels
    assert "replaces the row_batch construct with the unbatched JIT kernel" in forced_out
    assert "construct_jit=1 row_batch=1" in forced_out

    assert auto_md5 == off_md5 == forced_md5
