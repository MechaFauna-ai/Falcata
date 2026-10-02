# coding: utf-8
"""Packing the 4-bit row data on the device must produce exactly the bytes the host path produces.

cuda_plan key gpu_rowpack builds the packed row-major matrix (two 4-bit bins per byte) on the GPU from the Dataset's
columns when a Booster is created, instead of packing it on the host and uploading the result. The trained model must
be byte-identical either way, and under FALCATA_VERIFY=1 Falcata compares the device bytes with the host-packed bytes
itself, which also proves the device path ran.
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
    "max_bin": 15,
    "verbosity": -1,
    "seed": 3,
}


def _data(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "narrow":  # few columns, several values each
        X = rng.integers(0, 9, size=(50_001, 24)).astype(np.float32)
    elif kind == "many":  # several partitions
        X = rng.integers(0, 6, size=(30_003, 700)).astype(np.float32)
    elif kind == "odd-rows":  # odd row count: the last 4-bit column byte holds one row
        X = rng.integers(0, 5, size=(9_999, 51)).astype(np.float32)
    elif kind == "tall":  # 4-bit columns over 1 MiB: copied to the device one by one
        X = rng.integers(0, 7, size=(2_100_001, 6)).astype(np.float32)
    elif kind == "long-runs":  # small columns totalling more than one 16 MiB host gather per partition
        X = rng.integers(0, 7, size=(200_001, 400)).astype(np.float32)
    else:
        raise ValueError(kind)
    y = X[:, :5] @ rng.standard_normal(5) + rng.standard_normal(len(X))
    return X, y.astype(np.float32)


def _train(kind, params):
    X, y = _data(kind)
    p = {**BASE, **params}
    model = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=8).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint", "none"])
@pytest.mark.parametrize("kind", ["narrow", "many", "odd-rows", "tall", "long-runs"])
def test_gpu_rowpack_is_bit_identical_cuda(kind, quant):
    params = {"quant_mode": quant, "deterministic": True} if quant == "none" else {"quant_mode": quant}
    if quant == "none":
        params["cuda_plan"] = "auto,graph_loop:off"
        off = "auto,graph_loop:off,gpu_rowpack:off"
    else:
        params["cuda_plan"] = "auto"
        off = "auto,gpu_rowpack:off"
    on_model = _train(kind, params)
    off_model = _train(kind, {**params, "cuda_plan": off})
    assert on_model == off_model


_PROBE = """
import sys
sys.path.insert(0, {here!r})
from test_gpu_rowpack import _train
_train({kind!r}, {{"quant_mode": "stochastic", "verbosity": 1}})
"""


@_REQUIRES_CUDA
@pytest.mark.parametrize("kind", ["narrow", "many", "odd-rows", "tall", "long-runs"])
def test_gpu_rowpack_matches_host_bytes_cuda(kind):
    """FALCATA_VERIFY=1 packs on the host as well and compares every byte; the log line also proves engagement."""
    code = _PROBE.format(here=os.path.dirname(os.path.abspath(__file__)), kind=kind)
    env = {**os.environ, "FALCATA_VERIFY": "1"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    assert "device-packed 4-bit row data matches the host-packed bytes" in out, out[-2000:]
