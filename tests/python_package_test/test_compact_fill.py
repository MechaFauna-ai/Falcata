# coding: utf-8
"""The tiled 4-bit compact fill must write exactly the bytes of the per-cell fill.

cuda_plan key tiled_fill switches the per-tree compact-column fill between a
shared-memory tiled kernel and the per-cell kernel. Both gather the same
nibbles, so the trained model must be byte-identical whichever runs. The sweep
covers what the tiled kernel branches on: the source layout (column-major copy
vs packed rows), row counts that do not fill the last tile, odd sampled-column
counts (padding slots), several feature partitions, and more byte slots than
one tile holds (which keeps the per-cell kernel). Only the deterministic
(quantized) modes are compared: fp64 histograms use float atomics and differ in
the last bits run to run even with the same plan.
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
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 50,
    "max_bin": 15,
    "learning_rate": 0.1,
    "verbosity": -1,
    "seed": 3,
}


def _train(rows, feats, params, seed):
    rng = np.random.default_rng(seed)
    # 14 values: bins span the full nibble, so a dropped or masked high bit changes the model
    X = rng.integers(0, 14, size=(rows, feats)).astype(np.int8)
    y = (X[:, :8].astype(np.float64) @ rng.standard_normal(8) + rng.standard_normal(rows)).astype(np.float32)
    p = {**BASE, **params}
    model = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=12).model_to_string()
    # the parameter dump records the plan string itself
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


# (rows, features, feature_fraction): what the tiled kernel branches on, and the gate around it
SHAPES = [
    pytest.param(50_001, 120, 0.15, id="few-slots"),  # one partition, odd row count
    pytest.param(20_000, 121, 0.55, id="odd-columns"),  # odd sampled-column counts: padding slots
    pytest.param(300_000, 64, 0.5, id="many-tiles"),
    pytest.param(30_001, 1200, 0.2, id="multi-partition"),  # ~120 slots over several partitions, odd rows
    pytest.param(30_000, 1200, 0.42, id="at-gate"),  # ~252 slots: just under the 256-slot tile
    pytest.param(40_003, 2000, 0.1, id="more-partitions"),
    pytest.param(131, 600, 0.3, id="one-tile-plus"),  # barely more than one 128-row tile
    pytest.param(30_000, 1200, 0.5, id="over-gate"),  # > 256 slots: the per-cell kernel in both arms
]


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "quant", [{"quant_mode": "stochastic"}, {"quant_mode": "fixedpoint"}], ids=["stochastic", "fixedpoint"]
)
@pytest.mark.parametrize("colmajor", ["on", "off"])
@pytest.mark.parametrize(("rows", "feats", "feature_fraction"), SHAPES)
def test_tiled_fill_is_bit_identical_cuda(rows, feats, feature_fraction, colmajor, quant):
    params = {"feature_fraction": feature_fraction, **quant}
    tiled = _train(rows, feats, {**params, "cuda_plan": f"auto,colmajor_fill:{colmajor}"}, seed=rows + feats)
    per_cell = _train(
        rows, feats, {**params, "cuda_plan": f"auto,colmajor_fill:{colmajor},tiled_fill:off"}, seed=rows + feats
    )
    assert tiled == per_cell


_PROBE = """
import sys
sys.path.insert(0, {here!r})
from test_compact_fill import _train
_train({rows}, {feats}, {{"feature_fraction": {ff}, "quant_mode": "stochastic", "verbosity": 1}}, seed=0)
"""


def _fill_kernels(rows, feats, feature_fraction):
    """(kernel, slots, partitions) of every compact fill in a training run, from the FALCATA_DEBUG=diag log."""
    code = _PROBE.format(here=os.path.dirname(os.path.abspath(__file__)), rows=rows, feats=feats, ff=feature_fraction)
    env = {**os.environ, "FALCATA_DEBUG": "diag"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    found = re.findall(r"compact fill: (\S+) 4-bit kernel, (\d+) byte slots in (\d+) partitions", out)
    return [(k, int(s), int(p)) for k, s, p in found]


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("rows", "feats", "feature_fraction", "kernel", "multi_partition"),
    [
        (30_001, 1200, 0.2, "tiled", True),
        (30_000, 1200, 0.42, "tiled", True),
        (40_003, 2000, 0.1, "tiled", True),
        (30_000, 1200, 0.5, "per-cell", True),
    ],
)
def test_shapes_reach_the_intended_fill_kernel_cuda(rows, feats, feature_fraction, kernel, multi_partition):
    """The bit-identity sweep only means something if its shapes take the path they are named for."""
    fills = _fill_kernels(rows, feats, feature_fraction)
    assert fills, "no compact fill was logged"
    assert all(k == kernel for k, _, _ in fills), fills
    assert all((p > 1) == multi_partition for _, _, p in fills), fills
