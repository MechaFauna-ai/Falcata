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


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "quant", [{"quant_mode": "stochastic"}, {"quant_mode": "fixedpoint"}], ids=["stochastic", "fixedpoint"]
)
@pytest.mark.parametrize("colmajor", ["on", "off"])
@pytest.mark.parametrize(
    ("rows", "feats", "feature_fraction"),
    [
        (50_001, 120, 0.15),  # few slots, odd row count
        (20_000, 121, 0.55),  # odd sampled-column counts: padding slots
        (300_000, 64, 0.5),  # many tiles
        (30_000, 1200, 0.5),  # > 256 byte slots and several partitions: per-cell kernel
    ],
    ids=["few-slots", "odd-columns", "many-tiles", "wide"],
)
def test_tiled_fill_is_bit_identical_cuda(rows, feats, feature_fraction, colmajor, quant):
    params = {"feature_fraction": feature_fraction, **quant}
    tiled = _train(rows, feats, {**params, "cuda_plan": f"auto,colmajor_fill:{colmajor}"}, seed=rows + feats)
    per_cell = _train(
        rows, feats, {**params, "cuda_plan": f"auto,colmajor_fill:{colmajor},tiled_fill:off"}, seed=rows + feats
    )
    assert tiled == per_cell
