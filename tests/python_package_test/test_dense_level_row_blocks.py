# coding: utf-8
"""dense_level_row_blocks and dense_block_map regroup the rows of the quantized dense batched construct and never
change the model.

cuda_plan key dense_level_row_blocks runs every leaf of a level at the largest leaf's rows per thread of the launched
grid; dense_block_map launches only the block rows that have rows and maps each back to its pair and block row. Both
change only which block adds which row to the packed integer histogram, so every model must be byte-identical with
either key off. The shapes cover 8-bit columns (the kernel's main user), 4-bit columns with pair_hist off, bagging,
high quant_bins (the packed partials' overflow guard), a level with more pairs than the block map's prefix holds
(launched as one mapped grid per prefix-full of pairs),
a 3-leaf tree (one pair per level: neither key engages) and the complete device tree.
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
    "num_leaves": 255,
    "max_depth": 8,
    "min_data_in_leaf": 5,
    "learning_rate": 0.1,
    "max_bin": 255,
    "quant_mode": "stochastic",
    "verbosity": -1,
    "seed": 5,
}

# (rows, columns, extra params)
SHAPES = {
    "bins255": (60_001, 30, {}),
    "fixedpoint_bagged": (50_003, 20, {"quant_mode": "fixedpoint", "bagging_fraction": 0.6, "bagging_freq": 1}),
    "quant_bins64": (40_001, 12, {"num_grad_quant_bins": 64, "min_data_in_leaf": 1}),
    "bins15_no_pair": (40_001, 40, {"max_bin": 15, "feature_fraction": 0.5, "cuda_plan": "pair_hist:off"}),
    # device_tree:off: depth 12 is the complete device tree's by default; keep covering the >256-pair prefix path
    "pairs_over_prefix": (
        90_001,
        6,
        {"num_leaves": 4095, "max_depth": 12, "min_data_in_leaf": 1, "cuda_plan": "device_tree:off"},
    ),
    "three_leaves": (5_001, 4, {"num_leaves": 3, "max_depth": 2}),
    # the complete device tree: the rows-per-thread floor is derived on the device from the level's sizes
    "device_tree": (60_001, 30, {"num_leaves": 63, "max_depth": 6}),
}

OFF = "dense_level_row_blocks:off,dense_block_map:off"


def _train(X, y, params, plan_extra):
    p = dict(params)
    plan = p.pop("cuda_plan", "")
    if plan_extra:
        plan = f"{plan},{plan_extra}" if plan else plan_extra
    if plan:
        p["cuda_plan"] = plan
    ds = flc.Dataset(X, label=y, params=p)
    model = flc.train(p, ds, num_boost_round=6).model_to_string()
    # the plan string itself is recorded with the parameters
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_dense_level_row_blocks_bit_identical(shape):
    rows, cols, extra = SHAPES[shape]
    rng = np.random.default_rng(17)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    y = (X[:, 0] + 0.5 * np.sin(3 * X[:, 1]) + 0.3 * rng.standard_normal(rows)).astype(np.float32)
    params = {**BASE, **extra}
    on = _train(X, y, params, "")
    assert on == _train(X, y, params, OFF)
    assert on == _train(X, y, params, "dense_block_map:off")
