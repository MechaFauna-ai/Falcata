# coding: utf-8
"""The one-launch quantized fix + subtract must leave exactly the histograms of the two sequential kernels.

cuda_plan key fix_subtract_masked runs the most-frequent-bin fix of the smaller leaf and the parent-minus-smaller
subtract of the larger leaf in one launch: the subtract blocks skip the need-fix features' most-frequent bins, and
the fix block of each such feature subtracts its fixed value. Integer arithmetic per bin in the same bit-width
cases, so the trained model must be byte-identical with the key on or off. The sweep covers where it runs: the
complete device tree, the host-launched level path (device_tree:off), the classic leaf-wise loop (no depth
limit; columns stored sparse), feature sampling with columns wider than fix_subtract_fused serves, bagging, and leaves
large enough for 32-bit histogram bins (the bit-change buffer case).
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
    "use_quantized_grad": True,
    "num_leaves": 63,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 11,
}


def _data(rows, density, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, 40)).astype(np.float32)
    X[:, :20] *= rng.random((rows, 20)) < density  # zero-heavy columns: their most-frequent bin needs the fix
    X[:, 20:25] = np.round(X[:, 20:25] * 2)
    X[:, 30:][rng.random((rows, 10)) < 0.05] = np.nan
    y = np.nan_to_num(X[:, :8]) @ rng.standard_normal(8) + rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _train(rows, density, params, rounds=12):
    X, y = _data(rows, density)
    p = {**BASE, **params}
    model = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=rounds).model_to_string()
    # the parameter dump records the plan string itself
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


CASES = [
    pytest.param(60_001, 0.6, {}, id="level"),
    pytest.param(60_001, 0.6, {"num_leaves": 15, "max_depth": 4, "plan": "device_tree:off"}, id="prefix-level"),
    pytest.param(60_001, 0.12, {"num_leaves": 63, "max_depth": -1}, id="leaf-wise-sparse"),
    pytest.param(60_001, 0.6, {"feature_fraction": 0.5}, id="feature-sampling"),
    pytest.param(60_001, 0.6, {"bagging_fraction": 0.7, "bagging_freq": 1}, id="bagging"),
    pytest.param(1_200_000, 0.6, {"num_leaves": 15, "max_depth": 4}, id="32bit-leaf-bins"),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize(("rows", "density", "config"), CASES)
def test_fix_subtract_masked_is_bit_identical_cuda(rows, density, config, quant):
    config = dict(config)
    extra = config.pop("plan", "")
    plan = "auto," + extra + ("," if extra else "")
    params = {"quant_mode": quant, **config}
    on = _train(rows, density, {**params, "cuda_plan": plan + "fix_subtract_masked:on"})
    off = _train(rows, density, {**params, "cuda_plan": plan + "fix_subtract_masked:off"})
    assert on == off
