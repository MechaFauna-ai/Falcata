"""CUDA training at the largest num_leaves (131072).

A level of the batched (hybrid / selective / device-level) flows launches its kernels with the level's split pairs on
grid y, which CUDA caps at 65535. A tree that may reach 131072 leaves can reach a level of 65536 pairs: the fix/subtract
launch failed with "[CUDA] invalid argument" (cuda_pdl_launch.hpp). Such budgets now take the per-split loop.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


@pytest.mark.parametrize("quant_mode", ["fixedpoint", "none"])
def test_cuda_trains_with_the_largest_num_leaves(quant_mode):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300_000, 10)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * rng.standard_normal(300_000)
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "num_leaves": 131072,
        "max_bin": 15,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.0,
        "quant_mode": quant_mode,
        "deterministic": True,
        "verbose": -1,
    }
    booster = flc.train(params, flc.Dataset(X, y, params=params), num_boost_round=1)
    assert booster.num_trees() == 1
    assert np.all(np.isfinite(booster.predict(X[:1000])))
