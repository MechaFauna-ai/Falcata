# coding: utf-8
"""Quantized training must refuse negative hessians from a custom objective instead of training on corrupted sums.

The quantized path packs each hessian into an unsigned field (16-bit packed histograms, (grad << 32) | hess root
sums). On origin/master a custom objective whose hessians were -0.75 on a contiguous run of 3000 rows (+1 elsewhere)
silently produced a single-leaf tree under quant_mode=stochastic, while the same objective with all-positive hessians
grows the 4 leaves it asks for.
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _train(hess_sign, quant_mode):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=20000) * 0.1).astype(np.float32)
    hess = np.ones(20000)
    hess[:3000] = 0.75 * hess_sign

    def fobj(preds, ds):
        return preds - ds.get_label(), hess.copy()

    params = {
        "objective": fobj,
        "device_type": "cuda",
        "num_leaves": 4,
        "learning_rate": 1.0,
        "verbose": -1,
        "deterministic": True,
        "seed": 0,
        "min_data_in_leaf": 1,
        "quant_mode": quant_mode,
    }
    return flc.train(params, flc.Dataset(X, label=y), num_boost_round=1)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint"])
def test_quantized_custom_objective_rejects_negative_hessians(quant_mode):
    with pytest.raises(flc.basic.FalcataError, match="negative hessians"):
        _train(-1.0, quant_mode)


@_REQUIRES_CUDA
def test_quantized_custom_objective_with_positive_hessians_still_trains():
    booster = _train(1.0, "stochastic")
    assert booster.dump_model()["tree_info"][0]["num_leaves"] == 4
