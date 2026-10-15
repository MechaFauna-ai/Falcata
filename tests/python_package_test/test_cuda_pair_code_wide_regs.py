# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key pair_code_wide_regs.

The fused fill packs the 6-bit (pair_code_wide) and the mixed-radix (pair_code_radix) code views with per-slot
coefficients (span(hi), has_hi, times the slot's place value) in registers and dp2a sums, two words per thread
(with pair_code5_pack2; off: the generic pack), instead of the generic per-slot pack. Both packs must write the same
words: the model is byte-identical with the key off, for the radix view and (pair_code_radix off) the 6-bit view,
over 6 x 6 cells, mixed widths (3 to 7 values), shallow and deep trees, both quantization modes, bagging, and row
widths that are not a multiple of a word's slots.
"""

import hashlib
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _model_md5(params, X, y, rounds):
    ds = flc.Dataset(X, label=y, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("values", "nan_frac", "num_features", "feature_fraction", "leaves", "depth", "quant", "bagging"),
    [
        ((5, 6), 0.08, 1200, 0.3, 63, 6, "stochastic", 1.0),  # 5 values + NaN: 6 x 6 cells
        ((6, 7), 0.0, 1200, 0.3, 63, 6, "stochastic", 1.0),  # 6 values, 6 x 6 cells
        ((3, 8), 0.05, 1200, 0.3, 255, 10, "fixedpoint", 1.0),  # 3..7 values (+ NaN): mixed radices
        ((5, 6), 0.08, 1210, 0.3, 255, 10, "stochastic", 0.6),  # bagged, deep
        ((5, 6), 0.08, 1180, 0.25, 127, 8, "fixedpoint", 1.0),
    ],
)
def test_pair_code_wide_regs_is_bit_identical_cuda(
    values, nan_frac, num_features, feature_fraction, leaves, depth, quant, bagging
):
    rows = 8000
    rng = np.random.default_rng(23)
    levels = rng.integers(values[0], values[1], size=num_features)
    X = (rng.random((rows, num_features)) * levels).astype(np.int64).astype(np.float64)
    if nan_frac > 0:
        X[rng.random(X.shape) < nan_frac] = np.nan
    w = rng.standard_normal(40)
    y = np.nan_to_num(X[:, :40]) @ w + rng.standard_normal(rows)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": leaves,
        "max_depth": depth,
        "max_bin": 15,
        "feature_fraction": feature_fraction,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "quant_mode": quant,
        "verbose": -1,
        "seed": 42,
        "metric": "None",
    }
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    for base in (
        "auto",
        "auto,pair_code5_pack2:off",
        "auto,pair_code_radix:off",
        "auto,pair_code_radix:off,pair_code5_pack2:off",
    ):
        on = _model_md5({**params, "cuda_plan": base}, X, y, 15)
        off = _model_md5({**params, "cuda_plan": base + ",pair_code_wide_regs:off"}, X, y, 15)
        assert on == off
