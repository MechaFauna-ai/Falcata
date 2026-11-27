# coding: utf-8
"""Tests for cuda_plan key warp_find_fp32 (all-FP32 quantized warp split finder).

Under cuda_precision=fp32, warp_find_fp32 routes eligible quantized levels to an all-FP32
instantiation of FindBestSplitsDiscretizedForLevelWarpKernel instead of falling back to the
block-per-task FP32 kernel.

These tests verify:
1. Default cuda_precision=fp64 is bit-identical with warp_find_fp32 on vs off.
2. Under cuda_precision=fp32, warp_find_fp32:on is deterministic across runs and bit-identical
   across internal tier controls (find_split_small:off, warp_find_mid_ppl:off) for narrow,
   wide, NaN-bearing, multiclass, and max_delta_step workloads.
3. Quality parity: under cuda_precision=fp32, warp_find_fp32:on achieves matching validation
   loss relative to both warp_find_fp32:off (block FP32) and default fp64.
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


def _model_md5(booster):
    lines = [
        line
        for line in booster.model_to_string().splitlines()
        if "cuda_plan" not in line and "cuda_precision" not in line
    ]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _make_data(*, rows, cols, objective, max_bin, with_nans, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    if max_bin <= 15:
        X = np.floor(np.clip(X, -2.0, 2.0) + 2.0).astype(np.float32)
    if with_nans:
        X[rng.random(X.shape) < 0.05] = np.nan
    z = (
        np.nan_to_num(X[:, 0])
        + 0.7 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2])
        - 0.5 * np.nan_to_num(X[:, 3])
        + 0.4 * rng.normal(size=rows)
    )
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > 0.0).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


_CASES = [
    pytest.param(
        {"cols": 24, "max_bin": 255, "objective": "regression", "quant": "fixedpoint", "with_nans": False, "cap": 0.0},
        id="reg_wide_fixed",
    ),
    pytest.param(
        {"cols": 24, "max_bin": 255, "objective": "binary", "quant": "stochastic", "with_nans": True, "cap": 0.0},
        id="bin_wide_nan_stoch",
    ),
    pytest.param(
        {"cols": 24, "max_bin": 255, "objective": "multiclass", "quant": "fixedpoint", "with_nans": False, "cap": 0.0},
        id="multi_wide_fixed",
    ),
    pytest.param(
        {"cols": 60, "max_bin": 7, "objective": "regression", "quant": "stochastic", "with_nans": True, "cap": 0.0},
        id="reg_narrow_nan_stoch",
    ),
    pytest.param(
        {"cols": 24, "max_bin": 255, "objective": "regression", "quant": "fixedpoint", "with_nans": False, "cap": 0.5},
        id="reg_wide_max_delta_step",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", _CASES)
def test_warp_find_fp32_internal_determinism_and_parity(case):
    """Verify fp64 invariance, fp32 internal determinism, and prediction parity for warp_find_fp32."""
    X, y = _make_data(
        rows=20003,
        cols=case["cols"],
        objective=case["objective"],
        max_bin=case["max_bin"],
        with_nans=case["with_nans"],
    )
    objective = case["objective"]
    base = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 127,
        "max_depth": 7,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "max_bin": case["max_bin"],
        "quant_mode": case["quant"],
        "max_delta_step": case["cap"],
        "verbose": -1,
        "seed": 11,
    }
    if objective == "multiclass":
        base["num_class"] = 4

    # 1. Default fp64 must be bit-identical regardless of warp_find_fp32.
    fp64_on = flc.train(
        {**base, "cuda_precision": "fp64", "cuda_plan": "auto,warp_find_fp32:on"},
        flc.Dataset(X, label=y, params=base),
        num_boost_round=12,
    )
    fp64_off = flc.train(
        {**base, "cuda_precision": "fp64", "cuda_plan": "auto,warp_find_fp32:off"},
        flc.Dataset(X, label=y, params=base),
        num_boost_round=12,
    )
    assert _model_md5(fp64_on) == _model_md5(fp64_off)

    # 2. Under fp32, warp_find_fp32:on is deterministic and bit-identical across
    #    find_split_small / warp_find_mid_ppl tier toggles.
    fp32_base = {**base, "cuda_precision": "fp32"}
    fp32_on_1 = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:on"},
        flc.Dataset(X, label=y, params=fp32_base),
        num_boost_round=12,
    )
    fp32_on_2 = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:on"},
        flc.Dataset(X, label=y, params=fp32_base),
        num_boost_round=12,
    )
    fp32_no_split = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:on,find_split_small:off"},
        flc.Dataset(X, label=y, params=fp32_base),
        num_boost_round=12,
    )
    fp32_no_mid = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:on,warp_find_mid_ppl:off"},
        flc.Dataset(X, label=y, params=fp32_base),
        num_boost_round=12,
    )
    md5_on = _model_md5(fp32_on_1)
    assert md5_on == _model_md5(fp32_on_2)
    assert md5_on == _model_md5(fp32_no_split)
    assert md5_on == _model_md5(fp32_no_mid)

    # 3. Quality parity vs block fp32 and default fp64.
    fp32_off = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:off"},
        flc.Dataset(X, label=y, params=fp32_base),
        num_boost_round=12,
    )
    p_fp64 = fp64_on.predict(X)
    p_on = fp32_on_1.predict(X)
    p_off = fp32_off.predict(X)
    if objective == "regression":
        rmse_fp64 = float(np.sqrt(np.mean((p_fp64 - y) ** 2)))
        rmse_on = float(np.sqrt(np.mean((p_on - y) ** 2)))
        rmse_off = float(np.sqrt(np.mean((p_off - y) ** 2)))
        assert abs(rmse_on - rmse_fp64) < 0.02 * rmse_fp64 + 1e-4
        assert abs(rmse_on - rmse_off) < 0.02 * rmse_off + 1e-4
    elif objective == "binary":
        eps = 1e-15
        ll_fp64 = float(
            -np.mean(y * np.log(np.clip(p_fp64, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p_fp64, eps, 1 - eps)))
        )
        ll_on = float(
            -np.mean(y * np.log(np.clip(p_on, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p_on, eps, 1 - eps)))
        )
        assert abs(ll_on - ll_fp64) < 0.02 * ll_fp64 + 1e-3
    else:
        acc_fp64 = float(np.mean(np.argmax(p_fp64, axis=1) == y.astype(int)))
        acc_on = float(np.mean(np.argmax(p_on, axis=1) == y.astype(int)))
        assert abs(acc_on - acc_fp64) < 0.02
