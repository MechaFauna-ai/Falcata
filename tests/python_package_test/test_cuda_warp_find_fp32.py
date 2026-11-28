# coding: utf-8
"""Tests for cuda_plan key warp_find_fp32 (all-FP32 quantized warp split finder).

Under cuda_precision=fp32, warp_find_fp32 routes eligible quantized levels to an all-FP32
instantiation of FindBestSplitsDiscretizedForLevelWarpKernel instead of falling back to the
block-per-task FP32 kernel.

These tests verify:
1. Dispatch via FALCATA_DEBUG=diag: warp_find_fp32:on routes to the warp kernel across narrow,
   wide (including find_split_small SPLIT tier), and mixed tiers, while warp_find_fp32:off falls
   back to the block kernel.
2. Default cuda_precision=fp64 is bit-identical with warp_find_fp32 on vs off.
3. Under cuda_precision=fp32, warp_find_fp32:on is deterministic across runs and bit-identical
   across internal tier controls (find_split_small:off, warp_find_mid_ppl:off) for narrow
   reverse-only, narrow mixed-direction, wide, mixed bin-width boundary (8/9, 64/65, 128/129),
   NaN-bearing, lambda_l2, min_gain_to_split, GOSS (hessian_ridge_quanta), multiclass, and
   max_delta_step workloads.
4. Held-out test quality parity: under cuda_precision=fp32, warp_find_fp32:on achieves matching
   held-out loss (RMSE for regression, logloss for binary and multiclass) against both
   warp_find_fp32:off (block FP32) and default fp64.
"""

import hashlib
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


def _model_md5(booster):
    lines = [
        line
        for line in booster.model_to_string().splitlines()
        if "cuda_plan" not in line and "cuda_precision" not in line
    ]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _make_data(*, kind, rows, objective, seed=7):
    rng = np.random.default_rng(seed)
    if kind == "wide":
        X = rng.standard_normal((rows, 24)).astype(np.float32)
    elif kind == "wide_nan":
        X = rng.standard_normal((rows, 24)).astype(np.float32)
        X[:, :12][rng.random((rows, 12)) < 0.1] = np.nan
    elif kind == "narrow_rev":
        # 5 values, zero most frequent (mfb_offset=1), no NaNs -> narrow reverse-only scans
        X = rng.integers(0, 5, size=(rows, 40)).astype(np.float32)
        X[rng.random(X.shape) < 0.04] = 0.0
    elif kind == "narrow_mixed_dir":
        # 5 values with NaNs in half the columns -> narrow mixed-direction (reverse + shifted forward) scans
        X = rng.integers(0, 5, size=(rows, 40)).astype(np.float32)
        X[rng.random(X.shape) < 0.04] = 0.0
        X[:, :20][rng.random((rows, 20)) < 0.15] = np.nan
    elif kind == "boundwidth":
        # Columns spanning tier boundaries (8/9, 64/65, 128/129 positions) so warp_find_mid_ppl
        # exercises PPL=2, PPL=4, and PPL=8 alongside narrow 4-lane groups.
        cols = []
        for k in (8, 9, 64, 65, 128, 129):
            nan_col = rng.integers(1, k + 1, size=rows).astype(np.float32)
            nan_col[rng.random(rows) < 0.12] = np.nan
            cols += [rng.integers(0, k, size=rows), rng.integers(1, k + 1, size=rows), nan_col]
        X = np.column_stack(cols + [rng.standard_normal((rows, 4))]).astype(np.float32)
    else:
        raise ValueError(kind)

    w = rng.standard_normal(min(8, X.shape[1]))
    z = (
        np.nan_to_num(X[:, : len(w)]) @ w
        + 0.7 * np.nan_to_num(X[:, 0]) * np.nan_to_num(X[:, 1])
        + 0.5 * rng.normal(size=rows)
    )
    if objective == "multiclass":
        y = np.digitize(z, np.quantile(z, [0.25, 0.5, 0.75])).astype(np.float32)
    elif objective == "binary":
        y = (z > np.median(z)).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


_DIAG_PROBE = """
import sys
sys.path.insert(0, {here!r})
import falcata as flc
from test_cuda_warp_find_fp32 import _make_data
X, y = _make_data(kind={kind!r}, rows=16003, objective="regression", seed=13)
params = {{
    "device_type": "cuda",
    "objective": "regression",
    "num_leaves": 63,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "quant_mode": "fixedpoint",
    "cuda_precision": "fp32",
    "verbosity": 1,
    "seed": 5,
    **{extra!r},
}}
ds = flc.Dataset(X, label=y, params=params)
flc.train(params, ds, num_boost_round=3)
"""


def _probe_diag(kind, extra, debug="diag"):
    code = _DIAG_PROBE.format(
        here=os.path.dirname(os.path.abspath(__file__)),
        kind=kind,
        extra=extra,
    )
    env = {**os.environ, "FALCATA_DEBUG": debug}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("kind", "plan", "expected_tier"),
    [
        pytest.param("wide", "auto,warp_find_fp32:on", "warp (tier: wide)", id="wide-on"),
        pytest.param("wide", "auto,warp_find_fp32:off", "block", id="wide-off"),
        pytest.param("narrow_rev", "auto,warp_find_fp32:on", "warp (tier: few-position)", id="narrow-rev-on"),
        pytest.param("narrow_mixed_dir", "auto,warp_find_fp32:on", "warp (tier: few-position)", id="narrow-mixed-on"),
        pytest.param("narrow_mixed_dir", "auto,warp_find_fp32:off", "block", id="narrow-mixed-off"),
        pytest.param("boundwidth", "auto,warp_find_fp32:on", "warp (tier: few-position + wide)", id="boundwidth-on"),
        pytest.param("boundwidth", "auto,warp_find_fp32:off", "block", id="boundwidth-off"),
    ],
)
def test_warp_find_fp32_diag_dispatch(kind, plan, expected_tier):
    """FALCATA_DEBUG=diag confirms warp_find_fp32:on dispatches to warp tiers and :off falls back to block."""
    out = _probe_diag(kind, {"cuda_plan": plan}, debug="diag")
    found = [f"{k}{t}" for k, t in re.findall(r"split finder: (\S+) kernel for this tree's quantized levels(.*)", out)]
    assert found, f"no split finder diag lines in output:\n{out}"
    assert all(line == expected_tier for line in found), found


@_REQUIRES_CUDA
def test_warp_find_fp32_reaches_split_small_cap():
    """Wide FP32 launches query find_split_cap for the 4-warp find_split_small SPLIT tier."""
    out = _probe_diag("wide", {"cuda_plan": "auto,warp_find_fp32:on,find_split_small:on"}, debug="smcount=170")
    assert "find_split_cap" in out, f"expected find_split_cap SM query in output:\n{out}"


_CASES = [
    pytest.param(
        {"kind": "wide", "objective": "regression", "quant": "fixedpoint", "extra": {}},
        id="reg_wide_fixed",
    ),
    pytest.param(
        {
            "kind": "wide",
            "objective": "regression",
            "quant": "fixedpoint",
            "extra": {"lambda_l2": 2.0, "min_gain_to_split": 0.05},
        },
        id="reg_wide_l2_mingain",
    ),
    pytest.param(
        {
            "kind": "wide",
            "objective": "regression",
            "quant": "fixedpoint",
            "extra": {"data_sample_strategy": "goss", "top_rate": 0.2, "other_rate": 0.1},
        },
        id="reg_wide_goss_ridge",
    ),
    pytest.param(
        {"kind": "wide_nan", "objective": "binary", "quant": "stochastic", "extra": {}},
        id="bin_wide_nan_stoch",
    ),
    pytest.param(
        {"kind": "wide", "objective": "multiclass", "quant": "fixedpoint", "extra": {"lambda_l2": 1.0}},
        id="multi_wide_fixed",
    ),
    pytest.param(
        {"kind": "narrow_rev", "objective": "regression", "quant": "stochastic", "extra": {}},
        id="reg_narrow_rev_stoch",
    ),
    pytest.param(
        {"kind": "narrow_mixed_dir", "objective": "regression", "quant": "stochastic", "extra": {}},
        id="reg_narrow_mixed_dir_stoch",
    ),
    pytest.param(
        {"kind": "boundwidth", "objective": "binary", "quant": "fixedpoint", "extra": {}},
        id="bin_boundwidth_mid_ppl",
    ),
    pytest.param(
        {"kind": "wide", "objective": "regression", "quant": "fixedpoint", "extra": {"max_delta_step": 0.5}},
        id="reg_wide_max_delta_step",
    ),
]


def _logloss_binary(y_true, p):
    eps = 1e-15
    p_clip = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p_clip) + (1.0 - y_true) * np.log(1.0 - p_clip)))


def _logloss_multiclass(y_true, p):
    eps = 1e-15
    yi = y_true.astype(np.int64)
    probs = np.clip(p[np.arange(len(yi)), yi], eps, 1.0)
    return float(-np.mean(np.log(probs)))


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", _CASES)
def test_warp_find_fp32_internal_determinism_and_parity(case):
    """Verify fp64 invariance, fp32 internal tier determinism, and held-out metric parity."""
    X_all, y_all = _make_data(kind=case["kind"], rows=20003, objective=case["objective"])
    n_train = 16003
    X_tr, y_tr = np.ascontiguousarray(X_all[:n_train]), np.ascontiguousarray(y_all[:n_train])
    X_va, y_va = np.ascontiguousarray(X_all[n_train:]), np.ascontiguousarray(y_all[n_train:])

    objective = case["objective"]
    base = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 127,
        "max_depth": 7,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "max_bin": 255,
        "quant_mode": case["quant"],
        "verbose": -1,
        "seed": 11,
        **case["extra"],
    }
    if objective == "multiclass":
        base["num_class"] = 4

    # 1. Default fp64 must be bit-identical regardless of warp_find_fp32.
    fp64_on = flc.train(
        {**base, "cuda_precision": "fp64", "cuda_plan": "auto,warp_find_fp32:on"},
        flc.Dataset(X_tr, label=y_tr, params=base),
        num_boost_round=12,
    )
    fp64_off = flc.train(
        {**base, "cuda_precision": "fp64", "cuda_plan": "auto,warp_find_fp32:off"},
        flc.Dataset(X_tr, label=y_tr, params=base),
        num_boost_round=12,
    )
    assert _model_md5(fp64_on) == _model_md5(fp64_off)

    # 2. Under fp32, warp_find_fp32:on is deterministic and bit-identical across
    #    find_split_small / warp_find_mid_ppl tier toggles.
    fp32_base = {**base, "cuda_precision": "fp32"}
    fp32_on_1 = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:on"},
        flc.Dataset(X_tr, label=y_tr, params=fp32_base),
        num_boost_round=12,
    )
    fp32_on_2 = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:on"},
        flc.Dataset(X_tr, label=y_tr, params=fp32_base),
        num_boost_round=12,
    )
    fp32_no_split = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:on,find_split_small:off"},
        flc.Dataset(X_tr, label=y_tr, params=fp32_base),
        num_boost_round=12,
    )
    fp32_no_mid = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:on,warp_find_mid_ppl:off"},
        flc.Dataset(X_tr, label=y_tr, params=fp32_base),
        num_boost_round=12,
    )
    md5_on = _model_md5(fp32_on_1)
    assert md5_on == _model_md5(fp32_on_2)
    assert md5_on == _model_md5(fp32_no_split)
    assert md5_on == _model_md5(fp32_no_mid)

    # 3. Held-out quality parity vs BOTH block fp32 (:off) and default fp64.
    fp32_off = flc.train(
        {**fp32_base, "cuda_plan": "auto,warp_find_fp32:off"},
        flc.Dataset(X_tr, label=y_tr, params=fp32_base),
        num_boost_round=12,
    )
    p_fp64 = fp64_on.predict(X_va)
    p_on = fp32_on_1.predict(X_va)
    p_off = fp32_off.predict(X_va)
    assert np.all(np.isfinite(p_on))
    if objective == "regression":
        rmse_fp64 = float(np.sqrt(np.mean((p_fp64 - y_va) ** 2)))
        rmse_on = float(np.sqrt(np.mean((p_on - y_va) ** 2)))
        rmse_off = float(np.sqrt(np.mean((p_off - y_va) ** 2)))
        assert abs(rmse_on - rmse_fp64) < 0.02 * rmse_fp64 + 1e-4
        assert abs(rmse_on - rmse_off) < 0.02 * rmse_off + 1e-4
    elif objective == "binary":
        ll_fp64 = _logloss_binary(y_va, p_fp64)
        ll_on = _logloss_binary(y_va, p_on)
        ll_off = _logloss_binary(y_va, p_off)
        assert abs(ll_on - ll_fp64) < 0.02 * ll_fp64 + 1e-3
        assert abs(ll_on - ll_off) < 0.02 * ll_off + 1e-3
    else:
        ll_fp64 = _logloss_multiclass(y_va, p_fp64)
        ll_on = _logloss_multiclass(y_va, p_on)
        ll_off = _logloss_multiclass(y_va, p_off)
        assert abs(ll_on - ll_fp64) < 0.02 * ll_fp64 + 1e-3
        assert abs(ll_on - ll_off) < 0.02 * ll_off + 1e-3
        acc_fp64 = float(np.mean(np.argmax(p_fp64, axis=1) == y_va.astype(int)))
        acc_on = float(np.mean(np.argmax(p_on, axis=1) == y_va.astype(int)))
        acc_off = float(np.mean(np.argmax(p_off, axis=1) == y_va.astype(int)))
        assert abs(acc_on - acc_fp64) < 0.02
        assert abs(acc_on - acc_off) < 0.02
