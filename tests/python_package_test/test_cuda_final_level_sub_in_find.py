"""Bit-identity tests for final_level_sub_in_find and hist_subtract_vec4.

cuda_plan key final_level_sub_in_find eliminates SubtractHistogram and
CopyChangedNumBitHistogram on the final searched level (level == num_levels - 2)
of complete depth-limited quantized device trees by computing parent - smaller
on the fly inside WarpFindBest.

cuda_plan key hist_subtract_vec4 vectorizes histogram subtract and bit-change
copy kernels using 128-bit loads/stores (int4 / longlong2 / double2 / float4).

Both must be 100% byte-identical to their off states across all tree depths,
histogram bit-width transitions (16/16, 32/16->16, 32/16->32, 32/32), MFB /
need-fix columns, zero_as_missing, NaNs, feature sampling, bagging, multiclass,
and odd total bin counts.
"""

import os
import re

import falcata as flc
import numpy as np
import pytest

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

BASE = {
    "objective": "regression",
    "device_type": "cuda",
    "use_quantized_grad": True,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 17,
}


def _make_data(rows, cols=37, sparse_zero_cols=12, nan_cols=8, n_classes=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    if sparse_zero_cols > 0:
        # Zero-heavy columns: triggers both most_freq_bin != 0 (need_fix) and mfb == 0
        X[:, :sparse_zero_cols] *= rng.random((rows, sparse_zero_cols)) < 0.45
    # Low-cardinality columns (narrow 4-lane warp_find tier) mixed with 255-bin columns (wide tier)
    X[:, sparse_zero_cols : sparse_zero_cols + 6] = np.round(
        X[:, sparse_zero_cols : sparse_zero_cols + 6] * 2.0
    )
    if nan_cols > 0:
        X[:, -nan_cols:][rng.random((rows, nan_cols)) < 0.08] = np.nan
    lin = np.nan_to_num(X[:, :8]) @ rng.standard_normal(8)
    if n_classes > 1:
        logits = np.stack([lin * (k + 1) * 0.3 + rng.standard_normal(rows) for k in range(n_classes)], axis=1)
        y = np.argmax(logits, axis=1).astype(np.float32)
    else:
        y = (lin + rng.standard_normal(rows)).astype(np.float32)
    return X, y


def _train(X, y, params, rounds=10):
    p = {**BASE, **params}
    ds = flc.Dataset(X, label=y, params=p)
    model = flc.train(p, ds, num_boost_round=rounds).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.MULTILINE)


DEPTH_CASES = [
    pytest.param(1, 2, id="depth1"),
    pytest.param(2, 4, id="depth2"),
    pytest.param(6, 63, id="depth6"),
    pytest.param(10, 1023, id="depth10"),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize(("max_depth", "num_leaves"), DEPTH_CASES)
def test_final_level_sub_in_find_depths_cuda(max_depth, num_leaves, quant):
    X, y = _make_data(30_003, cols=37, sparse_zero_cols=12, nan_cols=8, seed=1)
    params = {"quant_mode": quant, "max_depth": max_depth, "num_leaves": num_leaves}
    on = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:on"}, rounds=8)
    off = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:off"}, rounds=8)
    assert on == off


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_depth", [2, 6])
def test_final_level_sub_in_find_32bit_to_16bit_transitions_cuda(max_depth):
    # Large row count with high num_grad_quant_bins forces 32-bit parent histograms transitioning
    # to 16-bit smaller/larger children (and 32-bit larger with 16-bit smaller) on the final level.
    X, y = _make_data(260_000, cols=19, sparse_zero_cols=6, nan_cols=4, seed=2)
    params = {
        "quant_mode": "fixedpoint",
        "num_grad_quant_bins": 32,
        "max_depth": max_depth,
        "num_leaves": (1 << max_depth) - 1,
    }
    on = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:on"}, rounds=6)
    off = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:off"}, rounds=6)
    assert on == off


FEATURE_AND_DATA_CASES = [
    pytest.param({"zero_as_missing": True}, id="zero-as-missing"),
    pytest.param({"feature_fraction": 0.4}, id="feature-fraction"),
    pytest.param({"feature_fraction_bynode": 0.5}, id="feature-fraction-bynode"),
    pytest.param({"bagging_fraction": 0.7, "bagging_freq": 1}, id="bagging"),
    pytest.param({"objective": "multiclass", "num_class": 4}, id="multiclass"),
    pytest.param({"extra_trees": True}, id="extra-trees"),
    pytest.param({"cuda_precision": "fp32"}, id="fp32-gain"),
    pytest.param({"num_leaves": 25}, id="num-leaves-early-stop"),
    pytest.param({"plan": "device_level_graph:off"}, id="no-graph"),
    pytest.param({"plan": "fix_subtract_masked:off"}, id="no-fix-masked"),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("cfg", FEATURE_AND_DATA_CASES)
def test_final_level_sub_in_find_features_and_edge_cases_cuda(cfg):
    cfg = dict(cfg)
    extra_plan = cfg.pop("plan", "")
    n_classes = cfg.get("num_class", 0)
    X, y = _make_data(24_001, cols=31, sparse_zero_cols=10, nan_cols=6, n_classes=n_classes, seed=3)
    params = {
        "quant_mode": "fixedpoint",
        "max_depth": 6,
        "num_leaves": 63,
        **cfg,
    }
    prefix = f"auto,{extra_plan}," if extra_plan else "auto,"
    on = _train(X, y, {**params, "cuda_plan": prefix + "final_level_sub_in_find:on"}, rounds=6)
    off = _train(X, y, {**params, "cuda_plan": prefix + "final_level_sub_in_find:off"}, rounds=6)
    assert on == off


@_REQUIRES_CUDA
@pytest.mark.parametrize("sparse_zero_cols", [0, 12], ids=["no-need-fix", "with-need-fix"])
@pytest.mark.parametrize("max_bin", [15, 63, 255])
def test_final_level_sub_in_find_mfb_and_bin_widths_cuda(sparse_zero_cols, max_bin):
    # Exercises both need_fix_histogram_features_ empty and non-empty, narrow & wide tiers, and odd total bins.
    X, y = _make_data(20_005, cols=23, sparse_zero_cols=sparse_zero_cols, nan_cols=5, seed=4)
    params = {
        "quant_mode": "fixedpoint",
        "max_depth": 6,
        "num_leaves": 63,
        "max_bin": max_bin,
    }
    on = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:on"}, rounds=6)
    off = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:off"}, rounds=6)
    assert on == off


VEC4_CASES = [
    pytest.param({"quant_mode": "fixedpoint", "max_depth": 6, "num_leaves": 63}, id="quant-device-tree"),
    pytest.param(
        {"quant_mode": "fixedpoint", "max_depth": 6, "num_leaves": 63, "plan": "device_tree:off"},
        id="quant-host-level",
    ),
    pytest.param(
        {"quant_mode": "fixedpoint", "max_depth": 6, "num_leaves": 63, "feature_fraction": 0.4},
        id="quant-sampled",
    ),
    pytest.param(
        {
            "quant_mode": "fixedpoint",
            "max_depth": 6,
            "num_leaves": 63,
            "plan": "fix_subtract_masked:off,final_level_sub_in_find:off",
        },
        id="quant-unmasked",
    ),
    pytest.param(
        {
            "use_quantized_grad": False,
            "quant_mode": "none",
            "deterministic": True,
            "max_depth": 6,
            "num_leaves": 63,
        },
        id="nonquant-det-fp64",
    ),
    pytest.param(
        {
            "use_quantized_grad": False,
            "quant_mode": "none",
            "deterministic": True,
            "max_depth": 6,
            "num_leaves": 63,
            "feature_fraction": 0.5,
        },
        id="nonquant-det-sampled",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("cfg", VEC4_CASES)
def test_hist_subtract_vec4_bit_identical_cuda(cfg):
    cfg = dict(cfg)
    extra_plan = cfg.pop("plan", "")
    # cols=29 + max_bin=31 yields odd num_total_bin not divisible by 4 (tests alignment + tail handling).
    X, y = _make_data(40_001, cols=29, sparse_zero_cols=10, nan_cols=5, seed=5)
    params = {"max_bin": 31, **cfg}
    prefix = f"auto,{extra_plan}," if extra_plan else "auto,"
    on = _train(X, y, {**params, "cuda_plan": prefix + "hist_subtract_vec4:on"}, rounds=8)
    off = _train(X, y, {**params, "cuda_plan": prefix + "hist_subtract_vec4:off"}, rounds=8)
    assert on == off
