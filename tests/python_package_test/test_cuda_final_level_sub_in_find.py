"""Bit-identity and route-activation tests for final_level_sub_in_find and hist_subtract_vec4.

cuda_plan key final_level_sub_in_find eliminates SubtractHistogram and
CopyChangedNumBitHistogram on the final searched level (level == num_levels - 2)
of complete depth-limited quantized device trees by computing parent - smaller
on the fly inside WarpFindBest.

cuda_plan key hist_subtract_vec4 vectorizes histogram subtract and bit-change
copy kernels using 128-bit loads/stores (int4 / longlong2 / double2 / float4).

Both must be 100% byte-identical to their off states across all tree depths,
histogram bit-width transitions (16/16, 32/16->16, 32/16->32, 32/32), MFB /
need-fix columns, zero_as_missing, NaNs, feature sampling, bagging, multiclass,
pair-chunked final levels, live-grid routes, non-quantized fp32/fp64 paths, and
odd total bin counts.
"""

import json
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
    "use_quantized_grad": True,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 17,
}


def _make_data(rows, *, cols=37, sparse_zero_cols=12, nan_cols=8, n_classes=0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    if sparse_zero_cols > 0:
        # Zero-heavy signed columns: most_freq_bin != 0 (exercises need_fix_histogram_features_).
        X[:, :sparse_zero_cols] *= rng.random((rows, sparse_zero_cols)) < 0.45
    # Low-cardinality columns (narrow 4-lane warp_find tier) mixed with 255-bin columns (wide tier).
    narrow_end = min(cols, sparse_zero_cols + 6)
    if narrow_end > sparse_zero_cols:
        X[:, sparse_zero_cols:narrow_end] = np.round(X[:, sparse_zero_cols:narrow_end] * 2.0)
    if nan_cols > 0:
        X[:, -nan_cols:][rng.random((rows, nan_cols)) < 0.08] = np.nan
    lin = np.nan_to_num(X[:, : min(8, cols)]) @ rng.standard_normal(min(8, cols))
    if n_classes > 1:
        logits = np.stack([lin * (k + 1) * 0.3 + rng.standard_normal(rows) for k in range(n_classes)], axis=1)
        y = np.argmax(logits, axis=1).astype(np.float32)
    else:
        y = (lin + rng.standard_normal(rows)).astype(np.float32)
    return X, y


def _make_mfb0_data(rows, cols=29, nan_cols=4, seed=0):
    """Non-negative zero-heavy columns so every feature has most_freq_bin == 0 (need_fix empty)."""
    rng = np.random.default_rng(seed)
    X = np.abs(rng.standard_normal((rows, cols)).astype(np.float32)) + 0.01
    X *= rng.random((rows, cols)) < 0.70  # >=30% exact zeros -> bin 0 is strictly most frequent on every column
    if nan_cols > 0:
        X[:, -nan_cols:][rng.random((rows, nan_cols)) < 0.05] = np.nan
    y = (np.nan_to_num(X[:, : min(8, cols)]) @ rng.standard_normal(min(8, cols)) + rng.standard_normal(rows)).astype(
        np.float32
    )
    return X, y


def _train(X, y, params, rounds=10, assert_odd_bins=False):
    p = {**BASE, **params}
    ds = flc.Dataset(X, label=y, params=p)
    ds.construct()
    if assert_odd_bins:
        total_bins = sum(ds.feature_num_bin(i) for i in range(ds.num_feature()))
        assert total_bins % 4 != 0, f"expected total_bins % 4 != 0 to test vec4 tail, got {total_bins}"
    model = flc.train(p, ds, num_boost_round=rounds).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.MULTILINE)


def _diag_probe(kind):
    """Subprocess entry point for FALCATA_DEBUG=diag assertions (FalcataDebug is cached per process)."""
    if kind == "fix_none":
        X, y = _make_mfb0_data(20_003, cols=23, nan_cols=4, seed=10)
        params = {"quant_mode": "fixedpoint", "max_depth": 6, "num_leaves": 63, "max_bin": 63}
        on = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:on"}, rounds=4)
        off = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:off"}, rounds=4)
        assert on == off
    elif kind == "fix_used_features":
        # max_bin=7 (<=8) + feature_fraction=0.4 + zero-heavy signed cols -> FixOnlyUsedFeaturesBatchedKernel
        X, y = _make_data(24_001, cols=48, sparse_zero_cols=40, nan_cols=0, seed=11)
        params = {
            "quant_mode": "fixedpoint",
            "max_depth": 6,
            "num_leaves": 63,
            "max_bin": 7,
            "feature_fraction": 0.4,
        }
        on = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:on"}, rounds=4)
        off = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:off"}, rounds=4)
        assert on == off
    elif kind == "fix_batched":
        X, y = _make_data(20_003, cols=24, sparse_zero_cols=16, nan_cols=4, seed=12)
        params = {"quant_mode": "fixedpoint", "max_depth": 4, "num_leaves": 15, "max_bin": 255}
        on = _train(
            X,
            y,
            {**params, "cuda_plan": "auto,device_tree_live_grid:4096,final_level_sub_in_find:on"},
            rounds=4,
        )
        off = _train(
            X,
            y,
            {**params, "cuda_plan": "auto,device_tree_live_grid:4096,final_level_sub_in_find:off"},
            rounds=4,
        )
        assert on == off
    elif kind == "fix_live":
        # Wide need_fix list + depth 8 + device_tree_live_grid:1 -> FixHistogramDiscretizedLiveKernel
        X, y = _make_data(30_001, cols=64, sparse_zero_cols=48, nan_cols=4, seed=13)
        params = {"quant_mode": "fixedpoint", "max_depth": 8, "num_leaves": 255, "max_bin": 255}
        on = _train(
            X,
            y,
            {**params, "cuda_plan": "auto,device_tree_live_grid:1,final_level_sub_in_find:on"},
            rounds=3,
        )
        off = _train(
            X,
            y,
            {**params, "cuda_plan": "auto,device_tree_live_grid:1,final_level_sub_in_find:off"},
            rounds=3,
        )
        assert on == off
    elif kind == "vec4_live_masked":
        # Wide columns + depth 8 + device_tree_live_grid:1 with BOTH final_level_sub_in_find:on and
        # hist_subtract_vec4:on exercises FixSubtractDiscretizedMaskedLiveKernel<true> and
        # CopyChangedNumBitHistogramLiveKernel<true> on earlier levels while skipping final level.
        X, y = _make_data(55_001, cols=110, sparse_zero_cols=60, nan_cols=4, seed=14)
        params = {
            "quant_mode": "fixedpoint",
            "num_grad_quant_bins": 16,
            "max_depth": 8,
            "num_leaves": 255,
            "max_bin": 255,
        }
        on = _train(
            X,
            y,
            {**params, "cuda_plan": "auto,device_tree_live_grid:1,final_level_sub_in_find:on,hist_subtract_vec4:on"},
            rounds=3,
        )
        off = _train(
            X,
            y,
            {**params, "cuda_plan": "auto,device_tree_live_grid:1,final_level_sub_in_find:on,hist_subtract_vec4:off"},
            rounds=3,
        )
        assert on == off
    elif kind == "vec4_live_subtract":
        # fix_subtract_masked:off exercises SubtractHistogramDiscretizedLiveKernel<true> and
        # CopyChangedNumBitHistogramLiveKernel<true> on level D-3 (32 pairs * ~8 blocks > 170).
        X, y = _make_mfb0_data(55_001, cols=120, nan_cols=4, seed=15)
        params = {
            "quant_mode": "fixedpoint",
            "num_grad_quant_bins": 16,
            "max_depth": 8,
            "num_leaves": 255,
            "max_bin": 255,
        }
        on = _train(
            X,
            y,
            {
                **params,
                "cuda_plan": (
                    "auto,device_tree_live_grid:1,fix_subtract_masked:off,"
                    "final_level_sub_in_find:on,hist_subtract_vec4:on"
                ),
            },
            rounds=3,
        )
        off = _train(
            X,
            y,
            {
                **params,
                "cuda_plan": (
                    "auto,device_tree_live_grid:1,fix_subtract_masked:off,"
                    "final_level_sub_in_find:on,hist_subtract_vec4:off"
                ),
            },
            rounds=3,
        )
        assert on == off
    elif kind == "pair_chunked":
        # Run with FALCATA_DEBUG=diag,vramfree=4 to force chunks > 1 on max_depth=9 device_tree.
        X, y = _make_data(35_003, cols=45, sparse_zero_cols=16, nan_cols=4, seed=16)
        params = {"quant_mode": "fixedpoint", "max_depth": 9, "num_leaves": 511, "max_bin": 63}
        on = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:on"}, rounds=3)
        off = _train(X, y, {**params, "cuda_plan": "auto,final_level_sub_in_find:off"}, rounds=3)
        assert on == off
    elif kind == "nonquant_fp32":
        # Non-quantized fp32 histogram layout (float4 subtract path): verify activation, finite preds, and parity.
        X, y = _make_data(25_003, cols=29, sparse_zero_cols=8, nan_cols=4, seed=17)
        for small_leaf in ("on", "off"):
            p_base = {
                **BASE,
                "use_quantized_grad": False,
                "quant_mode": "none",
                "cuda_precision": "fp32",
                "deterministic": False,
                "max_depth": 6,
                "num_leaves": 63,
                "max_bin": 31,
            }
            p_on = {
                **p_base,
                "cuda_plan": f"auto,nonquant_hist_cost:off,small_leaf_construct:{small_leaf},hist_subtract_vec4:on",
            }
            p_off = {
                **p_base,
                "cuda_plan": f"auto,nonquant_hist_cost:off,small_leaf_construct:{small_leaf},hist_subtract_vec4:off",
            }
            ds = flc.Dataset(X, label=y, params=p_on)
            ds.construct()
            assert sum(ds.feature_num_bin(i) for i in range(ds.num_feature())) % 4 != 0
            bst_on = flc.train(p_on, ds, num_boost_round=6)
            bst_off = flc.train(p_off, ds, num_boost_round=6)
            pred_on = bst_on.predict(X[:2048])
            pred_off = bst_off.predict(X[:2048])
            assert np.all(np.isfinite(pred_on))
            assert np.all(np.isfinite(pred_off))
            # Float-atomic accumulation order is nondeterministic, so predictions must agree closely.
            rmse_diff = float(np.sqrt(np.mean((pred_on - pred_off) ** 2)))
            y_std = float(np.std(y[:2048]))
            assert rmse_diff < 0.05 * y_std, f"fp32 vec4/scalar pred divergence too large: {rmse_diff} vs std {y_std}"
    else:
        raise ValueError(f"unknown probe kind: {kind}")
    print(json.dumps({"ok": True, "kind": kind}))


def _run_diag_probe(kind, debug="diag"):
    code = "import runpy, sys; runpy.run_path(sys.argv[1])['_diag_probe'](sys.argv[2])"
    env = dict(os.environ, FALCATA_DEBUG=debug)
    proc = subprocess.run(
        [sys.executable, "-c", code, __file__, kind],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"probe {kind} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc.stderr


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
@pytest.mark.parametrize(
    ("rows", "quant_bins", "max_depth", "num_leaves"),
    [
        # ~45K rows with num_grad_quant_bins=16 -> 16-bit threshold ~2047 rows; root/upper levels are 32-bit,
        # and depth 4 (final searched level) crosses 32->16, 32->32+16, 32->32, and 16->16 simultaneously.
        pytest.param(45_000, 16, 5, 31, id="32to16-depth5"),
        pytest.param(260_000, 32, 2, 3, id="32bit-depth2"),
        pytest.param(260_000, 32, 6, 63, id="32to16-depth6"),
    ],
)
def test_final_level_sub_in_find_32bit_to_16bit_transitions_cuda(rows, quant_bins, max_depth, num_leaves):
    X, y = _make_data(rows, cols=19, sparse_zero_cols=6, nan_cols=4, seed=2)
    params = {
        "quant_mode": "fixedpoint",
        "num_grad_quant_bins": quant_bins,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
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
@pytest.mark.parametrize("mfb_mode", ["mfb0-only", "with-need-fix"])
@pytest.mark.parametrize("max_bin", [15, 63, 255])
def test_final_level_sub_in_find_mfb_and_bin_widths_cuda(mfb_mode, max_bin):
    # Exercises both need_fix_histogram_features_ empty (mfb0-only) and non-empty, narrow & wide tiers.
    if mfb_mode == "mfb0-only":
        X, y = _make_mfb0_data(20_005, cols=23, nan_cols=5, seed=4)
    else:
        X, y = _make_data(20_005, cols=23, sparse_zero_cols=12, nan_cols=5, seed=4)
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
    # cols=30 + max_bin=31 yields num_total_bin % 4 != 0 (explicitly asserted inside _train).
    X, y = _make_data(40_001, cols=30, sparse_zero_cols=10, nan_cols=5, seed=5)
    params = {"max_bin": 31, **cfg}
    prefix = f"auto,{extra_plan}," if extra_plan else "auto,"
    on = _train(X, y, {**params, "cuda_plan": prefix + "hist_subtract_vec4:on"}, rounds=8, assert_odd_bins=True)
    off = _train(X, y, {**params, "cuda_plan": prefix + "hist_subtract_vec4:off"}, rounds=8)
    assert on == off


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("kind", "expected_pattern"),
    [
        pytest.param("fix_none", r"final_level_sub_in_find: pairs=16 fix=none", id="fix-none"),
        pytest.param("fix_used_features", r"final_level_sub_in_find: pairs=16 fix=used_features", id="fix-used"),
        pytest.param("fix_batched", r"final_level_sub_in_find: pairs=4 fix=batched", id="fix-batched"),
        pytest.param("fix_live", r"final_level_sub_in_find: pairs=64 fix=live", id="fix-live"),
    ],
)
def test_final_level_sub_in_find_diag_routes_cuda(kind, expected_pattern):
    diag = _run_diag_probe(kind)
    assert re.search(expected_pattern, diag), f"expected /{expected_pattern}/ in diag:\n{diag}"


@_REQUIRES_CUDA
def test_hist_subtract_vec4_live_grid_routes_cuda():
    diag_masked = _run_diag_probe("vec4_live_masked")
    assert re.search(r"device_tree_live_grid:.*fix_subtract=live-masked.*copy=live.*vec4=1", diag_masked), diag_masked
    assert re.search(r"device_tree_live_grid:.*fix_subtract=live-masked.*copy=live.*vec4=0", diag_masked), diag_masked
    assert re.search(r"final_level_sub_in_find: pairs=64 fix=live", diag_masked), diag_masked

    diag_sub = _run_diag_probe("vec4_live_subtract")
    assert re.search(r"device_tree_live_grid:.*fix_subtract=live-subtract.*copy=live.*vec4=1", diag_sub), diag_sub
    assert re.search(r"device_tree_live_grid:.*fix_subtract=live-subtract.*copy=live.*vec4=0", diag_sub), diag_sub
    assert re.search(r"final_level_sub_in_find: pairs=64 fix=none", diag_sub), diag_sub


@_REQUIRES_CUDA
def test_final_level_sub_in_find_pair_chunked_cuda():
    diag = _run_diag_probe("pair_chunked", debug="diag,vramfree=4")
    m = re.search(r"device_tree: scratch estimate .* (\d+) searched pairs, (\d+) chunks of (\d+) pairs", diag)
    assert m is not None, f"expected scratch estimate in diag:\n{diag}"
    assert int(m.group(2)) > 1, f"expected pair chunking (chunks > 1) in diag:\n{diag}"
    assert re.search(r"final_level_sub_in_find: pairs=\d+ fix=", diag), diag


@_REQUIRES_CUDA
def test_hist_subtract_vec4_nonquant_fp32_cuda():
    diag = _run_diag_probe("nonquant_fp32")
    assert re.search(r"nonquant_subtract: pairs=\d+ hist_fp32=1 vec4=1", diag), diag
    assert re.search(r"nonquant_subtract: pairs=\d+ hist_fp32=1 vec4=0", diag), diag
