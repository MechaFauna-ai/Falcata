# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key device_graph_indirect.

device_graph_indirect moves per-tree-varying runtime scalars into CUDADeviceTreeGraphParams
(uploaded once per tree with CUDADeviceLevelConfig), passes CUDADeviceLevelConfig by device
pointer to the device-level control kernels, and buckets dynamic shared memory and launch
grids/blocks so feature-sampled heterogeneous datasets hit one cached CUDA graph instead of
churning captures/updates or falling back to host enqueue.
"""

import hashlib
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


def _model_md5(params, X, y, rounds):
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _mixbins_wide_data(n=6000, m=1200, with_nan=False, seed=42):
    rng = np.random.default_rng(seed)
    X = np.floor(rng.random((n, m)) * rng.integers(3, 7, size=m)).astype(np.float32)
    if with_nan:
        nan_cols = int(0.6 * m)
        mask = rng.random((n, nan_cols)) < 0.05
        X[:, :nan_cols] = np.where(mask, np.nan, X[:, :nan_cols])
    y = (np.nan_to_num(X[:, :40]) @ rng.standard_normal(40) + rng.standard_normal(n)).astype(np.float32)
    return X, y


def _compact8_data(n=5000, m=400, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    X[rng.random((n, m)) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, :20]) @ rng.standard_normal(20) + rng.standard_normal(n)).astype(np.float32)
    return X, y


def _assert_same_model(params, X, y, rounds):
    off = _model_md5({**params, "cuda_plan": "auto,device_graph_indirect:off"}, X, y, rounds)
    on = _model_md5({**params, "cuda_plan": "auto,device_graph_indirect:on"}, X, y, rounds)
    assert on == off


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("max_depth", "num_leaves", "with_nan"),
    [
        (-1, 31, False),  # r24-churn-style selective mixed-bins
        (6, 63, False),  # complete device tree mixed-bins
        (6, 63, True),  # complete device tree mixed-bins + NaN (variable used tasks)
        (-1, 15, True),  # selective mixed-bins + NaN
    ],
)
def test_device_graph_indirect_mixbins_bit_identical_cuda(max_depth, num_leaves, with_nan):
    X, y = _mixbins_wide_data(n=6000, m=1200, with_nan=with_nan, seed=11)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "max_bin": 15,
        "feature_fraction": 0.2,
        "min_data_in_leaf": 20,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 42,
    }
    _assert_same_model(params, X, y, rounds=25)


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_bin", [15, 63])
def test_device_graph_indirect_compact_sampled_bit_identical_cuda(max_bin):
    X, y = _compact8_data(n=5000, m=400, seed=13)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": max_bin,
        "feature_fraction": 0.2,
        "min_data_in_leaf": 20,
        "use_quantized_grad": True,
        "quant_mode": "fixedpoint",
        "verbose": -1,
        "seed": 42,
    }
    _assert_same_model(params, X, y, rounds=25)


@_REQUIRES_CUDA
def test_device_graph_indirect_goss_bit_identical_cuda():
    X, y = _compact8_data(n=6000, m=200, seed=19)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "data_sample_strategy": "goss",
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": 63,
        "feature_fraction": 0.3,
        "min_data_in_leaf": 20,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 42,
    }
    _assert_same_model(params, X, y, rounds=25)


@_REQUIRES_CUDA
def test_device_graph_indirect_bynode_bit_identical_cuda():
    X, y = _mixbins_wide_data(n=5000, m=400, with_nan=True, seed=23)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": 15,
        "feature_fraction": 0.3,
        "feature_fraction_bynode": 0.7,
        "min_data_in_leaf": 20,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 42,
    }
    _assert_same_model(params, X, y, rounds=20)


@_REQUIRES_CUDA
def test_device_graph_indirect_nonquant_and_multiclass_guard_cuda():
    rng = np.random.default_rng(29)
    X = rng.standard_normal((4000, 160)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    y_reg = (np.nan_to_num(X[:, :10]) @ rng.standard_normal(10) + rng.standard_normal(4000)).astype(np.float32)
    y_mc = rng.integers(0, 4, size=4000).astype(np.float32)

    nq_params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": 63,
        "feature_fraction": 0.25,
        "use_quantized_grad": False,
        "deterministic": True,
        "verbose": -1,
        "seed": 42,
    }
    _assert_same_model(nq_params, X, y_reg, rounds=15)

    mc_params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": 4,
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": 63,
        "feature_fraction": 0.25,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 42,
    }
    _assert_same_model(mc_params, X, y_mc, rounds=10)


@_REQUIRES_CUDA
def test_device_graph_indirect_eliminates_churn_diag_cuda():
    complete_code = (
        "import sys, json, numpy as np, falcata as flc\n"
        "rng = np.random.default_rng(42)\n"
        "n, m = 6000, 1200\n"
        "X = rng.integers(0, 5, size=(n, m)).astype(np.float32)\n"
        "X[:, :720] = np.where(rng.random((n, 720)) < 0.05, np.nan, X[:, :720])\n"
        "y = (np.nan_to_num(X[:, :40]) @ rng.standard_normal(40) + rng.standard_normal(n)).astype(np.float32)\n"
        "p = dict(json.loads(sys.argv[2]), cuda_plan=sys.argv[1])\n"
        "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=30)\n"
    )
    sel_code = (
        "import sys, json, numpy as np, falcata as flc\n"
        "rng = np.random.default_rng(42)\n"
        "n, m = 6000, 1200\n"
        "X = np.floor(rng.random((n, m)) * rng.integers(3, 7, size=m)).astype(np.float32)\n"
        "y = (X[:, :40] @ rng.standard_normal(40) + rng.standard_normal(n)).astype(np.float32)\n"
        "p = dict(json.loads(sys.argv[2]), cuda_plan=sys.argv[1])\n"
        "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=30)\n"
    )
    env = dict(os.environ, FALCATA_DEBUG="diag")

    def run_diag(code, plan, base_params):
        out = subprocess.run(
            [sys.executable, "-c", code, plan, json.dumps(base_params)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stderr

    # 1. Complete device tree (Numerai-NaN-style): off churns misses; on stays at ~100% steady-state hits.
    complete_params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 63,
        "max_depth": 6,
        "max_bin": 15,
        "feature_fraction": 0.2,
        "min_data_in_leaf": 20,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 42,
    }
    off_err = run_diag(complete_code, "auto,device_graph_indirect:off", complete_params)
    on_err = run_diag(complete_code, "auto,device_graph_indirect:on", complete_params)

    off_matches = re.findall(r"complete depth=\d+.*hits=(\d+) misses=(\d+)", off_err)
    on_matches = re.findall(r"complete depth=\d+.*hits=(\d+) misses=(\d+)", on_err)
    assert off_matches
    assert on_matches
    off_hits, off_misses = map(int, off_matches[-1])
    on_hits, on_misses = map(int, on_matches[-1])
    assert off_misses > 15, f"expected churn with device_graph_indirect:off, got hits={off_hits} misses={off_misses}"
    assert on_misses <= 8, f"expected low misses with device_graph_indirect:on, got hits={on_hits} misses={on_misses}"
    assert on_hits >= 22, f"expected high hits with device_graph_indirect:on, got hits={on_hits} misses={on_misses}"
    off_updates = [line for line in off_err.splitlines() if "device_level_graph_update: updates=" in line]
    on_updates = [line for line in on_err.splitlines() if "device_level_graph_update: updates=" in line]
    assert off_updates
    assert on_updates
    assert "updates=0 " not in off_updates[-1]
    assert "updates=0 " in on_updates[-1]

    # 2. Selective tree (r24-churn-style, max_depth=-1): off disables graphs after 9 misses; on keeps graphs active.
    sel_params = {**complete_params, "num_leaves": 31, "max_depth": -1}
    sel_off_err = run_diag(sel_code, "auto,device_graph_indirect:off", sel_params)
    sel_on_err = run_diag(sel_code, "auto,device_graph_indirect:on", sel_params)
    assert "graphs off" in sel_off_err
    assert "graphs off" not in sel_on_err


@_REQUIRES_CUDA
def test_device_graph_indirect_categorical_cuda():
    # Exercises DeviceLevelCatBitsetsIndirectKernel + captured graph replay on mixed categorical + numerical wide data.
    rng = np.random.default_rng(31)
    n, m = 5000, 300
    X = np.floor(rng.random((n, m)) * rng.integers(3, 7, size=m)).astype(np.float32)
    cat_cols = list(range(12))
    for c in cat_cols:
        X[:, c] = rng.integers(0, 4 + (c % 8), size=n).astype(np.float32)
    X[:, 12:180] = np.where(rng.random((n, 168)) < 0.05, np.nan, X[:, 12:180])
    cat_signal = (X[:, 0] == 1).astype(np.float32) - (X[:, 1] == 2).astype(np.float32)
    y = (cat_signal + 0.2 * np.nan_to_num(X[:, 12:24]).sum(axis=1) + rng.standard_normal(n)).astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": 15,
        "feature_fraction": 0.25,
        "min_data_in_leaf": 20,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 42,
    }
    ds_off = flc.Dataset(
        X, label=y, params={**params, "cuda_plan": "auto,device_graph_indirect:off"}, categorical_feature=cat_cols
    )
    ds_on = flc.Dataset(
        X, label=y, params={**params, "cuda_plan": "auto,device_graph_indirect:on"}, categorical_feature=cat_cols
    )
    b_off = flc.train({**params, "cuda_plan": "auto,device_graph_indirect:off"}, ds_off, num_boost_round=25)
    b_on = flc.train({**params, "cuda_plan": "auto,device_graph_indirect:on"}, ds_on, num_boost_round=25)
    md5_off = hashlib.md5(
        "\n".join(line for line in b_off.model_to_string().splitlines() if "cuda_plan" not in line).encode()
    ).hexdigest()
    md5_on = hashlib.md5(
        "\n".join(line for line in b_on.model_to_string().splitlines() if "cuda_plan" not in line).encode()
    ).hexdigest()
    assert md5_on == md5_off

    # Verify categorical device-tree graph replays cleanly via FALCATA_DEBUG=diag.
    code = (
        "import sys, json, numpy as np, falcata as flc\n"
        "rng = np.random.default_rng(31)\n"
        "n, m = 5000, 300\n"
        "X = np.floor(rng.random((n, m)) * rng.integers(3, 7, size=m)).astype(np.float32)\n"
        "cat_cols = list(range(12))\n"
        "for c in cat_cols:\n"
        "    X[:, c] = rng.integers(0, 4 + (c % 8), size=n).astype(np.float32)\n"
        "X[:, 12:180] = np.where(rng.random((n, 168)) < 0.05, np.nan, X[:, 12:180])\n"
        "cat_signal = (X[:, 0] == 1).astype(np.float32) - (X[:, 1] == 2).astype(np.float32)\n"
        "y = (cat_signal + 0.2 * np.nan_to_num(X[:, 12:24]).sum(axis=1) + rng.standard_normal(n)).astype(np.float32)\n"
        "p = dict(json.loads(sys.argv[1]), cuda_plan='auto,device_graph_indirect:on')\n"
        "ds = flc.Dataset(X, label=y, params=p, categorical_feature=cat_cols)\n"
        "flc.train(p, ds, num_boost_round=25)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, json.dumps(params)],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    matches = re.findall(r"complete depth=\d+.*graph=1 hits=(\d+) misses=(\d+)", out.stderr)
    assert matches
    hits, misses = map(int, matches[-1])
    assert hits >= 15, f"expected categorical graph hits, got hits={hits} misses={misses}"


@_REQUIRES_CUDA
@pytest.mark.parametrize("feature_fraction", [0.30, 0.314])
def test_device_graph_indirect_near_48kb_smem_cuda(feature_fraction):
    # 300 columns of 15-16 distinct values sampled at ff=0.30..0.314 (90-94 cols = 45-47 byte slots) places
    # pair-joint dynamic SMEM in (44 KB, 48 KB], guarding against rounding past the 48 KB static+dynamic limit.
    rng = np.random.default_rng(37)
    n, m = 4000, 300
    spans = rng.integers(15, 17, size=m)
    X = np.floor(rng.random((n, m)) * spans).astype(np.float32)
    y = (X[:, :20] @ rng.standard_normal(20) + rng.standard_normal(n)).astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": 16,
        "feature_fraction": feature_fraction,
        "min_data_in_leaf": 20,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 42,
    }
    _assert_same_model(params, X, y, rounds=20)


@_REQUIRES_CUDA
@pytest.mark.parametrize("find_split_waves", [0.05, 0.25, 1.0, 4.0])
def test_device_graph_indirect_find_split_small_boundary_cuda(find_split_waves):
    # Exercises warp-finder integer warp_split_max_blocks threshold across shallow/mid levels where
    # may_split straddles split=1 vs split=4 on variable-task-count wide NaN data.
    rng = np.random.default_rng(41)
    n, m = 4000, 200
    X = rng.standard_normal((n, m)).astype(np.float32)
    X[:, :80] = np.where(rng.random((n, 80)) < 0.1, np.nan, X[:, :80])
    y = (np.nan_to_num(X[:, :20]) @ rng.standard_normal(20) + rng.standard_normal(n)).astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "max_bin": 255,
        "feature_fraction": 0.35,
        "min_data_in_leaf": 20,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 42,
    }
    wave = f"find_split_waves:{find_split_waves}"
    off = _model_md5({**params, "cuda_plan": f"auto,{wave},device_graph_indirect:off"}, X, y, 20)
    on = _model_md5({**params, "cuda_plan": f"auto,{wave},device_graph_indirect:on"}, X, y, 20)
    assert on == off
