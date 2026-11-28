# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key partition_pack_gh.

The key maintains a parallel ping-pong partition-ordered quantized (grad, hess) buffer in lockstep
with cuda_data_indices_ during the batched level apply, so level >= 1 quantized histogram constructs
read packed grad/hess sequentially alongside data_indices instead of randomly gathering
cuda_gradients_and_hessians[idx]. Explicit partition_pack_gh:on forces the path at test scale;
the resulting model must be byte-identical to partition_pack_gh:off across device_tree, host-level,
bagging + fixedpoint, selective (max_depth=-1), gap_copy_once:off, final_map_only:off, small leaves
(rows < 1024), 4-bit pair-joint, 5-bin code5, and multiclass training.
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
    ds = flc.Dataset(X, label=y, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


_CASES = [
    pytest.param(
        {"cols": 32, "max_bin": 255, "objective": "binary", "leaves": 63, "depth": 6, "quant": "stochastic"},
        id="device_tree",
    ),
    pytest.param(
        {
            "cols": 32,
            "max_bin": 255,
            "objective": "binary",
            "leaves": 63,
            "depth": 6,
            "quant": "stochastic",
            "plan_prefix": "auto,device_tree:off,device_levels:off",
        },
        id="host_level",
    ),
    pytest.param(
        {
            "cols": 32,
            "max_bin": 255,
            "objective": "binary",
            "leaves": 63,
            "depth": 6,
            "bagging": 0.7,
            "quant": "fixedpoint",
        },
        id="bagged_fixedpoint",
    ),
    pytest.param(
        {"cols": 32, "max_bin": 255, "objective": "binary", "leaves": 31, "depth": -1, "quant": "stochastic"},
        id="selective_device_tree",
    ),
    pytest.param(
        {
            "cols": 32,
            "max_bin": 255,
            "objective": "binary",
            "leaves": 31,
            "depth": -1,
            "quant": "stochastic",
            "plan_prefix": "auto,device_tree:off",
        },
        id="selective_host",
    ),
    pytest.param(
        {
            "cols": 32,
            "max_bin": 255,
            "objective": "binary",
            "leaves": 31,
            "depth": -1,
            "quant": "stochastic",
            "plan_prefix": "auto,gap_copy_once:off",
        },
        id="gap_copy_once_off",
    ),
    pytest.param(
        {
            "cols": 32,
            "max_bin": 255,
            "objective": "binary",
            "leaves": 63,
            "depth": 6,
            "quant": "stochastic",
            "plan_prefix": "auto,final_map_only:off",
        },
        id="final_map_only_off",
    ),
    pytest.param(
        {
            "rows": 512,
            "cols": 32,
            "max_bin": 255,
            "objective": "binary",
            "leaves": 31,
            "depth": 6,
            "min_data_in_leaf": 5,
            "quant": "stochastic",
        },
        id="small_rows_512",
    ),
    pytest.param(
        {
            "cols": 64,
            "max_bin": 15,
            "objective": "regression",
            "leaves": 63,
            "depth": 6,
            "feature_fraction": 0.5,
            "quant": "stochastic",
        },
        id="pair_joint_4bit",
    ),
    pytest.param(
        {
            "cols": 64,
            "max_bin": 5,
            "objective": "regression",
            "leaves": 63,
            "depth": 6,
            "feature_fraction": 0.5,
            "quant": "stochastic",
        },
        id="code5_5bin",
    ),
    pytest.param(
        {"cols": 32, "max_bin": 255, "objective": "multiclass", "leaves": 63, "depth": 6, "quant": "fixedpoint"},
        id="multiclass_fixedpoint",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", _CASES)
def test_partition_pack_gh_is_bit_identical_cuda(case):
    rows = case.get("rows", 12007)
    cols = case["cols"]
    max_bin = case["max_bin"]
    objective = case["objective"]
    rng = np.random.default_rng(11)
    if max_bin <= 5:
        X = rng.integers(0, max_bin, size=(rows, cols)).astype(np.float32)
    else:
        X = rng.standard_normal((rows, cols)).astype(np.float32)
        if max_bin <= 15:
            X = np.floor(X * 2.0).astype(np.float32)
        X[rng.random(X.shape) < 0.02] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > 0).astype(np.float32)
    else:
        y = z.astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": case["leaves"],
        "max_depth": case["depth"],
        "min_data_in_leaf": case.get("min_data_in_leaf", 10),
        "learning_rate": 0.1,
        "max_bin": max_bin,
        "feature_fraction": case.get("feature_fraction", 1.0),
        "quant_mode": case["quant"],
        "verbose": -1,
        "seed": 42,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    bagging = case.get("bagging", 1.0)
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    plan_prefix = case.get("plan_prefix", "auto")
    on = _model_md5({**params, "cuda_plan": f"{plan_prefix},partition_pack_gh:on"}, X, y, 12)
    off = _model_md5({**params, "cuda_plan": f"{plan_prefix},partition_pack_gh:off"}, X, y, 12)
    assert on == off


_DIAG_CODE = (
    "import sys, json, numpy as np, falcata as flc\n"
    "rng = np.random.default_rng(1)\n"
    "X = rng.standard_normal((20000, 10)).astype(np.float32)\n"
    "y = (X[:, 0] + X[:, 1] * X[:, 2] + 0.3 * rng.standard_normal(20000)).astype(np.float32)\n"
    "p = dict(json.loads(sys.argv[2]), cuda_plan=sys.argv[1])\n"
    "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=12)\n"
)


@_REQUIRES_CUDA
def test_partition_pack_gh_diag_activation_and_graph_parity_cuda():
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 32,
        "max_depth": 5,  # odd depth: verifies index_parity_canonical still yields 1 graph capture with pack_gh:on
        "min_data_in_leaf": 5,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "verbose": -1,
        "seed": 3,
    }
    on = subprocess.run(
        [sys.executable, "-c", _DIAG_CODE, "auto,partition_pack_gh:on", json.dumps(params)],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    off = subprocess.run(
        [sys.executable, "-c", _DIAG_CODE, "auto,partition_pack_gh:off", json.dumps(params)],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[hybrid-diag] partition_pack_gh: active" in on.stderr, on.stderr[-2000:]
    assert "[hybrid-diag] partition_pack_gh: active" not in off.stderr, off.stderr[-2000:]
    found_on = re.findall(r"device_tree: complete .* hits=(\d+) misses=(\d+)", on.stderr)
    assert len(found_on) == 12, on.stderr[-2000:]
    assert int(found_on[-1][1]) == 1
