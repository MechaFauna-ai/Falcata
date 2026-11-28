# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key partition_pack_gh.

The key maintains a parallel ping-pong partition-ordered quantized (grad, hess) buffer in lockstep
with cuda_data_indices_ during the batched level apply, so level >= 1 quantized histogram constructs
read packed grad/hess sequentially alongside data_indices instead of randomly gathering
cuda_gradients_and_hessians[idx]. Explicit partition_pack_gh:on forces the path at test scale;
the resulting model must be byte-identical to partition_pack_gh:off across device_tree, host-level,
bagging + fixedpoint, selective (max_depth=-1), 4-bit pair-joint, and multiclass training.
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
        {"cols": 32, "max_bin": 255, "objective": "multiclass", "leaves": 63, "depth": 6, "quant": "fixedpoint"},
        id="multiclass_fixedpoint",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", _CASES)
def test_partition_pack_gh_is_bit_identical_cuda(case):
    rows = 12007
    cols = case["cols"]
    max_bin = case["max_bin"]
    objective = case["objective"]
    rng = np.random.default_rng(11)
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
        "min_data_in_leaf": 10,
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
