# coding: utf-8
"""sync_used_narrow moves the level sync of narrow data (a task list that fits one sync block) to the used-task
kernel, which also mirrors the leaf entries into the readback staging (readback_fused). It picks the same winner, so
every model must be byte-identical with the key off. The shapes cover one feature, a few, two hundred features,
multiclass, bagging, feature sampling (already on the used-task kernel), a non-quantized training and a shape
with more tasks than one sync block (not affected)."""

import os
import re

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
    "num_leaves": 63,
    "max_depth": 7,
    "min_data_in_leaf": 5,
    "learning_rate": 0.1,
    "max_bin": 255,
    "quant_mode": "stochastic",
    "verbosity": -1,
    "seed": 5,
}

# (rows, columns, extra params)
SHAPES = {
    "one_feature": (20_001, 1, {"num_leaves": 15}),
    "few_features": (40_001, 5, {}),
    "two_hundred_features": (30_001, 200, {"max_bin": 63}),
    "multiclass": (30_001, 20, {"objective": "multiclass", "num_class": 4, "quant_mode": "fixedpoint"}),
    "bagged": (40_001, 12, {"bagging_fraction": 0.6, "bagging_freq": 1}),
    "feature_fraction": (40_001, 30, {"feature_fraction": 0.5}),
    "not_quantized": (30_001, 10, {"use_quantized_grad": False}),
    "one_block_limit": (2_001, 1_024, {"max_bin": 15, "num_leaves": 7, "min_data_in_leaf": 1}),
    "more_tasks_than_one_block": (3_001, 1_200, {"max_bin": 15, "num_leaves": 15, "min_data_in_leaf": 1}),
}


def _train(X, y, params, plan_extra):
    p = dict(params)
    plan = p.pop("cuda_plan", "")
    if plan_extra:
        plan = f"{plan},{plan_extra}" if plan else plan_extra
    if plan:
        p["cuda_plan"] = plan
    ds = flc.Dataset(X, label=y, params=p)
    model = flc.train(p, ds, num_boost_round=6).model_to_string()
    # the plan string itself is recorded with the parameters
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_sync_used_narrow_bit_identical(shape):
    rows, cols, extra = SHAPES[shape]
    rng = np.random.default_rng(23)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    signal = X[:, 0] + 0.5 * np.sin(3 * X[:, min(1, cols - 1)]) + 0.3 * rng.standard_normal(rows)
    y = signal.astype(np.float32)
    params = {**BASE, **extra}
    if params["objective"] == "multiclass":
        y = np.digitize(signal, np.quantile(signal, [0.25, 0.5, 0.75])).astype(np.float32)
    assert _train(X, y, params, "") == _train(X, y, params, "sync_used_narrow:off")
