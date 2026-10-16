# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key pdl_level_chain.

The key launches the level chain's consumer kernels (level sync, offset aggregate, split-inner partition, quantized
fix + subtract) as programmatic dependents of the kernel before them on the same stream; each consumer waits for
its producer's completion and memory before touching memory, so the model is byte-identical with the key off. The
sweep covers quantized training (stochastic, fixed point), feature sampling (the used-feature fix
+ subtract and the used-task sync), bagging, missing values, a leaf budget that binds before max_depth, and the
multiclass objective (several trees per round).
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
    ("objective", "leaves", "depth", "quant", "feature_fraction", "bagging"),
    [
        ("regression", 1023, 10, "stochastic", 1.0, 1.0),
        ("binary", 255, 10, "fixedpoint", 0.5, 1.0),
        ("binary", 511, 9, "stochastic", 1.0, 0.6),
        ("multiclass", 127, 7, "fixedpoint", 1.0, 1.0),
        ("regression", 255, 8, "stochastic", 0.7, 0.5),
    ],
)
def test_pdl_level_chain_is_bit_identical_cuda(objective, leaves, depth, quant, feature_fraction, bagging):
    rows = 30011
    rng = np.random.default_rng(13)
    X = rng.standard_normal((rows, 24)).astype(np.float32)
    X[:, 20:] = rng.integers(0, 5, size=(rows, 4))
    X[rng.random(X.shape) < 0.03] = np.nan
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
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "feature_fraction": feature_fraction,
        "verbose": -1,
        "quant_mode": quant,
        "seed": 5,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 12)
    off = _model_md5({**params, "cuda_plan": "auto,pdl_level_chain:off"}, X, y, 12)
    assert on == off
