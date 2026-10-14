# coding: utf-8
"""CUDA objective gradients must train the same model with or without their fast paths.

cuda_plan keys gradient_fp64_lean (binary logloss: a reduced-FP64 evaluation whose floats are stored only where every
double within its error bound rounds to the same float, else the original double expression) and softmax_registers
(multiclass: the row's class scores in registers instead of a global scratch) change only how the same stored floats
are computed, so the trained model must be byte-identical with each key off. The sweep covers sigmoid values that
move many rows into the lean path's fallbacks (large |x|, the hessian's cancellation), un-quantized and quantized
gradients, sample weights and is_unbalance (the original kernels), and class counts on both register capacities and
above them (the scratch kernel).
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

KEYS = ("gradient_fp64_lean", "softmax_registers")


def _model_md5(params, X, y, rounds, weight=None):
    ds = flc.Dataset(X, label=y, weight=weight, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "objective, extra, quantized, weighted",
    [
        ("binary", {}, True, False),
        ("binary", {}, False, False),
        ("binary", {"sigmoid": 0.3}, True, False),
        ("binary", {"sigmoid": 4.0}, False, False),
        ("binary", {"is_unbalance": True}, True, False),
        ("binary", {}, True, True),
        ("multiclass", {"num_class": 3}, True, False),
        ("multiclass", {"num_class": 7}, False, True),
        ("multiclass", {"num_class": 12}, True, False),
        ("multiclass", {"num_class": 20}, True, False),
    ],
)
def test_gradient_keys_are_bit_identical_cuda(objective, extra, quantized, weighted):
    rows, cols = 30011, 12
    rng = np.random.default_rng(5)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.3 * rng.standard_normal(rows)
    if objective == "multiclass":
        k = extra["num_class"]
        y = np.digitize(signal, np.quantile(signal, np.linspace(0, 1, k + 1)[1:-1])).astype(float)
    else:
        y = (signal > 0.4).astype(float)
    weight = rng.uniform(0.5, 2.0, size=rows) if weighted else None
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 63,
        "max_depth": 7,
        "min_data_in_leaf": 5,
        "learning_rate": 0.5,
        "use_quantized_grad": quantized,
        "verbosity": -1,
        "seed": 7,
        **extra,
    }
    reference = _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=30, weight=weight)
    for key in KEYS:
        assert _model_md5({**params, "cuda_plan": f"auto,{key}:off"}, X, y, rounds=30, weight=weight) == reference, key
