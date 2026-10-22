# coding: utf-8
"""CUDA objective gradients must train the same model with or without their fast paths.

cuda_plan keys gradient_fp64_lean (binary logloss, weighted or not: a reduced-FP64 evaluation whose floats are stored only where every
double within its error bound rounds to the same float, else the original double expression) and softmax_registers
(multiclass: the row's class scores in registers instead of a global scratch) change only how the same stored floats
are computed, so the trained model must be byte-identical with each key off. The sweep covers sigmoid values that
move many rows into the lean path's fallbacks (large |x|, the hessian's cancellation), un-quantized and quantized
gradients, sample weights and is_unbalance (the weighted lean variants), and class counts on every register capacity and
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
    ("objective", "extra", "quantized", "weighted"),
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
        ("multiclass", {"num_class": 40}, True, False),
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


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "extra",
    [
        {"use_quantized_grad": True, "bagging_fraction": 0.6, "bagging_freq": 1},
        {"use_quantized_grad": False, "cuda_plan": "auto,device_tree:off"},
        {"use_quantized_grad": True, "objective": "multiclassova", "num_class": 4},
    ],
)
def test_wide_weight_binary_training_is_deterministic_cuda(extra):
    """Sample weights over 2^58 (exp(U(-20, 20)) as float32) do not sum exactly in double, so the binary init score's
    label and weight sums depend on the order the per-block partials are added. That order is now the block order
    (it was the order of the blocks' atomicAdds, which made three trainings give three models): the same data must
    train the same model every time, quantized and bagged, non-quantized, and one-vs-all (the binary init per class)."""
    rows = 100_000
    rng = np.random.default_rng(1)
    X = rng.standard_normal((rows, 20)).astype(np.float32)
    signal = X[:, :8] @ rng.standard_normal(8)
    weight = np.exp(rng.uniform(-20, 20, size=rows)).astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": "binary",
        "is_unbalance": True,
        "sigmoid": 1.7,
        "num_leaves": 63,
        "max_depth": 7,
        "min_data_in_leaf": 5,
        "learning_rate": 0.2,
        "verbosity": -1,
        "seed": 3,
        **extra,
    }
    if params["objective"] == "multiclassova":
        params.pop("is_unbalance")
        y = np.digitize(signal, np.quantile(signal, [0.25, 0.5, 0.75])).astype(float)
    else:
        y = (signal + rng.standard_normal(rows) > np.median(signal)).astype(float)
    md5s = {_model_md5(params, X, y, rounds=20, weight=weight) for _ in range(3)}
    assert len(md5s) == 1, md5s
