# coding: utf-8
"""Non-quantized CUDA training (quant_mode=none) trains the same model on every run (cuda_plan key
nonquant_deterministic, default on).

The batched level construct accumulates in 64-bit fixed point there (integer atomics, order-free) where the float
atomics used to run (the graph level loop, compact column views, levels wider than the deterministic slot slab); the
per-leaf and the host batched deterministic constructs keep running where they ran. The sweep trains each shape
three times in one process and requires one model: dense depth-limited (graph loop) regression, binary and
multiclass, feature sampling (compact view), bagging, sparse input, a leaf-wise tree (per-leaf construct),
cuda_precision=fp32, a wide shape on the host level loop whose deep levels outgrow the deterministic slab, and the
requests deterministic=true keeps on the non-quantized path (monotone and interaction constraints). The
key must not touch quantized training: key on and off train the same quantized model.
"""

import hashlib
import os

import numpy as np
import pytest
import scipy.sparse as sp

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _model_md5(params, X, y, rounds):
    ds = flc.Dataset(X, label=y, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    text = booster.model_to_string().split("\nparameters:")[0]
    lines = [line for line in text.splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(objective, rows, cols, seed, sparse=False):
    rng = np.random.default_rng(seed)
    if sparse:
        X = sp.random(rows, cols, density=0.05, format="csr", random_state=seed, dtype=np.float64)
        dense = X[:, :4].toarray()
        z = dense[:, 0] - dense[:, 1] + 0.5 * dense[:, 2] * dense[:, 3] + 0.05 * rng.normal(size=rows)
    else:
        X = rng.standard_normal((rows, cols)).astype(np.float32)
        X[rng.random(X.shape) < 0.03] = np.nan
        Xn = np.nan_to_num(X)
        z = Xn[:, 0] + 0.5 * Xn[:, 1] * Xn[:, 2] - 0.3 * Xn[:, 3] + 0.3 * rng.normal(size=rows)
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > np.median(z)).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("name", "objective", "rows", "cols", "extra"),
    [
        ("graph-regression", "regression", 40000, 20, {"num_leaves": 63, "max_depth": 6}),
        ("graph-binary-deep", "binary", 40000, 20, {"num_leaves": 1023, "max_depth": 10}),
        ("graph-multiclass", "multiclass", 30000, 20, {"num_leaves": 63, "max_depth": 6, "num_class": 4}),
        ("compact-view", "regression", 30000, 200, {"num_leaves": 255, "max_depth": 8, "feature_fraction": 0.3}),
        (
            "bagged",
            "binary",
            40000,
            20,
            {"num_leaves": 255, "max_depth": 8, "bagging_fraction": 0.7, "bagging_freq": 1},
        ),
        ("sparse", "regression", 30000, 60, {"num_leaves": 63, "max_depth": 6}),
        ("leaf-wise", "regression", 30000, 20, {"num_leaves": 63, "max_depth": -1}),
        ("fp32", "binary", 30000, 20, {"num_leaves": 255, "max_depth": 8, "cuda_precision": "fp32"}),
        (
            "wide-host-loop",
            "regression",
            12000,
            1500,
            {"num_leaves": 255, "max_depth": 8, "cuda_plan": "auto,graph_loop:off"},
        ),
        (
            "monotone",
            "regression",
            30000,
            20,
            {"num_leaves": 63, "max_depth": 6, "monotone_constraints": [1, -1] + [0] * 18},
        ),
        (
            "interaction",
            "binary",
            30000,
            20,
            {"num_leaves": 63, "max_depth": 6, "interaction_constraints": [[0, 1, 2], [3, 4, 5, 6, 7]]},
        ),
    ],
)
def test_nonquant_training_is_run_to_run_deterministic_cuda(name, objective, rows, cols, extra):
    X, y = _data(objective, rows, cols, seed=17, sparse=(name == "sparse"))
    params = {
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": "none",
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "seed": 7,
        **extra,
    }
    md5s = {_model_md5(params, X, y, 12) for _ in range(3)}
    assert len(md5s) == 1, f"{name}: {len(md5s)} different models in 3 trainings"


@_REQUIRES_CUDA
def test_nonquant_deterministic_off_still_trains_cuda():
    X, y = _data("regression", 20000, 20, seed=5)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "none",
        "num_leaves": 63,
        "max_depth": 6,
        "verbose": -1,
        "seed": 7,
        "cuda_plan": "auto,nonquant_deterministic:off",
    }
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=10)
    pred = booster.predict(X)
    assert np.all(np.isfinite(pred))
    assert np.corrcoef(pred, y)[0, 1] > 0.5


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "quant", "extra"),
    [
        ("regression", "stochastic", {"num_leaves": 63, "max_depth": 6}),
        ("binary", "fixedpoint", {"num_leaves": 255, "max_depth": 8, "feature_fraction": 0.5}),
        (
            "multiclass",
            "stochastic",
            {"num_leaves": 31, "max_depth": 5, "num_class": 4, "bagging_fraction": 0.7, "bagging_freq": 1},
        ),
    ],
)
def test_nonquant_deterministic_key_does_not_touch_quantized_cuda(objective, quant, extra):
    X, y = _data(objective, 30000, 20, seed=11)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": quant,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 7,
        **extra,
    }
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 12)
    off = _model_md5({**params, "cuda_plan": "auto,nonquant_deterministic:off"}, X, y, 12)
    assert on == off
