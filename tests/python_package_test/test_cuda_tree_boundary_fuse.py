# coding: utf-8
"""CUDA tree-boundary fusion keys must train bit-identical models with each key on or off.

Covers the four Lever 3 cuda_plan keys (all default ON):
  - robust_scale_async: async PDL-chained log-magnitude histogram + in-kernel reset (fixedpoint)
  - discretize_root_sums: fused Chunk256 discretization + root chunk sums + plane scale snapshot
  - const_hess_no_write: skip redundant hessian=1.0f writes on rounds >= 1 (unweighted L2/L1/Huber/Quantile)
  - grad_minmax_fuse: fuse per-1024-row min/max partials into GetGradients kernels

Verifies exact model MD5 identity for all four keys off vs all on, as well as each individual key
toggled alone, across L2 (stochastic, fixedpoint, nonquant deterministic, weighted, bagged, selective),
L1, Huber, Quantile, Binary (stochastic, imbalanced fixedpoint+robust, weighted), Multiclass Softmax
(fixedpoint, stochastic), and Multiclass OVA.
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

L3_KEYS = (
    "discretize_root_sums",
    "robust_scale_async",
    "const_hess_no_write",
    "grad_minmax_fuse",
)


def _model_md5(params, X, y, rounds, weight=None):
    ds = flc.Dataset(X, label=y, weight=weight, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


@pytest.fixture(scope="module")
def boundary_data():
    rng = np.random.default_rng(12345)
    # Non-multiple of 1024 to exercise partial tail blocks in Chunk256 kernels.
    n, f = 5000, 16
    X = np.ascontiguousarray(rng.standard_normal((n, f)), dtype=np.float32)
    y_reg = np.ascontiguousarray(
        X[:, 0] * 1.5 - X[:, 1] + 0.5 * X[:, 2] ** 2 + rng.normal(0, 0.5, n),
        dtype=np.float32,
    )
    y_bin = np.ascontiguousarray((y_reg > np.median(y_reg)).astype(np.float32))
    y_imb = np.ascontiguousarray((y_reg > np.quantile(y_reg, 0.99)).astype(np.float32))
    y_mc = np.ascontiguousarray(rng.integers(0, 7, size=n), dtype=np.float32)
    weight = np.ascontiguousarray(rng.uniform(0.5, 2.5, size=n), dtype=np.float32)
    return {
        "X": X,
        "reg": y_reg,
        "bin": y_bin,
        "imb": y_imb,
        "mc": y_mc,
        "weight": weight,
    }


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("case_name", "target_key", "extra", "weighted"),
    [
        ("l2_stoch", "reg", {"objective": "regression", "quant_mode": "stochastic"}, False),
        ("l2_fixed", "reg", {"objective": "regression", "quant_mode": "fixedpoint"}, False),
        ("l2_nonquant_det", "reg", {"objective": "regression", "quant_mode": "none", "deterministic": True}, False),
        ("l2_weighted_stoch", "reg", {"objective": "regression", "quant_mode": "stochastic"}, True),
        (
            "l2_bagged_fixed",
            "reg",
            {"objective": "regression", "quant_mode": "fixedpoint", "bagging_fraction": 0.7, "bagging_freq": 1},
            False,
        ),
        ("l1_fixed", "reg", {"objective": "regression_l1", "quant_mode": "fixedpoint"}, False),
        ("huber_stoch", "reg", {"objective": "huber", "quant_mode": "stochastic"}, False),
        ("quantile_fixed", "reg", {"objective": "quantile", "quant_mode": "fixedpoint"}, False),
        ("bin_stoch", "bin", {"objective": "binary", "quant_mode": "stochastic"}, False),
        ("bin_imb_fixed_robust", "imb", {"objective": "binary", "quant_mode": "fixedpoint"}, False),
        ("bin_weighted_fixed", "bin", {"objective": "binary", "quant_mode": "fixedpoint"}, True),
        ("mc_softmax_fixed", "mc", {"objective": "multiclass", "num_class": 7, "quant_mode": "fixedpoint"}, False),
        ("mc_softmax_stoch", "mc", {"objective": "multiclass", "num_class": 7, "quant_mode": "stochastic"}, False),
        ("mc_ova_fixed", "mc", {"objective": "multiclassova", "num_class": 7, "quant_mode": "fixedpoint"}, False),
        (
            "selective_l2_fixed",
            "reg",
            {"objective": "regression", "quant_mode": "fixedpoint", "max_depth": -1, "num_leaves": 31},
            False,
        ),
    ],
)
def test_tree_boundary_fuse_keys_bit_identical_cuda(boundary_data, case_name, target_key, extra, weighted):
    X = boundary_data["X"]
    y = boundary_data[target_key]
    weight = boundary_data["weight"] if weighted else None

    base_params = {
        "device_type": "cuda",
        "verbose": -1,
        "metric": "None",
        "seed": 42,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 5,
        **extra,
    }

    all_off = "auto," + ",".join(f"{k}:off" for k in L3_KEYS)
    reference = _model_md5({**base_params, "cuda_plan": all_off}, X, y, rounds=15, weight=weight)

    # All 4 keys ON (default auto)
    assert _model_md5({**base_params, "cuda_plan": "auto"}, X, y, rounds=15, weight=weight) == reference, (
        f"{case_name}: auto (all on) diverged from all_off"
    )

    # Each key ON individually (other 3 OFF)
    for key in L3_KEYS:
        others_off = ",".join(f"{k}:off" for k in L3_KEYS if k != key)
        plan = f"auto,{others_off}"
        assert _model_md5({**base_params, "cuda_plan": plan}, X, y, rounds=15, weight=weight) == reference, (
            f"{case_name}: only {key} on diverged from all_off"
        )
