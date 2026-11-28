# coding: utf-8
"""CUDA tree-boundary fusion keys must train bit-identical models with each key on or off.

Covers the four Lever 3 cuda_plan keys (all default ON):
  - robust_scale_async: async PDL-chained log-magnitude histogram + in-kernel reset (fixedpoint)
  - discretize_root_sums: fused Chunk256 discretization + root chunk sums + plane scale snapshot
  - const_hess_no_write: skip redundant hessian=1.0f writes on rounds >= 1 (unweighted L2/L1/Huber/Quantile)
  - grad_minmax_fuse: fuse per-1024-row min/max partials into GetGradients kernels

Verifies exact model MD5 identity for all four keys off vs all on, as well as each individual key
toggled alone, across objectives, boundary row counts (512/1023/1024/1025), outlier-anchored
robust_scale triggers, and booster lifecycles (init_model continuation, refit, skipped multiclass
classes, pdl_level_chain:off, async_tree_start:off).
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

ALL_OFF_SUFFIX = ",".join(f"{k}:off" for k in L3_KEYS)


def _strip_plan_md5(model_str: str) -> str:
    lines = [line for line in model_str.splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _model_md5(params, X, y, rounds, weight=None):
    ds = flc.Dataset(X, label=y, weight=weight, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    return _strip_plan_md5(booster.model_to_string())


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

    all_off = f"auto,{ALL_OFF_SUFFIX}"
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


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_rows", [512, 1023, 1024, 1025])
@pytest.mark.parametrize(
    ("objective", "quant_mode", "extra"),
    [
        ("regression", "stochastic", {}),
        ("regression", "fixedpoint", {}),
        ("binary", "stochastic", {}),
        ("multiclass", "fixedpoint", {"num_class": 4}),
    ],
)
def test_tree_boundary_fuse_chunk_boundaries_cuda(num_rows, objective, quant_mode, extra):
    """Exact 1024-row chunk boundary sizes (512, 1023, 1024, 1025) must stay bit-identical."""
    rng = np.random.default_rng(2026 + num_rows)
    X = np.ascontiguousarray(rng.standard_normal((num_rows, 12)), dtype=np.float32)
    signal = X[:, 0] - 0.8 * X[:, 1] + 0.3 * rng.standard_normal(num_rows)
    if objective == "multiclass":
        k = extra["num_class"]
        y = np.ascontiguousarray(
            np.digitize(signal, np.quantile(signal, np.linspace(0, 1, k + 1)[1:-1])),
            dtype=np.float32,
        )
    elif objective == "binary":
        y = np.ascontiguousarray((signal > 0.0).astype(np.float32))
    else:
        y = np.ascontiguousarray(signal, dtype=np.float32)

    params = {
        "device_type": "cuda",
        "objective": objective,
        "quant_mode": quant_mode,
        "verbose": -1,
        "metric": "None",
        "seed": 17,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "max_depth": 4,
        **extra,
    }
    ref = _model_md5({**params, "cuda_plan": f"auto,{ALL_OFF_SUFFIX}"}, X, y, rounds=12)
    got = _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=12)
    assert got == ref, f"rows={num_rows} obj={objective} mode={quant_mode}"


@_REQUIRES_CUDA
def test_robust_scale_async_reanchor_outlier_cluster_cuda():
    """Verify robust_scale actually re-anchors on a wide outlier gap AND robust_scale_async matches sync bit-for-bit."""
    rng = np.random.default_rng(999)
    n, f = 5000, 12
    X = np.ascontiguousarray(rng.standard_normal((n, f)), dtype=np.float32)
    y = np.ascontiguousarray(X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.2, n), dtype=np.float32)
    # 10 rows (0.2% << 5% max_outlier_fraction) separated by >10 octaves from bulk -> triggers gap re-anchoring
    y[:10] = 1.0e4

    params = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "fixedpoint",
        "verbose": -1,
        "metric": "None",
        "seed": 42,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 5,
    }
    md5_no_robust = _model_md5({**params, "cuda_plan": "auto,robust_scale:off"}, X, y, rounds=15)
    md5_sync_robust = _model_md5({**params, "cuda_plan": f"auto,{ALL_OFF_SUFFIX}"}, X, y, rounds=15)
    md5_async_only = _model_md5(
        {
            **params,
            "cuda_plan": "auto,discretize_root_sums:off,const_hess_no_write:off,grad_minmax_fuse:off",
        },
        X,
        y,
        rounds=15,
    )
    md5_all_on = _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=15)

    # Confirm the fixture genuinely engaged robust_scale re-anchoring
    assert md5_sync_robust != md5_no_robust, "outlier fixture did not trigger robust_scale re-anchoring"
    assert md5_async_only == md5_sync_robust
    assert md5_all_on == md5_sync_robust


@_REQUIRES_CUDA
@pytest.mark.parametrize("extra_plan", ["pdl_level_chain:off", "async_tree_start:off"])
def test_tree_boundary_fuse_with_plan_toggles_cuda(boundary_data, extra_plan):
    """L3 keys stay bit-identical when composed with pdl_level_chain:off or async_tree_start:off."""
    X = boundary_data["X"]
    y = boundary_data["reg"]
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "fixedpoint",
        "verbose": -1,
        "metric": "None",
        "seed": 42,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 5,
    }
    ref = _model_md5({**params, "cuda_plan": f"auto,{extra_plan},{ALL_OFF_SUFFIX}"}, X, y, rounds=12)
    got = _model_md5({**params, "cuda_plan": f"auto,{extra_plan}"}, X, y, rounds=12)
    assert got == ref, extra_plan


@_REQUIRES_CUDA
def test_tree_boundary_fuse_lifecycle_init_model_and_refit_cuda(boundary_data):
    """Continued training (init_model) and refit stay bit-identical with L3 keys on vs off."""
    X = boundary_data["X"]
    y = boundary_data["reg"]
    rng = np.random.default_rng(77)
    y_refit = np.ascontiguousarray(y + rng.normal(0, 0.3, len(y)), dtype=np.float32)

    base_params = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "fixedpoint",
        "verbose": -1,
        "metric": "None",
        "seed": 42,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 5,
    }

    def run_lifecycle(plan: str):
        p = {**base_params, "cuda_plan": plan}
        ds = flc.Dataset(X, label=y, params=p, free_raw_data=False)
        b1 = flc.train(p, ds, num_boost_round=8)
        b2 = flc.train(p, ds, num_boost_round=8, init_model=b1)
        md5_cont = _strip_plan_md5(b2.model_to_string())
        b_refit = b2.refit(X, y_refit)
        md5_refit = _strip_plan_md5(b_refit.model_to_string())
        return md5_cont, md5_refit

    ref_cont, ref_refit = run_lifecycle(f"auto,{ALL_OFF_SUFFIX}")
    got_cont, got_refit = run_lifecycle("auto")
    assert got_cont == ref_cont
    assert got_refit == ref_refit


@_REQUIRES_CUDA
def test_tree_boundary_fuse_multiclass_skipped_class_cuda(boundary_data):
    """Multiclass with an absent class (class_need_train_ == false) stays bit-identical."""
    X = boundary_data["X"]
    rng = np.random.default_rng(55)
    # Classes {0, 1, 3, 4} present, class 2 completely absent -> class_need_train_[2] == False
    present = np.array([0.0, 1.0, 3.0, 4.0], dtype=np.float32)
    y = np.ascontiguousarray(rng.choice(present, size=len(X)))
    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": 5,
        "quant_mode": "fixedpoint",
        "verbose": -1,
        "metric": "None",
        "seed": 42,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "max_depth": 4,
    }
    ref = _model_md5({**params, "cuda_plan": f"auto,{ALL_OFF_SUFFIX}"}, X, y, rounds=10)
    got = _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=10)
    assert got == ref
