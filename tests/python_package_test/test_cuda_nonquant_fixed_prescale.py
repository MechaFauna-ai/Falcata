# coding: utf-8
"""Bit-identity regression tests for cuda_plan key nonquant_fixed_prescale.

Verifies exact model MD5 parity between nonquant_fixed_prescale:on (default) and
nonquant_fixed_prescale:off (baseline per-row scalbnf + emulated carry atomics)
across dense 8-bit device_tree, 4-bit USE_REG_BINS (numerai-like), full-view
4-bit, 16-bit bins (max_bin=1023), carry/sign stress, bagging + feature_fraction,
multiclass, host level loop (device_tree:off), selective (max_depth=-1),
FALCATA_TEST_FORCE_NONQUANT_64K=1 fallback, and nonquant_hist_cost auto-selection.
"""

import hashlib
import json
import os
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
    text = booster.model_to_string().split("\nparameters:")[0]
    lines = [line for line in text.splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _make_data(objective, rows, cols, seed=17, kind="dense"):
    rng = np.random.default_rng(seed)
    if kind == "discrete_5bin":
        X = rng.integers(0, 5, size=(rows, cols), dtype=np.int8).astype(np.float32)
        z = 0.4 * (X[:, 0] - 2.0) - 0.3 * (X[:, 1] - 2.0) + 0.2 * (X[:, 2] - 2.0) * (X[:, 3] - 2.0)
        z = z + 0.5 * rng.standard_normal(rows)
    elif kind == "carry_stress":
        # Low-cardinality bins + heavy-tailed both-sign targets to force repeated lo-word wraps
        # and negative two's-complement carries under shared-atomic contention.
        X = rng.integers(0, 7, size=(rows, cols), dtype=np.int8).astype(np.float32)
        sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=rows)
        mag = np.where(rng.random(rows) < 0.05, 1e4, 1.0)
        z = sign * mag * (0.5 * (X[:, 0] - 3.0) - 0.4 * (X[:, 1] - 3.0) + rng.standard_normal(rows))
    else:
        X = rng.standard_normal((rows, cols)).astype(np.float32)
        X[rng.random(X.shape) < 0.03] = np.nan
        Xn = np.nan_to_num(X)
        z = Xn[:, 0] + 0.5 * Xn[:, 1] * Xn[:, 2] - 0.3 * Xn[:, 3] + 0.3 * rng.standard_normal(rows)
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > np.median(z)).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


def _subprocess_md5_and_diag(params, shape, rounds):
    """Run one training in a fresh process with FALCATA_DEBUG=diag; return (md5, stderr)."""
    code = (
        "import json, runpy, sys; "
        "mod = runpy.run_path(sys.argv[1]); "
        "p = json.loads(sys.argv[2]); "
        "rows, cols, kind = json.loads(sys.argv[3]); "
        "X, y = mod['_make_data'](p['objective'], rows, cols, seed=42, kind=kind); "
        "print(mod['_model_md5'](p, X, y, int(sys.argv[4])))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, json.dumps(params), json.dumps(list(shape)), str(rounds)],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()[-1], result.stderr


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("name", "objective", "shape", "base_plan", "extra"),
    [
        (
            "dense-8bit-device-tree",
            "regression",
            (40000, 28, "dense"),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": 6, "max_bin": 255},
        ),
        (
            "4bit-reg-bins-numerai",
            "regression",
            (40000, 120, "discrete_5bin"),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 127, "max_depth": 7, "max_bin": 15, "feature_fraction": 0.3},
        ),
        (
            "4bit-full-view",
            "binary",
            (40000, 20, "dense"),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": 6, "max_bin": 15},
        ),
        (
            "uint16-bins-max-bin-1023",
            "regression",
            (30000, 12, "dense"),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": 6, "max_bin": 1023},
        ),
        (
            "carry-and-sign-stress",
            "regression",
            (50000, 24, "carry_stress"),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": 6, "max_bin": 15},
        ),
        (
            "bagging-feature-fraction",
            "binary",
            (40000, 80, "dense"),
            "auto,nonquant_deterministic:on",
            {
                "num_leaves": 127,
                "max_depth": 7,
                "max_bin": 255,
                "bagging_fraction": 0.7,
                "bagging_freq": 1,
                "feature_fraction": 0.4,
            },
        ),
        (
            "multiclass-device-tree",
            "multiclass",
            (30000, 24, "dense"),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": 6, "num_class": 4, "max_bin": 255},
        ),
        (
            "host-level-device-tree-off",
            "regression",
            (30000, 100, "dense"),
            "auto,device_tree:off,nonquant_deterministic:on",
            {"num_leaves": 127, "max_depth": 7, "max_bin": 255, "feature_fraction": 0.4},
        ),
        (
            "selective-unbounded-depth",
            "regression",
            (30000, 30, "dense"),
            "auto,nonquant_deterministic:on",
            {"num_leaves": 63, "max_depth": -1, "max_bin": 255},
        ),
    ],
)
def test_nonquant_fixed_prescale_matches_baseline_cuda(name, objective, shape, base_plan, extra):
    rows, cols, kind = shape
    X, y = _make_data(objective, rows, cols, seed=42, kind=kind)
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
    on_md5 = _model_md5({**params, "cuda_plan": f"{base_plan},nonquant_fixed_prescale:on"}, X, y, 12)
    off_md5 = _model_md5({**params, "cuda_plan": f"{base_plan},nonquant_fixed_prescale:off"}, X, y, 12)
    assert on_md5 == off_md5, f"{name}: prescale:on ({on_md5}) != prescale:off ({off_md5})"


@_REQUIRES_CUDA
def test_nonquant_fixed_prescale_diag_engagement_cuda():
    # Confirm via FALCATA_DEBUG=diag that prescale:on actually activates the prescaled path and :off does not.
    base = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "none",
        "num_leaves": 63,
        "max_depth": 6,
        "max_bin": 255,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "seed": 7,
    }
    on_md5, on_diag = _subprocess_md5_and_diag(
        {**base, "cuda_plan": "auto,nonquant_deterministic:on,nonquant_fixed_prescale:on"},
        (30000, 24, "dense"),
        8,
    )
    off_md5, off_diag = _subprocess_md5_and_diag(
        {**base, "cuda_plan": "auto,nonquant_deterministic:on,nonquant_fixed_prescale:off"},
        (30000, 24, "dense"),
        8,
    )
    assert on_md5 == off_md5, (on_md5, off_md5)
    assert "[hybrid-diag] nonquant_fixed_prescale: active" in on_diag, on_diag[-2000:]
    assert "[hybrid-diag] nonquant_fixed_prescale: active" not in off_diag, off_diag[-2000:]


@_REQUIRES_CUDA
def test_nonquant_fixed_prescale_forced_64k_fallback_cuda(monkeypatch, capfd):
    # With 64 KiB shared-memory fault injection, fixed-point construct is unsupported and falls back to classic det.
    monkeypatch.setenv("FALCATA_TEST_FORCE_NONQUANT_64K", "1")
    X, y = _make_data("regression", 12000, 40, seed=23, kind="dense")
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "none",
        "num_leaves": 255,
        "max_depth": 8,
        "max_bin": 255,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.5,
        "verbose": 1,
        "seed": 7,
        "num_threads": 4,
    }
    on_md5 = _model_md5({**params, "cuda_plan": "auto,nonquant_deterministic:on,nonquant_fixed_prescale:on"}, X, y, 10)
    off_md5 = _model_md5(
        {**params, "cuda_plan": "auto,nonquant_deterministic:on,nonquant_fixed_prescale:off"}, X, y, 10
    )
    ref_md5 = _model_md5({**params, "cuda_plan": "auto,hybrid:off,nonquant_deterministic:on"}, X, y, 10)
    assert on_md5 == off_md5 == ref_md5, (on_md5, off_md5, ref_md5)
    log = capfd.readouterr()
    assert "TEST ONLY: nonquant shared limit 65536 bytes, classic fallback=1" in log.out + log.err


@_REQUIRES_CUDA
def test_nonquant_fixed_prescale_graph_and_hist_cost_cuda():
    # Shape where nonquant_hist_cost auto-selects fixed-point without explicit nonquant_deterministic:on
    X, y = _make_data("regression", 80000, 12, seed=31)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "none",
        "num_leaves": 63,
        "max_depth": 6,
        "max_bin": 15,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 7,
    }
    auto_on = _model_md5({**params, "cuda_plan": "auto,nonquant_fixed_prescale:on"}, X, y, 10)
    auto_off = _model_md5({**params, "cuda_plan": "auto,nonquant_fixed_prescale:off"}, X, y, 10)
    no_graph = _model_md5({**params, "cuda_plan": "auto,device_level_graph:off,nonquant_fixed_prescale:on"}, X, y, 10)
    assert auto_on == auto_off == no_graph, (auto_on, auto_off, no_graph)
