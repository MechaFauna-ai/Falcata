# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key dense_vec_narrow.

The key pads single-partition narrow 8-bit quantized dense row data (<= 32 columns, <= 12288 bins, not 4-bit packed,
feature_fraction >= 1.0) to a power-of-two 4-byte multiple (4, 8, 16, or 32 bytes) and runs a warp-cooperative
sub-warp construct kernel (TX x {32,64} threads with TX = 1/2/4/8 threads per row, aligned uint32 loads, dynamically
sized shared histogram). Integer sums over the same rows are order-invariant, so the model is byte-identical with
the key off.
"""

import hashlib
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
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _make_dataset(rows, cols, objective, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.02] = np.nan
    z = (
        np.nan_to_num(X[:, 0])
        + (0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, -1]) if cols > 1 else 0.0)
        + 0.3 * rng.normal(size=rows)
    )
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > 0).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "case",
    [
        (1, "binary", 63, 6, 1.0, 1.0, "stochastic", "", {}),
        (4, "binary", 63, 6, 1.0, 1.0, "stochastic", "", {}),
        (5, "regression", 63, 6, 1.0, 1.0, "fixedpoint", "", {}),
        (13, "binary", 255, 8, 1.0, 1.0, "stochastic", "", {}),
        (13, "binary", 127, 7, 1.0, 1.0, "stochastic", "", {"gpu_use_dp": True}),
        (16, "regression", 127, 7, 1.0, 0.7, "fixedpoint", "", {}),
        (17, "binary", 127, 7, 1.0, 1.0, "stochastic", "", {}),
        (28, "binary", 255, 8, 1.0, 1.0, "stochastic", "", {}),
        (28, "binary", 63, 6, 1.0, 0.8, "stochastic", "", {"boosting": "rf"}),
        (29, "binary", 255, 8, 1.0, 1.0, "fixedpoint", "", {}),
        (32, "multiclass", 63, 6, 1.0, 1.0, "fixedpoint", "", {}),
        (29, "binary", 63, 6, 1.0, 1.0, "stochastic", ",device_tree:off", {}),
        (29, "binary", 63, 6, 1.0, 1.0, "fixedpoint", ",device_levels:off", {}),
        (29, "binary", 63, 6, 0.3, 1.0, "stochastic", "", {}),
        (29, "binary", 63, 6, 0.7, 0.7, "stochastic", "", {}),
        (29, "binary", 63, -1, 1.0, 1.0, "stochastic", "", {}),
    ],
)
def test_dense_vec_narrow_is_bit_identical_cuda(monkeypatch, case):
    """Verify dense_vec_narrow is bit-identical to key-off across narrow widths and flows."""
    cols, objective, leaves, depth, feature_fraction, bagging, quant, extra_plan, extra_params = case
    monkeypatch.setenv("FALCATA_TEST_FORCE_DENSE_VEC_NARROW", "1")
    X, y = _make_dataset(20011, cols, objective)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "max_bin": 255,
        "feature_fraction": feature_fraction,
        "quant_mode": quant,
        "verbose": -1,
        "seed": 11,
        **extra_params,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": f"auto{extra_plan},dense_vec_narrow:on"}, X, y, 10)
    off = _model_md5({**params, "cuda_plan": f"auto{extra_plan},dense_vec_narrow:off"}, X, y, 10)
    assert on == off


_DIAG_PROBE = """
import sys
sys.path.insert(0, {here!r})
import falcata as flc
from test_cuda_dense_vec_narrow import _make_dataset

X, y = _make_dataset(20011, {cols}, "binary")
params = {{
    "device_type": "cuda",
    "objective": "binary",
    "num_leaves": 63,
    "max_depth": 6,
    "min_data_in_leaf": 10,
    "learning_rate": 0.1,
    "max_bin": 255,
    "feature_fraction": {feature_fraction},
    "quant_mode": "stochastic",
    "verbose": 1,
    "seed": 11,
    "cuda_plan": {plan!r},
}}
ds = flc.Dataset(X, label=y, params=params)
flc.train(params, ds, num_boost_round=3)
"""


def _run_diag_probe(cols, plan, feature_fraction, force_narrow, debug="diag"):
    code = _DIAG_PROBE.format(
        here=os.path.dirname(os.path.abspath(__file__)),
        cols=cols,
        feature_fraction=feature_fraction,
        plan=plan,
    )
    env = {**os.environ, "FALCATA_DEBUG": debug}
    if force_narrow:
        env["FALCATA_TEST_FORCE_DENSE_VEC_NARROW"] = "1"
    else:
        env.pop("FALCATA_TEST_FORCE_DENSE_VEC_NARROW", None)
    res = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=True,
        timeout=120,
    )
    return res.stdout


@_REQUIRES_CUDA
def test_dense_vec_narrow_diag_activation_cuda():
    """Verify FALCATA_DEBUG=diag confirms kernel engagement, key-off, natural L2 gate, ff gate, and VRAM gate."""
    # Forced L2 spill + key on + ff=1.0: narrow kernel engages.
    out_on = _run_diag_probe(28, "auto,dense_vec_narrow:on", feature_fraction=1.0, force_narrow=True)
    assert "quantized construct: dense_vec_narrow_32B kernel" in out_on, out_on

    # Key off: narrow kernel does not engage.
    out_off = _run_diag_probe(28, "auto,dense_vec_narrow:off", feature_fraction=1.0, force_narrow=True)
    assert "quantized construct:" in out_off, out_off
    assert "dense_vec_narrow_" not in out_off, out_off

    # Natural L2 gate on small data (20011 x 28 ~ 0.56 MB << L2): cleanly declines without force env var.
    out_small = _run_diag_probe(28, "auto,dense_vec_narrow:on", feature_fraction=1.0, force_narrow=False)
    assert "quantized construct:" in out_small, out_small
    assert "dense_vec_narrow_" not in out_small, out_small

    # feature_fraction < 1.0: padding/narrow kernel cleanly declines.
    out_ff = _run_diag_probe(28, "auto,dense_vec_narrow:on", feature_fraction=0.7, force_narrow=True)
    assert "quantized construct:" in out_ff, out_ff
    assert "dense_vec_narrow_" not in out_ff, out_ff

    # Tight VRAM headroom (vramfree=1 MiB): padding/narrow kernel cleanly declines.
    out_vram = _run_diag_probe(
        28, "auto,dense_vec_narrow:on", feature_fraction=1.0, force_narrow=True, debug="diag,vramfree=1"
    )
    assert "quantized construct:" in out_vram, out_vram
    assert "dense_vec_narrow_" not in out_vram, out_vram
