# coding: utf-8
"""Modernized compact_prefill_dt must coexist with device_tree, AsyncTreeStart, and pair_code5 bit-identically.

cuda_plan key compact_prefill_dt overlaps the next tree's compact bin matrix fill (4-bit nibble, pair_code5/6/radix,
8-bit compact view, or non-quantized compact view) and metadata staging on a low-priority non-blocking stream while
the current tree executes device_tree, using non-mutating ColSampler::PeekNextByTree so RNG order is never corrupted
under reset_parameter or early stopping. It skips GPU compact prefill on unbagged single-pass fused_root_hist trees
to avoid adding an extra DRAM pass.
"""

import hashlib
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _model_md5(params, X, y, rounds, *, valid_sets=None, callbacks=None):
    ds = flc.Dataset(X, label=y, params=params)
    vsets = None
    if valid_sets is not None:
        vsets = [flc.Dataset(Xv, label=yv, reference=ds, params=params) for Xv, yv in valid_sets]
    booster = flc.train(params, ds, num_boost_round=rounds, valid_sets=vsets, callbacks=callbacks)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _make_dataset(kind, rows, cols, seed):
    rng = np.random.default_rng(seed)
    if kind == "fewbin":
        X = rng.integers(0, 5, size=(rows, cols), dtype=np.int8).astype(np.float32)
        y = (X[:, :16] @ rng.standard_normal(16) + rng.normal(0.0, 1.0, size=rows)).astype(np.float32)
    elif kind == "fewbin-nan":
        X = rng.integers(0, 5, size=(rows, cols), dtype=np.int8).astype(np.float32)
        X[rng.random(X.shape) < 0.06] = np.nan
        y = (np.nan_to_num(X[:, :16]) @ rng.standard_normal(16) + rng.normal(0.0, 1.0, size=rows)).astype(np.float32)
    else:
        X = rng.standard_normal(size=(rows, cols)).astype(np.float32)
        y = (X[:, :16] @ rng.standard_normal(16) + rng.normal(0.0, 1.0, size=rows)).astype(np.float32)
    return X, y


_BIT_IDENTITY_CASES = [
    pytest.param(
        {
            "kind": "fewbin",
            "cols": 600,
            "ff": 0.2,
            "bag_frac": 0.8,
            "bag_freq": 1,
            "quant_mode": "stochastic",
            "deterministic": False,
        },
        id="bagged-4bit-code5",
    ),
    pytest.param(
        {
            "kind": "fewbin-nan",
            "cols": 400,
            "ff": 0.2,
            "bag_frac": 0.7,
            "bag_freq": 1,
            "quant_mode": "stochastic",
            "deterministic": False,
        },
        id="bagged-4bit-nan",
    ),
    pytest.param(
        {
            "kind": "dense",
            "cols": 120,
            "ff": 0.3,
            "bag_frac": 1.0,
            "bag_freq": 0,
            "quant_mode": "fixedpoint",
            "deterministic": False,
        },
        id="8bit-compact-ff03",
    ),
    pytest.param(
        {
            "kind": "fewbin",
            "cols": 200,
            "ff": 0.25,
            "bag_frac": 1.0,
            "bag_freq": 0,
            "quant_mode": "none",
            "deterministic": True,
        },
        id="nonquant-det-4bit",
    ),
    pytest.param(
        {
            "kind": "fewbin",
            "cols": 600,
            "ff": 0.2,
            "bag_frac": 1.0,
            "bag_freq": 0,
            "quant_mode": "stochastic",
            "deterministic": False,
        },
        id="unbagged-4bit-guard",
    ),
    pytest.param(
        {
            "kind": "dense",
            "cols": 40,
            "ff": 1.0,
            "bag_frac": 1.0,
            "bag_freq": 0,
            "quant_mode": "fixedpoint",
            "deterministic": False,
        },
        id="unsampled-guard",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", _BIT_IDENTITY_CASES)
def test_compact_prefill_dt_bit_identical_cuda(case):
    X, y = _make_dataset(case["kind"], rows=12000, cols=case["cols"], seed=17)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 63,
        "max_depth": 6,
        "min_data_in_leaf": 20,
        "feature_fraction": case["ff"],
        "bagging_fraction": case["bag_frac"],
        "bagging_freq": case["bag_freq"],
        "quant_mode": case["quant_mode"],
        "deterministic": case["deterministic"],
        "verbosity": -1,
        "seed": 42,
    }
    md5_off = _model_md5({**params, "cuda_plan": "auto,compact_prefill_dt:off"}, X, y, rounds=15)
    md5_on = _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=15)
    assert md5_on == md5_off


@_REQUIRES_CUDA
def test_compact_prefill_dt_rng_order_reset_parameter_cuda():
    """PeekNextByTree must never mutate ColSampler RNG state across reset_parameter or early stopping."""
    X, y = _make_dataset("fewbin", rows=6000, cols=80, seed=31)
    Xv, yv = _make_dataset("fewbin", rows=1500, cols=80, seed=32)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.35,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "quant_mode": "stochastic",
        "verbosity": -1,
        "seed": 7,
    }

    def run(plan):
        cbs = [flc.reset_parameter(learning_rate=lambda i: 0.1 if i < 6 else 0.05)]
        return _model_md5({**params, "cuda_plan": plan}, X, y, rounds=14, valid_sets=[(Xv, yv)], callbacks=cbs)

    md5_off = run("auto,compact_prefill_dt:off")
    assert run("auto") == md5_off
    assert run("auto,compact_prefill:on") == md5_off


_DIAG_PROBE = """
import hashlib
import numpy as np
import falcata as flc

rng = np.random.default_rng(5)
kind = {kind!r}
rows, cols = 8000, {cols}
if kind == "fewbin":
    X = rng.integers(0, 5, size=(rows, cols), dtype=np.int8).astype(np.float32)
else:
    X = rng.standard_normal(size=(rows, cols)).astype(np.float32)
y = (X[:, :24] @ rng.standard_normal(24) + rng.standard_normal(rows)).astype(np.float32)
params = {{
    "device_type": "cuda",
    "objective": "regression",
    "num_leaves": 63,
    "max_depth": 6,
    "max_bin": {max_bin},
    "feature_fraction": {ff},
    "bagging_fraction": {bag_frac},
    "bagging_freq": {bag_freq},
    "quant_mode": "stochastic",
    "verbosity": 1,
    "seed": 42,
    "cuda_plan": {plan!r},
}}
bst = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=8)
print("MD5", hashlib.md5(bst.model_to_string().split("\\nparameters:")[0].encode()).hexdigest())
"""


def _run_diag(spec, plan):
    code = _DIAG_PROBE.format(**spec, plan=plan)
    env = {**os.environ, "FALCATA_DEBUG": "diag"}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    combined = p.stdout + "\n" + p.stderr
    md5 = re.findall(r"^MD5 (\w+)$", p.stdout, flags=re.M)[0]
    hits = re.findall(r"compact fill: prefill hit \(([^)]+)\)", combined)
    return md5, hits


_DIAG_CASES = [
    pytest.param(
        {"kind": "fewbin", "cols": 1000, "max_bin": 15, "ff": 0.2, "bag_frac": 0.8, "bag_freq": 1},
        True,
        "4-bit",
        id="bagged-4bit-code5",
    ),
    pytest.param(
        {"kind": "dense", "cols": 120, "max_bin": 255, "ff": 0.3, "bag_frac": 1.0, "bag_freq": 0},
        True,
        "8-bit",
        id="8bit-compact-ff03",
    ),
    pytest.param(
        {"kind": "fewbin", "cols": 1000, "max_bin": 15, "ff": 0.2, "bag_frac": 1.0, "bag_freq": 0},
        False,
        "",
        id="unbagged-4bit-skip",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize(("spec", "expect_hits", "expect_substr"), _DIAG_CASES)
def test_compact_prefill_dt_diag_activation_cuda(spec, expect_hits, expect_substr):
    md5_on, hits_on = _run_diag(spec, "auto")
    md5_off, hits_off = _run_diag(spec, "auto,compact_prefill_dt:off")
    assert md5_on == md5_off
    assert hits_off == []
    if expect_hits:
        assert len(hits_on) >= 5, hits_on
        assert all(expect_substr in h for h in hits_on), hits_on
    else:
        assert hits_on == [], hits_on
