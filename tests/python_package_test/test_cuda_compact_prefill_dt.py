# coding: utf-8
"""Modernized compact_prefill_dt must coexist with device_tree, AsyncTreeStart, and pair_code5 bit-identically.

cuda_plan key compact_prefill_dt overlaps the next tree's compact bin matrix fill (4-bit nibble, pair_code5/6/radix,
8-bit compact view, or non-quantized compact view) and metadata staging on a low-priority non-blocking stream while
the current tree executes device_tree, using non-mutating ColSampler::PeekNextByTree so RNG order is never corrupted
under reset_parameter or early stopping. It skips GPU compact prefill on unbagged single-pass fused_root_hist trees
to avoid adding an extra DRAM pass, and gracefully declines optional prefill under tight VRAM without OOM.
"""

import gc
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


def _model_md5(params, X, y, rounds, *, valid_sets=None, callbacks=None, init_model=None):
    ds = flc.Dataset(X, label=y, params=params)
    vsets = None
    if valid_sets is not None:
        vsets = [flc.Dataset(Xv, label=yv, reference=ds, params=params) for Xv, yv in valid_sets]
    booster = flc.train(
        params,
        ds,
        num_boost_round=rounds,
        valid_sets=vsets,
        callbacks=callbacks,
        init_model=init_model,
    )
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest(), booster


def _make_dataset(kind, rows, cols, seed, *, num_class=1):
    rng = np.random.default_rng(seed)
    if kind == "fewbin":
        X = rng.integers(0, 5, size=(rows, cols), dtype=np.int8).astype(np.float32)
        base = X[:, :16] @ rng.standard_normal(16) + rng.normal(0.0, 1.0, size=rows)
    elif kind == "fewbin-nan":
        X = rng.integers(0, 5, size=(rows, cols), dtype=np.int8).astype(np.float32)
        X[rng.random(X.shape) < 0.06] = np.nan
        base = np.nan_to_num(X[:, :16]) @ rng.standard_normal(16) + rng.normal(0.0, 1.0, size=rows)
    else:
        X = rng.standard_normal(size=(rows, cols)).astype(np.float32)
        base = X[:, :16] @ rng.standard_normal(16) + rng.normal(0.0, 1.0, size=rows)
    if num_class > 1:
        y = np.floor(np.mod(np.abs(base) * 3.0, num_class)).astype(np.float32)
    else:
        y = base.astype(np.float32)
    return X, y


_BIT_IDENTITY_CASES = [
    pytest.param(
        {
            "kind": "fewbin",
            "cols": 1200,
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
            "cols": 1200,
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
    pytest.param(
        {
            "kind": "fewbin",
            "cols": 200,
            "ff": 0.25,
            "bag_frac": 0.8,
            "bag_freq": 1,
            "quant_mode": "stochastic",
            "deterministic": False,
            "objective": "multiclass",
            "num_class": 3,
        },
        id="multiclass-bagged-4bit",
    ),
    pytest.param(
        {
            "kind": "fewbin",
            "cols": 200,
            "ff": 0.25,
            "bag_frac": 1.0,
            "bag_freq": 0,
            "quant_mode": "stochastic",
            "deterministic": False,
            "extra": {"data_sample_strategy": "goss", "top_rate": 0.2, "other_rate": 0.1},
        },
        id="goss-4bit",
    ),
    pytest.param(
        {
            "kind": "fewbin",
            "cols": 200,
            "ff": 0.4,
            "bag_frac": 0.8,
            "bag_freq": 1,
            "quant_mode": "stochastic",
            "deterministic": False,
            "extra": {"feature_fraction_bynode": 0.7},
        },
        id="bynode-bagged-4bit",
    ),
    pytest.param(
        {
            "kind": "fewbin",
            "cols": 1200,
            "ff": 0.2,
            "bag_frac": 0.8,
            "bag_freq": 1,
            "quant_mode": "stochastic",
            "deterministic": False,
            "plan_suffix": ",fused_root_hist:off",
        },
        id="bagged-4bit-fused-root-off",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", _BIT_IDENTITY_CASES)
def test_compact_prefill_dt_bit_identical_cuda(case):
    num_class = case.get("num_class", 1)
    X, y = _make_dataset(case["kind"], rows=12000, cols=case["cols"], seed=17, num_class=num_class)
    params = {
        "device_type": "cuda",
        "objective": case.get("objective", "regression"),
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
        **case.get("extra", {}),
    }
    if num_class > 1:
        params["num_class"] = num_class
    suffix = case.get("plan_suffix", "")
    md5_off, _ = _model_md5({**params, "cuda_plan": f"auto,compact_prefill_dt:off{suffix}"}, X, y, rounds=15)
    md5_on, _ = _model_md5({**params, "cuda_plan": f"auto{suffix}"}, X, y, rounds=15)
    assert md5_on == md5_off


@_REQUIRES_CUDA
def test_compact_prefill_legacy_mode_bit_identical_cuda():
    """Legacy compact_prefill:on with compact_prefill_dt:off must still match bit-identically."""
    X, y = _make_dataset("fewbin", rows=8000, cols=160, seed=23)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.3,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "quant_mode": "stochastic",
        "verbosity": -1,
        "seed": 19,
    }
    md5_ref, _ = _model_md5({**params, "cuda_plan": "auto,compact_prefill_dt:off"}, X, y, rounds=12)
    md5_legacy, _ = _model_md5(
        {**params, "cuda_plan": "auto,compact_prefill:on,compact_prefill_dt:off"}, X, y, rounds=12
    )
    assert md5_legacy == md5_ref


@_REQUIRES_CUDA
def test_compact_prefill_dt_rng_reset_early_stop_and_init_model_cuda():
    """PeekNextByTree must preserve exact RNG draw order across reset_parameter(feature_fraction),
    early_stopping across multiple valid_sets, and init_model continuation."""
    X, y = _make_dataset("fewbin", rows=6000, cols=80, seed=31)
    Xv1, yv1 = _make_dataset("fewbin", rows=1500, cols=80, seed=32)
    Xv2, yv2 = _make_dataset("fewbin", rows=1500, cols=80, seed=33)
    # Noisy second validation set encourages early stopping after a few rounds.
    yv2 = yv2 + np.random.default_rng(34).normal(0.0, 25.0, size=yv2.shape).astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.25,
        "feature_fraction": 0.35,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "quant_mode": "stochastic",
        "verbosity": -1,
        "seed": 7,
    }

    def run_sequence(plan):
        cbs = [
            flc.reset_parameter(
                learning_rate=lambda i: 0.25 if i < 4 else 0.15,
                feature_fraction=lambda i: 0.35 if i < 5 else 0.5,
            ),
            flc.early_stopping(stopping_rounds=3, verbose=False),
        ]
        md5_stage1, bst1 = _model_md5(
            {**params, "cuda_plan": plan},
            X,
            y,
            rounds=25,
            valid_sets=[(Xv1, yv1), (Xv2, yv2)],
            callbacks=cbs,
        )
        # Continue training from the early-stopped booster (init_model) with a fresh booster.
        md5_stage2, _ = _model_md5(
            {**params, "feature_fraction": 0.4, "cuda_plan": plan},
            X,
            y,
            rounds=6,
            init_model=bst1,
        )
        return bst1.current_iteration(), md5_stage1, md5_stage2

    iters_off, s1_off, s2_off = run_sequence("auto,compact_prefill_dt:off")
    iters_on, s1_on, s2_on = run_sequence("auto")
    iters_pre, s1_pre, s2_pre = run_sequence("auto,compact_prefill:on")
    assert iters_off < 25, "expected early_stopping to trigger before max rounds"
    assert (iters_on, s1_on, s2_on) == (iters_off, s1_off, s2_off)
    assert (iters_pre, s1_pre, s2_pre) == (iters_off, s1_off, s2_off)


@_REQUIRES_CUDA
def test_compact_prefill_dt_teardown_with_pending_prefill_cuda():
    """Destroying a booster immediately while a next-tree prefill may be in flight must drain cleanly."""
    X, y = _make_dataset("fewbin", rows=6000, cols=200, seed=41)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.25,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "quant_mode": "stochastic",
        "verbosity": -1,
        "seed": 41,
        "cuda_plan": "auto",
    }
    for rounds in (1, 2, 5):
        ds = flc.Dataset(X, label=y, params=params)
        bst = flc.train(params, ds, num_boost_round=rounds)
        assert bst.num_trees() == rounds
        del bst, ds
        gc.collect()


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


def _run_diag(spec, plan, *, debug_env="diag"):
    code = _DIAG_PROBE.format(**spec, plan=plan)
    env = {**os.environ, "FALCATA_DEBUG": debug_env}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    combined = p.stdout + "\n" + p.stderr
    md5 = re.findall(r"^MD5 (\w+)$", p.stdout, flags=re.M)[0]
    hits = re.findall(r"compact fill: prefill hit \(([^)]+)\)", combined)
    return md5, hits


_DIAG_CASES = [
    pytest.param(
        {"kind": "fewbin", "cols": 1200, "max_bin": 15, "ff": 0.2, "bag_frac": 0.8, "bag_freq": 1},
        True,
        ("4-bit", "code5=1"),
        id="bagged-4bit-code5",
    ),
    pytest.param(
        {"kind": "dense", "cols": 120, "max_bin": 255, "ff": 0.3, "bag_frac": 1.0, "bag_freq": 0},
        True,
        ("8-bit",),
        id="8bit-compact-ff03",
    ),
    pytest.param(
        {"kind": "fewbin", "cols": 1200, "max_bin": 15, "ff": 0.2, "bag_frac": 1.0, "bag_freq": 0},
        False,
        (),
        id="unbagged-4bit-skip",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize(("spec", "expect_hits", "expect_substrs"), _DIAG_CASES)
def test_compact_prefill_dt_diag_activation_cuda(spec, expect_hits, expect_substrs):
    md5_on, hits_on = _run_diag(spec, "auto")
    md5_off, hits_off = _run_diag(spec, "auto,compact_prefill_dt:off")
    assert md5_on == md5_off
    assert hits_off == []
    if expect_hits:
        assert len(hits_on) >= 5, hits_on
        for substr in expect_substrs:
            assert all(substr in h for h in hits_on), (substr, hits_on)
    else:
        assert hits_on == [], hits_on


@_REQUIRES_CUDA
def test_compact_prefill_dt_tight_vram_graceful_fallback_cuda():
    """Under tight VRAM headroom (vramfree=1 MiB), optional prefill must decline cleanly without OOM
    and produce the exact same model."""
    spec = {"kind": "dense", "cols": 120, "max_bin": 255, "ff": 0.3, "bag_frac": 1.0, "bag_freq": 0}
    md5_ref, _ = _run_diag(spec, "auto,compact_prefill_dt:off", debug_env="diag")
    md5_tight, hits_tight = _run_diag(spec, "auto", debug_env="diag,vramfree=1")
    assert md5_tight == md5_ref
    assert hits_tight == [], hits_tight
