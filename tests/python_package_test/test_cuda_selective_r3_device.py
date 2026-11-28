# coding: utf-8
"""Bit-identity and activation gate tests for cuda_plan key selective_r3_device.

Under ``selective_r3_device:on`` (the default), selective trees with ``num_leaves >= 128`` hand off
directly on-device from the dynamic R1 prefix into the captured two-level parallel R3 device loop.
Every tested configuration must produce an identical model ``md5`` to ``selective_r3_device:off``,
including tie-heavy discrete datasets where parallel R3's exact tie-break fallback must engage, and
tiny trees (``num_leaves < 128``) must keep the device R3 loop off.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

_DIAG_RE = re.compile(r"selective_r3_device: complete .*?displaced=(\d+) resumes=(\d+) graph_hits=(\d+)")

_TRAIN = (
    "import hashlib, json, sys, numpy as np, falcata as flc\n"
    "d = np.load(sys.argv[1])\n"
    "p = json.loads(sys.argv[2])\n"
    "ds = flc.Dataset(d['X'], label=d['y'], params=p)\n"
    "ds.construct()\n"
    "b = flc.train(p, ds, num_boost_round=int(sys.argv[3]))\n"
    "md5 = hashlib.md5(b.model_to_string().split('\\nparameters:')[0].encode()).hexdigest()[:12]\n"
    "print(json.dumps({'md5': md5, 'trees': b.num_trees()}))\n"
)


def _run_diag(X, y, params, rounds):
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "data.npz")
        np.savez(data_path, X=np.ascontiguousarray(X), y=np.ascontiguousarray(y))
        env = {k: v for k, v in os.environ.items() if k != "FALCATA_DEBUG"}
        env["FALCATA_DEBUG"] = "diag"
        proc = subprocess.run(
            [sys.executable, "-c", _TRAIN, data_path, json.dumps(params), str(rounds)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    matches = [_DIAG_RE.search(line) for line in proc.stderr.splitlines()]
    found = [m for m in matches if m is not None]
    stats = {
        "r3_complete": len(found),
        "displaced": sum(int(m.group(1)) for m in found),
        "resumes": sum(int(m.group(2)) for m in found),
        "graph_hits": max((int(m.group(3)) for m in found), default=0),
    }
    return info["md5"], info["trees"], stats


def _make_dataset(kind):
    rng = np.random.default_rng(2026)
    if kind == "continuous":
        X = rng.standard_normal((24000, 16)).astype(np.float32)
        z = X[:, 0] * X[:, 1] + 0.7 * np.sin(X[:, 2]) - 0.5 * X[:, 3]
        y = (z + rng.normal(0.0, 0.5, size=len(X))).astype(np.float32)
        return X, y
    if kind == "binary":
        X = rng.standard_normal((24000, 16)).astype(np.float32)
        z = X[:, 0] * X[:, 1] + 0.6 * np.sin(X[:, 2]) - 0.5 * X[:, 3]
        y = (z + rng.normal(0.0, 0.6, size=len(X)) > 0.0).astype(np.float32)
        return X, y
    if kind == "multiclass":
        X = rng.standard_normal((16000, 14)).astype(np.float32)
        logits = np.stack(
            [
                X[:, 0] + 0.5 * X[:, 1],
                -X[:, 0] + 0.5 * X[:, 2],
                X[:, 2] - 0.6 * X[:, 3],
                -X[:, 1] + 0.6 * X[:, 3],
            ],
            axis=1,
        )
        y = np.argmax(logits + rng.normal(0.0, 0.4, size=logits.shape), axis=1).astype(np.float32)
        return X, y
    if kind == "tie_mirror":
        half = 10000
        Xh = rng.integers(0, 8, size=(half, 10)).astype(np.float32)
        yh = (5.0 + Xh[:, 1] + Xh[:, 2] * (Xh[:, 3] > 3) + rng.normal(0.0, 0.5, size=half)).astype(np.float32)
        Xh[:, 0] = 0.0
        X2 = Xh.copy()
        X2[:, 0] = 1.0
        return np.vstack([Xh, X2]), np.concatenate([yh, -yh])
    if kind == "tie_binary":
        X = rng.integers(0, 2, size=(18000, 14)).astype(np.float32)
        y = ((X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]) % 2).astype(np.float32)
        return X, y
    if kind == "tie_quant3":
        X = rng.integers(0, 3, size=(20000, 12)).astype(np.float32)
        y = ((X[:, 0] == 1).astype(np.float32) - (X[:, 1] == 2).astype(np.float32)).astype(np.float32)
        return X, y
    raise ValueError(f"unknown kind: {kind}")


_CASES = [
    (
        "unbounded_255",
        "continuous",
        {
            "objective": "regression",
            "num_leaves": 255,
            "max_depth": -1,
            "max_bin": 63,
            "min_data_in_leaf": 10,
            "quant_mode": "stochastic",
        },
        8,
        True,
    ),
    (
        "budget_255_d12",
        "binary",
        {
            "objective": "binary",
            "num_leaves": 255,
            "max_depth": 12,
            "max_bin": 63,
            "min_data_in_leaf": 10,
            "quant_mode": "stochastic",
        },
        8,
        True,
    ),
    (
        "tie_mirror_255",
        "tie_mirror",
        {
            "objective": "regression",
            "num_leaves": 255,
            "max_depth": -1,
            "max_bin": 15,
            "min_data_in_leaf": 5,
            "quant_mode": "fixedpoint",
        },
        6,
        True,
    ),
    (
        "tie_binary_255",
        "tie_binary",
        {
            "objective": "regression",
            "num_leaves": 255,
            "max_depth": -1,
            "max_bin": 15,
            "min_data_in_leaf": 5,
            "quant_mode": "fixedpoint",
        },
        6,
        True,
    ),
    (
        "tie_quant3_255",
        "tie_quant3",
        {
            "objective": "regression",
            "num_leaves": 255,
            "max_depth": -1,
            "max_bin": 15,
            "min_data_in_leaf": 5,
            "quant_mode": "fixedpoint",
        },
        6,
        True,
    ),
    (
        "bagging_fixedpoint_255",
        "continuous",
        {
            "objective": "regression",
            "num_leaves": 255,
            "max_depth": -1,
            "max_bin": 63,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.75,
            "bagging_freq": 1,
            "quant_mode": "fixedpoint",
        },
        6,
        True,
    ),
    (
        "multiclass_255",
        "multiclass",
        {
            "objective": "multiclass",
            "num_class": 4,
            "num_leaves": 255,
            "max_depth": -1,
            "max_bin": 63,
            "min_data_in_leaf": 10,
            "lambda_l2": 1.0,
            "quant_mode": "stochastic",
        },
        4,
        True,
    ),
    (
        "tiny_gate_off_63",
        "continuous",
        {
            "objective": "regression",
            "num_leaves": 63,
            "max_depth": -1,
            "max_bin": 63,
            "min_data_in_leaf": 10,
            "quant_mode": "stochastic",
        },
        6,
        False,
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize(("name", "kind", "extra", "rounds", "expect_r3"), _CASES, ids=[c[0] for c in _CASES])
def test_selective_r3_device_bit_identity_and_activation_cuda(name, kind, extra, rounds, expect_r3):
    X, y = _make_dataset(kind)
    base = {
        "device_type": "cuda",
        "verbose": -1,
        "metric": "None",
        "num_threads": 24,
        "seed": 42,
        **extra,
    }
    md5_off, trees_off, stats_off = _run_diag(X, y, {**base, "cuda_plan": "auto,selective_r3_device:off"}, rounds)
    md5_on, trees_on, stats_on = _run_diag(X, y, {**base, "cuda_plan": "auto,selective_r3_device:on"}, rounds)
    assert md5_on == md5_off, f"{name}: md5 mismatch on={md5_on} off={md5_off}"
    assert trees_on == trees_off
    assert stats_off["r3_complete"] == 0, f"{name}: selective_r3_device fired with key off ({stats_off})"
    if expect_r3:
        assert stats_on["r3_complete"] > 0, f"{name}: expected selective_r3_device: complete diag lines, got 0"
        if "feature_fraction" not in extra:
            assert stats_on["graph_hits"] > 0, f"{name}: expected CUDA graph replay hits, got {stats_on}"
        if name in ("unbounded_255", "bagging_fixedpoint_255", "multiclass_255"):
            assert stats_on["displaced"] > 0, f"{name}: expected displaced splits, got {stats_on}"
        if name in ("tie_binary_255", "tie_quant3_255"):
            assert stats_on["resumes"] > 0, f"{name}: expected exact tie-break resumes, got {stats_on}"
    else:
        assert stats_on["r3_complete"] == 0, f"{name}: expected gate off for num_leaves<128, got {stats_on}"
