# coding: utf-8
"""Packed root statistics must retain low hessian bits during CUDA quantized training."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _normalized_model(booster):
    """Remove exactly one serialized cuda_plan line, preserving every other model byte."""
    raw = booster.model_to_string(num_iteration=-1).encode("utf-8")
    kept = []
    removed = 0
    for line in raw.splitlines(keepends=True):
        if re.fullmatch(rb"\[cuda_plan: [^\r\n]*\]\r?\n?", line):
            removed += 1
        else:
            assert not line.startswith(b"[cuda_plan"), "malformed cuda_plan line"
            kept.append(line)
    assert removed == 1
    return b"".join(kept)


def _first_tree_quanta(target, seed):
    """CPU oracle for first-tree fixedpoint256, max|gradient|128, EF on, robust scale off.

    Philox4x32-10 supplies each row's phase. Integer products use uint64 to recover both uint32 halves;
    float32 additions precede the kernel's double rounding offsets. First-tree residuals are zero.
    This models the unchanged discretizer, not the packed root reduction being tested.
    """
    x = np.arange(len(target), dtype=np.uint64)
    y = np.full_like(x, 0x243F6A88)
    z = np.full_like(x, 0x85A308D3)
    w = np.full_like(x, 0x13198A2E)
    k0, k1 = seed, seed ^ 0xC2B2AE35
    mask = np.uint64(0xFFFFFFFF)
    for _ in range(10):
        p0, p1 = np.uint64(0xD2511F53) * x, np.uint64(0xCD9E8D57) * z
        x, y, z, w = (
            (p1 >> np.uint64(32)) ^ y ^ np.uint64(k0),
            p1 & mask,
            (p0 >> np.uint64(32)) ^ w ^ np.uint64(k1),
            p0 & mask,
        )
        k0, k1 = (k0 + 0x9E3779B9) & 0xFFFFFFFF, (k1 + 0xBB67AE85) & 0xFFFFFFFF
    gphase = (x >> np.uint64(8)).astype(np.float32) * np.float32(2**-24) - np.float32(0.5)
    hphase = (y >> np.uint64(8)).astype(np.float32) * np.float32(2**-24) - np.float32(0.5)
    gradient = (-target.astype(np.float32) + gphase).astype(np.float32)
    hessian = (np.float32(256) + hphase).astype(np.float32)
    # Both fixture targets are positive, hence gradients are negative. robust_scale:off means no gradient clamp.
    qg = np.trunc(gradient.astype(np.float64) - 0.5).astype(np.int64)
    qh = np.trunc(hessian.astype(np.float64) + 0.5).astype(np.int64)
    return qg, qh


def _probe(tmp_path):
    rows = 131072
    X = np.repeat(np.array([0, 1], dtype=np.float32), rows // 2).reshape(-1, 1)
    y = 127 + X[:, 0]
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "num_leaves": 2,
        "max_depth": 1,
        "min_data_in_leaf": 1,
        "max_bin": 3,
        "feature_pre_filter": False,
        "enable_bundle": False,
        "is_enable_sparse": False,
        "feature_fraction": 1.0,
        "learning_rate": 0.001,
        "use_quantized_grad": True,
        "quant_mode": "fixedpoint",
        "quant_bins": 256,
        "cuda_precision": "fp64",
        "boost_from_average": False,
        "quant_train_renew_leaf": False,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "path_smooth": 0.0,
        "max_delta_step": 0.0,
        "seed": 17,
        "num_threads": 2,
        "metric": "None",
        "verbosity": -1,
        "cuda_plan": "auto,device_levels:off,device_tree:off,device_level_graph:off,quant_ef:on,robust_scale:off",
    }
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=1)

    # Preserve full Native evidence before any model/value oracle. The module helper is also a retention hook
    # for validation wrappers authenticating the loaded library and keeping raw/normalized model sidecars.
    (tmp_path / "root.raw.txt").write_bytes(booster.model_to_string(num_iteration=-1).encode("utf-8"))
    dump = booster.dump_model()
    (tmp_path / "root.dump.json").write_text(json.dumps(dump, allow_nan=True), encoding="utf-8")
    model = _normalized_model(booster)

    qg, qh = _first_tree_quanta(y, params["seed"])
    groups = (slice(0, rows // 2), slice(rows // 2, rows))
    sum_g = [int(qg[group].sum(dtype=np.int64)) for group in groups]
    sum_h = [int(qh[group].sum(dtype=np.int64)) for group in groups]
    expected = [-g / (h / 256.0) * params["learning_rate"] for g, h in zip(sum_g, sum_h, strict=True)]
    packed = sum(sum_g) * (1 << 32) + sum(sum_h)
    rounded = int(float(packed))
    (tmp_path / "root.oracle.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "seed": params["seed"],
                "branch_counts": [rows // 2, rows // 2],
                "branch_gradient_quanta": sum_g,
                "branch_hessian_quanta": sum_h,
                "packed_root": str(packed),
                "double_roundtrip_root": str(rounded),
                "roundtrip_lost_hessian_quanta": sum(sum_h) - (rounded & 0xFFFFFFFF),
                "expected_leaf_values": expected,
            }
        ),
        encoding="utf-8",
    )
    assert -(1 << 63) < packed < 0
    assert abs(packed) > (1 << 53)
    assert 0 < sum(sum_h) < (1 << 32)
    assert packed != rounded, "fixture does not discriminate packed-root precision; inspect root.oracle.json"

    for line in (
        b"[quant_mode: fixedpoint]\n",
        b"[num_grad_quant_bins: 256]\n",
        b"[cuda_precision: fp64]\n",
        b"[quant_train_renew_leaf: 0]\n",
    ):
        assert model.count(line) == 1
    assert X.shape == (rows, 1)
    assert booster.current_iteration() == booster.num_trees() == len(dump["tree_info"]) == 1
    tree = dump["tree_info"][0]
    node = tree["tree_structure"]
    assert tree["num_leaves"] == 2
    assert node["internal_count"] == rows
    assert node["split_index"] == 0
    assert node["split_feature"] == 0
    assert node["decision_type"] == "<="
    assert 0 <= node["threshold"] < 1
    leaves = [node["left_child"], node["right_child"]]
    for index, leaf in enumerate(leaves):
        assert "split_index" not in leaf
        assert leaf["leaf_index"] == index
        assert leaf["leaf_count"] == rows // 2
    # A single binary split fixes branch membership independently of model leaf IDs or packed statistics.
    # Check quantized branch outputs, including legitimate EF dither, with no leaf-renewal assumption.
    np.testing.assert_allclose([leaf["leaf_value"] for leaf in leaves], expected, rtol=0.0, atol=1e-12)


@_REQUIRES_CUDA
def test_cuda_quant_root_precision(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--probe", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    (tmp_path / "probe.stdout.log").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "probe.stderr.log").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    assert len(sys.argv) == 3
    assert sys.argv[1] == "--probe"
    _probe(Path(sys.argv[2]))
