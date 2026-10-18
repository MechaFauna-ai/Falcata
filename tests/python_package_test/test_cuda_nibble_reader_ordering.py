# coding: utf-8
"""Pending nibble fills must precede bounded whole-tree readers; run through gpuq only."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import falcata as flc

_ROUNDS = 500
_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _probe(arm, model_path):
    rng = np.random.default_rng(23)
    # Multiple source partitions are required for the interleaved row layout used by the code fill.
    levels = 5 + np.arange(3555) % 2
    X = np.floor(rng.random((4001, len(levels))) * levels).astype(np.float32)
    X[rng.random(X.shape) < 0.08] = np.nan
    y = np.nan_to_num(X[:, :16]) @ rng.standard_normal(16) + rng.standard_normal(len(X))
    # The fill still writes six-bit codes, but the per-byte descendant reader supports five-bit codes only.
    # Fix this reader choice in ALL arms so the fixture requires deferred nibbles, regardless of device occupancy.
    plan = "auto,view_mode:compact,pair_code_radix:off,pair_code5_words:off,device_levels:off"
    plan += (
        ",device_tree:off,device_level_graph:off" if arm == "host" else (",device_tree:on,device_level_graph:" + arm)
    )
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "num_leaves": 255,
        "max_depth": 8,
        "max_bin": 15,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.1,
        "learning_rate": 0.01,
        "use_quantized_grad": True,
        "quant_mode": "stochastic",
        "seed": 42,
        "num_threads": 2,
        "metric": "None",
        "verbosity": 1,  # Info diagnostics prove the pending fill and actual reader engaged.
        "cuda_plan": plan,
    }
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=_ROUNDS)
    assert booster.current_iteration() == _ROUNDS
    model = booster.model_to_string(num_iteration=-1)
    assert len(re.findall(r"^Tree=\d+$", model, re.MULTILINE)) == _ROUNDS
    leaves = [int(n) for n in re.findall(r"^num_leaves=(\d+)$", model, re.MULTILINE)]
    assert len(leaves) == _ROUNDS
    assert all(1 < n <= 255 for n in leaves)
    assert "[num_leaves: 255]\n" in model
    assert "[max_depth: 8]\n" in model
    # Remove exactly the serialized plan line; preserve every other byte, including all tree text and newlines.
    model, removed = re.subn(r"^\[cuda_plan: [^\r\n]*\]\n", "", model, flags=re.MULTILINE)
    assert removed == 1
    Path(model_path).write_bytes(model.encode("utf-8"))
    print(json.dumps({"rounds": booster.current_iteration(), "leaves": leaves}), flush=True)


@_REQUIRES_CUDA
def test_cuda_nibble_reader_ordering(tmp_path):
    models = []
    child = "import runpy,sys; runpy.run_path(sys.argv[1])['_probe'](*sys.argv[2:])"
    for arm in ("host", "off", "on"):
        model_path = tmp_path / (arm + ".model")
        result = subprocess.run(
            [sys.executable, "-c", child, __file__, arm, str(model_path)],
            env=dict(os.environ, FALCATA_DEBUG="diag", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1"),
            capture_output=True,
            text=True,
            check=False,
        )
        diag = result.stdout + "\n" + result.stderr
        assert result.returncode == 0, diag
        data = json.loads(result.stdout.splitlines()[-1])
        assert data["rounds"] == _ROUNDS
        # Materialization logs only when a valid code view left the nibble fill pending, not just when requested.
        assert len(re.findall(r"compact fill: pair_code5 on \([^\n]*6 bits per code\)", diag)) == _ROUNDS, diag
        assert diag.count("compact fill: pair_code5 nibble view materialized for another reader") == _ROUNDS, diag
        events = re.findall(
            r"device_tree: complete depth=(\d+) applied_levels=(\d+) splits=(\d+) "
            r"growth_readbacks=1 host_tail=0 graph=(\d+) hits=(\d+) misses=(\d+)",
            diag,
        )
        if arm == "host":
            assert not events, diag
        else:
            assert len(events) == _ROUNDS, diag
            assert "device_tree: host fallback" not in diag
            assert "quantized construct: pair_hist (device levels) kernel" in diag
            assert "pair_code5_words (device levels)" not in diag
            for event, leaves in zip(events, data["leaves"], strict=True):
                assert int(event[0]) == 8
                assert 0 < int(event[1]) <= 8
                assert int(event[2]) + 1 == leaves
                # With graph:on, a pending nibble reader must already be capturable on the very first tree.
                assert int(event[3]) == int(arm == "on"), diag
        models.append(model_path.read_bytes())
    assert models[0] == models[1], "complete host/whole graph-off model bytes differ"
    assert models[0] == models[2], "complete host/whole graph-on model bytes differ"
