"""Exact DL launch resources, dense fallback, graph reuse and complete-model equality.

Small synthetic correctness fixtures only; these are not performance measurements.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")
_HOST = "device_levels:off,device_tree:off"
_ROUTES = {"prefix": "device_levels:on,device_tree:off", "whole": "device_levels:off,device_tree:on"}


def _model(booster):
    # Controller intent is the only model parameter allowed to differ.
    return "\n".join(line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line)


def _boundary_probe(padded, route, graph, quant_mode):
    rng = np.random.default_rng(314159)
    x = rng.integers(0, 8, size=(8192, 386), dtype=np.int8).astype(np.float32)
    x[:8] = np.arange(8, dtype=np.float32)[:, None]
    y = (x[:, 0] + 2 * x[:, 1] - x[:, 2] + (x[:, 3] > 3)).astype(np.float32)
    # One source partition, with a strict sample so the compact view is built.
    # With padding: 188/189 eight-by-eight tables -> 48,880/49,140 bytes.
    # Without padding: 191/192 tables -> 48,896/49,152 bytes. The latter leaves no static shared-memory space.
    lower, upper = (376, 378) if padded else (382, 384)
    common = (
        f"auto,view_mode:compact,pair_pad:{'on' if padded else 'off'},"
        f"pair_block_rows:off,pair_capped_rows:off,pair_code5:off,device_level_graph:{'on' if graph else 'off'}"
    )
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "gpu_use_dp": False,
        "quant_mode": quant_mode,
        "quant_bins": 64,
        "max_bin": 8,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": len(x),
        "max_depth": 5,
        "num_leaves": 31,
        "min_data_in_leaf": 16,
        "feature_fraction": lower / x.shape[1],
        "feature_pre_filter": False,
        "enable_bundle": False,
        "is_enable_sparse": False,
        "use_missing": False,
        "num_threads": 2,
        "verbosity": 1,
        "seed": 271828,
        "cuda_plan": common + "," + _HOST,
    }
    dataset = flc.Dataset(x, label=y, params=params, free_raw_data=False).construct()
    host = flc.Booster(params, train_set=dataset)
    device = flc.Booster({**params, "cuda_plan": common + "," + _ROUTES[route]}, train_set=dataset)
    # Back to the first geometry after rejecting another one, on retained allocations and learner state.
    # Two updates per phase also exercise repeated graph / shape use.
    for phase, selected in enumerate((lower, upper, lower)):
        if phase:
            for booster in (host, device):
                booster.reset_parameter({"feature_fraction": selected / x.shape[1]})
        for _ in range(2):
            host.update()
            device.update()
            assert _model(host) == _model(device), f"complete model differs at phase {phase}"
    assert host.num_trees() == device.num_trees() == 6
    assert all(tree["num_leaves"] > 1 for tree in device.dump_model()["tree_info"])
    print(json.dumps({"equal": True}))


def _run(*args):
    code = "import json,runpy,sys; runpy.run_path(sys.argv[1])['_boundary_probe'](*json.loads(sys.argv[2]))"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, json.dumps(args)],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    output = result.stdout + "\nSTDERR:\n" + result.stderr
    assert result.returncode == 0, output
    assert json.loads(result.stdout.splitlines()[-1])["equal"], output
    assert "CUDA error" not in output, output
    assert "invalid argument" not in output, output
    return output


def _admissions(diag):
    result = {}
    for line in diag.splitlines():
        if "[hybrid-diag] device_pair_joint:" in line:
            fields = dict(re.findall(r"(\w+)=([\w-]+)", line))
            result[int(fields["dynamic_shared"])] = fields
    return result


@_REQUIRES_CUDA
@pytest.mark.parametrize("padded", [False, True], ids=["full-48k", "12-byte-slack"])
@pytest.mark.parametrize("route", ["prefix", "whole"])
@pytest.mark.parametrize(("graph", "quant_mode"), [(True, "fixedpoint"), (False, "stochastic")])
def test_device_pair_joint_shared_limit_and_resets_cuda(padded, route, graph, quant_mode):
    diag = _run(padded, route, graph, quant_mode)
    admissions = _admissions(diag)
    low_bytes, high_bytes = (48880, 49140) if padded else (48896, 49152)
    assert low_bytes in admissions, diag
    assert high_bytes in admissions, diag
    low, high = admissions[low_bytes], admissions[high_bytes]
    assert low["kernel"] == high["kernel"] == "0", diag  # plain DL gather, not a code-view specialization
    assert low["route"] == "pair", diag
    assert low["reason"] == "ok", diag
    assert int(low["resident_blocks"]) > 0, diag
    # Prove that the compatible neighbor still runs the device pair kernel; disabling the family is not a fix.
    assert "quantized construct: pair_hist (device levels) kernel" in diag, diag
    assert int(high["static_shared"]) > 0, diag
    available = min(int(high["max_dynamic_shared"]), int(high["shared_limit"]) - int(high["static_shared"]))
    if not padded:
        # Exact 48 KiB dynamic allocation is incompatible with the DL kernel's additional static allocation.
        assert high_bytes > available, diag
    if high_bytes > available:
        assert (high["route"], high["reason"]) == ("dense", "shared_memory"), diag
    else:
        # A compiler/device that fits the 12-byte-slack case may keep the pair route; no fixed static-size guess.
        assert (high["route"], high["reason"]) == ("pair", "ok"), diag
    assert "host fallback" not in diag, diag
    assert f"controller: requested=explicit resolved={route} reason=explicit_controller_overrides" in diag, diag
    if route == "whole":
        complete = [line for line in diag.splitlines() if "device_tree: complete" in line]
        assert len(complete) == 6, diag
        assert all(f"graph={int(graph)}" in line for line in complete), diag
    else:
        assert "device_levels: 4 device-driven levels per tree" in diag, diag
        assert "device_tree: complete" not in diag, diag
