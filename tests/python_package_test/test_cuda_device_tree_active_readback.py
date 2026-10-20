# coding: utf-8
"""UNRUN source projection: active whole-tree readback must preserve model and live score bytes.

Keep the original124 native cases and four old bad-root controls unchanged.
These12 extra cases require a future authorized CUDA queue reservation.
"""

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _active_readback_probe(graph, setting):
    original = runpy.run_path(str(Path(__file__).with_name("test_cuda_device_tree.py")))
    if setting in {"weighted", "renewal"}:
        # Reuse the already-passing fixture's exact model/live-score assertions;
        # this projection adds explicit no-active-readback proof for these gates.
        original["_final_all_map_probe"](graph, setting, "")
        return
    X, _, params, _ = original["_case"]("full")
    # Exactly three genuine leaves: root X0, then only X0=1 splits X1. Odd N exercises
    # double/int tails; odd L5/7/1031 changes full-stride alignment and descriptor offsets.
    y = (1 + 7 * X[:, 0] + 2 * X[:, 0] * X[:, 1]).astype(np.float32)
    params.update(feature_pre_filter=False, boost_from_average=False)
    gate = "" if setting == "enabled" else "," + setting + ":off"
    stages = [
        (5, 2, 10.0, len(y)),
        (5, 2, 10.0, len(y)),
        (5, 2, 10.0, len(y)),  # Same geometry must actually reuse capture when graph is on.
        (5, 2, 1e12, len(y)),  # Stump must not publish stale staged bytes.
        (1031, 10, 10.0, len(y)),  # Odd capacity growth, positive early exhaustion below depth10.
        (7, 2, 10.0, len(y) // 2),  # Different actual rows; reset dataset and shrink logical capacity.
        (7, 2, 10.0, len(y) // 2),
        (5, 2, 10.0, len(y)),  # Restore dataset and offsets after another capacity change.
    ]
    snapshots = []
    for controller in ["off", "on"]:
        first_L, first_depth, first_gain, first_rows = stages[0]
        p = {
            **params,
            "num_leaves": first_L,
            "max_depth": first_depth,
            "min_gain_to_split": first_gain,
            "cuda_plan": "auto,device_levels:off,device_tree:" + controller + ",device_level_graph:" + graph + gate,
        }
        ds = flc.Dataset(X[:first_rows], label=y[:first_rows], params=p)
        booster = flc.Booster(p, train_set=ds)
        previous = stages[0]
        arm = []
        for max_leaves, depth, gain, rows in stages:
            if (max_leaves, depth, gain) != previous[:3]:
                booster.reset_parameter({"num_leaves": max_leaves, "max_depth": depth, "min_gain_to_split": gain})
            changed_data = rows != previous[3]
            if changed_data:
                ds = flc.Dataset(X[:rows], label=y[:rows], reference=ds, params=p)
            previous = (max_leaves, depth, gain, rows)
            empty = booster.update(train_set=ds if changed_data else None)
            if gain < 1e12:
                assert not empty, "odd-prefix fixture must actually produce another tree"
                current = booster.dump_model()["tree_info"][-1]
                assert current["num_leaves"] == 3, "fixture must reach odd actual N, not an accidental stump/full tree"
                assert current["tree_structure"]["internal_count"] == rows
            else:
                assert empty, "stump stage must actually stop splitting"
            live = []

            def capture(preds, _, scores=live):
                scores.append(preds.copy())
                return "capture", 0.0, False

            booster.eval_train(feval=capture)
            assert len(live) == 1
            prediction = booster.predict(X[:rows], raw_score=True, use_fil=False, num_threads=32)
            np.testing.assert_allclose(live[0], prediction, rtol=0.0, atol=1e-12)
            model = "\n".join(line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line)
            arm.append((empty, model, live[0]))
        snapshots.append(arm)
    for off, on in zip(*snapshots, strict=True):
        assert off[:2] == on[:2], "active-prefix readback changed the exact full model or empty-tree result"
        np.testing.assert_array_equal(off[2], on[2])
    print(json.dumps({"stages": len(stages), "expected_actual_leaves": [3, 3, 3, 1, 3, 3, 3, 3]}))


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize(
    "setting", ["enabled", "readback_kernel", "final_readback_first", "async_tree_start", "weighted", "renewal"]
)
def test_device_tree_active_readback_cuda(graph, setting):
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_active_readback_probe'](*sys.argv[2:])"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, graph, setting],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\nSTDERR:\n" + result.stderr
    data = json.loads(result.stdout.splitlines()[-1])
    diag = result.stderr
    events = re.findall(
        r"device_tree: complete depth=(\d+) applied_levels=(\d+) splits=(\d+) "
        r"growth_readbacks=1 host_tail=0 graph=(\d+) hits=(\d+) misses=(\d+)",
        diag,
    )
    active = re.findall(r"device_tree_active_readback: queued=(\d+) ready=(\d+) leaves=(\d+)", diag)
    assert "device_tree: host fallback" not in diag
    if setting in {"weighted", "renewal"}:
        assert len(events) == len(active) == data["stages"] == 1, diag
        assert int(events[0][2]) > 0, "disabled route must still grow a real whole tree"
        assert tuple(map(int, active[0])) == (0, 0, int(events[0][2]) + 1)
        return
    assert len(events) == len(active) == data["stages"] == 8, diag
    enabled = setting == "enabled"
    for event, row, expected_N in zip(events, active, data["expected_actual_leaves"], strict=True):
        splits = int(event[2])
        queued, ready, leaves = map(int, row)
        assert leaves == splits + 1 == expected_N
        assert queued == int(enabled)
        assert ready == int(enabled and splits > 0)
    assert int(events[4][0]) == 10
    assert int(events[4][1]) == 2, "positive early exhaustion must stop at its two real splitting levels"
    if graph == "off":
        assert all(int(event[3]) == 0 for event in events)
    else:
        assert int(events[2][4]) > 0, "identical consecutive geometry must reuse capture"
