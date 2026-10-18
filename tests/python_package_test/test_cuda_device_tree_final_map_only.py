# coding: utf-8
"""UNAPPLIED/UNRUN final-map-only projection; retain existing136+4 and default441 unchanged.

These12 additional cases reuse established full-model/live-score/gap assertions.
The standalone native probe additionally proves absent out-index stores and exact lazy repair.
No CUDA import, pytest collection or execution was performed while preparing this proposal.
"""

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _map_only_probe(graph, setting):
    original = runpy.run_path(str(Path(__file__).with_name("test_cuda_device_tree.py")))
    cases = {
        "full": ("full", ""),
        "history": ("history", ""),
        "history_pdl_off": ("history", ",pdl_level_chain:off,apply_row_batch:off"),
        "end_gate": ("history", ",final_readback_first:off"),
        "weighted": ("weighted", ""),
        "renewal": ("renewal", ""),
    }
    scenario, gate = cases[setting]
    # Keeps exact model bytes, live-vs-CPU score checks and nonzero-ID heterogeneous terminal-gap assertions.
    original["_final_all_map_probe"](graph, scenario, gate)


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize("setting", ["full", "history", "history_pdl_off", "end_gate", "weighted", "renewal"])
def test_device_tree_final_map_only_cuda(graph, setting):
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_map_only_probe'](*sys.argv[2:])"
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
    complete = re.findall(
        r"device_tree: complete depth=(\d+) applied_levels=(\d+) splits=(\d+) "
        r"growth_readbacks=1 host_tail=0 graph=(\d+) hits=(\d+) misses=(\d+)",
        diag,
    )
    mode = re.findall(
        r"device_tree_final_map_only: enabled=(\d+) deferred=(\d+) splits=(\d+) gaps=(\d+) flat=(\d+)", diag
    )
    inline = re.findall(
        r"device_tree_final_all_map: enabled=(\d+) complete=(\d+) final_splits=(\d+) leaves=(\d+)", diag
    )
    assert len(complete) == len(mode) == len(inline) == data["stages"], diag
    assert "device_tree: host fallback" not in diag
    enabled = setting not in {"end_gate", "weighted", "renewal"}
    for event, receipt, mapped in zip(complete, mode, inline, strict=True):
        depth, applied, total = map(int, event[:3])
        queued, deferred, splits, gaps, flat = map(int, receipt)
        assert queued == int(enabled)
        assert deferred == int(enabled and applied == depth and total > 0)
        if deferred:
            assert int(mapped[1]) == 1
            assert splits == int(mapped[2]) > 0
            assert flat >= splits
            assert splits + gaps <= total + 1 - splits
        else:
            assert splits == gaps == flat == 0
    if setting == "full":
        assert tuple(map(int, mode[0][:4])) == (1, 1, 512, 0)
        assert int(inline[0][3]) == 1024
        assert int(complete[0][1]) == 10
    elif setting in {"history", "history_pdl_off"}:
        assert all(int(m[1]) == 1 and int(m[3]) > 0 for m in mode[:3]), "no terminal gaps were deferred"
        assert all(data["nonzero_distinct_terminal_gaps"][:3]), "gap fixture lost its discriminating nonzero ID"
        assert int(complete[3][2]) == 0, "stump inherited a pending final partition"
        assert int(mode[3][1]) == 0, "stump inherited a pending final partition"
        assert 0 < int(complete[4][1]) < 10, "early stop got static-final parity"
        assert int(mode[4][1]) == 0, "early stop got static-final parity"
        assert tuple(map(int, mode[6][:4])) == (1, 1, 1, 0), "odd-depth shallow completion lost its correction"
        assert len(mode) == 10, "data grow/shrink/reset history was shortened"
        if graph == "on":
            assert int(complete[2][4]) > 0, "same geometry must really reuse the graph"
    if graph == "off":
        assert all(int(event[3]) == 0 for event in complete)
