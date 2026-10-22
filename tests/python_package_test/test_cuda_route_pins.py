# coding: utf-8
"""Route pins: which controller grows each golden route cell, and why it is not the complete device tree.

Under FALCATA_DEBUG=diag the CUDA learner reports every tree the complete device tree did not grow as
`device_tree: fallback (<reason>) route=<route>[ prefix=<reason>]` (once per route change), and every tree it grew as
`device_tree: complete ...`. The reasons and routes form a fixed vocabulary, FALCATA_DEVICE_TREE_REASONS and
FALCATA_GROWTH_ROUTES in include/Falcata/cuda/cuda_device_tree_reason.hpp; this test reads it from there.

The cells are the lattice's `route/*` golden cells (tests/gates/lattice.py build_cells, md5s in md5_lattice.json),
trained through the lattice's own worker, so the pin and the golden describe the same training. ROUTES_EXPECTED holds,
per cell, the sequence of distinct (reason, route[, prefix]) tuples the run must report, in first-seen order;
`("complete", "device_tree")` is a complete device tree. A change that moves a cell to another controller (a port to
the device, a deleted fallback) updates its row here, in the same change.

No cell reaches compact_16bit or large_bin_layout (r14b is refused by the global-memory finder first), sparse_layout
only non-quantized (r14a; quantized sparse data trains on the dense row bin), and none root_tail_parity
(test_cuda_device_tree_root_tail.py asserts it where its fixture reaches a host tail).
"""

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

_ROOT = Path(__file__).resolve().parents[2]
_HEADER = _ROOT / "include" / "Falcata" / "cuda" / "cuda_device_tree_reason.hpp"
_GATES = _ROOT / "tests" / "gates"


def _vocabulary():
    text = _HEADER.read_text()
    reasons = re.findall(r'X\(k\w+, "(\w+)", "[\w-]+"\)', text)
    routes = re.findall(r'X\(k\w+, "(\w+)"\)', text)
    return reasons, routes


def _lattice():
    sys.path.insert(0, str(_GATES))
    try:
        return importlib.import_module("lattice")
    finally:
        sys.path.pop(0)


REASONS, ROUTES = _vocabulary()
LATTICE = _lattice()
CELLS = {c["id"]: c for c in LATTICE.build_cells() if c["id"].startswith("route/")}

_COMPLETE = ("complete", "device_tree")
_CLASSIC = "classic"

ROUTES_EXPECTED = {
    # any depth: a level wider than the shared select sorts in global scratch (P4)
    "route/r01-depth13": [_COMPLETE],
    # vramfree=1: the per-pair search scratch runs in pair chunks instead of the former memory gate (P4)
    "route/r02-memgate": [_COMPLETE],
    "route/r03a-budget": [("selective_budget", "selective_prefix")],
    "route/r03b-unbounded": [("selective_unbounded", "selective_prefix")],
    "route/r04a-nq-l15": [("nonquant_l15", "l15_graph")],
    "route/r04b-nq-onesync": [("nonquant_onesync", "one_sync")],
    "route/r05-nq-selective": [("nonquant_selective", "selective_one_sync")],
    "route/r06a-cat": [("categorical", "host_two_sync")],
    "route/r06b-cat-mc": [("categorical", "host_two_sync")],
    "route/r06c-cat-unbounded": [("categorical", "selective_two_sync")],
    "route/r07a-bynode-sel": [("bynode_outside_bound", "selective_two_sync")],
    # in the depth bound now that it has no depth limit (P4): the device draws the keyed masks
    "route/r07b-bynode-d13": [_COMPLETE],
    "route/r08a-ic": [("interaction_constraints", _CLASSIC)],
    "route/r08b-ic-bynode": [("interaction_constraints", _CLASSIC)],
    # 25 rounds in groups of 4: the last group's only tree runs alone (forest_single_job_device)
    "route/r09a-rf-g4": [("forest_slots", "forest_slots"), _COMPLETE],
    "route/r09b-rf-mc": [("forest_slots", "forest_slots")],
    "route/r12-vector": [("vector_leaves", "selective_two_sync")],
    "route/r13a-l1": [("lambda_l1", "host_two_sync")],
    "route/r13b-smooth": [("path_smooth", "host_two_sync")],
    "route/r13c-extra": [("extra_trees", "host_two_sync")],
    "route/r13d-catrand": [("cat_random_search", "host_two_sync")],
    # non-quantized sparse-mc keeps the CSR row data
    "route/r14a-sparse": [("sparse_layout", "host_two_sync")],
    # max_bin 511 without quantization: the global-memory finder refuses first (16-bit compact rows are not reached)
    "route/r14b-compact16": [("nonquant_gt_256_bins", _CLASSIC)],
    "route/r15-widebin": [("nonquant_gt_256_bins", _CLASSIC)],
    "route/r17-monotone": [("monotone", _CLASSIC)],
    "route/r17b-monotone-sel": [("monotone", _CLASSIC)],
    "route/r18-cegb": [("cegb", _CLASSIC)],
    "route/r19-forced": [("forced_splits", _CLASSIC)],
    "route/r20-linear-sel": [("linear_budget", _CLASSIC)],
    "route/r21-stump": [("num_leaves_le_2", _CLASSIC)],
    # r22 pins complete device trees (no known shape reaches the root tail, see the lattice comment); root-tail parity
    # is asserted in test_cuda_device_tree_root_tail.py
    "route/r22-roottail": [_COMPLETE],
    # the selective device prefix turns its graphs off after 9 misses; the host's selective loop grows the rest
    "route/r24-churn": [
        ("selective_unbounded", "selective_prefix"),
        ("selective_unbounded", "selective_two_sync", "prefix_graph_churn"),
    ],
    "route/r27a-goss": [_COMPLETE],
    "route/r27b-renew": [_COMPLETE],
}

_LINE = re.compile(r"\[hybrid-diag\] device_tree: (?:complete |fallback \((\w+)\) route=(\w+)(?: prefix=(\w+))?)")


def test_route_pins_cover_the_lattice_route_cells():
    assert sorted(CELLS) == sorted(ROUTES_EXPECTED)


def test_route_vocabulary_is_unique_and_pins_use_it():
    assert len(REASONS) == len(set(REASONS))
    assert {"none", "unclassified"} <= set(REASONS)
    assert len(ROUTES) == len(set(ROUTES))
    assert "device_tree" in ROUTES
    for cid, expected in ROUTES_EXPECTED.items():
        for reason, route, *prefix in expected:
            assert reason == "complete" or reason in REASONS, (cid, reason)
            assert route in ROUTES, (cid, route)
            assert all(token in REASONS for token in prefix), (cid, prefix)


def _reported(cell, tmp_path):
    debug = ",".join(["diag", *(cell.get("debug") or [])])
    proc = subprocess.run(
        [sys.executable, "-c", LATTICE._WORKER_SOURCE, json.dumps([cell])],
        capture_output=True,
        text=True,
        env={**os.environ, "FALCATA_DEBUG": debug},
        check=False,
        cwd=tmp_path,
    )
    records = [json.loads(line[5:]) for line in proc.stdout.splitlines() if line.startswith("CELL ")]
    assert records, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert records[0]["ok"], proc.stdout[-2000:] + proc.stderr[-2000:]
    seen = []
    for reason, route, prefix in _LINE.findall(proc.stderr):
        pair = ((reason, route, prefix) if prefix else (reason, route)) if reason else _COMPLETE
        if pair not in seen:
            seen.append(pair)
    return seen, records[0]["md5"], proc.stderr


@_REQUIRES_CUDA
@pytest.mark.parametrize("cell_id", sorted(ROUTES_EXPECTED))
def test_route_pin_cuda(cell_id, tmp_path):
    seen, md5, diag = _reported(CELLS[cell_id], tmp_path)
    assert seen == ROUTES_EXPECTED[cell_id], diag[-3000:]
    assert "unclassified" not in diag
    # the same training as the lattice's golden
    golden = json.loads(LATTICE.FINGERPRINT_FILE.read_text())["cells"].get(cell_id)
    if golden is not None:
        assert md5 == golden["md5"]
