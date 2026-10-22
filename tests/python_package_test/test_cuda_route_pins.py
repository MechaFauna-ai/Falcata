# coding: utf-8
"""Route pins: which controller grows each golden route configuration, and why it is not the complete device tree.

Under FALCATA_DEBUG=diag the CUDA learner reports every tree the complete device tree did not grow as
`device_tree: fallback (<reason>) route=<route>` (once per route change), and every tree it grew as
`device_tree: complete ...`. The reasons and routes form a fixed vocabulary, FALCATA_DEVICE_TREE_REASONS and
FALCATA_GROWTH_ROUTES in include/Falcata/cuda/cuda_device_tree_reason.hpp; this test reads it from there.

PINS holds the route cells of the full device controller program (its P0 golden corpus: one cell per row of the
refusal inventory that a single GPU can reproduce): the lattice profile and parameters, and the sequence of distinct
(reason, route) pairs the run must report, `("complete", "device_tree")` for a complete device tree. A change that
moves a configuration to another controller (a port to the device, a deleted fallback) updates its row here, in
the same change. The cells train through the lattice's own worker (tests/gates/lattice.py), so the models are the
lattice's.
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


REASONS, ROUTES = _vocabulary()

_FF20 = [1, -1] + [0] * 17 + [1]
_IC = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
_RF = {"boosting": "rf", "bagging_fraction": 0.632, "bagging_freq": 1}
_FORCED = {"feature": 0, "threshold": 0.0, "left": {"feature": 1, "threshold": 0.0}}
_COMPLETE = ("complete", "device_tree")

# id: (profile, params, rounds, extra FALCATA_DEBUG tokens, forced splits, [(reason, route[, prefix]), ...] in
# first-seen order)
PINS = {
    "r01-depth13": (
        "dense",
        {"num_leaves": 8191, "max_depth": 13, "min_data_in_leaf": 1},
        5,
        "",
        None,
        [("depth_gt_12", "prefix")],
    ),
    "r02-memgate": (
        "dense",
        {"num_leaves": 1023, "max_depth": 10, "min_data_in_leaf": 2},
        5,
        "vramfree=1",
        None,
        [("memory_gate", "prefix")],
    ),
    "r03a-budget": (
        "dense",
        {"num_leaves": 151, "max_depth": 10},
        25,
        "",
        None,
        [("selective_budget", "selective_prefix")],
    ),
    "r03b-unbounded": (
        "dense",
        {"num_leaves": 255, "max_depth": -1, "min_data_in_leaf": 5},
        25,
        "",
        None,
        [("selective_unbounded", "selective_prefix")],
    ),
    "r04a-nq-l15": ("dense", {"quant_mode": "none"}, 25, "", None, [("nonquant_l15", "l15_graph")]),
    "r04b-nq-onesync": (
        "dense",
        {"quant_mode": "none", "num_leaves": 4095, "max_depth": 12, "min_data_in_leaf": 1},
        5,
        "",
        None,
        [("nonquant_onesync", "one_sync")],
    ),
    "r05-nq-selective": (
        "dense",
        {"quant_mode": "none", "num_leaves": 31, "max_depth": -1},
        25,
        "",
        None,
        [("nonquant_selective", "selective_one_sync")],
    ),
    "r06a-cat": ("categorical", {"num_leaves": 255, "max_depth": 8}, 25, "", None, [("categorical", "host_two_sync")]),
    "r06b-cat-mc": ("categorical-mc", {}, 25, "", None, [("categorical", "host_two_sync")]),
    "r06c-cat-unbounded": (
        "categorical",
        {"num_leaves": 63, "max_depth": -1},
        25,
        "",
        None,
        [("categorical", "selective_two_sync")],
    ),
    "r07a-bynode-sel": (
        "dense",
        {"feature_fraction_bynode": 0.5, "num_leaves": 31, "max_depth": -1},
        25,
        "",
        None,
        [("bynode_outside_bound", "selective_two_sync")],
    ),
    "r07b-bynode-d13": (
        "dense",
        {"feature_fraction_bynode": 0.5, "num_leaves": 8191, "max_depth": 13, "min_data_in_leaf": 1},
        5,
        "",
        None,
        [("bynode_outside_bound", "host_two_sync")],
    ),
    "r08a-ic": ("dense", {"interaction_constraints": _IC}, 25, "", None, [("interaction_constraints", "classic")]),
    "r08b-ic-bynode": (
        "dense",
        {"interaction_constraints": _IC, "feature_fraction_bynode": 0.6},
        25,
        "",
        None,
        [("interaction_constraints", "classic")],
    ),
    "r09a-rf-g4": (
        "imbalanced",
        {**_RF, "cuda_plan": "auto,forest_group:4"},
        25,
        "",
        None,
        # 25 rounds in groups of 4: the last group's only tree runs alone (forest_single_job_device)
        [("forest_slots", "forest_slots"), _COMPLETE],
    ),
    "r09b-rf-mc": ("rare-mc-b", dict(_RF), 10, "", None, [("forest_slots", "forest_slots")]),
    "r12-vector": (
        "vector-leaf",
        {"num_leaves": 31, "max_depth": -1},
        25,
        "",
        None,
        [("vector_leaves", "selective_two_sync")],
    ),
    "r13a-l1": ("dense", {"lambda_l1": 1.0}, 25, "", None, [("lambda_l1", "host_two_sync")]),
    "r13b-smooth": ("dense", {"path_smooth": 5.0}, 25, "", None, [("path_smooth", "host_two_sync")]),
    "r13c-extra": ("dense", {"extra_trees": True}, 25, "", None, [("extra_trees", "host_two_sync")]),
    "r13d-catrand": ("categorical", {"cat_random_search": 4}, 25, "", None, [("cat_random_search", "host_two_sync")]),
    # sparse-mc trains on the dense row-wise bin (dense_rowdata): the complete device tree, not the sparse layout
    "r14a-sparse": ("sparse-mc", {"num_leaves": 31, "max_depth": 5}, 25, "", None, [_COMPLETE]),
    # max_bin 511 without quantization: the global-memory finder refuses first (compact 16-bit rows are not reached)
    "r14b-compact16": (
        "dense",
        {"max_bin": 511, "feature_fraction": 0.5, "quant_mode": "none"},
        25,
        "",
        None,
        [("nonquant_gt_256_bins", "classic")],
    ),
    "r15-widebin": (
        "missing-mfb0-wide",
        {"quant_mode": "none", "max_bin": 511},
        25,
        "",
        None,
        [("nonquant_gt_256_bins", "classic")],
    ),
    # monotone constraints are refused with quantized gradients on CUDA: the cells train quant_mode=none
    "r17-monotone": (
        "dense",
        {"quant_mode": "none", "monotone_constraints": _FF20},
        25,
        "",
        None,
        [("monotone", "classic")],
    ),
    "r17b-monotone-sel": (
        "dense",
        {"quant_mode": "none", "monotone_constraints": _FF20, "num_leaves": 31, "max_depth": -1},
        25,
        "",
        None,
        [("monotone", "classic")],
    ),
    "r18-cegb": (
        "dense",
        {"cegb_penalty_split": 0.05, "cegb_penalty_feature_coupled": [1.0] * 20, "cegb_tradeoff": 0.5},
        25,
        "",
        None,
        [("cegb", "classic")],
    ),
    "r19-forced": ("dense", {}, 25, "", _FORCED, [("forced_splits", "classic")]),
    "r20-linear-sel": (
        "dense",
        {"linear_tree": True, "num_leaves": 31, "max_depth": -1},
        25,
        "",
        None,
        [("linear_budget", "classic")],
    ),
    "r21-stump": ("dense", {"num_leaves": 2, "max_depth": -1}, 25, "", None, [("num_leaves_le_2", "classic")]),
    # no root tail found on sampledwide / sampled at feature_fraction 0.05 down to 0.005 (P1 notes): every tree complete
    "r22-roottail": (
        "sampledwide",
        {"feature_fraction": 0.05, "num_leaves": 63, "max_depth": 6},
        25,
        "",
        None,
        [_COMPLETE],
    ),
    # the plan's `sampled` profile keeps hitting the prefix graph cache; on `sampledwide` the selective device prefix
    # turns its graphs off after 9 misses and the host's selective loop grows the rest of the trees
    "r24-churn": (
        "sampledwide",
        {"num_leaves": 31, "max_depth": -1},
        40,
        "",
        None,
        [
            ("selective_unbounded", "selective_prefix"),
            ("selective_unbounded", "selective_two_sync", "prefix_graph_churn"),
        ],
    ),
    "r27a-goss": ("dense", {"data_sample_strategy": "goss"}, 25, "", None, [_COMPLETE]),
    "r27b-renew": ("dense", {"objective": "l1", "quant_train_renew_leaf": True}, 25, "", None, [_COMPLETE]),
}

_LINE = re.compile(r"\[hybrid-diag\] device_tree: (?:complete |fallback \((\w+)\) route=(\w+)(?: prefix=(\w+))?)")


def test_route_vocabulary_is_unique_and_pins_use_it():
    assert len(REASONS) == len(set(REASONS))
    assert {"none", "unclassified"} <= set(REASONS)
    assert len(ROUTES) == len(set(ROUTES))
    assert "device_tree" in ROUTES
    for cid, (*_, expected) in PINS.items():
        for reason, route, *prefix in expected:
            assert reason == "complete" or reason in REASONS, (cid, reason)
            assert route in ROUTES, (cid, route)
            assert all(token in REASONS for token in prefix), (cid, prefix)


def _reported(cell_id, tmp_path):
    profile, params, rounds, debug, forced, _ = PINS[cell_id]
    params = dict(params)
    if forced is not None:
        path = tmp_path / "forced.json"
        path.write_text(json.dumps(forced))
        params["forcedsplits_filename"] = str(path)
    sys.path.insert(0, str(_GATES))
    try:
        lattice = importlib.import_module("lattice")
    finally:
        sys.path.pop(0)
    cell = {"id": cell_id, "profile": profile, "params": params, "rounds": rounds}
    env = {**os.environ, "FALCATA_DEBUG": "diag" + ("," + debug if debug else "")}
    proc = subprocess.run(
        [sys.executable, "-c", lattice._WORKER_SOURCE, json.dumps([cell])],
        capture_output=True,
        text=True,
        env=env,
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
    return seen, proc.stderr


@_REQUIRES_CUDA
@pytest.mark.parametrize("cell_id", list(PINS))
def test_route_pin_cuda(cell_id, tmp_path):
    seen, diag = _reported(cell_id, tmp_path)
    assert seen == PINS[cell_id][-1], diag[-3000:]
    assert "unclassified" not in diag
