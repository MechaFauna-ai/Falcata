# coding: utf-8
"""Serialized models are internally consistent and route-faithful.

Regression coverage for the stale-candidate resurrection bug: the CUDA
best-split sync reductions (SyncBestSplitForLevelKernel and
SyncBestSplitForLeafKernel) defaulted their nothing-found read index to the
smaller role's task-0 output slot. When a leaf's fresh search found no valid
split while its sibling's search was skipped (siblings parked at exactly
min_data_in_leaf skip the find, leaving their output slots stale), the stale
slot's surviving is_valid flag resurrected an EARLIER level's candidate as
the leaf's best split -- always tagged with the lowest real feature index.
The partition then applied a split the finder never scored on those rows:
recorded leaf_count below the min_data floor, leaf_weight != leaf_count, and
descendant leaves reachable by (almost) no rows.

Three invariants, checked per parameterized case:

  1. recorded counts: under L2 the hessian is 1 per row, so every leaf's
     leaf_weight equals its leaf_count and no leaf sits below the floor;
  2. round-trip fidelity: in-process predictions == text reload == binary
     (pickle/falb) reload, bit-exact;
  3. reachability: routing the training matrix through the reloaded model
     reproduces every recorded leaf_count exactly (zero dead leaves).

The data is numerai-shaped (about five distinct small-integer values per
feature, optional NaN), the config budget-limited (num_leaves < 2^max_depth,
so CUDA takes the selective grow-then-prune flow), the target mostly noise
so late trees reach the tiny-gain regime where empty searches sit next to
floor-pinned siblings -- the exact conditions that fired the bug. The
heavier, trigger-tuned version of this check runs nightly as
tests/gates/leaf_integrity.py.
"""

import os
import pickle
import re

import numpy as np
import pytest

import falcata as lgb

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

ROWS, FEATS = 120_000, 200
FLOOR = ROWS // 56  # the production min_data-to-rows ratio


def _data(with_nan, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 5, size=(ROWS, FEATS)).astype(np.float32)
    for j in range(0, FEATS, 7):  # skewed columns pin children at the floor
        col = x[:, j]
        col[rng.random(ROWS) < 0.75] = 2.0
        x[:, j] = col
    if with_nan:
        x[rng.random((ROWS, FEATS)) < 0.15] = np.nan
    signal = np.nan_to_num(x[:, 0]) - np.nan_to_num(x[:, 1])
    y = (0.05 * signal + rng.normal(size=ROWS)).astype(np.float32)
    return x, y


def _params(device, quant_mode):
    params = {
        "objective": "regression",
        "device": device,
        "learning_rate": 0.02,
        "num_leaves": 16384,  # budget-limited: selective grow-then-prune on CUDA
        "max_depth": 15,
        "min_data_in_leaf": FLOOR,
        "feature_fraction": 0.15,
        "max_bin": 255,
        "seed": 7,
        "verbosity": -1,
        "num_threads": 4,
    }
    if device == "cuda":
        params["quant_mode"] = quant_mode
        params["cuda_precision"] = "fp32"
    return params


def _parse_model(text):
    floor = int(re.search(r"\[min_data_in_leaf:\s*(\d+)\]", text).group(1))
    counts, weights = [], []
    for block in text.split("\nTree=")[1:]:
        c = re.search(r"\nleaf_count=([^\n]*)", block)
        w = re.search(r"\nleaf_weight=([^\n]*)", block)
        if not c or not w:
            continue
        counts.append(np.array([int(v) for v in c.group(1).split()]))
        weights.append(np.array([float(v) for v in w.group(1).split()]))
    return floor, counts, weights


def _train(device, quant_mode, with_nan, trees):
    x, y = _data(with_nan)
    params = _params(device, quant_mode)
    booster = lgb.train(params, lgb.Dataset(x, label=y, params=params), num_boost_round=trees)
    return x, booster


def _assert_recorded_counts(text):
    floor, counts, weights = _parse_model(text)
    assert counts, "no leaf counts serialized -- the checks below would pass vacuously"
    for tree_index, (c, w) in enumerate(zip(counts, weights, strict=True)):
        assert c.size == w.size
        bad = np.where(np.abs(w - c) > 0.5)[0]
        assert bad.size == 0, (
            f"tree {tree_index}: leaf_weight != leaf_count at leaves {bad.tolist()[:5]} "
            f"(count {c[bad[:5]].tolist()}, weight {w[bad[:5]].tolist()}) -- "
            "a split was scored on rows the leaf never held"
        )
        below = np.where(c < floor)[0]
        assert below.size == 0, (
            f"tree {tree_index}: leaves {below.tolist()[:5]} recorded below "
            f"min_data_in_leaf={floor}: {c[below[:5]].tolist()}"
        )


def _assert_round_trip(booster, text, x_probe):
    p_live = booster.predict(x_probe)
    p_text = lgb.Booster(model_str=text).predict(x_probe)
    np.testing.assert_array_equal(p_live, p_text, err_msg="text reload changed predictions")
    p_bin = pickle.loads(pickle.dumps(booster)).predict(x_probe)
    np.testing.assert_array_equal(p_live, p_bin, err_msg="binary reload changed predictions")


def _assert_reachability(text, x):
    booster = lgb.Booster(model_str=text)
    _, counts, _ = _parse_model(text)
    occ = [np.zeros(c.size, dtype=np.int64) for c in counts]
    for start in range(0, len(x), 60_000):
        leaves = np.asarray(booster.predict(x[start : start + 60_000], pred_leaf=True))
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        for t in range(leaves.shape[1]):
            np.add.at(occ[t], leaves[:, t].astype(np.int64), 1)
    for tree_index, (c, o) in enumerate(zip(counts, occ, strict=True)):
        drift = np.where(c != o)[0]
        assert drift.size == 0, (
            f"tree {tree_index}: routed occupancy != recorded leaf_count at leaves "
            f"{drift.tolist()[:5]} (recorded {c[drift[:5]].tolist()}, routed "
            f"{o[drift[:5]].tolist()}) -- training routed rows the serialized rules do not"
        )
        assert int((o == 0).sum()) == 0, f"tree {tree_index}: dead (unreachable) leaves"


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant_mode", ["fixedpoint", "none"])
@pytest.mark.parametrize("with_nan", [True, False])
def test_cuda_leaf_integrity_late_boosting(quant_mode, with_nan):
    # enough trees to sit deep in the tiny-gain regime where the stale-slot
    # resurrection fired; the nightly gate runs the heavier trigger-tuned form
    x, booster = _train("cuda", quant_mode, with_nan, trees=1200)
    text = booster.model_to_string()
    _assert_recorded_counts(text)
    _assert_round_trip(booster, text, x[:30_000])
    _assert_reachability(text, x)


def test_cpu_leaf_integrity_round_trip():
    x, booster = _train("cpu", None, True, trees=300)
    text = booster.model_to_string()
    _assert_recorded_counts(text)
    _assert_round_trip(booster, text, x[:30_000])
    _assert_reachability(text, x)
