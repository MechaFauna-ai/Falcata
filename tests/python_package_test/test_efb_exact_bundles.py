# coding: utf-8
"""Exclusive feature bundling must not move rows to the other side of a split.

Without bagging every training row is in exactly one leaf of every tree, so
routing the training matrix through the model by its raw values
(``pred_leaf``) must reproduce each leaf's recorded ``leaf_count``. Two
defects in the bundle search broke that whenever features shared a group:

1. a single-value group admitted up to ``total_sample_cnt / 10000`` sampled
   rows on which two of its features are both off their most frequent bin
   (LightGBM's tolerance). Such a row keeps the bin of only one of them, so
   the other feature's splits sent it the wrong way in training;
2. ``BinMapper::FindBin`` sorted the sampled values in place, after which the
   search paired the value of one row with the row id of another when it
   worked out which rows of a column are off the most frequent bin (needed
   when that bin is not the zero bin, e.g. a column that is mostly 0.5 or
   mostly 1). The search then saw the wrong rows and bundled columns that
   really collide.

Each case below fired one of them on the CPU and on CUDA before the fix.
"""

import os

import numpy as np
import pytest

import falcata as lgb

_DEVICES = ["cpu"] + (["cuda"] if os.environ.get("TASK", "") == "cuda" else [])


def _conflict_data(rng, n):
    # two sparse 0/1 columns whose ones overlap on 5 rows: within the old
    # n / 10000 tolerance, so they used to share a single-value group
    x = np.zeros((n, 4), np.float32)
    a = rng.choice(n, n // 50, replace=False)
    rest = np.setdiff1d(np.arange(n), a)
    b = np.concatenate([rng.choice(rest, n // 50 - 5, replace=False), a[:5]])
    x[a, 0] = 1
    x[b, 1] = 1
    x[:, 2] = rng.standard_normal(n)
    x[:, 3] = rng.standard_normal(n)
    y = 5 * x[:, 0] + x[:, 1] + 0.1 * rng.standard_normal(n)
    return x, y


def _pairing_data(rng, n):
    # column 0 is 0 (5%), 0.5 (90%) or 1 (5%): its most frequent bin is not the
    # zero bin. Column 1 is 1 on a quarter of the early rows where column 0 is
    # 1; with the values sorted under their row ids the search saw the late
    # rows of column 0 instead and bundled the two
    x = np.zeros((n, 4), np.float32)
    x[:, 0] = rng.choice([0.0, 0.5, 1.0], size=n, p=[0.05, 0.9, 0.05])
    ones = np.where(x[:, 0] == 1.0)[0]
    early = ones[ones < n // 2]
    x[early[: len(early) // 4], 1] = 1
    x[:, 2] = rng.standard_normal(n)
    x[:, 3] = rng.standard_normal(n)
    y = 3 * x[:, 1] - 2 * (x[:, 0] == 1) + 0.1 * rng.standard_normal(n)
    return x, y


def _one_hot_data(rng, n):
    # truly exclusive columns (a 12-level one-hot block) beside dense noise:
    # these must still bundle
    level = rng.integers(0, 12, size=n)
    x = np.zeros((n, 14), np.float32)
    x[np.arange(n), level] = 1
    x[:, 12] = rng.standard_normal(n)
    x[:, 13] = rng.standard_normal(n)
    y = (level % 3 == 0) * 2.0 + x[:, 12] + 0.1 * rng.standard_normal(n)
    return x, y


def _leaf_counts(node, out):
    if "leaf_index" in node:
        out[node["leaf_index"]] = node.get("leaf_count", 0)
    else:
        _leaf_counts(node["left_child"], out)
        _leaf_counts(node["right_child"], out)
    return out


def _misplaced_rows(booster, x):
    leaves = booster.predict(x, pred_leaf=True)
    moved = 0
    for t, info in enumerate(booster.dump_model()["tree_info"]):
        recorded = _leaf_counts(info["tree_structure"], {})
        routed = np.bincount(leaves[:, t].astype(int), minlength=len(recorded))
        want = np.array([recorded.get(i, 0) for i in range(len(routed))])
        moved += int(np.abs(routed - want).sum()) // 2
    return moved


def _num_groups(dataset, tmp_path):
    path = tmp_path / "dump.txt"
    dataset._dump_text(path)
    with open(path) as f:
        for line in f:
            if line.startswith("num_groups:"):
                return int(line.split(":")[1])
    raise AssertionError("no num_groups in the dataset dump")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("case", ["conflict", "pairing", "one_hot"])
def test_bundled_rows_follow_their_raw_values(case, device, tmp_path):
    rng = np.random.default_rng(0)
    x, y = {"conflict": _conflict_data, "pairing": _pairing_data, "one_hot": _one_hot_data}[case](rng, 100_000)
    params = {
        "objective": "regression",
        "num_leaves": 8,
        "min_data_in_leaf": 1,
        "learning_rate": 0.3,
        "device_type": device,
        "enable_bundle": True,
        "verbose": -1,
        "seed": 1,
    }
    dataset = lgb.Dataset(x, label=y, params=params, free_raw_data=False)
    booster = lgb.train(params, dataset, num_boost_round=10)
    assert _misplaced_rows(booster, x) == 0
    if case == "one_hot":
        assert _num_groups(dataset, tmp_path) < x.shape[1]  # the exclusive columns still share groups
