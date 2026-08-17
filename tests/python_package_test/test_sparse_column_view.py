# coding: utf-8
"""Per-tree feature sampling must not change which rows a split routes.

Falcata keeps two device representations of the same bins: a row-major matrix
the histograms are built from, and per-column buffers the split-apply kernels
read. They agree on dense columns. They do NOT agree on a column whose bins
came from a sparse bin iterator: that one spells the feature's most-frequent
bin as 0, where the row matrix carries the bin's real index.

With feature_fraction < 1 the tree learner publishes a per-tree column view
gathered from the ROW matrix. Publishing those bytes for a sparse column made
the apply route that feature's rows by the wrong rule, so a split was scored on
rows the leaf never received: leaves below min_data_in_leaf (down to zero
rows), leaf values computed from one row set and applied to another, and --
since the error feeds the next iteration's residual -- leaf values that grow
without bound (6.8M x 3555 numerai at feature_fraction=0.15 reached inf leaf
values by tree ~1500).

The invariant below needs no reference run: under L2 the hessian is exactly 1
per row, so a leaf's sum of hessians IS its row count, and leaf_weight !=
leaf_count means the split finder and the data partition disagree about which
rows the leaf holds.

COVERAGE NOTE: a sparse-encoded column only reaches CUDA from a dataset whose
groups were built with sparse bins elsewhere -- constructing (or loading) with
device_type=cuda forces dense storage today, so these tests cover the sampled
path generally, not that specific trigger. The trigger itself is exercised by
the numerai dataset in the nightly.
"""

import os
import re

import numpy as np
import pytest

import falcata as lgb

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

ROWS, FEATS = 200_000, 120
PARAMS = {
    "objective": "regression",
    "num_leaves": 64,
    "max_depth": 8,
    "min_data_in_leaf": 2_000,
    "feature_fraction": 0.15,  # well below 1, so the per-tree compact view is built
    "max_bin": 5,
    "learning_rate": 0.05,
    "verbosity": -1,
    "num_threads": 4,
    "seed": 0,
}


def _data(seed=0):
    """Low-cardinality features, some of them dominated by a single bin.

    A feature that sits in one bin for most rows is the shape that gets sparse
    storage wherever sparse storage is available at all.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, size=(ROWS, FEATS)).astype(np.float32)
    for j in range(0, FEATS, 8):
        col = X[:, j]
        col[rng.random(ROWS) < 0.9] = 3.0
        X[:, j] = col
    y = rng.normal(size=ROWS)
    return X, y


def _leaf_stats(booster):
    """(leaves, leaves whose recorded weight != count, smallest leaf)."""
    leaves = mismatched = 0
    smallest = np.inf
    for block in booster.model_to_string().split("\nTree=")[1:]:
        counts = re.search(r"\nleaf_count=([^\n]*)", block)
        weights = re.search(r"\nleaf_weight=([^\n]*)", block)
        if not counts or not weights:
            continue
        c = np.array([float(v) for v in counts.group(1).split()])
        w = np.array([float(v) for v in weights.group(1).split()])
        leaves += c.size
        mismatched += int((np.abs(w - c) / np.maximum(c, 1.0) > 1e-6).sum())
        smallest = min(smallest, c.min())
    return leaves, mismatched, smallest


@_REQUIRES_CUDA
def test_sampled_trees_score_the_rows_they_route_cuda():
    X, y = _data()
    params = {**PARAMS, "device_type": "cuda"}
    booster = lgb.train(params, lgb.Dataset(X, label=y, params=params), num_boost_round=30)

    leaves, mismatched, smallest = _leaf_stats(booster)
    assert leaves > 0
    assert mismatched == 0, (
        f"{mismatched} of {leaves} leaves recorded a hessian sum that disagrees with their row "
        "count: the split was scored on rows the leaf did not get"
    )
    assert smallest >= PARAMS["min_data_in_leaf"], (
        f"smallest leaf holds {smallest:.0f} rows against min_data_in_leaf={PARAMS['min_data_in_leaf']}"
    )


@_REQUIRES_CUDA
def test_sampled_trees_match_cpu_cuda():
    """Both devices must build the same trees here, not merely plausible ones:
    a mis-served column changes which splits win."""
    X, y = _data(seed=1)
    cuda_params = {**PARAMS, "device_type": "cuda"}
    cpu_params = {**PARAMS, "device_type": "cpu"}
    on_cuda = lgb.train(cuda_params, lgb.Dataset(X, label=y, params=cuda_params), num_boost_round=15)
    on_cpu = lgb.train(cpu_params, lgb.Dataset(X, label=y, params=cpu_params), num_boost_round=15)

    pred_cuda = on_cuda.predict(X)
    assert np.isfinite(pred_cuda).all()
    np.testing.assert_allclose(pred_cuda, on_cpu.predict(X), rtol=1e-4, atol=1e-5)


def test_sampled_trees_score_the_rows_they_route_cpu():
    """The same invariant on the device that always held it, so a failure above
    is read as a CUDA defect rather than a bad expectation."""
    X, y = _data()
    params = {**PARAMS, "device_type": "cpu"}
    booster = lgb.train(params, lgb.Dataset(X, label=y, params=params), num_boost_round=10)
    leaves, mismatched, smallest = _leaf_stats(booster)
    assert leaves > 0
    assert mismatched == 0
    assert smallest >= PARAMS["min_data_in_leaf"]
