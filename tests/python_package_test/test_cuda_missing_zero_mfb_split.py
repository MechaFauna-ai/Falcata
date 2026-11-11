# coding: utf-8
"""A zero-as-missing feature whose most frequent value is not zero must split the same way on CUDA as on the CPU.

Rows at such a feature's most frequent bin are stored outside the feature's bin range. The CPU (DenseBin::SplitInner)
and the CUDA row-to-leaf map send them to the most frequent bin's side of the threshold, and send them to the
missing-value default only when the most frequent bin is also the zero bin (missing_is_zero AND mfb_is_zero). The CUDA
bit-vector kernel and its hybrid replica tested (missing_is_zero OR mfb_is_zero), so for a zero-as-missing feature
with a non-zero most frequent value they sent those rows to the missing-value default. The child row sets then
disagreed with the row-to-leaf map and with the CPU learner. Every flow (classic per-leaf loop, host level flow,
device levels, complete device tree) must grow the trees the CPU learner grows.
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")

_PLANS = {
    "classic": "auto,hybrid:off",
    "host": "auto,device_levels:off",
    "device_levels": "auto,device_levels:on,device_tree:off",
    "device_tree": "auto,device_levels:on,device_tree:on",
}


def _data(bundled):
    rng = np.random.default_rng(0)
    rows = 20000
    # (only the bundled column reads the feature through an out-of-range bin window; the single column is a control)
    # mostly 5.0, with some zeros (missing under zero_as_missing) and a spread of other values: the most frequent bin
    # is neither the zero bin nor bin 0
    x = np.full(rows, 5.0, dtype=np.float32)
    r = rng.random(rows)
    x[r < 0.15] = 0.0
    spread = r > 0.7
    x[spread] = rng.uniform(-10.0, 10.0, size=int(spread.sum())).astype(np.float32)
    cols = [x]
    if bundled:
        # a second feature that is non-default only where x is at its most frequent value, so EFB bundles the two
        # and x is read through the bundle's [min_bin, max_bin] window
        w = np.full(rows, 5.0, dtype=np.float32)
        at_mfb = (x == 5.0) & (rng.random(rows) < 0.3)
        w[at_mfb] = rng.uniform(-10.0, 10.0, size=int(at_mfb.sum())).astype(np.float32)
        cols.append(w)
    X = np.column_stack(cols + [rng.standard_normal(rows).astype(np.float32)])
    y = np.where(x == 5.0, 3.0, 0.0) + np.where(x == 0.0, -2.0, 0.0) + 0.2 * x + 0.1 * rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _train(X, y, device, plan=None):
    params = {
        "objective": "regression",
        "device_type": device,
        "zero_as_missing": True,
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.2,
        "min_data_in_leaf": 20,
        "seed": 0,
        "num_threads": 8,
        "verbose": -1,
    }
    if plan is not None:
        params["cuda_plan"] = plan
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=10, keep_training_booster=True)
    trees = booster.trees_to_dataframe()
    structure = trees[["tree_index", "node_depth", "split_feature", "count"]].astype(str).to_numpy().tolist()
    return structure, booster.predict(X), np.array(booster._Booster__inner_predict(data_idx=0))


@_REQUIRES_CUDA
@pytest.mark.parametrize("bundled", [False, True])
def test_missing_zero_with_nonzero_mfb_matches_cpu_cuda(bundled):
    X, y = _data(bundled)
    cpu_structure, cpu_pred, _ = _train(X, y, "cpu")
    for name, plan in _PLANS.items():
        structure, pred, train_score = _train(X, y, "cuda", plan)
        assert structure == cpu_structure, f"{name}: trees (features and leaf counts) differ from the CPU learner's"
        np.testing.assert_allclose(pred, cpu_pred, rtol=1e-4, atol=1e-4, err_msg=name)
        # the training scores come from the row-to-leaf map, the predictions from walking the trees: both must agree
        np.testing.assert_allclose(train_score, pred, rtol=1e-4, atol=1e-4, err_msg=name)
