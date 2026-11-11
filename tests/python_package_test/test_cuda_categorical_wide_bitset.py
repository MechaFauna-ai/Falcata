# coding: utf-8
"""A categorical split with more than 1024 categories on one side must keep all of them on CUDA.

CUDABitsetLen sizes a split's category bitset from the largest category in it: one block per 1024 thresholds finds
its block's maximum, then ReduceBlockMaxLen reduces the block maxima. ReduceBlockMaxLen stored thread 0's partial
maximum (block 0 alone, for up to 1024 blocks) instead of the block reduction, so when the largest category sat in a
later block the bitset was cut short and the categories past its end went to the other side. The classic per-leaf
loop builds these bitsets; it must grow the trees the CPU learner grows.
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _data():
    rng = np.random.default_rng(0)
    # category c appears less often as c grows, so its bin index (bins follow descending frequency) grows with c
    num_categories = 4000
    counts = 20 + (num_categories - np.arange(num_categories)) // 40
    cat = rng.permutation(np.repeat(np.arange(num_categories), counts))
    rows = cat.size
    # categories below 1500 carry a positive signal that weakens as c grows, the rest a negative one: the best split
    # sends categories 0..1499 left, so the thresholds sorted by gradient ratio start with the 1024 lowest bins
    # (block 0 of CUDABitsetLen) and the largest bins come in the next block; the 2500 negative categories exceed
    # max_cat_threshold, so the split from the negative side cannot win
    target = np.where(np.arange(num_categories) < 1500, 3.0 - np.arange(num_categories) / 1000.0, -2.0)
    y = target[cat] + 0.05 * rng.standard_normal(rows)
    X = np.column_stack([cat.astype(np.float64), rng.standard_normal(rows)])
    return X, y


def _train(device, plan=None):
    X, y = _data()
    params = {
        "objective": "regression",
        "device_type": device,
        "max_bin": 4100,
        "max_cat_threshold": 2000,
        "max_cat_to_onehot": 4,
        "min_data_per_group": 1,
        "cat_smooth": 1.0,
        "cat_l2": 1.0,
        "num_leaves": 4,
        "min_data_in_leaf": 20,
        "learning_rate": 0.5,
        "seed": 0,
        "num_threads": 8,
        "verbose": -1,
    }
    if plan is not None:
        params["cuda_plan"] = plan
    ds = flc.Dataset(X, label=y, params=params, categorical_feature=[0])
    booster = flc.train(params, ds, num_boost_round=2)
    return booster, booster.predict(X), X


@_REQUIRES_CUDA
def test_categorical_split_over_1024_categories_cuda():
    cpu_booster, cpu_pred, X = _train("cpu")
    root = cpu_booster.dump_model()["tree_info"][0]["tree_structure"]
    assert root["split_feature"] == 0
    assert len(str(root["threshold"]).split("||")) > 1024, "the root split must hold more than 1024 categories"
    for name, plan in {"classic": "auto,hybrid:off", "auto": "auto"}.items():
        _, pred, _ = _train("cuda", plan)
        np.testing.assert_allclose(pred, cpu_pred, rtol=1e-4, atol=1e-4, err_msg=name)
