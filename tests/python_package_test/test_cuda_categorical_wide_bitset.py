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
    num_categories = 3000
    rows = 120000
    cat = rng.integers(0, num_categories, size=rows)
    # a random half of the categories carries the signal, so the best split puts about 1500 of them on one side
    positive = rng.permutation(num_categories)[: num_categories // 2]
    is_positive = np.zeros(num_categories, dtype=bool)
    is_positive[positive] = True
    y = np.where(is_positive[cat], 2.0, -2.0) + 0.1 * rng.standard_normal(rows)
    X = np.column_stack([cat.astype(np.float64), rng.standard_normal(rows)])
    return X, y


def _train(device, plan=None):
    X, y = _data()
    params = {
        "objective": "regression",
        "device_type": device,
        "max_bin": 4000,
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
