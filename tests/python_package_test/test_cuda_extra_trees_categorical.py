# coding: utf-8
"""CUDA extra_trees restricts many-vs-many categorical splits to the drawn threshold above 256 bins too.

A categorical feature with more bins than the split finder's block (256 threads) is searched by the global-memory
kernel, which drew the extra_trees threshold but never applied it: its candidate loops evaluated every threshold,
so extra_trees only advanced the generator and the search stayed exhaustive. The shared-memory kernel (<= 256 bins)
and the CPU learner evaluate only the drawn threshold in each direction.

Stumps on one categorical feature make the drawn threshold visible: the root split's category count is the drawn
position + 1 whenever that candidate is admissible (every candidate is, with the minimum-size limits at 1). Both
constructions below keep max_cat_threshold the binding limit (max_num_cat = 64 for 200 and for 400 categories), so
both kernels draw from the same range with the same seed, and the >256-bin feature must reproduce the <=256-bin
feature's sequence of category counts.
"""

import hashlib
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _data(num_categories, rows_per_category=60, seed=0):
    rng = np.random.default_rng(seed)
    cats = np.repeat(np.arange(num_categories), rows_per_category)
    rng.shuffle(cats)
    effect = rng.standard_normal(num_categories)
    y = effect[cats] + 0.5 * rng.standard_normal(cats.size)
    return cats.reshape(-1, 1).astype(np.float64), y


def _train(num_categories, extra_trees):
    X, y = _data(num_categories)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 2,
        "learning_rate": 0.01,
        "extra_trees": extra_trees,
        "max_cat_threshold": 64,
        "max_cat_to_onehot": 1,
        "min_data_per_group": 1,
        "cat_smooth": 1,
        "min_data_in_leaf": 1,
        "max_bin": 1024,
        "seed": 7,
        "verbose": -1,
    }
    ds = flc.Dataset(X, label=y, categorical_feature=[0], params=params)
    return flc.train(params, ds, num_boost_round=40)


def _root_category_counts(booster):
    counts = []
    for tree in booster.dump_model()["tree_info"]:
        root = tree["tree_structure"]
        assert root.get("decision_type") == "==", "expected a categorical root split"
        counts.append(len(str(root["threshold"]).split("||")))
    return counts


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_categories", [200, 400])
def test_extra_trees_restricts_many_vs_many_categorical_threshold_cuda(num_categories):
    exhaustive = _root_category_counts(_train(num_categories, extra_trees=False))
    drawn = _root_category_counts(_train(num_categories, extra_trees=True))
    # The exhaustive search keeps choosing the same best set size; the drawn threshold varies with the draw.
    assert len(set(exhaustive)) == 1
    assert len(set(drawn)) >= 10, f"extra_trees left the categorical search exhaustive: {drawn}"


@_REQUIRES_CUDA
def test_extra_trees_categorical_above_256_bins_matches_shared_memory_kernel_cuda():
    shared = _root_category_counts(_train(200, extra_trees=True))
    global_memory = _root_category_counts(_train(400, extra_trees=True))
    assert global_memory == shared


@_REQUIRES_CUDA
def test_extra_trees_categorical_above_256_bins_is_deterministic_cuda():
    hashes = {hashlib.md5(_train(400, extra_trees=True).model_to_string().encode()).hexdigest() for _ in range(2)}
    assert len(hashes) == 1
