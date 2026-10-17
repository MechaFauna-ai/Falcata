# coding: utf-8
"""CUDA extra_trees training with many-vs-many categorical splits trains the same model on every run.

The categorical split search draws its random threshold from the task's CUDARandom once per node evaluation. When
every thread of the block made that draw, the threads raced on the shared generator state and on the shared
threshold, so the threshold depended on warp scheduling.
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


def _data(rows, seed):
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, 40, size=(rows, 4))
    num = rng.standard_normal((rows, 4))
    effect = rng.standard_normal((4, 40))
    z = sum(effect[j, cats[:, j]] for j in range(4)) + num[:, 0] + 0.3 * rng.standard_normal(rows)
    y = np.digitize(z, np.quantile(z, [0.25, 0.5, 0.75]))
    return np.column_stack([cats, num]), y


@_REQUIRES_CUDA
def test_categorical_extra_trees_is_run_to_run_deterministic_cuda():
    X, y = _data(12000, seed=3)
    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": 4,
        "extra_trees": True,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "seed": 7,
        "verbose": -1,
    }
    hashes = set()
    for _ in range(4):
        ds = flc.Dataset(X, label=y, categorical_feature=[0, 1, 2, 3], params=params)
        booster = flc.train(params, ds, num_boost_round=20)
        hashes.add(hashlib.md5(booster.model_to_string().encode()).hexdigest())
    assert len(hashes) == 1, f"{len(hashes)} different models from 4 identical runs"
