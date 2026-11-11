# coding: utf-8
"""Deterministic CUDA lambdarank on queries longer than 2048 items must be reproducible and must learn.

Queries above 2048 items take the globally sorted kernel. Its deterministic variant accumulates each item's lambdas
and hessians over the item's pairs in the CPU's (i, j) order, visiting only the pairs that involve the item. Two runs
must give identical models, and the training NDCG must improve. (The CUDA lambdarank kernels use log2(2 + rank) as
the pair discount where the CPU objective uses its reciprocal, so the models are not compared with the CPU's.)
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _data():
    rng = np.random.default_rng(0)
    group = [2600, 3100, 2300]
    rows = sum(group)
    X = rng.standard_normal((rows, 6))
    relevance = X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(rows)
    y = np.digitize(relevance, [-1.0, 0.0, 0.8, 1.6]).astype(np.float64)
    return X, y, group


def _train(device, rounds=5):
    X, y, group = _data()
    params = {
        "objective": "lambdarank",
        "device_type": device,
        "deterministic": True,
        "lambdarank_truncation_level": 40,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "seed": 0,
        "num_threads": 8,
        "verbose": -1,
        "metric": "ndcg",
        "eval_at": [10],
    }
    ds = flc.Dataset(X, label=y, group=group, params=params)
    evals = {}
    booster = flc.train(
        params,
        ds,
        num_boost_round=rounds,
        valid_sets=[ds],
        valid_names=["train"],
        callbacks=[flc.record_evaluation(evals)],
    )
    return booster.predict(X), evals["train"]["ndcg@10"]


@_REQUIRES_CUDA
def test_lambdarank_queries_over_2048_items_deterministic_cuda():
    pred, ndcg = _train("cuda")
    pred_again, _ = _train("cuda")
    np.testing.assert_array_equal(pred, pred_again)
    assert np.all(np.isfinite(pred))
    assert ndcg[-1] > ndcg[0]
