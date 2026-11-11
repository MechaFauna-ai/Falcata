# coding: utf-8
"""Deterministic CUDA lambdarank on queries longer than 2048 items must match the CPU objective.

Queries above 2048 items take the globally sorted kernel. Its deterministic variant accumulates each item's lambdas
and hessians over the item's pairs in the CPU's (i, j) order; this test pins those values against the CPU objective
(through the trained model) and checks that two runs are identical.
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


def _train(device):
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
    }
    booster = flc.train(params, flc.Dataset(X, label=y, group=group, params=params), num_boost_round=5)
    return booster.predict(X)


@_REQUIRES_CUDA
def test_lambdarank_queries_over_2048_items_deterministic_cuda():
    cuda_pred = _train("cuda")
    np.testing.assert_array_equal(cuda_pred, _train("cuda"))
    np.testing.assert_allclose(cuda_pred, _train("cpu"), rtol=1e-4, atol=1e-4)
