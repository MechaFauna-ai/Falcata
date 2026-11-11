# coding: utf-8
"""CUDA lambdarank must compute the CPU objective's gradients.

The CPU objective weights a pair by |discount(high_rank) - discount(low_rank)| with DCGCalculator::GetDiscount,
1 / log2(2 + rank), the same discount behind the inverse max DCG it normalizes by. Every CUDA lambdarank kernel used
log2(2 + rank) instead, so CUDA trained a different objective. With deterministic=True the CUDA kernels accumulate in
the CPU's pair order; the models must match the CPU's for queries handled by each kernel family (up to 1024 items,
up to 2048, and the globally sorted kernel beyond).
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _data(group):
    rng = np.random.default_rng(0)
    rows = sum(group)
    X = rng.standard_normal((rows, 6))
    relevance = X[:, 0] + 0.5 * X[:, 1] + 0.8 * rng.standard_normal(rows)
    y = np.digitize(relevance, [-1.0, 0.0, 0.8, 1.6]).astype(np.float64)
    return X, y


def _train(device, group, extra):
    X, y = _data(group)
    params = {
        "objective": "lambdarank",
        "device_type": device,
        "deterministic": True,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "seed": 0,
        "num_threads": 8,
        "verbose": -1,
        **extra,
    }
    booster = flc.train(params, flc.Dataset(X, label=y, group=group, params=params), num_boost_round=5)
    return booster.predict(X)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "group",
    [[200] * 40, [1500, 1300, 1700], [2600, 3100]],
    ids=["up_to_1024", "up_to_2048", "sorted"],
)
@pytest.mark.parametrize("extra", [{}, {"lambdarank_norm": False}, {"lambdarank_truncation_level": 10}])
def test_lambdarank_matches_cpu_cuda(group, extra):
    np.testing.assert_allclose(_train("cuda", group, extra), _train("cpu", group, extra), rtol=1e-4, atol=1e-4)
