"""Ranking workspaces must not retain data across objective instances or query sizes."""

import numpy as np
import pytest

import falcata as lgb

from .utils import BuildInfo


@pytest.mark.parametrize("objective", ["lambdarank", "rank_xendcg"])
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("position_bias", [False, True])
def test_ranking_scratch_reuse(objective, threads, position_bias):
    if position_bias and BuildInfo.has_cuda:
        pytest.skip("Positions in learning to rank is not supported in CUDA version yet")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(256, 4))
    labels = rng.integers(0, 4, len(x))
    params = {
        "objective": objective,
        "num_threads": threads,
        "num_leaves": 3,
        "min_data_in_leaf": 3,
        "seed": 7,
        # the CUDA lambdarank gradients accumulate pair lambdas with float atomics, so without it two identical runs
        # differ by about 1e-8 (repeat at the same query size on cuda); the CPU objective is unaffected
        "deterministic": True,
        "verbosity": -1,
    }
    expected = {}
    for query_size in [2, 16, 64, 16, 2, 64]:
        positions = np.tile(np.arange(query_size), len(x) // query_size) if position_bias else None
        data = lgb.Dataset(x, label=labels, group=[query_size] * (len(x) // query_size), position=positions)
        model = lgb.train(params, data, num_boost_round=4)
        actual = model.predict(x, num_threads=threads)
        if query_size in expected:
            np.testing.assert_array_equal(actual, expected[query_size])
        else:
            expected[query_size] = actual
