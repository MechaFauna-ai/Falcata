"""Regression coverage for MAP prefix selection and stable score ties."""

import numpy as np
import pytest

import falcata as lgb


@pytest.mark.parametrize("cutoffs", [[1, 3, 5], [1, 3, 5, 10001], [5, 1]])
def test_map_prefix_matches_stable_sort(cutoffs):
    rng = np.random.default_rng(7)
    n = 10000
    labels = rng.integers(0, 2, n)
    scores = rng.integers(0, 8, n).astype(float)
    data = lgb.Dataset(rng.normal(size=(n, 2)), label=labels, group=[n], init_score=scores)

    def stationary_objective(predictions, dataset):
        return np.zeros_like(predictions), np.ones_like(predictions)

    model = lgb.train(
        {"objective": stationary_objective, "metric": "map", "eval_at": cutoffs, "verbosity": -1, "num_threads": 1},
        data,
        num_boost_round=1,
        valid_sets=[data],
    )
    ranked = labels[np.argsort(-scores, kind="stable")]
    precision = np.cumsum(ranked) / np.arange(1, n + 1)
    for k in cutoffs:
        expected = sum(precision[:k] * ranked[:k]) / min(labels.sum(), k)
        assert model.best_score["training"][f"map@{k}"] == expected
