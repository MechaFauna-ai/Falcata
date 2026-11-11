"""Multiclass metric workspaces must preserve every row and weight path."""

import numpy as np
import pytest

import falcata as lgb


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("objective", ["multiclass", "multiclassova", "custom"])
def test_multiclass_metric_buffers(objective, weighted, threads):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(256, 4))
    labels = np.arange(len(x)) % 3
    weights = rng.uniform(0.1, 2, len(x)) if weighted else None
    initial = rng.uniform(0.1, 1, (len(x), 3))
    initial /= initial.sum(axis=1, keepdims=True)
    data = lgb.Dataset(x, label=labels, weight=weights, init_score=initial if objective == "custom" else None)

    def stationary_objective(predictions, dataset):
        return np.zeros_like(predictions), np.ones_like(predictions)

    model = lgb.train(
        {
            "objective": stationary_objective if objective == "custom" else objective,
            "metric": "multi_logloss",
            "num_class": 3,
            "num_threads": threads,
            "num_leaves": 3,
            "verbosity": -1,
        },
        data,
        num_boost_round=3,
        valid_sets=[data],
    )
    probabilities = initial if objective == "custom" else model.predict(x, num_threads=threads)
    losses = -np.log(probabilities[np.arange(len(x)), labels])
    expected = np.average(losses, weights=weights.astype(np.float32).astype(np.float64) if weighted else None)
    assert model.best_score["training"]["multi_logloss"] == pytest.approx(expected, rel=1e-12)
