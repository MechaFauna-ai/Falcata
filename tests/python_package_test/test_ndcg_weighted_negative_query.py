"""Weighted NDCG weights an all-negative query by its query weight.

A query whose labels are all 0 has NDCG 1 by convention. The weighted branch added a
plain ``1.0`` for it, while the denominator was the sum of the query weights, so weighted
NDCG could exceed 1 and constant weights changed the metric.
"""

import numpy as np
import pytest

import falcata as flc


def _ndcg(weight):
    X = np.arange(12.0).reshape(6, 2)
    y = np.array([0, 0, 0, 2, 1, 0], dtype=float)  # the first query has only 0 labels
    ds = flc.Dataset(X, y, group=[3, 3], weight=weight, params={"verbose": -1, "min_data_in_bin": 1})
    evals = {}
    flc.train(
        {"objective": "lambdarank", "metric": "ndcg", "eval_at": [1, 3], "min_data_in_leaf": 1, "verbose": -1},
        ds,
        num_boost_round=1,
        valid_sets=[ds],
        callbacks=[flc.record_evaluation(evals)],
    )
    return {k: v[-1] for k, v in evals["training"].items()}


@pytest.mark.parametrize("w", [0.25, 4.0])
def test_constant_weights_do_not_change_ndcg(w):
    unweighted = _ndcg(None)
    weighted = _ndcg(np.full(6, w))
    for name, value in unweighted.items():
        assert weighted[name] == pytest.approx(value)
        assert weighted[name] <= 1.0


def test_weight_of_negative_query_counts():
    # query 0 (all-negative, NDCG 1) has weight 3, query 1 has weight 1
    weighted = _ndcg(np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0]))
    unweighted = _ndcg(None)
    for name, value in unweighted.items():
        ndcg_q1 = 2 * value - 1  # unweighted is the plain mean of 1 and ndcg_q1
        assert weighted[name] == pytest.approx((3 * 1.0 + ndcg_q1) / 4)
