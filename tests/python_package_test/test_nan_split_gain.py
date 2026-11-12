"""A split candidate with a NaN gain is not taken.

Opposite extreme init scores (+-1e300, the clamp for +-inf) make the gradients non-finite,
so a categorical split candidate gets a NaN gain. SplitInfo ranks NaN as kMinScore, so the
candidate could still win the ArgMax over leaves without any candidate, and the
``gain <= 0`` check let NaN through: the tree learner split on it with an invalid threshold
(BinMapper::BinToValue read out of bounds; segfault).
"""

import numpy as np

import falcata as flc


def test_nan_gain_split_is_not_taken():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    X[:, 0] = rng.integers(0, 10, n)
    y = rng.normal(size=n)
    init = np.zeros(n)
    init[5] = -1e300
    init[7] = 1e300
    ds = flc.Dataset(
        X, y, init_score=init, categorical_feature=[0], params={"bin_construct_sample_cnt": 1, "verbose": -1}
    )
    params = {"objective": "regression", "num_leaves": 6, "min_data_in_leaf": 5, "num_threads": 1, "verbose": -1}
    booster = flc.train(params, ds, num_boost_round=3)
    # training stops at a stump instead of splitting on the NaN-gain candidate
    assert booster.num_trees() >= 1
    assert np.all(np.isfinite(booster.predict(X)))
