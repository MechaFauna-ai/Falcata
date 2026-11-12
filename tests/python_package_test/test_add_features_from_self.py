"""Adding a Dataset's features to itself is rejected.

Dataset::AddFeaturesFrom appended the source's per-feature vectors to the target's while
iterating them, so with source == target the iteration read freed memory
(heap-use-after-free under ASan, a crash otherwise).
"""

import numpy as np
import pytest

import falcata as flc


def test_add_features_from_self_is_rejected():
    X = np.random.default_rng(0).normal(size=(30, 3))
    ds = flc.Dataset(X, X[:, 0], free_raw_data=False).construct()
    with pytest.raises(flc.basic.FalcataError, match="Cannot add features from a Dataset to itself"):
        ds.add_features_from(ds)
    # the dataset is unchanged and still trains
    assert ds.num_feature() == 3
    booster = flc.train({"objective": "regression", "verbose": -1}, ds, num_boost_round=2)
    assert booster.num_feature() == 3
