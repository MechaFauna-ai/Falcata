# coding: utf-8
"""A rejected reset_parameter that enabled CEGB leaves no CEGB state behind.

The tree learner made its CEGB object (and sized its per-leaf split cache for the new config's num_leaves,
4 with max_depth=2) before the cegb_penalty_feature_coupled length check failed. The rollback restored the
old config (31 leaves, no CEGB) but kept the object, and training wrote past its cache for every leaf past
the fourth (heap-buffer-overflow, a crash in a plain build).
"""

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError


@pytest.mark.parametrize("enable_cegb_first", [False, True])
def test_rejected_cegb_reset_keeps_training(enable_cegb_first):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 10))
    y = X[:, 0] + X[:, 1] * X[:, 2]
    params = {"objective": "regression", "num_threads": 2, "verbose": -1, "device_type": "cpu"}
    booster = flc.Booster(params, flc.Dataset(X, y))
    booster.update()
    if enable_cegb_first:
        # CEGB on and then off again: the object it made must not outlive it either
        booster.reset_parameter({"max_depth": 2, "cegb_penalty_split": 0.1})
        booster.update()
        booster.reset_parameter({"max_depth": -1, "cegb_penalty_split": 0.0})
    with pytest.raises(FalcataError, match="cegb_penalty_feature_coupled should be the same size"):
        booster.reset_parameter({"max_depth": 2, "cegb_penalty_feature_coupled": [1, 1, 1]})
    for _ in range(3):
        booster.update()
    expected_trees = 5 if enable_cegb_first else 4
    assert booster.current_iteration() == expected_trees
    assert np.isfinite(booster.predict(X)).all()
    assert max(t["num_leaves"] for t in booster.dump_model()["tree_info"]) > 4
