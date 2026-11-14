"""Enabling bagging in two reset_parameter steps reads the current config, not a freed one.

BaggingSampleStrategy::ResetSampleConfig updated its config pointer only when bagging was
active. A reset that left bagging off (e.g. bagging_freq=1 with bagging_fraction still 1)
kept the pointer to the config GBDT::ResetConfig had just freed, and the next reset that
turned bagging on compared against it (heap-use-after-free under ASan).
"""

import numpy as np

import falcata as flc


def test_bagging_enabled_in_two_resets():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] * 3 + rng.normal(size=200) * 0.1
    params = {"verbose": -1, "seed": 1, "deterministic": True, "num_threads": 1}
    booster = flc.Booster(params, flc.Dataset(X, y))
    booster.reset_parameter({"bagging_freq": 1})
    booster.reset_parameter({"bagging_fraction": 0.5})
    for _ in range(5):
        booster.update()

    reference = flc.Booster({**params, "bagging_freq": 1, "bagging_fraction": 0.5}, flc.Dataset(X, y))
    for _ in range(5):
        reference.update()
    np.testing.assert_allclose(booster.predict(X), reference.predict(X))
