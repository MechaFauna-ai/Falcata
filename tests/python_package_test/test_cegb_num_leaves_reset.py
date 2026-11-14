"""CEGB keeps its per-leaf split cache large enough when reset_parameter raises num_leaves.

CostEfficientGradientBoosting::Init sized splits_per_leaf_ (num_leaves x num_features) only
on the first call. SerialTreeLearner::ResetConfig calls Init again after a reset, so raising
num_leaves left the cache at the old size and DeltaGain wrote past it
(heap-buffer-overflow under ASan).
"""

import numpy as np

import falcata as flc


def test_cegb_after_raising_num_leaves():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(270, 4))
    y = X[:, 0] * 3 + rng.normal(size=270)
    params = {"verbose": -1, "cegb_tradeoff": 0.5, "min_data_in_leaf": 0, "num_threads": 1, "deterministic": True}
    booster = flc.Booster(params, flc.Dataset(X, y, params={"min_data_in_leaf": 0, "verbose": -1}))
    booster.reset_parameter({"num_leaves": 64})
    for _ in range(3):
        booster.update()
    reference = flc.Booster(
        {**params, "num_leaves": 64}, flc.Dataset(X, y, params={"min_data_in_leaf": 0, "verbose": -1})
    )
    for _ in range(3):
        reference.update()
    np.testing.assert_allclose(booster.predict(X), reference.predict(X))
