"""set_leaf_output changes the model, not the scores the Booster keeps for its training and validation data.

This is the documented behaviour (Booster.set_leaf_output, FLC_BoosterSetLeafValue): eval_train / eval_valid keep the
scores computed before the change, while predictions and data added afterwards see the new leaf value.
"""

import numpy as np

import falcata as flc


def _l2(booster, X, y):
    return float(np.mean((booster.predict(X) - y) ** 2))


def test_set_leaf_output_keeps_cached_scores_and_changes_predictions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    y = X[:, 0] + rng.normal(scale=0.1, size=500)
    params = {"objective": "regression", "verbose": -1, "num_leaves": 4, "device_type": "cpu"}
    train = flc.Dataset(X, y, params=params)
    valid = flc.Dataset(X[:100], y[:100], reference=train)
    booster = flc.Booster(params, train)
    booster.add_valid(valid, "valid")
    for _ in range(3):
        booster.update()
    train_before, valid_before = booster.eval_train(), booster.eval_valid()
    for leaf in range(booster.dump_model()["tree_info"][0]["num_leaves"]):
        booster.set_leaf_output(0, leaf, 100.0)

    # the cached scores are not updated
    assert booster.eval_train() == train_before
    assert booster.eval_valid() == valid_before
    # predictions use the new value
    assert _l2(booster, X, y) > 1000 * train_before[0][2]
    # a Dataset added after the change is scored with the current trees
    fresh = flc.Dataset(X[:100], y[:100], reference=train)
    ((_, _, fresh_l2, _),) = booster.eval(fresh, "fresh")
    np.testing.assert_allclose(fresh_l2, _l2(booster, X[:100], y[:100]), rtol=1e-6)
