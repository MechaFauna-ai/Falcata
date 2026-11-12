"""A parameter reset that changes nothing does not change the trees grown after it.

Every reset re-drew the feature_fraction sample (ColSampler::SetConfig called ResetByTree, an extra draw on top of
the one each tree makes) and reseeded every feature's extra_trees generator (HistogramPool::ResetConfig). So a reset
of an unrelated parameter -- including the one Booster.update(fobj=...) makes to switch to a custom objective -- gave
later trees other features and other random thresholds.
"""

import numpy as np
import pytest

import falcata as flc

# device_type=cpu: the CPU column sampler and histogram pool are under test
BASE = {
    "objective": "regression",
    "device_type": "cpu",
    "deterministic": True,
    "num_threads": 1,
    "num_leaves": 7,
    "seed": 1,
    "verbose": -1,
}
RANDOMIZED = [
    {"feature_fraction": 0.5},
    {"feature_fraction_bynode": 0.5},
    {"extra_trees": True},
    {"extra_trees": True, "feature_fraction": 0.7},
]


def _data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 10))
    return x, x[:, 0] + x[:, 1] + rng.normal(size=1000) * 0.3


@pytest.mark.parametrize("params", RANDOMIZED)
@pytest.mark.parametrize("reset", [{"verbose": -1}, {"learning_rate": 0.1}, {"min_gain_to_split": 0.0}])
def test_noop_reset_keeps_the_model(params, reset):
    x, y = _data()

    def boosted(with_reset):
        booster = flc.Booster({**BASE, **params}, flc.Dataset(x, y))
        for i in range(10):
            if with_reset and i == 3:
                booster.reset_parameter(reset)
            booster.update()
        return booster.predict(x)

    np.testing.assert_array_equal(boosted(with_reset=True), boosted(with_reset=False))


@pytest.mark.parametrize("params", RANDOMIZED)
def test_custom_objective_with_builtin_gradients_matches_builtin(params):
    x, y = _data()
    p = {**BASE, **params, "boost_from_average": False}

    def l2(preds, data):
        return preds - data.get_label(), np.ones_like(preds)

    builtin = flc.train(p, flc.Dataset(x, y), num_boost_round=10)
    custom = flc.train({**p, "objective": l2}, flc.Dataset(x, y), num_boost_round=10)
    np.testing.assert_allclose(custom.predict(x), builtin.predict(x), rtol=0, atol=1e-10)


def test_reset_feature_fraction_still_takes_effect():
    x, y = _data()
    booster = flc.Booster({**BASE, "feature_fraction": 0.1}, flc.Dataset(x, y))
    for _ in range(3):
        booster.update()
    booster.reset_parameter({"feature_fraction": 1.0})
    for _ in range(5):
        booster.update()
    used = booster.dump_model()["tree_info"]
    # feature_fraction=0.1 of 10 features leaves one feature per tree; after the reset trees may split on several
    later = {node for tree in used[3:] for node in _split_features(tree["tree_structure"])}
    assert len(later) > 1


def _split_features(node):
    if "split_feature" not in node:
        return set()
    return {node["split_feature"]} | _split_features(node["left_child"]) | _split_features(node["right_child"])
