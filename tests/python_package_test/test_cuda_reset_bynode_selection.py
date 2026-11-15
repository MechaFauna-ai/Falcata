# coding: utf-8
"""reset_parameter of interaction_constraints / feature_fraction_bynode takes effect on the CUDA learner.

The CUDA learner decided once, in Init, whether to select features per node (select_features_by_node_), and sized
its best-split finder for that. A reset that added interaction constraints, or lowered feature_fraction_bynode, on a
booster created without them was ignored on CUDA (#522 fixed the CPU column sampler only): later trees mixed
features the new constraints forbid.
"""

import os

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

BASE = {
    "objective": "regression",
    "device_type": "cuda",
    "deterministic": True,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "verbose": -1,
}
NO_INTERACTIONS = [[0], [1], [2], [3]]


def _data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1500, 4))
    return x, x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3] + rng.normal(size=1500) * 0.1


def _paths_mixing_features(booster, start):
    count = 0

    def walk(node, features):
        nonlocal count
        if "split_feature" in node:
            features = features | {node["split_feature"]}
            count += len(features) > 1
            walk(node["left_child"], features)
            walk(node["right_child"], features)

    for tree in booster.dump_model()["tree_info"][start:]:
        walk(tree["tree_structure"], set())
    return count


def _continued(params_before, reset):
    x, y = _data()
    booster = flc.train(
        params_before, flc.Dataset(x, y, free_raw_data=False), num_boost_round=3, keep_training_booster=True
    )
    initial = booster.model_to_string()
    booster.reset_parameter(reset)
    for _ in range(4):
        booster.update()
    expected = flc.train(
        {**params_before, **reset}, flc.Dataset(x, y), num_boost_round=4, init_model=flc.Booster(model_str=initial)
    )
    return booster, expected, x


@pytest.mark.parametrize(
    ("before", "after"),
    [(None, NO_INTERACTIONS), (NO_INTERACTIONS, [[0, 1, 2, 3]]), ([[0, 1], [2, 3]], NO_INTERACTIONS)],
)
def test_reset_interaction_constraints_cuda(before, after):
    params = dict(BASE) if before is None else {**BASE, "interaction_constraints": before}
    booster, expected, x = _continued(params, {"interaction_constraints": after})
    if after == NO_INTERACTIONS:
        assert _paths_mixing_features(booster, start=3) == 0
    # CUDA float sums: a continuation from the saved model differs in the last digits, which can flip a near-tied
    # split for a few rows; the structural check above is the one an ignored reset fails
    np.testing.assert_allclose(booster.predict(x), expected.predict(x), rtol=1e-3, atol=1e-3)


def test_reset_feature_fraction_bynode_cuda():
    # the per-node sample after the reset continues the column sampler's stream, so it is not that of a fresh
    # continuation; it must still change the trees compared with no reset (on master the reset was ignored)
    x, y = _data()

    def boosted(reset):
        booster = flc.train(BASE, flc.Dataset(x, y, free_raw_data=False), num_boost_round=3, keep_training_booster=True)
        if reset:
            booster.reset_parameter({"feature_fraction_bynode": 0.25})
        for _ in range(4):
            booster.update()
        return booster.predict(x)

    assert not np.array_equal(boosted(reset=True), boosted(reset=False))
