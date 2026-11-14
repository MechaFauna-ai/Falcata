"""reset_parameter(interaction_constraints=...) takes effect on the trees grown after the reset.

The column sampler copied the constraints once, at construction, so a reset that added, changed or removed them was
ignored: later trees kept the constraints the booster was created with.
"""

import numpy as np
import pytest

import falcata as flc

# device_type=cpu: the CPU column sampler is under test
BASE = {
    "objective": "regression",
    "device_type": "cpu",
    "deterministic": True,
    "num_threads": 1,
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


@pytest.mark.parametrize(
    ("before", "after"),
    [(None, NO_INTERACTIONS), (NO_INTERACTIONS, [[0, 1, 2, 3]]), ([[0, 1], [2, 3]], NO_INTERACTIONS)],
)
def test_reset_interaction_constraints_matches_continued_training(before, after):
    x, y = _data()
    params = dict(BASE) if before is None else {**BASE, "interaction_constraints": before}
    booster = flc.train(params, flc.Dataset(x, y, free_raw_data=False), num_boost_round=3, keep_training_booster=True)
    initial = booster.model_to_string()
    booster.reset_parameter({"interaction_constraints": after})
    for _ in range(4):
        booster.update()
    expected = flc.train(
        {**BASE, "interaction_constraints": after},
        flc.Dataset(x, y),
        num_boost_round=4,
        init_model=flc.Booster(model_str=initial),
    )
    np.testing.assert_allclose(booster.predict(x), expected.predict(x), rtol=0, atol=1e-12)
    if after == NO_INTERACTIONS:
        assert _paths_mixing_features(booster, start=3) == 0
