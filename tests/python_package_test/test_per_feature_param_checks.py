"""Per-feature parameters that do not match the training data fail with a clear error.

interaction_constraints indices outside the features were read as features (heap-buffer-overflow in
Dataset::InnerFeatureIndex from ColSampler::GetByNode, or a segfault), and a wrong number of monotone_constraints or
feature_contri values was a bare CHECK.
"""

import numpy as np
import pytest

import falcata as flc

# device_type=cpu: the CPU column sampler read the out-of-range index
BASE = {"objective": "regression", "device_type": "cpu", "num_leaves": 15, "verbose": -1}


def _data():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 4))
    return x, x[:, 0] - x[:, 1] + x[:, 2] * x[:, 3] + rng.normal(size=2000) * 0.2


@pytest.mark.parametrize(
    ("constraints", "feature"), [([[0, 1], [2, 4]], 4), ([[-1, 0]], -1), ([[0, 1], [2, 99999999]], 99999999)]
)
def test_interaction_constraints_index_out_of_range(constraints, feature):
    x, y = _data()
    with pytest.raises(flc.basic.FalcataError, match=f"interaction_constraints refers to feature {feature}, but"):
        flc.train({**BASE, "interaction_constraints": constraints}, flc.Dataset(x, y), num_boost_round=2)


def test_interaction_constraints_index_out_of_range_on_reset():
    x, y = _data()
    booster = flc.Booster(BASE, flc.Dataset(x, y))
    with pytest.raises(flc.basic.FalcataError, match="interaction_constraints refers to feature 7, but"):
        booster.reset_parameter({"interaction_constraints": [[0, 7]]})


@pytest.mark.parametrize("param", ["monotone_constraints", "feature_contri"])
@pytest.mark.parametrize("size", [1, 6])
def test_per_feature_parameter_size(param, size):
    x, y = _data()
    values = [1] * size if param == "monotone_constraints" else [1.0] * size
    with pytest.raises(
        flc.basic.FalcataError, match=f"{param} has {size} values, but the training data has 4 features"
    ):
        flc.train({**BASE, param: values}, flc.Dataset(x, y), num_boost_round=2)


def test_valid_per_feature_parameters_still_train():
    x, y = _data()
    params = {
        **BASE,
        "interaction_constraints": [[0, 1], [2, 3]],
        "monotone_constraints": [1, -1, 0, 0],
        "feature_contri": [1.0, 1.0, 0.5, 0.5],
    }
    booster = flc.train(params, flc.Dataset(x, y), num_boost_round=3)
    assert np.isfinite(booster.predict(x)).all()
