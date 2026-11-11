"""CUDA linear trees keep the constant fallback of a leaf without linear features when a split feature has NaN.

With a NaN in a split feature, the CPU LinearTreeLearner fits each leaf on its complete rows, and a leaf whose path
holds only categorical features counts none: it keeps its leaf output as the constant. The CUDA learner counted
every row of such a leaf and fitted a constant of its own.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


def _leaves(booster):
    def walk(node):
        if "leaf_index" in node:
            yield node
        else:
            yield from walk(node["left_child"])
            yield from walk(node["right_child"])

    # the first tree is constant on every device
    return [leaf for tree in booster.dump_model()["tree_info"][1:] for leaf in walk(tree["tree_structure"])]


def test_cuda_linear_leaf_without_features_keeps_constant_with_nan_splits():
    rng = np.random.default_rng(3)
    n = 4000
    c = rng.integers(0, 4, n).astype(float)
    x = rng.normal(size=n)
    # c < 2 is constant per category (its leaves split on c only); c >= 2 is linear in x
    y = np.where(c < 2, 4.0 * c, 20.0 + 3 * x) + 0.1 * rng.normal(size=n)
    x[rng.random(n) < 0.1] = np.nan
    X = np.column_stack([c, x])
    params = {
        "objective": "regression",
        "linear_tree": True,
        "num_leaves": 4,
        "min_data_in_leaf": 20,
        "device_type": "cuda",
        "deterministic": True,
        "num_threads": 2,
        "verbosity": -1,
    }
    booster = flc.train(params, flc.Dataset(X, label=y, categorical_feature=[0]), num_boost_round=3)
    leaves = _leaves(booster)
    no_features = [leaf for leaf in leaves if not leaf.get("leaf_features")]
    assert no_features, "the data must give leaves whose path holds only the categorical feature"
    assert any(leaf.get("leaf_features") for leaf in leaves), "and leaves with a linear feature that has NaN"
    for leaf in no_features:
        assert leaf["leaf_const"] == leaf["leaf_value"]


def test_cuda_linear_tree_fits_the_reset_training_data():
    """After update(train_set=...), CUDA linear leaves fit the new rows' raw values (and NaN flags), as the CPU."""
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.normal(size=(n, 2))
    x_new = rng.normal(size=(n, 2))
    x_new[rng.random(n) < 0.1, 1] = np.nan
    models = {}
    for device in ["cpu", "cuda"]:
        params = {
            "objective": "regression",
            "linear_tree": True,
            "num_leaves": 4,
            "device_type": device,
            "deterministic": True,
            "num_threads": 2,
            "verbosity": -1,
        }
        data = flc.Dataset(x, label=x[:, 0] + x[:, 1], params=params, free_raw_data=False)
        booster = flc.train(params, data, num_boost_round=2, keep_training_booster=True)
        new = flc.Dataset(x_new, label=3 * x_new[:, 0] - np.nan_to_num(x_new[:, 1]), reference=data, params=params)
        for _ in range(2):
            booster.update(train_set=new)
        models[device] = booster
    np.testing.assert_allclose(models["cuda"].predict(x_new), models["cpu"].predict(x_new), rtol=1e-4, atol=1e-4)
