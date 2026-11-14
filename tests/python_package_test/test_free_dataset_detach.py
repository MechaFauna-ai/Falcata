"""Booster.free_dataset() on a training booster detaches the native booster from the freed Datasets.

free_dataset() dropped the Python Datasets, which frees their native handles, but the native training booster kept
pointers to them: a later reset_parameter() or add_valid() read freed memory (heap-use-after-free under ASan in
ColSampler::SetConfig and Dataset::CheckAlign; a crash or garbage otherwise). add_valid() on a booster without
training data dereferenced a null Dataset.
"""

import gc

import numpy as np
import pytest

import falcata as flc

# device_type=cpu: the Python-level lifecycle is under test
PARAMS = {"objective": "quantile", "alpha": 0.5, "device_type": "cpu", "num_leaves": 7, "verbose": -1}


def _booster(num_updates=2):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(800, 4))
    y = x[:, 0] + rng.normal(size=800) * 0.3
    train = flc.Dataset(x, y)
    booster = flc.Booster(PARAMS, train)
    booster.add_valid(flc.Dataset(x, y, reference=train), "valid")
    for _ in range(num_updates):
        booster.update()
    return booster, x


def test_free_dataset_keeps_the_model():
    booster, x = _booster()
    expected = booster.predict(x)
    booster.free_dataset()
    gc.collect()
    np.testing.assert_array_equal(booster.predict(x), expected)
    assert booster.current_iteration() == 2
    assert booster.num_trees() == 2


def test_free_dataset_without_trees():
    booster, x = _booster(num_updates=0)
    booster.free_dataset()
    gc.collect()
    assert booster.num_trees() == 0


@pytest.mark.parametrize("reset", [{"learning_rate": 0.5}, {"alpha": 0.2}])
def test_reset_parameter_after_free_dataset(reset):
    booster, x = _booster()
    expected = booster.predict(x)
    booster.free_dataset()
    gc.collect()
    booster.reset_parameter(reset)
    np.testing.assert_array_equal(booster.predict(x), expected)


def test_reset_objective_after_free_dataset_is_refused():
    booster, x = _booster()
    expected = booster.predict(x)
    booster.free_dataset()
    gc.collect()
    # changing the objective needs training data, which is gone
    with pytest.raises(flc.basic.FalcataError, match="no training data"):
        booster.reset_parameter({"objective": "huber"})
    np.testing.assert_array_equal(booster.predict(x), expected)


def test_add_valid_after_free_dataset_is_refused():
    booster, x = _booster()
    reference = flc.Dataset(x, x[:, 0], free_raw_data=False).construct()
    booster.free_dataset()
    gc.collect()
    with pytest.raises(flc.basic.FalcataError, match="Cannot add validation data to a Booster without training data"):
        booster.add_valid(flc.Dataset(x, x[:, 0], reference=reference), "valid")
