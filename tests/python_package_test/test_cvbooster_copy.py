"""Copying CV results must preserve the CVBooster wrapper."""

from copy import deepcopy

import numpy as np

import falcata as flc


def test_cvbooster_deepcopy_preserves_wrapper():
    data = np.arange(80, dtype=np.float64).reshape(40, 2)
    booster = flc.train(
        {"num_threads": 2, "verbosity": -1, "min_data_in_leaf": 1},
        flc.Dataset(data, label=np.arange(40)),
        num_boost_round=2,
    )
    original = flc.CVBooster()
    original.boosters = [booster]
    original.best_iteration = 2
    copied = deepcopy(original)
    assert isinstance(copied, flc.CVBooster)
    assert copied.best_iteration == 2
    assert copied.boosters[0] is not booster
    np.testing.assert_array_equal(copied.boosters[0].predict(data), booster.predict(data))
