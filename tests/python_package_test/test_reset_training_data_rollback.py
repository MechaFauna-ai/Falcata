"""A rejected Booster.update(train_set=...) leaves the Booster usable.

Booster::ResetTrainingData replaced its objective and metrics before GBDT checked the new
Dataset. When GBDT rejected it (different bin mappers), GBDT still pointed at the freed
objective, so the next update read freed memory (heap-use-after-free under ASan, a
segfault otherwise).
"""

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, _safe_call


def test_rejected_train_set_keeps_booster_usable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = X[:, 0] + rng.normal(size=50) * 0.1
    params = {"objective": "regression", "verbose": -1, "min_data_in_leaf": 2}
    booster = flc.Booster(params, flc.Dataset(X, y))
    booster.update()
    reference = flc.Booster(params, flc.Dataset(X, y))
    reference.update()

    X_other = rng.normal(size=(50, 6))
    with pytest.raises(flc.basic.FalcataError, match="different bin mappers"):
        booster.update(train_set=flc.Dataset(X_other, X_other[:, 0]))

    # still trains on the original data, exactly like a booster that never saw the reset
    booster.update()
    reference.update()
    np.testing.assert_allclose(booster.predict(X), reference.predict(X))


def test_accepted_train_set_still_switches_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = X[:, 0]
    full = flc.Dataset(X, y, free_raw_data=False).construct()
    subset = full.subset(list(range(30))).construct()
    booster = flc.Booster({"objective": "regression", "verbose": -1, "min_data_in_leaf": 2}, full)
    booster.update()
    booster.update(train_set=subset)
    assert booster.current_iteration() == 2
    assert np.all(np.isfinite(booster.predict(X)))


def test_reset_with_merged_loaded_trees_fails_cleanly():
    # trees merged in from a loaded model cannot score a Dataset; the reset used to fail halfway through the
    # rescoring, after GBDT had switched to the new data (heap corruption on the next update)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 3))
    y = X[:, 0]
    params = {"objective": "regression", "verbose": -1, "min_data_in_leaf": 2}
    full = flc.Dataset(X, y, free_raw_data=False).construct()
    booster = flc.Booster(params, full)
    for _ in range(3):
        booster.update()
    loaded = flc.Booster(model_str=booster.model_to_string())
    _safe_call(_LIB.FLC_BoosterMerge(booster._handle, loaded._handle))
    booster.shuffle_models()  # moves loaded trees into the positions the reset rescores
    with pytest.raises(flc.basic.FalcataError, match="training bin metadata"):
        booster.update(train_set=full.subset(list(range(40))).construct())
    booster.update()
    assert np.all(np.isfinite(booster.predict(X)))
