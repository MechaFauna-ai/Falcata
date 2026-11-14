"""A rejected add_valid leaves the Booster as it was.

Booster::AddValidData registered the validation Dataset and its metrics before GBDT checked
it. When GBDT rejected it ("different bin mappers"), the Booster still listed it, so a later
reset_parameter that changes the objective rebuilt metrics on that Dataset after Python had
freed it (heap-use-after-free under ASan), and the Booster and GBDT disagreed on the number of
validation sets.
"""

import numpy as np
import pytest

import falcata as flc


def test_rejected_valid_set_is_not_kept():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0]
    booster = flc.Booster({"objective": "regression", "verbose": -1}, flc.Dataset(X, y))
    booster.update()
    X_other = rng.normal(size=(50, 5))
    with pytest.raises(flc.basic.FalcataError, match="different bin mappers"):
        booster.add_valid(flc.Dataset(X_other, X_other[:, 0]), "bad")
    booster.reset_parameter({"objective": "huber"})
    booster.update()
    assert np.all(np.isfinite(booster.predict(X)))
    assert booster.eval_valid() == []
