# coding: utf-8
"""pred_leaf output stays inside its buffer when the Booster gains a tree during the prediction.

Python sizes the output with FLC_BoosterCalcNumPredict, then predicts in a second C API call. With pred_leaf the size
depends on the number of iterations; when another thread trained the Booster between the two calls, a range resolved
by the C++ side (num_iteration <= 0, or one past the end of the model) grew and the prediction wrote past the numpy
buffer (ASan heap-buffer-overflow in GBDT::PredictLeafIndexInRange). The tests train (or roll back) right before the
prediction call on the same thread, which is the interleaving a concurrent trainer produces.
"""

import numpy as np
import pytest

import falcata as flc
import falcata.basic as flc_basic


class _MutateBeforePredict:
    """Proxy for the C library that changes the Booster right before each FLC_BoosterPredictFor* call."""

    def __init__(self, lib, booster, mutate):
        self._lib = lib
        self._booster = booster
        self._mutate = mutate
        self.calls = 0

    def __getattr__(self, name):
        attr = getattr(self._lib, name)
        if not name.startswith("FLC_BoosterPredictFor"):
            return attr

        def mutate_then_predict(*args):
            self._mutate(self._booster)
            self.calls += 1
            return attr(*args)

        return mutate_then_predict


def _train(booster):
    booster.update()


def _booster(num_class=1):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    if num_class == 1:
        y = X[:, 0] + rng.normal(scale=0.1, size=300)
        params = {"objective": "regression"}
    else:
        y = rng.integers(0, num_class, size=300)
        params = {"objective": "multiclass", "num_class": num_class}
    params.update({"verbose": -1, "num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1})
    booster = flc.Booster(params, flc.Dataset(X, y, params={"verbose": -1}))
    for _ in range(3):
        booster.update()
    return booster, X


@pytest.mark.parametrize("num_class", [1, 3])
@pytest.mark.parametrize(("start_iteration", "num_iteration"), [(0, -1), (0, 0), (1, -1), (0, 100), (2, 50)])
def test_pred_leaf_range_is_fixed_before_sizing(monkeypatch, num_class, start_iteration, num_iteration):
    booster, X = _booster(num_class)
    expected_iterations = min(3 - start_iteration, num_iteration) if num_iteration > 0 else 3 - start_iteration
    proxy = _MutateBeforePredict(flc_basic._LIB, booster, _train)
    monkeypatch.setattr(flc_basic, "_LIB", proxy)
    leaves = booster.predict(X[:50], pred_leaf=True, start_iteration=start_iteration, num_iteration=num_iteration)
    monkeypatch.undo()
    assert proxy.calls == 1
    assert booster.current_iteration() == 4
    assert leaves.shape == (50, expected_iterations * num_class)
    # the leaves are those of the iterations that existed when the prediction started
    expected = booster.predict(
        X[:50], pred_leaf=True, start_iteration=start_iteration, num_iteration=expected_iterations
    )
    np.testing.assert_array_equal(leaves, expected)


def test_pred_leaf_empty_range(monkeypatch):
    booster, X = _booster()
    proxy = _MutateBeforePredict(flc_basic._LIB, booster, _train)
    monkeypatch.setattr(flc_basic, "_LIB", proxy)
    leaves = booster.predict(X[:50], pred_leaf=True, start_iteration=3)
    monkeypatch.undo()
    assert proxy.calls == 1
    assert leaves.shape == (50, 0)
    assert leaves.dtype == np.int32
    # without training in between, an empty range still gives no columns (and validates the data)
    assert booster.predict(X[:50], pred_leaf=True, start_iteration=10).shape == (50, 0)
    with pytest.raises(flc.basic.FalcataError, match="number of features"):
        booster.predict(X[:50, :2], pred_leaf=True, start_iteration=10)


def test_pred_leaf_rollback_in_between_is_reported(monkeypatch):
    booster, X = _booster()
    monkeypatch.setattr(flc_basic, "_LIB", _MutateBeforePredict(flc_basic._LIB, booster, flc.Booster.rollback_one_iter))
    # a rollback in between makes the result shorter than its buffer: reported, never written past
    with pytest.raises(ValueError, match="Wrong length for predict results"):
        booster.predict(X[:50], pred_leaf=True)
