"""Dataset replacement must retain its previous owner through native validation."""

import weakref

import numpy as np
import pytest

import falcata as lgb
from falcata.basic import _LIB


def _data():
    rows = np.arange(96)
    x = np.column_stack((rows % 23 / 23.0, rows % 7 / 7.0))
    return x, (2 * x[:, 0] - 3 * x[:, 1]).astype(np.float32)


def _dataset(x, y):
    return lgb.Dataset(
        x,
        y,
        params={"num_threads": 2, "min_data_in_bin": 1},
        free_raw_data=False,
    ).construct()


def _objective(preds, data):
    return preds - data.get_label(), np.ones_like(preds)


def _booster(x, y, fobj=None):
    # No external strong reference to this Dataset survives the call.
    model = lgb.Booster(
        {
            "device_type": "cpu",
            "quant_mode": "none",
            "objective": "regression",
            "num_threads": 2,
            "num_leaves": 5,
            "min_data_in_leaf": 3,
            "min_data_in_bin": 1,
            "verbosity": -1,
        },
        train_set=_dataset(x, y),
    )
    for _ in range(2):
        assert not model.update(fobj=fobj)
    return model


def _bookkeeping(model):
    return (
        model.train_set_version,
        model._Booster__init_predictor,
        model._Booster__inner_predict_buffer[0],
    )


def _assert_bookkeeping(model, previous, state):
    assert previous() is not None, "previous Dataset owner absent at native reset boundary"
    assert model.train_set is previous()
    assert model.train_set_version == state[0]
    assert model._Booster__init_predictor is state[1]
    assert model._Booster__inner_predict_buffer[0] is state[2]


def _assert_models_equal(actual, control, x):
    assert actual.dump_model() == control.dump_model()
    assert actual.model_to_string() == control.model_to_string()
    prediction = actual.predict(x, num_threads=2)
    np.testing.assert_array_equal(prediction, control.predict(x, num_threads=2))
    assert np.isfinite(prediction).all()
    assert np.ptp(prediction) > 0


@pytest.mark.parametrize("custom_objective", [False, True])
def test_replacement_retains_owner_at_native_boundary(monkeypatch, custom_objective):
    x, y = _data()
    fobj = _objective if custom_objective else None
    model = _booster(x, y, fobj)
    control = _booster(x, y, fobj)
    retained = [control.train_set]
    _assert_models_equal(model, control, x)
    native_reset = _LIB.FLC_BoosterResetTrainingData

    for offset in (0.25, -0.5, 1.0):
        previous = weakref.ref(model.train_set)
        state = _bookkeeping(model)
        # Independent construction on identical features yields compatible bins
        # without a reference Dataset that would mask the ownership regression.
        replacement = _dataset(x.copy(), y + offset)
        assert replacement.reference is None
        calls = []

        def checked_reset(handle, data_handle, *, previous=previous, state=state, replacement=replacement, calls=calls):
            # On the old wrapper this assertion stops BEFORE any unsafe call.
            _assert_bookkeeping(model, previous, state)
            assert data_handle is replacement._handle
            calls.append(True)
            return native_reset(handle, data_handle)

        with monkeypatch.context() as patch:
            patch.setattr(_LIB, "FLC_BoosterResetTrainingData", checked_reset)

            def checked_objective(preds, data, *, replacement=replacement):
                assert data is replacement
                return _objective(preds, data)

            assert not model.update(train_set=replacement, fobj=checked_objective if fobj else None)
        assert calls == [True]
        assert previous() is None
        assert model.train_set is replacement
        assert model.train_set_version == replacement.version
        assert model._Booster__init_predictor is state[1]
        if fobj is None:
            assert model._Booster__inner_predict_buffer[0] is None
        retained.append(_dataset(x.copy(), y + offset))
        assert not control.update(train_set=retained[-1], fobj=fobj)
        _assert_models_equal(model, control, x)
        assert not model.update(fobj=fobj)
        assert not control.update(fobj=fobj)
        _assert_models_equal(model, control, x)
        # Only the Booster should own its current Dataset on the next reset.
        del replacement, checked_reset, checked_objective


def test_failed_alignment_preserves_python_bookkeeping(monkeypatch):
    x, y = _data()
    model = _booster(x, y, _objective)
    previous = weakref.ref(model.train_set)
    state = _bookkeeping(model)
    replacement = _dataset(x[:, :1].copy(), y)
    replacement.set_label(y + 1)
    assert replacement.version != state[0]
    native_reset = _LIB.FLC_BoosterResetTrainingData
    calls = []

    def checked_reset(handle, data_handle):
        _assert_bookkeeping(model, previous, state)
        calls.append(True)
        return native_reset(handle, data_handle)

    monkeypatch.setattr(_LIB, "FLC_BoosterResetTrainingData", checked_reset)
    with pytest.raises(lgb.basic.FalcataError, match="different bin mappers"):
        model.update(train_set=replacement)
    assert calls == [True]
    _assert_bookkeeping(model, previous, state)
    # Native reset is not transactional; do not continue native training here.


def test_failed_construction_preserves_python_bookkeeping(monkeypatch):
    x, y = _data()
    model = _booster(x, y, _objective)
    previous = weakref.ref(model.train_set)
    state = _bookkeeping(model)
    replacement = lgb.Dataset(x, y)

    def fail_construct():
        raise ValueError("construction failed")

    def unexpected_reset(*args):
        pytest.fail("native reset called after failed construction")

    monkeypatch.setattr(replacement, "construct", fail_construct)
    monkeypatch.setattr(_LIB, "FLC_BoosterResetTrainingData", unexpected_reset)
    with pytest.raises(ValueError, match="construction failed"):
        model.update(train_set=replacement)
    _assert_bookkeeping(model, previous, state)


def test_different_predictor_preserves_python_bookkeeping(monkeypatch):
    x, y = _data()
    model = _booster(x, y)
    previous = weakref.ref(model.train_set)
    state = _bookkeeping(model)
    replacement = _dataset(x.copy(), y)
    replacement._predictor = object()

    def unexpected_reset(*args):
        pytest.fail("native reset called with a different predictor")

    monkeypatch.setattr(_LIB, "FLC_BoosterResetTrainingData", unexpected_reset)
    with pytest.raises(lgb.basic.FalcataError, match="same predictor"):
        model.update(train_set=replacement)
    _assert_bookkeeping(model, previous, state)


@pytest.mark.parametrize("explicit_dataset", [False, True])
def test_same_dataset_version_reset(monkeypatch, explicit_dataset):
    x, y = _data()
    model = _booster(x, y)
    control = _booster(x, y)
    previous = weakref.ref(model.train_set)
    state = _bookkeeping(model)
    native_reset = _LIB.FLC_BoosterResetTrainingData
    calls = []

    def checked_reset(handle, data_handle):
        _assert_bookkeeping(model, previous, state)
        assert data_handle is model.train_set._handle
        calls.append(True)
        return native_reset(handle, data_handle)

    monkeypatch.setattr(_LIB, "FLC_BoosterResetTrainingData", checked_reset)
    for _ in range(2):
        assert not model.update(train_set=model.train_set if explicit_dataset else None)
        assert not control.update()
    assert calls == []
    model.train_set.set_label(y + 0.75)
    control.train_set.set_label(y + 0.75)
    assert model.train_set.version != model.train_set_version
    assert not model.update(train_set=model.train_set if explicit_dataset else None)
    assert calls == [True]
    assert model.train_set_version == model.train_set.version
    # The control's reset must not be intercepted by the model's boundary guard.
    monkeypatch.setattr(_LIB, "FLC_BoosterResetTrainingData", native_reset)
    assert not control.update()
    _assert_models_equal(model, control, x)
