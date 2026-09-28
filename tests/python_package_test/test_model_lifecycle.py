"""Regressions for safe Booster model replacement and reset failures."""

import numpy as np
import pytest

import falcata as lgb
from falcata import basic


def _trained_model(target_sign: float = 1.0):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(180, 5)).astype(np.float32)
    y = target_sign * (x[:, 0] - 0.4 * x[:, 1])
    params = {"objective": "regression", "verbosity": -1, "num_threads": 1, "device_type": "cpu"}
    model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=4)
    return model, x


def test_failed_text_and_binary_replacement_keep_old_model():
    model, x = _trained_model()
    expected = model.predict(x)

    with pytest.raises(lgb.basic.FalcataError, match=".+"):
        model.model_from_string("not a model")
    np.testing.assert_array_equal(model.predict(x), expected)

    with pytest.raises(lgb.basic.FalcataError, match=".+"):
        model.model_from_binary(b"FALB" + b"\0" * 32)
    np.testing.assert_array_equal(model.predict(x), expected)


def test_model_replacement_invalidates_cached_fil_prediction(monkeypatch):
    model, x = _trained_model()
    replacement, _ = _trained_model(target_sign=-1.0)
    replacement_binary = replacement.model_to_binary()
    model.params["device_type"] = "cuda"
    monkeypatch.setenv("FALCATA_FIL", "1")

    class FakeFilModel:
        is_classifier = False

        def __init__(self, value):
            self.value = value

        def predict(self, data):
            return np.full(data.shape[0], self.value, dtype=np.float64)

    built = []

    def build_fake(**_kwargs):
        result = FakeFilModel(float(len(built) + 1))
        built.append(result)
        return result

    monkeypatch.setattr(basic, "_load_fil_modules", lambda: (object(), object()))
    monkeypatch.setattr(model, "_build_fil_model", build_fake)
    first = model.predict(x)
    np.testing.assert_array_equal(first, np.ones(x.shape[0]))

    model.model_from_binary(replacement_binary)
    model.params["device_type"] = "cuda"
    second = model.predict(x)
    np.testing.assert_array_equal(second, np.full(x.shape[0], 2.0))
    assert len(built) == 2


@pytest.mark.parametrize("populate_config_first", [False, True])
def test_loaded_model_rejects_forced_splits_without_poisoning_config(tmp_path, populate_config_first):
    trained, _ = _trained_model()
    model = lgb.Booster(model_str=trained.model_to_string())

    if populate_config_first:
        model.reset_parameter({"learning_rate": 0.05})
    with pytest.raises(lgb.basic.FalcataError, match="forcedsplits_filename.*training data"):
        model.reset_parameter({"forcedsplits_filename": str(tmp_path / "splits.json")})

    # The rejected setting must not be committed, and a later normal reset
    # should still work after the first reset populated the native config.
    model.reset_parameter({"learning_rate": 0.1})


def test_owned_serialization_frees_buffer_if_copy_fails(monkeypatch):
    model, _ = _trained_model()
    real_free = basic._LIB.FLC_FreeOwnedBuffer
    freed = []

    def free_and_record(pointer):
        freed.append(pointer)
        real_free(pointer)

    monkeypatch.setattr(basic._LIB, "FLC_FreeOwnedBuffer", free_and_record)
    monkeypatch.setattr(
        basic.ctypes,
        "string_at",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("copy failed")),
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        model.model_to_binary()
    assert len(freed) == 1
