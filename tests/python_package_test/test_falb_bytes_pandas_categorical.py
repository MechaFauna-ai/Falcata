"""The FALB bytes API must keep a model's pandas category mapping.

model_to_binary() returned the bare payload and model_from_binary() reset
pandas_categorical to None, so Booster(model_bin=booster.model_to_binary())
silently re-coded categorical DataFrame columns by the *new* frame's
categories and predicted something else. save_model(format="falb") and
pickle kept the mapping; the bytes API now does too.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

import falcata as flc


@pytest.fixture(scope="module")
def booster():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"c": pd.Categorical(rng.choice(["a", "b", "c", "d"], n)), "x": rng.standard_normal(n)})
    y = df["c"].isin(["b", "d"]).astype(float) * 3 + df["x"] * 0.1
    return flc.train({"verbose": -1, "num_threads": 1, "min_data_in_leaf": 5}, flc.Dataset(df, label=y), 5)


def _reordered():
    # categories in a different order (and a subset) than at training time
    return pd.DataFrame({"c": pd.Categorical(["d", "b", "a"], categories=["d", "b", "a"]), "x": [0.0, 0.0, 0.0]})


def test_model_bin_round_trip_keeps_categories(booster):
    restored = flc.Booster(model_bin=booster.model_to_binary())
    assert restored.pandas_categorical == booster.pandas_categorical
    np.testing.assert_array_equal(restored.predict(_reordered()), booster.predict(_reordered()))


def test_model_from_binary_reads_saved_file_bytes(booster, tmp_path):
    path = tmp_path / "m.falb"
    booster.save_model(path)
    restored = flc.Booster(model_str=booster.model_to_string()).model_from_binary(path.read_bytes())
    assert restored.pandas_categorical == booster.pandas_categorical
    np.testing.assert_array_equal(restored.predict(_reordered()), booster.predict(_reordered()))


def test_plain_model_bytes_are_unchanged():
    x = np.random.default_rng(1).standard_normal((200, 2))
    plain = flc.train({"verbose": -1, "num_threads": 1}, flc.Dataset(x, label=x[:, 0]), num_boost_round=2)
    blob = plain.model_to_binary()
    assert b"pandas_categorical" not in blob
    assert flc.Booster(model_bin=blob).pandas_categorical is None


def test_pickle_payload_stays_bare_falb(booster):
    state = booster.__getstate__()
    assert b"pandas_categorical" not in state["_handle"]
    restored = pickle.loads(pickle.dumps(booster))
    np.testing.assert_array_equal(restored.predict(_reordered()), booster.predict(_reordered()))
