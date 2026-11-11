"""Categorical airline prep reuses numeric input without mutating its cache."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np


def test_categorical_airline_cache_is_independent(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    spec = importlib.util.spec_from_file_location("common", directory / "common.py")
    common = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "common", common)
    spec.loader.exec_module(common)
    spec = importlib.util.spec_from_file_location("airline_datasets", directory / "datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE_DIR", str(tmp_path))
    arrays = [np.arange(260, dtype=np.float32).reshape(20, 13), np.arange(20, dtype=np.float32)]
    arrays += [arrays[0][:5].copy(), arrays[1][:5].copy()]
    monkeypatch.setattr(module, "_airline_split", lambda: tuple(a.copy() for a in arrays))
    saved = []
    monkeypatch.setattr(module, "save", lambda name, *values: saved.append([a.copy() for a in values]))
    module.prep_airline_cat()
    cache = tmp_path / "airline"
    cache.mkdir()
    names = ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy")
    for name, value in zip(names, arrays, strict=True):
        np.save(cache / name, value)
    monkeypatch.setattr(module, "_airline_split", Mock(side_effect=AssertionError("CSV parsed twice")))
    module.prep_airline_cat()
    for actual, expected in zip(saved[1], saved[0], strict=True):
        np.testing.assert_array_equal(actual, expected)
    for name, original in zip(names, arrays, strict=True):
        np.testing.assert_array_equal(np.load(cache / name), original)
