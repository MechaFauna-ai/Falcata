"""Epsilon densification should allocate float32 matrices directly."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import dump_svmlight_file


def test_epsilon_densifies_float32(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    old_common = sys.modules.get("common")
    common_spec = importlib.util.spec_from_file_location("common", directory / "common.py")
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    spec = importlib.util.spec_from_file_location("bench_datasets", directory / "datasets.py")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["common"] = common
        spec.loader.exec_module(module)
    finally:
        if old_common is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = old_common
    source = tmp_path / "epsilon.svm"
    dump_svmlight_file(np.array([[0.0, 1.0], [1.0, 0.0]]), [-1, 1], str(source))
    monkeypatch.setattr(module, "fetch", lambda name: str(source))
    saved = []
    monkeypatch.setattr(module, "save", lambda *args: saved.append(args))
    module.prep_epsilon()
    assert saved[0][1].dtype == np.float32
    assert saved[0][3].dtype == np.float32
