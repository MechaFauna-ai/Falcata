"""Reject misspelled fixture names before writing any artifacts."""

import importlib.util
from pathlib import Path

import pytest


def test_unknown_freeze_selection(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "gates" / "falb_compat.py"
    spec = importlib.util.spec_from_file_location("fixture_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FIXTURE_DIR", tmp_path)
    with pytest.raises(ValueError, match="regresion"):
        module.freeze(["regresion"])
    assert list(tmp_path.iterdir()) == []
