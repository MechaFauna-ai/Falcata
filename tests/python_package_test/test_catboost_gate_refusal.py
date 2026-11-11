"""The categorical refusal gate must reject unrelated conversion failures."""

import runpy
from pathlib import Path

import pytest

import falcata.importers


def test_catboost_gate_rejects_unrelated_valueerror(monkeypatch):
    pytest.importorskip("catboost")
    original = falcata.importers.from_catboost

    def importer(model, *args, **kwargs):
        if hasattr(model, "get_cat_feature_indices") and model.get_cat_feature_indices():
            raise ValueError("unrelated conversion failure")
        return original(model, *args, **kwargs)

    monkeypatch.setattr(falcata.importers, "from_catboost", importer)
    with pytest.raises(SystemExit) as exc:  # noqa: PT011 - check exit status
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "gates" / "import_catboost.py"))
    assert exc.value.code == 1
