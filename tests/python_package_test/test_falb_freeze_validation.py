"""A rejected fixture recipe cannot overwrite committed fixture contents."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_invalid_constant_recipe_preserves_fixture(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "gates" / "falb_compat.py"
    spec = importlib.util.spec_from_file_location("falb_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FIXTURE_DIR", tmp_path)
    name = "multiclass-constant"
    directory = tmp_path / name
    directory.mkdir()
    model_file = directory / "model.txt"
    model_file.write_text("original fixture")
    text = "\nTree=0\nnum_leaves=2\n"
    model = SimpleNamespace(
        predict=lambda data: np.zeros(len(data)),
        save_model=lambda filename, **kwargs: Path(filename).write_text(text),
        model_to_string=lambda **kwargs: text,
    )
    monkeypatch.setattr(module.flc, "__version__", "test", raising=False)
    monkeypatch.setattr(module.flc, "train", lambda *args, **kwargs: model)
    monkeypatch.setattr(module, "make_data", lambda *args, **kwargs: (np.zeros((2, 1)), np.zeros(2), []))
    with pytest.raises(SystemExit, match="constant trees"):
        module.freeze([name])
    assert model_file.read_text() == "original fixture"
    assert sorted(item.name for item in directory.iterdir()) == ["model.txt"]
