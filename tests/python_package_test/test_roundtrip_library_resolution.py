"""The independent C API gate must explicitly select an uninstalled library."""

import ast
import builtins
import datetime
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("explicit", [False, True])
def test_uninstalled_library_requires_override(monkeypatch, tmp_path, explicit):
    path = Path(__file__).resolve().parents[1] / "gates" / "falb_roundtrip.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "resolve_lib"
    )
    namespace = {"os": os, "datetime": datetime}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "falcata.libpath":
            raise ImportError("package missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.delenv("FALCATA_LIB", raising=False)
    if explicit:
        library = str(tmp_path / "chosen.so")
        monkeypatch.setenv("FALCATA_LIB", library)
        assert namespace["resolve_lib"]() == library
    else:
        with pytest.raises(RuntimeError, match="FALCATA_LIB"):
            namespace["resolve_lib"]()
