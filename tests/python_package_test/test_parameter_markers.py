"""Parameter documentation generation must preserve malformed source files."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("text", ["intro", ".. start params list\n\nbody", "body\n\n.. end params list"])
def test_missing_parameter_markers_do_not_rewrite(tmp_path, text):
    path = Path(__file__).resolve().parents[2] / ".ci" / "parameter-generator.py"
    spec = importlib.util.spec_from_file_location("parameter_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "Parameters.rst"
    target.write_text(text)
    with pytest.raises(ValueError, match="marker"):
        module.gen_parameter_description([], [], target)
    assert target.read_text() == text
