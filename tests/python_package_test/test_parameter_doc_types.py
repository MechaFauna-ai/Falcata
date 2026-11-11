"""Fallback documentation types retain their container structure."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("inner", "expected"),
    [("std::vector<int>", "multi-int"), ("std::vector<std::vector<int>>", "multi-multi-int")],
)
def test_container_documentation_type(inner, expected, tmp_path):
    path = Path(__file__).resolve().parents[2] / ".ci" / "parameter-generator.py"
    spec = importlib.util.spec_from_file_location("parameter_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "Parameters.rst"
    output.write_text(".. start params list\n\n\n\n.. end params list")
    param = {"name": ["x"], "default": ["{}"], "inner_type": [inner], "desc": []}
    module.gen_parameter_description([("Test", 1)], [[param]], output)
    assert f"type = {expected}" in output.read_text()
