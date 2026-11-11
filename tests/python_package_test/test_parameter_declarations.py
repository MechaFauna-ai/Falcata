"""Configuration declarations allow ordinary C++ whitespace and comments."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [("int a=1;", "1"), ("int a = 1; // trailing", "1"), ('std::string a="x=y";', '"x=y"')],
)
def test_parameter_declaration(declaration, expected, tmp_path):
    path = Path(__file__).resolve().parents[2] / ".ci" / "parameter-generator.py"
    spec = importlib.util.spec_from_file_location("parameter_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    header = tmp_path / "config.h"
    header.write_text(
        "#pragma region Parameters\n#pragma region Test Parameters\n"
        + declaration
        + "\n#pragma endregion\n#pragma endregion\n"
    )
    _, sections = module.get_parameter_infos(header)
    assert sections[0][0]["name"] == ["a"]
    assert sections[0][0]["default"] == [expected]
