"""C compatibility forwarders must use argument names, never type tokens."""

import importlib.util
from pathlib import Path

import pytest


def generator():
    path = Path(__file__).resolve().parents[2] / ".ci" / "gen-c-api-compat.py"
    spec = importlib.util.spec_from_file_location("compat_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sized_array_argument_names():
    assert generator().arg_names("int n[N], const double values[8]") == ["n", "values"]


@pytest.mark.parametrize(
    "declaration",
    [
        "int",
        "const char*",
        "DatasetHandle",
        "const DatasetHandle",
        "struct ArrowArray*",
        "enum Foo",
        "union U",
        "void (*)(DatasetHandle)",
    ],
)
def test_unnamed_arguments_are_rejected(declaration):
    with pytest.raises(ValueError, match="named"):
        generator().arg_names(declaration)


def test_named_function_pointer_and_void():
    assert generator().arg_names("void (*callback)(const char*, int), void* context") == ["callback", "context"]
    assert generator().arg_names("void") == []
