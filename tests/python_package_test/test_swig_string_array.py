"""Compile the standalone SWIG helper's ownership contract."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_string_array_is_not_copyable(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    source = tmp_path / "ownership.cpp"
    source.write_text(
        '#include "swig/StringArray.hpp"\n'
        "#include <type_traits>\n"
        'static_assert(!std::is_copy_constructible<StringArray>::value, "owns buffers");\n'
        'static_assert(!std::is_move_constructible<StringArray>::value, "must not copy on move");\n'
        'static_assert(!std::is_copy_assignable<StringArray>::value, "owns buffers");\n'
        'int main() { StringArray a(1, 4); return a.setitem(0, "abc"); }\n'
    )
    executable = tmp_path / "ownership"
    subprocess.run(
        [compiler, "-std=c++11", "-I", str(Path(__file__).resolve().parents[2]), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
