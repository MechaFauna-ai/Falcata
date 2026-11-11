"""Exercise the full allocated SWIG string capacity."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_string_capacity_includes_terminator(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    source = tmp_path / "capacity.cpp"
    source.write_text(
        "#include <cstring>\n"
        '#include "swig/StringArray.hpp"\n'
        "#include <cassert>\n"
        "int main() {\n"
        "StringArray a(1, 4);\n"
        'assert(a.setitem(0, "abcd") == 0);\n'
        'assert(std::strcmp(a.getitem(0), "abcd") == 0);\n'
        'assert(a.setitem(0, "abcde") == -1);\n'
        'assert(a.setitem(1, "") == -1);\n'
        'StringArray empty(1, 0); assert(empty.setitem(0, "") == 0);\n'
        "}\n"
    )
    executable = tmp_path / "capacity"
    subprocess.run(
        [compiler, "-std=c++11", "-I", str(Path(__file__).resolve().parents[2]), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
