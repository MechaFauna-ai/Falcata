"""Categorical prediction must not convert out-of-range doubles to int."""

import shutil
import subprocess
from pathlib import Path

import pytest

from falcata.basic import _LIB


def test_categorical_value_range(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "categorical_value_range"
    library = Path(_LIB._name).resolve()
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsanitize=float-cast-overflow",
            "-fno-sanitize-recover=all",
            f"-I{root / 'include'}",
            f"-I{root / 'external_libs/fast_double_parser/include'}",
            f"-I{root / 'external_libs/fmt/include'}",
            str(root / "tests/train_regression/categorical_value_range.cpp"),
            str(library),
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
    generated = tmp_path / "generated.cpp"
    code = subprocess.check_output([str(executable), "emit"], text=True)
    generated.write_text(
        "#include <cstdint>\n#include <vector>\n#include <unordered_map>\n#include <limits>\n#include <cmath>\n#include <cassert>\n"
        + code
        + """
int main() {
  for (double value : {2147483647.5, 2147483648.0, -2147483649.0, 1e100, -1e100,
                       std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN(), -1.0}) {
    assert(PredictTree0(&value) == 7.0);
    assert(PredictTree0ByMap({{0, value}}) == 7.0);
  }
  double value = -0.5;
  assert(PredictTree0(&value) == 3.0);
}
"""
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsanitize=float-cast-overflow",
            "-fno-sanitize-recover=all",
            str(generated),
            "-o",
            str(tmp_path / "generated"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "generated")], check=True)
