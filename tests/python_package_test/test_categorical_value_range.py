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
