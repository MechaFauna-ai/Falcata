"""The if-else model export (task=convert_model) must route values like Tree::Predict.

A numerical split whose missing type is Zero or NaN compared only "is the value
missing?" and dropped the threshold, so every non-missing value took the
non-default branch. A split with an infinite threshold also printed ``inf``,
which does not compile.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from falcata.basic import _LIB


def test_if_else_missing_split(tmp_path):
    if sys.platform != "linux" or os.getenv("TASK") in {"cuda", "gpu", "mpi"}:
        pytest.skip("Native harness requires a Linux CPU build")
    if _LIB.FLC_DatasetDeviceBuilderFree(None) == 0:
        pytest.skip("Private-header harness does not support CUDA library layouts")
    if b"requires a CUDA build" not in _LIB.FLC_GetLastError():
        pytest.skip("Cannot identify the loaded library as a CPU build")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    library = Path(_LIB._name).resolve()
    includes = [
        f"-I{root}",
        f"-I{root / 'include'}",
        f"-I{root / 'external_libs/fast_double_parser/include'}",
        f"-I{root / 'external_libs/fmt/include'}",
    ]
    link = [str(library), f"-Wl,-rpath,{library.parent}"]

    generator = tmp_path / "generate"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            *includes,
            str(root / "tests/train_regression/if_else_missing_split.cpp"),
            *link,
            "-o",
            str(generator),
        ],
        check=True,
    )
    generated = tmp_path / "generated.cpp"
    generated.write_text(subprocess.run([str(generator)], check=True, capture_output=True, text=True).stdout)

    checker = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", *includes, str(generated), *link, "-o", str(checker)], check=True)
    result = subprocess.run([str(checker)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout
