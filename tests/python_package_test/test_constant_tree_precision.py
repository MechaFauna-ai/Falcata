"""Constant-tree generated code must preserve the leaf value."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from falcata.basic import _LIB


def test_constant_tree_precision(tmp_path):
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
    executable = tmp_path / "constant_tree_precision"
    library = Path(_LIB._name).resolve()
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-I{root}",
            f"-I{root / 'include'}",
            f"-I{root / 'external_libs/fast_double_parser/include'}",
            f"-I{root / 'external_libs/fmt/include'}",
            str(root / "tests/train_regression/constant_tree_precision.cpp"),
            str(library),
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
