# coding: utf-8
"""CMake must honour an explicit CUDA architecture even when it equals nvcc's default target."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[2]
NVCC = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None
    or shutil.which("ninja") is None
    or NVCC is None
    or not (SOURCE_DIR / "external_libs" / "fmt" / "CMakeLists.txt").exists(),
    reason="needs cmake, ninja, nvcc and a source checkout with submodules",
)


def _configure(build_dir, *args, cudaarchs=None):
    env = {k: v for k, v in os.environ.items() if k != "CUDAARCHS"}
    if cudaarchs is not None:
        env["CUDAARCHS"] = cudaarchs
    out = subprocess.run(
        [
            "cmake",
            "-S",
            str(SOURCE_DIR),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DUSE_CUDA=ON",
            "-DUSE_CCACHE=OFF",
            f"-DCMAKE_CUDA_COMPILER={NVCC}",
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    archs = re.findall(r"^-- CUDA_ARCHITECTURES: (.*)$", out.stdout, re.MULTILINE)
    assert len(archs) == 1, out.stdout
    return archs[0]


def test_explicit_cuda_architecture_matching_nvcc_default_is_honoured(tmp_path):
    default = _configure(tmp_path / "default")
    assert default != "75"
    assert _configure(tmp_path / "default") == default
    assert _configure(tmp_path / "default", "-DCMAKE_CUDA_ARCHITECTURES=75") == "75"
    assert _configure(tmp_path / "explicit", "-DCMAKE_CUDA_ARCHITECTURES=75") == "75"
    assert _configure(tmp_path / "explicit") == "75"
    assert _configure(tmp_path / "env", cudaarchs="75") == "75"
