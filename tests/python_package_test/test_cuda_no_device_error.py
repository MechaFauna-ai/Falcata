"""device_type=cuda without a visible GPU raises FalcataError instead of aborting the process.

The dataset construction fails with "[CUDA] no CUDA-capable device is detected"; while that exception unwound, the
pinned-host allocator's deallocate called cudaPointerGetAttributes, which fails the same way, and threw from a
destructor: std::terminate.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires a CUDA build of Falcata")

_PROGRAM = """
import numpy as np
import falcata as flc
rng = np.random.default_rng(0)
try:
    flc.train({"device_type": "cuda", "verbose": -1}, flc.Dataset(rng.random((100, 3)), rng.random(100)), 2)
except flc.basic.FalcataError as error:
    print("raised:", error)
"""


def test_cuda_without_a_visible_device_raises():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-c", _PROGRAM], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "raised:" in result.stdout and "no CUDA-capable device" in result.stdout
