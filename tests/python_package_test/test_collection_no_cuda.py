# coding: utf-8
"""Collecting the test suite must not initialize CUDA.

The nightly runs the CUDA suite with ``pytest --forked``: the parent collects every module, then forks a child per
test. A process forked after its parent initialized CUDA cannot use the GPU, so one module that touches CUDA at import
(a device probe in a skip condition, a module-level training run) makes every CUDA test after it fail.

This collects the suite the way the nightly does (TASK=cuda) in a fresh interpreter and checks that the CUDA predictor
probe never ran there, which holds on every build and machine. On Linux it also checks the process maps and open file
descriptors: an initialized CUDA driver has libcuda mapped and /dev/nvidia* open.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]

_COLLECT = r"""
import json
import os
import sys

import pytest

rc = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", *sys.argv[1:]])

from falcata import basic

state = {"rc": int(rc), "probed": basic._CUDA_PREDICTOR_USABLE is not None, "libcuda": [], "nvidia_fds": []}
if sys.platform.startswith("linux"):
    with open("/proc/self/maps") as maps:
        state["libcuda"] = sorted({line.split()[-1] for line in maps if "libcuda.so" in line})
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            state["nvidia_fds"].append(target)
print("COLLECT_STATE " + json.dumps(state))
"""


def test_collecting_the_suite_does_not_initialize_cuda():
    result = subprocess.run(
        [sys.executable, "-c", _COLLECT, str(_TESTS / "python_package_test"), str(_TESTS / "c_api_test")],
        cwd=_TESTS.parent,
        env={**os.environ, "TASK": "cuda"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("COLLECT_STATE ")]
    assert lines, result.stdout[-4000:] + result.stderr[-4000:]
    state = json.loads(lines[-1].removeprefix("COLLECT_STATE "))
    assert state["rc"] == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert not state["probed"], "a test module calls basic._cuda_predictor_usable() at import"
    assert not state["libcuda"], f"collection loaded the CUDA driver: {state['libcuda']}"
    assert not state["nvidia_fds"], f"collection opened NVIDIA devices: {state['nvidia_fds']}"
