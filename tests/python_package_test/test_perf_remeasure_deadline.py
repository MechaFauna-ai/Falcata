"""The entire lattice process, including startup, has a remeasurement deadline."""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_remeasure_stalled_process_is_bounded(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "gates" / "perf_gate.py"
    spec = importlib.util.spec_from_file_location("perf_deadline", path)
    perf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf)
    run = subprocess.run
    calls = []

    def stalled(command, **kwargs):
        calls.append(command)
        assert kwargs.get("timeout", 0) > 0, "remeasurement has no deadline"
        kwargs["timeout"] = 0.1
        return run([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

    monkeypatch.setattr(perf.subprocess, "run", stalled)
    start = time.monotonic()
    with pytest.raises(perf.UnresolvedCUDATimeout, match="stop GPU job"):
        perf.remeasure("stalled")
    assert time.monotonic() - start < 5
    assert len(calls) == 1
