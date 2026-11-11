"""A hung telemetry query must not bypass the guard's bounded wait."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def test_gpu_guard_handles_query_timeout(monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "gates" / "gpu_guard.py"
    spec = importlib.util.spec_from_file_location("gpu_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def hung_query(*args, timeout=None, **kwargs):
        assert timeout is not None, "unbounded nvidia-smi query"
        assert timeout > 0
        raise subprocess.TimeoutExpired("nvidia-smi", timeout)

    monkeypatch.setattr(subprocess, "run", hung_query)
    monkeypatch.setattr(sys, "argv", ["gpu_guard"])
    # the default device: the guard refuses an ambiguous CUDA_VISIBLE_DEVICES before querying
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit) as exc:  # noqa: PT011 - exit status is the contract
        module.main()
    assert exc.value.code == 2
    assert "cannot query nvidia-smi" in capsys.readouterr().out
