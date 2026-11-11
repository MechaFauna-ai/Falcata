"""Perf remeasurement stops the whole process group on unresolved timeouts."""

import importlib.util
import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="strict gpuq supervision uses POSIX process groups")


@pytest.fixture
def perf():
    path = Path(__file__).resolve().parents[1] / "gates" / "perf_gate.py"
    spec = importlib.util.spec_from_file_location("perf_deadline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("outcome", ["timeout", "stop"])
def test_remeasurement_stops_on_timeout(perf, monkeypatch, outcome):
    process = Mock(pid=12345, returncode=125)
    process.communicate.side_effect = (
        [subprocess.TimeoutExpired(["lattice"], 7), ("", "")] if outcome == "timeout" else [("", "")]
    )
    spawn = Mock(return_value=process)
    kill = Mock()
    monkeypatch.setattr(perf.subprocess, "Popen", spawn)
    monkeypatch.setattr(perf.os, "killpg", kill)
    monkeypatch.setenv("FALCATA_REMEASURE_TIMEOUT_SEC", "7")

    def old_run(command, **kwargs):
        if outcome == "timeout":
            assert kwargs.get("timeout", 0) > 0
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 125)

    monkeypatch.setattr(perf.subprocess, "run", old_run)
    with pytest.raises(RuntimeError, match="CUDA timeout"):
        perf.remeasure("first")
    assert spawn.call_args.kwargs["start_new_session"] is True
    assert process.communicate.call_args_list[0].kwargs == {"timeout": 7}
    if outcome == "timeout":
        kill.assert_called_once_with(12345, signal.SIGKILL)
        assert process.communicate.call_count == 2
    else:
        kill.assert_not_called()
