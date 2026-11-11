"""Perf remeasurement stops the whole process group on unresolved timeouts."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="strict gpuq supervision uses Linux procfs")


@pytest.fixture
def perf():
    path = Path(__file__).resolve().parents[1] / "gates" / "perf_gate.py"
    spec = importlib.util.spec_from_file_location("perf_deadline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("outcome", ["timeout", "stop", "interrupt", "stop-failed"])
def test_remeasurement_stops_on_timeout(perf, monkeypatch, outcome):
    process = Mock(pid=12345, returncode=125)
    process.communicate.side_effect = (
        [subprocess.TimeoutExpired(["lattice"], 7), ("", "")]
        if outcome in ("timeout", "stop-failed")
        else [KeyboardInterrupt(), ("", "")]
        if outcome == "interrupt"
        else [("", "")]
    )
    spawn = Mock(return_value=process)
    kill = Mock()
    monkeypatch.setattr(perf.subprocess, "Popen", spawn)
    monkeypatch.setattr(perf.os, "kill", kill)
    monkeypatch.setattr(perf, "_children_pids", lambda pid: [12346] if pid == 12345 else [])
    if outcome == "stop-failed":
        monkeypatch.setattr(
            perf, "_wait_stopped", Mock(side_effect=perf.UnresolvedCUDATimeout("CUDA timeout stopping worker"))
        )
    else:
        monkeypatch.setattr(perf, "_wait_stopped", lambda pid: None)
    monkeypatch.setenv("FALCATA_REMEASURE_TIMEOUT_SEC", "7")

    expected = (
        pytest.raises(KeyboardInterrupt)
        if outcome == "interrupt"
        else pytest.raises(RuntimeError, match="CUDA timeout")
    )
    with expected:
        perf.remeasure("first")
    assert not spawn.call_args.kwargs.get("start_new_session", False)
    assert process.communicate.call_args_list[0].kwargs == {"timeout": 7}
    if outcome != "stop":
        assert [call.args for call in kill.call_args_list] == [
            (12345, signal.SIGSTOP),
            (12346, signal.SIGSTOP),
            (12346, signal.SIGKILL),
            (12345, signal.SIGKILL),
        ]
        assert process.communicate.call_count == 2
        assert process.communicate.call_args.kwargs == {"timeout": 10}
    else:
        kill.assert_not_called()


@pytest.mark.parametrize("value", ["bad", "0", "-1"])
def test_invalid_remeasurement_deadline_fails_before_execution(perf, monkeypatch, value):
    monkeypatch.setenv("FALCATA_REMEASURE_TIMEOUT_SEC", value)
    spawn = Mock(side_effect=AssertionError("worker launched before configuration validation"))
    monkeypatch.setattr(perf.subprocess, "Popen", spawn)
    assert perf.main(["--gpuq-strict"]) == 2
    spawn.assert_not_called()


def test_remeasurement_stops_real_cpu_descendants(perf, tmp_path, monkeypatch):
    marker = tmp_path / "pids.json"
    script = tmp_path / "lattice.py"
    child_code = f"import time; marker={str(tmp_path)!r}; time.sleep(60)"
    script.write_text(
        "import os,json,subprocess,sys,time\nfrom pathlib import Path\n"
        + f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
        + "Path(os.environ['FALCATA_TEST_PIDS']).write_text(json.dumps({'parent':os.getpid(),'child':child.pid,'pgrp':os.getpgrp()}))\n"
        + "time.sleep(60)\n"
    )
    monkeypatch.setattr(perf, "__file__", str(tmp_path / "perf_gate.py"))
    monkeypatch.setenv("FALCATA_TEST_PIDS", str(marker))
    monkeypatch.setenv("FALCATA_REMEASURE_TIMEOUT_SEC", "2")
    original_spawn = perf.subprocess.Popen

    def ready_spawn(*args, **kwargs):
        process = original_spawn(*args, **kwargs)
        deadline = time.monotonic() + 10
        while not marker.exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                perf._stop_lattice_tree(process)
                pytest.fail("CPU test worker failed to initialize")
            time.sleep(0.01)
        return process

    monkeypatch.setattr(perf.subprocess, "Popen", ready_spawn)
    try:
        with pytest.raises(perf.UnresolvedCUDATimeout, match="CUDA timeout"):
            perf.remeasure("first")
        pids = json.loads(marker.read_text())
        assert pids["pgrp"] == os.getpgrp()
        child_stat = Path(f"/proc/{pids['child']}/stat")
        deadline = time.monotonic() + 2
        while True:
            try:
                state = child_stat.read_text().rsplit(") ", 1)[1].split()[0]
            except FileNotFoundError:  # reaped between polls
                break
            if state in ("Z", "X"):
                break
            assert time.monotonic() < deadline, "worker descendant survived timeout cleanup"
            time.sleep(0.01)
    finally:
        if marker.exists():
            for key in ("child", "parent"):
                pid = json.loads(marker.read_text())[key]
                try:
                    if str(tmp_path).encode() in Path(f"/proc/{pid}/cmdline").read_bytes():
                        os.kill(pid, signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError):
                    pass


@pytest.mark.parametrize("success", [False, True])
def test_remeasurement_result_paths_are_preserved(perf, monkeypatch, success):
    process = Mock(returncode=0 if success else 1)
    process.communicate.return_value = ("", "")

    def spawn(command, **kwargs):
        if success:
            Path(kwargs["env"]["FALCATA_GATES_RESULTS"]).write_text(
                json.dumps({"results": {"first": {"construct_sec": 0.1, "train_sec": 0.2}}})
            )
        return process

    monkeypatch.setattr(perf.subprocess, "Popen", spawn)
    assert perf.remeasure("first") == (0.3 if success else None)
