"""A stalled CUDA worker aborts the lattice before any later GPU work."""

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def lattice():
    path = Path(__file__).resolve().parents[1] / "gates" / "lattice.py"
    spec = importlib.util.spec_from_file_location("lattice_timeout_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("alone", [False, True])
def test_lattice_worker_has_a_deadline(lattice, monkeypatch, alone):
    calls = []

    def stalled(command, **kwargs):
        calls.append(command)
        assert kwargs.get("timeout", 0) > 0
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(lattice.subprocess, "run", stalled)
    monkeypatch.setattr(lattice, "WORKER_CHUNK", 1)
    cells = [{"id": "first", "debug": ["probe"] if alone else []}, {"id": "later"}]
    with pytest.raises(subprocess.TimeoutExpired):
        lattice.run_cells(cells)
    assert len(calls) == 1


def test_lattice_timeout_returns_stop_status(lattice, monkeypatch, tmp_path):
    monkeypatch.setattr(lattice.sys, "argv", ["lattice", "--check"])
    monkeypatch.setattr(lattice, "build_cells", lambda: [{"id": "first"}])
    monkeypatch.setattr(
        lattice, "run_cells", Mock(side_effect=subprocess.TimeoutExpired(["worker", '[{"id":"first"}]'], 1))
    )
    monkeypatch.setattr(lattice, "RESULTS_FILE", tmp_path / "results.json")
    monkeypatch.setenv("FALCATA_LATTICE_TIMEOUT_SEC", "7")
    assert lattice.main() == 125
    assert lattice.WORKER_TIMEOUT_SEC == 7
    assert not lattice.RESULTS_FILE.exists()


def test_perf_remeasurement_propagates_lattice_stop(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "gates" / "perf_gate.py"
    spec = importlib.util.spec_from_file_location("perf_lattice_stop_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 125))
    with pytest.raises(module.UnresolvedCUDATimeout, match="CUDA timeout"):
        module.remeasure("first")
    monkeypatch.setattr(module, "StrictGPUQ", lambda: None)
    monkeypatch.setattr(module, "evaluate", Mock(side_effect=module.UnresolvedCUDATimeout("CUDA timeout")))
    assert module.main(["--gpuq-strict"]) == 125


@pytest.mark.parametrize("timeout", ["bad", "0", "-1"])
def test_lattice_rejects_invalid_deadline(lattice, monkeypatch, timeout):
    monkeypatch.setenv("FALCATA_LATTICE_TIMEOUT_SEC", timeout)
    monkeypatch.setattr(lattice.sys, "argv", ["lattice", "--check"])
    monkeypatch.setattr(lattice, "build_cells", lambda: [])
    with pytest.raises(SystemExit) as result:
        lattice.main()
    assert result.value.code == 2


def test_list_ignores_unused_deadline_override(lattice, monkeypatch):
    monkeypatch.setenv("FALCATA_LATTICE_TIMEOUT_SEC", "bad")
    monkeypatch.setattr(lattice.sys, "argv", ["lattice", "--list"])
    monkeypatch.setattr(lattice, "build_cells", lambda: [])
    assert lattice.main() == 0
