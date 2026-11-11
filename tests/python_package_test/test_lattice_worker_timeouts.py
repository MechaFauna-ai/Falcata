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
    monkeypatch.setattr(lattice, "run_cells", Mock(side_effect=subprocess.TimeoutExpired(["worker"], 1)))
    monkeypatch.setattr(lattice, "RESULTS_FILE", tmp_path / "results.json")
    assert lattice.main() == 125
    assert not lattice.RESULTS_FILE.exists()
