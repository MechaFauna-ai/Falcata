"""The GPU guard checks the selected device rather than physical row zero."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def guard():
    path = Path(__file__).resolve().parents[1] / "gates" / "gpu_guard.py"
    spec = importlib.util.spec_from_file_location("gpu_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_checks_visible_uuid(guard, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbbbbbb")

    def query(command, **kwargs):
        if "--query-gpu=uuid" in command:
            rows = "GPU-aaaaaaaa-1111-2222-3333-444444444444\nGPU-bbbbbbbb-1111-2222-3333-444444444444\n"
        else:
            rows = "9000, 0\n" if "--id=GPU-bbbbbbbb-1111-2222-3333-444444444444" in command else "100, 100\n9000, 0\n"
        return subprocess.CompletedProcess(command, 0, rows, "")

    monkeypatch.setattr(guard.subprocess, "run", query)
    assert guard.gpu_state() == (9000, 0)


def test_ambiguous_guard_selection_fails(guard, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        guard.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 0, "100, 100\n9000, 0\n", "")
    )
    with pytest.raises(ValueError, match="exactly one GPU"):
        guard.gpu_state()


def test_guard_samples_spanning_the_window(guard, monkeypatch):
    import types  # noqa: PLC0415

    clock = [0.0]
    samples = []

    def sample(device):
        samples.append(clock[0])
        return 9000, 0

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(guard, "time", types.SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(guard, "gpu_state", sample)
    monkeypatch.setattr(guard.sys, "argv", ["gpu_guard", "--sustain-sec", "15", "--poll-sec", "20"])
    with pytest.raises(SystemExit) as result:
        guard.main()
    assert result.value.code == 0
    assert samples == [0.0, 20.0]


def test_guard_explicit_hardware_id(guard, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "7000, 10\n", "")

    monkeypatch.setattr(guard.subprocess, "run", query)
    assert guard.gpu_state("1") == (7000, 10)
    assert "--id=1" in commands[0]


@pytest.mark.parametrize("visible", ["1", "MIG-example", ""])
def test_guard_rejects_unsupported_selection(guard, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="MIG|UUID"):
        guard.gpu_state()


def test_guard_rejects_ambiguous_uuid_prefix(guard, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-")
    monkeypatch.setattr(
        guard.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 0, "GPU-one\nGPU-two\n", "")
    )
    with pytest.raises(ValueError, match="exactly one"):
        guard.gpu_state()
