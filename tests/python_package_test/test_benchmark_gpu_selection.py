"""Benchmark monitoring follows an explicitly selected GPU."""

import ast
import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def monitor_class():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "ResourceMonitor"
    )
    ns = {"os": os, "subprocess": subprocess, "threading": threading}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["ResourceMonitor"]


def test_monitor_reads_selected_gpu(monkeypatch, monitor_class):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-second")
    monkeypatch.delenv("FALCATA_BENCH_GPU_ID", raising=False)

    def query(command, **kwargs):
        text = "10\n" if "--id=GPU-second" in command else "999\n10\n"
        return subprocess.CompletedProcess(command, 0, text, "")

    monkeypatch.setattr(subprocess, "run", query)
    monitor = monitor_class()
    monitor._stop = Mock()
    monitor._stop.is_set.side_effect = [False, True]
    monitor._run()
    assert monitor.gpu_peak_mb == 10


def test_ambiguous_monitor_selection_fails_before_training(monkeypatch, monitor_class):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("FALCATA_BENCH_GPU_ID", raising=False)
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, "999\n10\n", "")))
    monitor = monitor_class()
    with pytest.raises(ValueError, match="Multiple GPUs"):
        monitor.__enter__()
    assert not monitor._thread.is_alive()


def test_explicit_monitor_device_override(monkeypatch, monitor_class):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("FALCATA_BENCH_GPU_ID", "GPU-selected")
    query = Mock(return_value=subprocess.CompletedProcess([], 0, "12\n", ""))
    monkeypatch.setattr(subprocess, "run", query)
    assert monitor_class()._gpu_memory_mb() == 12
    assert "--id=GPU-selected" in query.call_args.args[0]


@pytest.mark.parametrize("visible", ["", "-1"])
def test_cpu_only_monitor_skips_gpu_query(monkeypatch, monitor_class, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.delenv("FALCATA_BENCH_GPU_ID", raising=False)
    query = Mock(side_effect=AssertionError("GPU query with CUDA disabled"))
    monkeypatch.setattr(subprocess, "run", query)
    assert monitor_class()._gpu_memory_mb() == 0
    query.assert_not_called()


def test_unavailable_monitor_memory_is_optional(monkeypatch, monitor_class):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("FALCATA_BENCH_GPU_ID", raising=False)
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, "[N/A]\n", "")))
    monitor = monitor_class()
    monitor._thread = Mock()
    monitor.__enter__()
    monitor._thread.start.assert_called_once()
    assert monitor.gpu_peak_mb == 0


@pytest.mark.parametrize("visible", ["1", "MIG-example"])
def test_ambiguous_or_mig_monitor_selection_is_explicit(monkeypatch, monitor_class, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.delenv("FALCATA_BENCH_GPU_ID", raising=False)
    with pytest.raises(ValueError, match="MIG|UUID"):
        monitor_class()
