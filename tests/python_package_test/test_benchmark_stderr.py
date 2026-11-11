"""Benchmark child logs retain a bounded diagnostic tail."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("exitcode", [0, 7])
def test_child_stderr_tail(exitcode, monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("benchmark_orchestrate", directory / "orchestrate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    status, tail = module.run_child(
        [sys.executable, "-c", f"import sys;sys.stderr.write('λ'*1048576+'TAIL');sys.exit({exitcode})"],
        timeout=10,
    )
    assert status == ("ok" if exitcode == 0 else "failed")
    assert tail == "λ" * 1996 + "TAIL"
    status, tail = module.run_child(
        [sys.executable, "-c", "import sys,time;sys.stderr.write('before timeout');sys.stderr.flush();time.sleep(5)"],
        timeout=0.2,
    )
    assert status == "timeout"
    assert tail == "before timeout"
