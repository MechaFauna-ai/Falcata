"""A crash found while isolating a batch timeout must keep its reproducer."""

import importlib.util
import subprocess
from pathlib import Path


def test_fuzz_timeout_isolation_keeps_crash(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "gates" / "falb_fuzz.py"
    spec = importlib.util.spec_from_file_location("falb_timeout_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "gate.py"))
    monkeypatch.setattr(module, "BATCH", 2)
    monkeypatch.setattr(module, "replay_crash_corpus", lambda: 0)
    monkeypatch.setattr(module, "build_seed_files", lambda _: [b"seed"])
    monkeypatch.setattr(module, "mutate", lambda *args: b"crash reproducer")
    monkeypatch.setattr(module.sys, "argv", ["fuzz", "--seconds", "1", "--seed", "7"])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 300, output=b"OK first\n")
        return subprocess.CompletedProcess(command, -11 if Path(command[-1]).name == "m1.falb" else 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main() == 1
    assert (tmp_path / "falb_fuzz_crash.falb").read_bytes() == b"crash reproducer"
    output = capsys.readouterr().out
    assert "exit -11" in output
    assert "mutation #1" in output
    assert "environment slowness" not in output
    assert len(calls) == 3
