"""CPU fixtures for strict-gpuq timing policy; never start a GPU workload."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GATES = Path(__file__).resolve().parents[1] / "gates"


@pytest.fixture
def perf():
    spec = importlib.util.spec_from_file_location("perf_policy_under_test", GATES / "perf_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def queue(tmp_path, monkeypatch):
    log = tmp_path / "contention.log"
    log.write_text("")
    monkeypatch.setenv("GPUQ_JOB_ID", "1234")
    monkeypatch.setenv("GPUQ_CONTENTION_LOG", str(log))
    return log


@pytest.fixture
def files(tmp_path):
    results, baseline = tmp_path / "results.json", tmp_path / "baseline.json"
    results.write_text(json.dumps({"results": {"a": {"ok": True, "construct_sec": 0.25, "train_sec": 0.75}}}))
    baseline.write_text(json.dumps({"a": [1.0] * 3, "_pending": []}) + "\n")
    return results, baseline


def invoke(perf, monkeypatch, files, *extra, record=True):
    results, baseline = files
    argv = ["perf_gate.py", "--gpuq-strict", "--results", str(results), "--baseline-file", str(baseline)]
    argv += ["--warn-pct", "4", "--fail-pct", "8", *extra]
    if record:
        argv.append("--record")
    monkeypatch.setattr(sys, "argv", argv)
    return perf.main()


def set_times(files, **times):
    files[0].write_text(
        json.dumps({"results": {cid: {"ok": True, "construct_sec": 0, "train_sec": t} for cid, t in times.items()}})
    )


def test_clean_scheduler_is_only_idle_authority(perf, queue, files, monkeypatch):
    probes = []

    def persistent_desktop_contexts(*args, **kwargs):
        probes.append(args)
        return subprocess.CompletedProcess(args, 0, "2727, 16\n7256, 503\n203829, 374\n", "")

    monkeypatch.setattr(perf.subprocess, "run", persistent_desktop_contexts)
    assert invoke(perf, monkeypatch, files) == 0
    assert json.loads(files[1].read_text())["a"] == [1.0] * 4
    assert probes == []  # No nvidia-smi, desktop or other independent idle query.


def test_broad_real_regression_fails_under_clean_strict(perf, queue, files, monkeypatch, capsys):
    files[1].write_text(json.dumps({"a": [1.0] * 3, "b": [1.0] * 3}))
    set_times(files, a=1.5, b=1.5)
    assert invoke(perf, monkeypatch, files) == 0
    first = json.loads(files[1].read_text())
    assert first == {"a": [1.0] * 3, "b": [1.0] * 3, "_pending": ["a", "b"]}
    calls = []
    monkeypatch.setattr(perf, "remeasure", lambda cid: calls.append(cid) or 1.5)
    assert invoke(perf, monkeypatch, files) == 1
    assert calls == ["a"] * 3 + ["b"] * 3
    assert "2nd consecutive run" in capsys.readouterr().out
    after = json.loads(files[1].read_text())
    assert after["a"] == after["b"] == [1.0] * 3


def test_warn_threshold_and_nonrecord_pending(perf, queue, files, monkeypatch, capsys):
    set_times(files, a=1.05)
    assert invoke(perf, monkeypatch, files) == 0
    assert "WARN a:" in capsys.readouterr().out
    assert json.loads(files[1].read_text())["a"][-1] == 1.05
    set_times(files, a=1.5)
    before = json.loads(files[1].read_text())["a"]
    assert invoke(perf, monkeypatch, files, record=False) == 0
    assert json.loads(files[1].read_text()) == {"a": before, "_pending": ["a"]}


@pytest.mark.parametrize("exists", [False, True])
def test_contended_job_cannot_seed_or_change_pending(perf, queue, files, monkeypatch, exists):
    queue.write_text("1792200000 desktop woke\n")
    files[1].write_text('{"a": [1,1,1], "_pending": ["a"]}\n')
    before = files[1].read_bytes() if exists else None
    if not exists:
        files[1].unlink()
    monkeypatch.setattr(perf, "remeasure", lambda _: pytest.fail("must not remeasure contended timings"))
    assert invoke(perf, monkeypatch, files) == 0
    assert (files[1].read_bytes() if files[1].exists() else None) == before


@pytest.mark.parametrize("bad", ["job_missing", "job_zero", "job_invalid", "path_missing", "log_missing", "directory"])
def test_invalid_context_rejected_before_history_changes(perf, queue, files, monkeypatch, bad):
    if bad == "job_missing":
        monkeypatch.delenv("GPUQ_JOB_ID")
    elif bad == "job_zero":
        monkeypatch.setenv("GPUQ_JOB_ID", "0")
    elif bad == "job_invalid":
        monkeypatch.setenv("GPUQ_JOB_ID", "relaxed")
    elif bad == "path_missing":
        monkeypatch.delenv("GPUQ_CONTENTION_LOG")
    elif bad == "log_missing":
        queue.unlink()
    else:
        queue.unlink()
        queue.mkdir()
    before = files[1].read_bytes()
    assert invoke(perf, monkeypatch, files) == 2
    assert files[1].read_bytes() == before


def test_unreadable_context_is_an_error(perf, queue, files, monkeypatch):
    original_open = Path.open

    def no_access(path, *args, **kwargs):
        if path == queue:
            raise PermissionError("fixture denies reading scheduler evidence")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_access)
    before = files[1].read_bytes()
    assert invoke(perf, monkeypatch, files) == 2
    assert files[1].read_bytes() == before


@pytest.mark.parametrize("again", [None, 1.02, 1.5])
def test_contention_during_remeasurement_invalidates_entire_run(perf, queue, files, monkeypatch, again):
    files[1].write_text('{"a": [1,1,1], "_pending": ["a"]}\n')
    set_times(files, a=1.5)
    before = files[1].read_bytes()
    calls = []

    def remeasure(cid):
        calls.append(cid)
        queue.write_text("1792200001 foreign compute\n")
        return again

    monkeypatch.setattr(perf, "remeasure", remeasure)
    assert invoke(perf, monkeypatch, files) == 0
    assert calls == ["a"]
    assert files[1].read_bytes() == before


@pytest.mark.parametrize("lost", ["missing", "replaced"])
def test_evidence_lost_during_remeasurement_is_an_error(perf, queue, files, monkeypatch, lost):
    files[1].write_text('{"a": [1,1,1], "_pending": ["a"]}\n')
    set_times(files, a=1.5)
    before = files[1].read_bytes()

    def remeasure(cid):
        if lost == "missing":
            queue.unlink()
        else:
            replacement = queue.with_suffix(".new")
            replacement.write_text("")
            replacement.replace(queue)
        return 1.0

    monkeypatch.setattr(perf, "remeasure", remeasure)
    assert invoke(perf, monkeypatch, files) == 2
    assert files[1].read_bytes() == before


def test_contention_immediately_before_history_write_is_not_recorded(perf, queue, files, monkeypatch):
    before = files[1].read_bytes()
    original_dumps = json.dumps

    def mark_during_serialization(*args, **kwargs):
        payload = original_dumps(*args, **kwargs)
        queue.write_text("1792200002 desktop woke before save\n")
        return payload

    monkeypatch.setattr(perf.json, "dumps", mark_during_serialization)
    assert invoke(perf, monkeypatch, files) == 0
    assert files[1].read_bytes() == before


def test_cleared_remeasurement_records_the_good_sample(perf, queue, files, monkeypatch, capsys):
    files[1].write_text('{"a": [1,1,1], "_pending": ["a"]}\n')
    set_times(files, a=1.5)
    monkeypatch.setattr(perf, "remeasure", lambda _: 1.02)
    assert invoke(perf, monkeypatch, files) == 0
    assert json.loads(files[1].read_text()) == {"a": [1, 1, 1, 1.02], "_pending": []}
    out = capsys.readouterr().out
    assert "cleared on re-measurement" in out
    assert "earlier timing was contention" not in out


def test_unavailable_remeasurement_still_fails(perf, queue, files, monkeypatch):
    files[1].write_text('{"a": [1,1,1], "_pending": ["a"]}\n')
    set_times(files, a=1.5)
    monkeypatch.setattr(perf, "remeasure", lambda _: None)
    assert invoke(perf, monkeypatch, files) == 1
    assert json.loads(files[1].read_text())["a"] == [1, 1, 1]


def test_history_window_stays_bounded(perf, queue, files, monkeypatch):
    files[1].write_text(json.dumps({"a": [1] * perf.HISTORY}))
    assert invoke(perf, monkeypatch, files) == 0
    assert len(json.loads(files[1].read_text())["a"]) == perf.HISTORY


def test_env_baseline_and_cli_precedence(perf, queue, files, monkeypatch, tmp_path):
    env_baseline = tmp_path / "staged.json"
    monkeypatch.setenv("FALCATA_PERF_BASELINE", str(env_baseline))
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "--gpuq-strict", "--results", str(files[0]), "--record"])
    assert perf.main() == 0
    assert json.loads(env_baseline.read_text())["a"] == [1.0]
    before = env_baseline.read_bytes()
    assert invoke(perf, monkeypatch, files) == 0
    assert env_baseline.read_bytes() == before


def test_old_standalone_baseline_path_is_fallback(perf, queue, files, monkeypatch, tmp_path):
    fallback = tmp_path / "old-cache" / "perf_baseline.json"
    monkeypatch.setattr(perf, "DEFAULT_BASELINE", fallback)
    monkeypatch.delenv("FALCATA_PERF_BASELINE", raising=False)
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "--gpuq-strict", "--results", str(files[0]), "--record"])
    assert perf.main() == 0
    assert json.loads(fallback.read_text())["a"] == [1.0]


def test_no_implicit_standalone_bypass(perf, queue, files, monkeypatch):
    before = files[1].read_bytes()
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", "--results", str(files[0]), "--record"])
    with pytest.raises(SystemExit, match="2"):
        perf.main()
    assert files[1].read_bytes() == before


@pytest.fixture
def nightly(tmp_path):
    """Run the real shell orchestration with CPU stubs for every GPU command."""
    if os.name != "posix":
        pytest.skip("native nightly shell fixtures require POSIX")
    repo, home, bin_dir = tmp_path / "repo", tmp_path / "home", tmp_path / "bin"
    (repo / "tests/gates").mkdir(parents=True)
    (repo / ".gates-venv/bin").mkdir(parents=True)
    home.mkdir()
    bin_dir.mkdir()
    queue = tmp_path / "contention.log"
    queue.write_text("")
    calls = tmp_path / "calls.jsonl"
    stub = f"""#!{sys.executable}
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps([name, *args]) + "\\n")
if name == "git":
    if args and args[0] == "rev-parse": print("3a17c70f")
    sys.exit(0)
if name != "python": sys.exit(0)
if args and args[0] == "tests/gates/bench_tier.py":
    p = pathlib.Path("bench-count")
    n = int(p.read_text()) + 1 if p.exists() else 1
    p.write_text(str(n))
    pathlib.Path(args[args.index("--out") + 1]).write_text(json.dumps({{"results": {{"a": {{"ok": True, "construct_sec": 0, "train_sec": n}}}}}}))
    if os.environ.get("MARK_CONTENDED"): pathlib.Path(os.environ["GPUQ_CONTENTION_LOG"]).write_text("1 desktop woke\\n")
    if str(n) == os.environ.get("FAIL_BENCH_PASS"): sys.exit(1)
elif args and args[0] == "tests/gates/perf_gate.py":
    sys.exit(subprocess.run([sys.executable, os.environ["REAL_PERF"], *args[1:]]).returncode)
"""
    for path in [bin_dir / "git", bin_dir / "bash", bin_dir / "gpuq", repo / ".gates-venv/bin/python"]:
        path.write_text(stub)
        path.chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "FALCATA_REPO": str(repo),
        "GPUQ_JOB_ID": "1234",
        "GPUQ_CONTENTION_LOG": str(queue),
        "CALLS": str(calls),
        "REAL_PERF": str(GATES / "perf_gate.py"),
    }
    for key in ("FALCATA_PERF_BASELINE", "FALCATA_PERF_SEED_RUNS", "FAIL_BENCH_PASS", "MARK_CONTENDED"):
        env.pop(key, None)

    def run(**updates):
        proc = subprocess.run(
            ["/bin/bash", str(GATES / "nightly_local.sh")],
            env={**env, **updates},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        commands = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        logs = list((home / ".cache/falcata-gates").glob("nightly-*.log"))
        return proc, commands, logs[0].read_text() if logs else ""

    return run, repo, home, queue


@pytest.mark.parametrize("passes", [None, "3"])
def test_nightly_seed_passes_are_explicit_independent_and_scoped(nightly, tmp_path, passes):
    run, repo, home, queue = nightly
    updates = (
        {}
        if passes is None
        else {"FALCATA_PERF_SEED_RUNS": passes, "FALCATA_PERF_BASELINE": str(tmp_path / "staged.json")}
    )
    proc, calls, log = run(**updates)
    n = int(passes or "1")
    assert proc.returncode == 0, log + proc.stderr
    scripts = [x[1] for x in calls if x[0] == "python"]
    assert scripts.count("tests/gates/bench_tier.py") == scripts.count("tests/gates/perf_gate.py") == n
    assert scripts.count("tests/gates/lattice.py") == scripts.count("tests/gates/canonical.py") - 1 == 1
    assert "tests/gates/gpu_guard.py" not in scripts
    perf_calls = [x for x in calls if len(x) > 1 and x[1] == "tests/gates/perf_gate.py"]
    assert all(
        "--gpuq-strict" in x and x[x.index("--warn-pct") + 1] == "4" and x[x.index("--fail-pct") + 1] == "8"
        for x in perf_calls
    )
    baseline = tmp_path / "staged.json" if passes else home / ".cache/falcata-gates/bench_baseline.json"
    assert json.loads(baseline.read_text())["a"] == list(range(1, n + 1))
    assert "policy=strict-gpuq job=1234" in log
    assert f"seed_runs={n}" in log
    assert str(baseline) in log


@pytest.mark.parametrize(
    "updates",
    [
        {"GPUQ_JOB_ID": ""},
        {"GPUQ_CONTENTION_LOG": ""},
        {"FALCATA_PERF_SEED_RUNS": "0"},
        {"FALCATA_PERF_SEED_RUNS": "4"},
    ],
)
def test_nightly_invalid_context_or_seed_count_aborts_before_build(nightly, updates):
    run, repo, home, queue = nightly
    proc, calls, log = run(**updates)
    assert proc.returncode == 1
    assert "ABORT" in log
    assert not any(x[0] in ("python", "bash") for x in calls)


def test_nightly_failed_bench_does_not_consume_stale_results(nightly):
    run, repo, home, queue = nightly
    proc, calls, log = run(FALCATA_PERF_SEED_RUNS="2", FAIL_BENCH_PASS="1")
    assert proc.returncode == 1
    assert "FAIL: bench tier" in log
    assert sum(len(x) > 1 and x[1] == "tests/gates/perf_gate.py" for x in calls) == 1
    assert json.loads((home / ".cache/falcata-gates/bench_baseline.json").read_text())["a"] == [2]


def test_nightly_scheduler_contention_never_seeds_history(nightly):
    run, repo, home, queue = nightly
    proc, calls, log = run(MARK_CONTENDED="1")
    assert proc.returncode == 0, log  # Parent still must reject publication after daemon completion.
    assert not (home / ".cache/falcata-gates/bench_baseline.json").exists()
    assert "timing history and pending unchanged" in log
