"""The pre-merge speed gate's statistics and verdicts (tests/gates/speed_gate.py), on synthetic run records."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

pytest.importorskip("scipy")

GATE = Path(__file__).resolve().parents[1] / "gates" / "speed_gate.py"
spec = importlib.util.spec_from_file_location("speed_gate", GATE)
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)


def _runs(cells, slow, pairs=6, noise=0.01, seed=0, mem=None, md5_head="a"):
    """Interleaved base/head records with head = base * (1 + slow[cell]) and multiplicative noise."""
    rng = random.Random(seed)
    out = []
    for c in cells:
        for i in range(pairs):
            drift = 1 + 0.03 * rng.random()  # shared by both arms of a pair, as on a real card
            for arm in ("base", "head"):
                f = 1 + (slow.get(c, 0.0) if arm == "head" else 0.0)
                rec = {
                    "cell": c,
                    "arm": arm,
                    "pair": i,
                    "ok": True,
                    "md5": "a" if arm == "base" else md5_head,
                    "host_gb_peak": 2.0,
                    "dev_gb_peak": 1.0,
                }
                for st, base in (("construct_s", 2.0), ("create_s", 0.1), ("trees_s", 5.0), ("predict_s", 0.5)):
                    rec[st] = base * f * drift * (1 + noise * rng.gauss(0, 1))
                if mem and arm == "head":
                    rec["dev_gb_peak"] = mem
                out.append(rec)
    return out


def _compare(tmp_path, runs, cells, **kw):
    res = {
        "schema": 1,
        "machine": {"gpu": "test", "cpu": "test", "host": "t"},
        "cells": cells,
        "runs": runs,
        "skipped": {},
        "fail_pct": 3.0,
        "mem_fail_pct": 5.0,
        "alpha": 0.05,
        "base_sha": "b" * 40,
        "head_sha": "h" * 40,
        **kw,
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(res))
    md = tmp_path / "out.md"
    hist = tmp_path / "hist.jsonl"
    rc = sg.main(["compare", str(p), "--markdown", str(md), "--history", str(hist)])
    return rc, md.read_text(), hist


def test_paired_ratio_recovers_a_known_slowdown():
    base = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]
    head = [b * 1.05 for b in base]
    s = sg.paired_ratio(base, head)
    assert s["n"] == 6
    assert s["ratio"] == pytest.approx(1.05)
    assert s["lo"] == pytest.approx(1.05)
    assert s["hi"] == pytest.approx(1.05)
    assert s["p_slower"] == 0.0


def test_paired_ratio_needs_two_pairs():
    s = sg.paired_ratio([1.0], [1.1])
    assert s["ratio"] is None
    assert "fewer than 2" in s["reason"]


def test_holm_is_monotone_and_capped():
    adj = sg.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert adj["d"] is None


def test_null_change_passes(tmp_path):
    cells = ["x", "y", "z"]
    rc, md, hist = _compare(tmp_path, _runs(cells, {}), cells)
    assert rc == 0, md
    assert "**PASS**" in md
    assert len(hist.read_text().splitlines()) == 1


def test_five_percent_slowdown_fails_with_reason(tmp_path):
    cells = ["x", "y"]
    rc, md, _ = _compare(tmp_path, _runs(cells, {"y": 0.05}), cells)
    assert rc == 1
    assert "**FAIL**" in md
    assert "y trees_s: +5" in md


def test_slowdown_below_threshold_does_not_fail(tmp_path):
    cells = ["x"]
    rc, md, _ = _compare(tmp_path, _runs(cells, {"x": 0.015}, noise=0.002), cells)
    assert rc == 0
    assert "SLOWER" in md


def test_speedup_is_reported(tmp_path):
    cells = ["x"]
    rc, md, _ = _compare(tmp_path, _runs(cells, {"x": -0.10}), cells)
    assert rc == 0
    assert "FASTER" in md


def test_noisy_slowdown_is_inconclusive_not_pass(tmp_path):
    cells = ["x"]
    rc, md, _ = _compare(tmp_path, _runs(cells, {"x": 0.05}, pairs=3, noise=0.2, seed=3), cells)
    assert "INCONCLUSIVE" in md or "FAIL" in md


def test_memory_growth_fails(tmp_path):
    cells = ["x"]
    rc, md, _ = _compare(tmp_path, _runs(cells, {}, mem=1.5), cells)
    assert rc == 1
    assert "dev_gb_peak: 1.00 -> 1.50 GB" in md


def test_model_change_is_reported_not_failed(tmp_path):
    cells = ["x"]
    rc, md, _ = _compare(tmp_path, _runs(cells, {}, md5_head="b"), cells)
    assert rc == 0
    assert "Model md5 changed" in md
    assert ": x" in md


def test_failed_run_fails_the_gate(tmp_path):
    cells = ["x"]
    runs = _runs(cells, {})
    runs.append({"cell": "x", "arm": "head", "pair": 9, "ok": False, "error": "RuntimeError: boom"})
    rc, md, _ = _compare(tmp_path, runs, cells)
    assert rc == 1
    assert "boom" in md


def test_contended_run_gives_no_verdict(tmp_path):
    cells = ["x"]
    rc, md, _ = _compare(tmp_path, _runs(cells, {"x": 0.2}), cells, contended="foreign GPU processes [1]")
    assert rc == 3


def test_needs_more_only_when_the_interval_straddles_the_threshold():
    assert sg.needs_more({"ratio": 1.03, "lo": 0.99, "hi": 1.07}, 3.0)
    assert not sg.needs_more({"ratio": 1.10, "lo": 1.06, "hi": 1.14}, 3.0)
    assert not sg.needs_more({"ratio": 0.99, "lo": 0.97, "hi": 1.01}, 3.0)


def test_cell_one_arm_cannot_run_is_not_comparable(tmp_path):
    cells = ["x", "p"]
    runs = _runs(["x"], {})
    runs += [
        {"cell": "p", "arm": "base", "pair": 0, "ok": False, "skip": "this build has no native CUDA predictor"},
        {"cell": "p", "arm": "head", "pair": 0, "ok": True, "predict_s": 0.1},
    ]
    rc, md, _ = _compare(tmp_path, runs, cells)
    assert rc == 0, md
    assert "not comparable" in md
    assert "no native CUDA predictor" in md


def test_predict_cells_wait_for_the_predictor_benchmark():
    cspec = importlib.util.spec_from_file_location("speed_cells", GATE.parent / "speed_cells.py")
    speed_cells = importlib.util.module_from_spec(cspec)
    cspec.loader.exec_module(speed_cells)
    cells = speed_cells.cells()
    pred = [c for k, c in cells.items() if k.startswith("predict/")]
    assert len(pred) == 12
    have = (GATE.parents[2] / "benchmarks" / "predict_gpu.py").exists()
    for c in pred:
        assert (c.missing() is None) or (not have and "#618" in c.missing()) or "CuPy" in c.missing()


def test_desktop_clients_do_not_count_as_contention(monkeypatch):
    out = (
        "2616, /usr/lib/voxtype/voxtype-osd-gtk4, 16\n"
        "3012068, /opt/google/chrome/chrome --type=gpu-process --flag=a,b, 533\n"
        "50588, .venv/bin/python, 6228\n"
    )
    monkeypatch.setattr(sg.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": out})())
    monkeypatch.delenv("GPUQ_CONTENTION_LOG", raising=False)
    assert sg._foreign_gpu_processes() == [(50588, "python", "6228")]
    assert "50588" in sg._contended()


def test_strict_gpuq_job_uses_only_its_contention_log(monkeypatch, tmp_path):
    log = tmp_path / "contention.log"
    log.write_text("")
    monkeypatch.setenv("GPUQ_CONTENTION_LOG", str(log))
    monkeypatch.setattr(sg, "_foreign_gpu_processes", lambda: [(1, "python", "100")])
    assert sg._contended() is None
    log.write_text("1794950000 desktop woke\n")
    assert "desktop woke" in sg._contended()
