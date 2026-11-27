"""The perf gate re-measures each cell through the tier that defines it; CPU only."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="strict gpuq supervision uses Linux procfs")

GATES = Path(__file__).resolve().parents[1] / "gates"

# Stands in for the CUDA package inside the re-measurement child: bench_tier.py
# runs unchanged, only training is instant.
FAKE_FALCATA = """
class Dataset:
    def __init__(self, data, label=None, params=None):
        self.params = params or {}

    def construct(self):
        return self


class _Booster:
    def __init__(self, trees):
        self.trees = trees

    def num_trees(self):
        return self.trees


def train(params, train_set, num_boost_round):
    return _Booster(num_boost_round * params.get("num_class", 1))
"""


def load(name, filename, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.syspath_prepend(str(GATES))  # bench_tier imports saved_dataset from its own directory
    spec = importlib.util.spec_from_file_location(name, GATES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def perf(monkeypatch):
    module = load("perf_tiers_under_test", "perf_gate.py")
    monkeypatch.setattr(module, "_validate_supervision", lambda: None)
    return module


@pytest.fixture
def bench(monkeypatch):
    return load("bench_tier_under_test", "bench_tier.py", monkeypatch)


def test_every_bench_cell_remeasures_through_bench_tier(perf, bench, tmp_path):
    out = tmp_path / "results.json"
    for cid, *_ in bench.CELLS:
        argv = perf.remeasure_argv(cid, out)
        assert Path(argv[1]) == GATES / "bench_tier.py", cid
        args = bench.build_parser().parse_args(argv[2:])
        assert args.only == [cid]
        assert Path(args.out) == out


def test_lattice_cells_remeasure_through_lattice(perf, tmp_path):
    lattice = load("lattice_under_test", "lattice.py")
    ids = [cell["id"] for cell in lattice.build_cells()]
    assert ids
    for cid in ids:
        argv = perf.remeasure_argv(cid, tmp_path / "results.json")
        assert Path(argv[1]) == GATES / "lattice.py", cid
        assert argv[2:] == ["--check", "--only", cid]


def test_bench_tier_rejects_unknown_cell(bench):
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--only", "bench/no-such-cell"])


def test_bench_remeasurement_runs_the_real_tier(perf, tmp_path, monkeypatch):
    fake = tmp_path / "fake_pkg"
    (fake / "falcata").mkdir(parents=True)
    (fake / "falcata" / "__init__.py").write_text(FAKE_FALCATA)
    cache = tmp_path / "cache"
    (cache / "covtype").mkdir(parents=True)
    np.save(cache / "covtype" / "X_train.npy", np.zeros((8, 2), dtype=np.float32))
    np.save(cache / "covtype" / "y_train.npy", np.zeros(8, dtype=np.float32))
    monkeypatch.setenv("PYTHONPATH", str(fake))
    monkeypatch.setenv("FALCATA_BENCH_CACHE", str(cache))

    seconds = perf.remeasure("bench/covtype-shallow-nonquant")

    assert seconds is not None
    assert seconds >= 0


def test_unavailable_bench_remeasurement_says_why(perf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FALCATA_BENCH_CACHE", str(tmp_path / "missing"))

    assert perf.remeasure("bench/covtype-shallow-nonquant") is None
    assert "re-measurement unavailable (KeyError('bench/covtype-shallow-nonquant')" in capsys.readouterr().out


def test_bench_tier_only_runs_selected_cells(bench, tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(bench, "CACHE", tmp_path)
    for _, dataset, *_ in bench.CELLS:
        (tmp_path / dataset).mkdir(exist_ok=True)
    monkeypatch.setattr(
        bench,
        "run_cell",
        lambda cid, *_: ran.append(cid) or {"id": cid, "ok": True, "construct_sec": 0, "train_sec": 1},
    )
    out = tmp_path / "out.json"

    assert bench.main(["--only", "bench/year-shallow-fp32", "--out", str(out)]) == 0
    assert ran == ["bench/year-shallow-fp32"]
    assert list(json.loads(out.read_text())["results"]) == ["bench/year-shallow-fp32"]
