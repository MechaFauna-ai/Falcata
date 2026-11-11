"""One malformed journal record must not prevent benchmark resumption."""

import ast
import importlib.util
import json
import sys
from pathlib import Path


def test_resume_skips_invalid_records(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    spec = importlib.util.spec_from_file_location("common", directory / "common.py")
    common = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "common", common)
    spec.loader.exec_module(common)
    spec = importlib.util.spec_from_file_location("benchmark_orchestrate", directory / "orchestrate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "runs.jsonl"
    valid = {"library": "falcata-stoch", "dataset": "higgs", "regime": "shallow", "kind": "timed1", "status": "ok"}
    records = [{}, [], None, {**valid, "library": []}, valid]
    path.write_text("broken JSON\n" + "\n".join(json.dumps(r) for r in records))
    assert module.load_done(path) == {("falcata-stoch", "higgs", "shallow", "timed1"): "ok"}

    node = next(
        n
        for n in ast.parse((directory / "bench.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "recorded_cells"
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "bench.py", "exec"), module.__dict__)
    assert module.recorded_cells(path, "falcata-stoch", "higgs") == {("shallow", "timed1"): "ok"}
