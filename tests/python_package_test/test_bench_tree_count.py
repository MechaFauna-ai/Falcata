"""Optimized Python must still reject incomplete benchmark training."""

import ast
import statistics
import sys
import time
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("function", ["run_cell", "run_h60_cell"])
@pytest.mark.parametrize("correct", [False, True])
def test_optimized_benchmark_rejects_wrong_tree_count(monkeypatch, function, correct):
    path = Path(__file__).resolve().parents[1] / "gates" / "bench_tier.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == function)
    booster = Mock()
    booster.num_trees.return_value = (6 if function == "run_cell" else 2) if correct else 0
    fake = types.SimpleNamespace(Dataset=Mock(), train=Mock(return_value=booster))
    monkeypatch.setitem(sys.modules, "falcata", fake)
    ns = {
        "time": time,
        "statistics": statistics,
        "load": lambda _: (None, None),
        "BASE": {},
        "REPEATS": 1,
        "MAX_REPEATS": 1,
        "MEASURE_BUDGET_SEC": 0,
    }
    ns["frozen_h60"] = lambda: {"path": Path("frozen.bin"), "params": {}, "y": None, "manifest_sha256": "test"}
    ns["check_loaded_h60"] = lambda *args: None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", optimize=1), ns)
    args = ("test", "small", {"num_class": 3}, 2) if function == "run_cell" else ("test", {}, 2)
    if correct:
        assert ns[function](*args)["ok"]
    else:
        with pytest.raises(RuntimeError, match="trees"):
            ns[function](*args)
