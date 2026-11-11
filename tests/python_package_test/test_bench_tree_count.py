"""Optimized Python must still reject incomplete benchmark training."""

import ast
import statistics
import sys
import time
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


def test_optimized_benchmark_rejects_wrong_tree_count(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "gates" / "bench_tier.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "run_cell")
    booster = Mock()
    booster.num_trees.return_value = 0
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
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", optimize=1), ns)
    with pytest.raises(RuntimeError, match="trees"):
        ns["run_cell"]("test", "small", {}, 2)
