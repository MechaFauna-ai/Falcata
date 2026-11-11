"""Sparse benchmark snapshots should still produce truthful plots."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["plot_cross_library", "plot_construct", "plot_memory"])
def test_performance_plot_handles_missing_cells(tmp_path, monkeypatch, name):
    pytest.importorskip("matplotlib")
    path = Path(__file__).resolve().parents[2] / "docs" / "perf-plots" / "generate.py"
    tree = ast.parse(path.read_text())
    # Load the script's setup and definitions without running its final render-all calls.
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and (node.value.func.id.startswith("plot_") or node.value.func.id == "print")
        )
    ]
    namespace = {"__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["MED"] = {}
    namespace["FAIL"] = {("catboost", "numerai", regime): "crash" for regime in ["numerai", "numerai-deep"]}
    namespace["HERE"] = str(tmp_path)
    plt = namespace["plt"]
    monkeypatch.setattr(plt, "close", lambda *args: None)
    namespace[name]()
    if name != "plot_cross_library":
        assert any(text.get_text() == "✗ crash" for axis in plt.gcf().axes for text in axis.texts)
