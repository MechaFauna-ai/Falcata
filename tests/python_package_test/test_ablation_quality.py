"""Benchmark quality metrics must treat tied scores independently of row order."""

import ast
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("labels", [[0, 1], [1, 0]])
def test_ablation_auc_tied_scores(labels):
    source = (Path(__file__).resolve().parents[2] / "benchmarks" / "ablation.py").read_text()
    tree = ast.parse(source)
    quality = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "quality")
    namespace = {}
    exec(compile(ast.Module(body=[quality], type_ignores=[]), "ablation.py", "exec"), namespace)

    class ConstantBooster:
        def predict(self, data):
            return np.zeros(len(data))

    score, metric = namespace["quality"]({"objective": "binary"}, ConstantBooster(), np.ones((2, 1)), np.array(labels))
    assert metric == "auc"
    assert score == 0.5
