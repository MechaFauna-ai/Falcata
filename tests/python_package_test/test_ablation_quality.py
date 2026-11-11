"""Benchmark quality metrics must treat tied scores independently of row order."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


@pytest.mark.parametrize(
    ("labels", "preds"),
    [([0, 1], [0.0, 0.0]), ([1, 0], [0.0, 0.0]), ([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9])],
)
def test_ablation_auc_tied_scores(labels, preds):
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "ablation.py"
    spec = importlib.util.spec_from_file_location("ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FixedBooster:
        def predict(self, data):
            return np.array(preds)

    score, metric = module.quality({"objective": "binary"}, FixedBooster(), np.ones((len(labels), 1)), np.array(labels))
    assert metric == "auc"
    assert score == roc_auc_score(labels, preds)
