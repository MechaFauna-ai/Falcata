"""Metric baselines must be compared only with the same quantity."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("fingerprint", [False, True])
@pytest.mark.parametrize("update", [False, True])
def test_lattice_metric_identity(fingerprint, update):
    path = Path(__file__).resolve().parents[1] / "gates" / "lattice.py"
    spec = importlib.util.spec_from_file_location("lattice", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cell = {"id": "case", "equal_to": None, "fingerprint": fingerprint}
    result = {"ok": True, "metric": 0.95, "metric_name": "accuracy", "higher_better": True, "md5": "same"}
    baseline = {"metric": 0.9, "metric_name": "auc", "md5": "same"}
    failures, new = module.check([cell], {"case": result}, {"case": baseline}, update_ids={"case"} if update else None)
    if update:
        assert failures == []
        assert new["case"]["metric_name"] == "accuracy"
    else:
        assert len(failures) == 1
        assert "metric" in failures[0]
        assert "auc" in failures[0]
        assert "accuracy" in failures[0]
