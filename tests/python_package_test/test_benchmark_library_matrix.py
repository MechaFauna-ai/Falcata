"""Benchmark selections must not expose the shared library matrix."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("regime", ["shallow", "numerai-deep", "numerai-leaf"])
def test_library_selection_is_independent(regime):
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "common.py"
    spec = importlib.util.spec_from_file_location("benchmark_common", path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    expected = common.libraries_for(regime).copy()
    selection = common.libraries_for(regime)
    selection.clear()
    assert common.libraries_for(regime) == expected
