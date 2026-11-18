"""The benchmark scores contiguous float32 slabs and records which predictor served them."""

import ast
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("on_cuda", [False, True])
def test_chunk_prediction_reports_backend(on_cuda):
    slabs = []

    class Booster:
        cuda_predict_stats = None

        def predict(self, data):
            assert data.dtype == np.float32
            assert data.flags["C_CONTIGUOUS"]
            slabs.append(len(data))
            if on_cuda:
                self.cuda_predict_stats = {"num_chunks": 1}
            return data.sum(axis=1)

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "predict_in_chunks"
    )
    ns = {"np": np, "os": os, "PREDICT_CHUNK_ROWS": 200000}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    data = np.arange(36, dtype=np.float64).reshape(12, 3)
    result, backend = ns["predict_in_chunks"](Booster(), data, chunk_rows=5)
    np.testing.assert_array_equal(result, data.sum(axis=1))
    assert backend == ("cuda" if on_cuda else "cpu")
    assert slabs == [5, 5, 2]
