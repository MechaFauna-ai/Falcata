"""CPU fallback needs no device round trip for an existing host slab."""

import ast
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("mode", ["missing", "fil", "runtime-failure"])
def test_chunk_prediction_transfer_paths(mode, monkeypatch):
    transfers = {"h2d": 0, "d2h": 0}

    class Device:
        def __init__(self, values):
            self.values = values

    def to_device(data):
        transfers["h2d"] += 1
        return Device(data)

    def to_host(data):
        transfers["d2h"] += 1
        return data.values

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(asarray=to_device, asnumpy=to_host))
    monkeypatch.delenv("FALCATA_FIL", raising=False)

    class Booster:
        best_iteration = -1
        params = {"device_type": "cuda"}

        def _fil_model(self, *args):
            return None if mode == "missing" else object()

        def _fil_predict(self, data, **kwargs):
            return self.predict(data) if mode == "fil" else None

        def predict(self, data):
            return Device(data.values.sum(axis=1)) if isinstance(data, Device) else data.sum(axis=1)

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "predict_in_chunks"
    )
    ns = {"np": np, "os": os, "PREDICT_CHUNK_ROWS": 200000}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    data = np.arange(36, dtype=np.float32).reshape(12, 3)
    result, backend = ns["predict_in_chunks"](Booster(), data, chunk_rows=4)
    np.testing.assert_array_equal(result, data.sum(axis=1))
    assert backend == ("fil" if mode == "fil" else "cpu")
    assert transfers == {
        "h2d": 3 if mode == "fil" else int(mode == "runtime-failure"),
        "d2h": 3 if mode == "fil" else 0,
    }
