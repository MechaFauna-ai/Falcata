# coding: utf-8
"""Categorical columns of float32/float64 matrices are binned on the device, byte for byte as the host path does.

The GPU float binner replicates BinMapper::ValueToBin for categorical columns through a per-column category->bin
table. The host truncates a value to int, so a fractional value lands in the category toward zero; NaN, values at or
below -1, categories the bin mapper dropped (past its 99% count cut) or never saw, and values beyond the int range all
go to bin 0. Every Dataset's saved binary must equal the host binning path's (gpu_construct:off) and the construct must
have run on the device, except where the category ids exceed the device tables, which leaves the matrix to the host
path.
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

PARAMS = {
    "objective": "regression",
    "device_type": "cuda",
    "num_leaves": 15,
    "learning_rate": 0.1,
    "max_bin": 63,
    "verbosity": 2,
    "seed": 3,
    "num_threads": 4,
    "quant_mode": "stochastic",
}
CATEGORICAL = [1, 2, 3]


def _matrix(dtype, order, rows=20_001, seed=0, big_category=None):
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=rows)
    common = rng.integers(0, 12, size=rows).astype(np.float64)
    # more categories than max_bin: the bin mapper drops the least frequent ones
    rare = rng.integers(0, 250, size=rows).astype(np.float64)
    edge = common.copy()
    u = rng.random(rows)
    edge[u < 0.05] = np.nan
    edge[(u >= 0.05) & (u < 0.08)] = -1.0
    edge[(u >= 0.08) & (u < 0.10)] = -0.5  # truncates to category 0
    edge[(u >= 0.10) & (u < 0.12)] = 3.7  # truncates to category 3
    edge[(u >= 0.12) & (u < 0.13)] = 3e9  # beyond the int range
    edge[(u >= 0.13) & (u < 0.135)] = np.inf
    edge[(u >= 0.135) & (u < 0.14)] = -np.inf
    edge[(u >= 0.14) & (u < 0.15)] = -7.0
    edge[(u >= 0.15) & (u < 0.16)] = 1000.0  # above every category of the column, so outside its table
    if big_category is not None:
        rare[u > 0.7] = big_category
    X = np.column_stack([numeric, common, rare, edge, rng.normal(size=rows)]).astype(dtype)
    y = numeric + np.nan_to_num(common % 3) + rng.normal(size=rows)
    return (np.asfortranarray(X) if order == "F" else np.ascontiguousarray(X)), y


class _Capture:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(str(msg))

    def warning(self, msg):
        self.lines.append(str(msg))


@pytest.fixture
def captured_log():
    previous = (flc.basic._LOGGER, flc.basic._INFO_METHOD_NAME, flc.basic._WARNING_METHOD_NAME)
    capture = _Capture()
    flc.register_logger(capture)
    yield capture
    flc.register_logger(*previous)


def _binary(X, y, params, path):
    ds = flc.Dataset(X, label=y, params=params, categorical_feature=CATEGORICAL, free_raw_data=False)
    ds.construct()
    ds.save_binary(path)
    return ds, path.read_bytes()


@_REQUIRES_CUDA
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
def test_categorical_float_matrix_binned_on_device_cuda(dtype, order, tmp_path, captured_log):
    X, y = _matrix(dtype, order)
    host_params = {**PARAMS, "cuda_plan": "auto,gpu_construct:off"}
    ds, device = _binary(X, y, PARAMS, tmp_path / "device.bin")
    assert any("GPU construct: binned" in line for line in captured_log.lines), "the device binner did not run"
    host_ds, host = _binary(X, y, host_params, tmp_path / "host.bin")
    assert device == host, "the device-binned Dataset differs from the host-binned one"
    # the tree section only: the parameter dump records the plan string
    models = [
        flc.train({**p, "verbosity": -1}, d, num_boost_round=5).model_to_string().split("parameters:")[0]
        for p, d in ((PARAMS, ds), (host_params, host_ds))
    ]
    assert models[0] == models[1]


@_REQUIRES_CUDA
def test_categorical_ids_beyond_the_device_table_use_the_host_path_cuda(tmp_path, captured_log):
    X, y = _matrix(np.float32, "C", big_category=float(1 << 25))
    _, device = _binary(X, y, PARAMS, tmp_path / "device.bin")
    assert any("too large for the device category table" in line for line in captured_log.lines)
    assert not any("GPU construct: binned" in line for line in captured_log.lines)
    _, host = _binary(X, y, {**PARAMS, "cuda_plan": "auto,gpu_construct:off"}, tmp_path / "host.bin")
    assert device == host
