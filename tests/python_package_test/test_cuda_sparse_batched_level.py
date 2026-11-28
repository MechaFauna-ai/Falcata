"""Batched per-level sparse histogram construction (cuda_plan key sparse_batched_level).

Unlocks batched level histograms and complete device_tree CUDA graph execution for
quantized training on sparse CSR row data whose partitions fit shared memory.
Integer atomics are order-invariant, so every quantized model must be bit-identical
to sparse_batched_level:off (per-leaf sparse kernel) and to the dense row-wise bin.
"""

import os
import re
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

BASE = {
    "objective": "regression",
    "device_type": "cuda",
    "num_leaves": 63,
    "max_depth": 6,
    "min_data_in_leaf": 10,
    "learning_rate": 0.1,
    "max_bin": 63,
    "verbosity": -1,
    "seed": 17,
}


def _make_sparse_data(rows, cols, sparsity, objective, *, seed=17, signed=True):
    rng = np.random.default_rng(seed)
    density = 1.0 - sparsity
    X_csr = sparse.random(
        rows,
        cols,
        density=density,
        format="csr",
        random_state=seed,
        dtype=np.float32,
        data_rvs=(lambda n: rng.standard_normal(n).astype(np.float32))
        if signed
        else (lambda n: rng.uniform(0.5, 5.0, size=n).astype(np.float32)),
    )
    w = rng.standard_normal(cols).astype(np.float32)
    z = np.asarray(X_csr @ w).ravel() + 0.35 * rng.standard_normal(rows).astype(np.float32)
    if objective == "binary":
        y = (z > np.median(z)).astype(np.float32)
    elif objective == "multiclass":
        y = np.digitize(z, np.quantile(z, [0.25, 0.5, 0.75])).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X_csr, y


def _train_model_str(X, y, params, plan, rounds=10):
    p = {**BASE, **params, "cuda_plan": plan}
    ds = flc.Dataset(X, label=y, params=p)
    model = flc.train(p, ds, num_boost_round=rounds).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.MULTILINE)


CASES = [
    pytest.param(("regression", 0.95, 40_001, 60, True, {}), id="reg-95"),
    pytest.param(("binary", 0.95, 40_001, 60, True, {"objective": "binary"}), id="bin-95"),
    pytest.param(
        ("multiclass", 0.95, 30_001, 50, True, {"objective": "multiclass", "num_class": 4}),
        id="mc-95",
    ),
    pytest.param(("regression", 0.85, 40_001, 60, True, {}), id="reg-85"),
    pytest.param(("regression", 0.99, 40_001, 120, True, {}), id="reg-99-wide"),
    pytest.param(
        ("regression", 0.95, 40_001, 80, True, {"feature_fraction": 0.6}),
        id="feature-fraction-0.6",
    ),
    pytest.param(
        ("binary", 0.95, 40_001, 60, True, {"objective": "binary", "bagging_fraction": 0.7, "bagging_freq": 1}),
        id="bagging",
    ),
    pytest.param(
        ("regression", 0.95, 40_001, 60, True, {"num_leaves": 31, "max_depth": -1}),
        id="selective-unbounded",
    ),
    pytest.param(
        ("regression", 0.95, 40_001, 60, True, {"zero_as_missing": True}),
        id="zero-as-missing",
    ),
    pytest.param(
        ("regression", 0.95, 30_001, 20, False, {"max_bin": 7}),
        id="8bit-sparse-mfb0",
    ),
    pytest.param(
        ("binary", 0.95, 70_001, 60, True, {"objective": "binary", "num_leaves": 31, "max_depth": 5}),
        id="32bit-leaf-bins",
    ),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize("spec", CASES)
def test_sparse_batched_level_bit_identical(spec, quant):
    objective, sparsity, rows, cols, signed, extra = spec
    X, y = _make_sparse_data(rows, cols, sparsity, objective, signed=signed)
    params = {"quant_mode": quant, **extra}
    on_model = _train_model_str(X, y, params, "auto,dense_rowdata:off,sparse_batched_level:on", rounds=10)
    off_model = _train_model_str(X, y, params, "auto,dense_rowdata:off,sparse_batched_level:off", rounds=10)
    assert on_model == off_model


@_REQUIRES_CUDA
def test_sparse_batched_level_matches_dense_rowdata():
    X, y = _make_sparse_data(40_001, 60, 0.95, "binary", signed=True)
    params = {
        "objective": "binary",
        "quant_mode": "fixedpoint",
        "feature_fraction": 0.6,
    }
    sparse_batched = _train_model_str(X, y, params, "auto,dense_rowdata:off,sparse_batched_level:on", rounds=10)
    dense = _train_model_str(X, y, params, "auto,dense_rowdata:on", rounds=10)
    assert sparse_batched == dense


_DIAG_PROBE = """
import hashlib, sys
sys.path.insert(0, {here!r})
from test_cuda_sparse_batched_level import _make_sparse_data, _train_model_str
X, y = _make_sparse_data(40_001, 60, 0.95, "binary", signed=True)
params = {{"objective": "binary", "quant_mode": "fixedpoint", "verbosity": 1}}
model = _train_model_str(X, y, params, {plan!r}, rounds=6)
print("MODEL_MD5", hashlib.md5(model.encode()).hexdigest(), flush=True)
"""


def _run_diag_probe(plan):
    code = _DIAG_PROBE.format(here=os.path.dirname(os.path.abspath(__file__)), plan=plan)
    env = {**os.environ, "FALCATA_DEBUG": "diag"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    combined = proc.stdout + "\n" + proc.stderr
    md5 = re.search(r"MODEL_MD5 (\w+)", combined).group(1)
    return combined, md5


@_REQUIRES_CUDA
def test_sparse_batched_level_diag_activation():
    on_log, on_md5 = _run_diag_probe("auto,dense_rowdata:off,sparse_batched_level:on")
    assert "batched_level_kernels=1" in on_log, on_log[-3000:]
    assert "quantized construct: sparse_batched" in on_log, on_log[-3000:]
    assert "sparse_layout" not in on_log, on_log[-3000:]

    off_log, off_md5 = _run_diag_probe("auto,dense_rowdata:off,sparse_batched_level:off")
    assert "batched_level_kernels=0" in off_log, off_log[-3000:]
    assert "quantized construct: sparse_batched" not in off_log, off_log[-3000:]
    assert "sparse_layout" in off_log, off_log[-3000:]

    assert on_md5 == off_md5
