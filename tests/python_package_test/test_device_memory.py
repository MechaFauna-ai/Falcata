# coding: utf-8
"""Device memory is allocated when it is used, not when data is loaded.

Regression tests for the subset blow-up: taking a subset used to copy
device-to-device, which required the parent's columns to be resident
alongside the child's. Dropping 0.9% of rows therefore doubled device
memory, and a 12GB binned dataset OOMed a 32GB card it fits on twice.

The CPU tests here pin the API contract and run everywhere; the CUDA
tests pin the memory behaviour that motivated the change.
"""

import os
import subprocess

import numpy as np
import pytest

import falcata as lgb

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

ROWS, FEATS = 50_000, 40
PARAMS = {"objective": "regression", "num_leaves": 15, "max_bin": 15, "verbosity": -1, "num_threads": 4}


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10, size=(ROWS, FEATS)).astype(np.float32)
    y = X[:, 0] * 0.4 + rng.normal(size=ROWS) * 0.1
    return X, y


def _gpu_used_bytes():
    """Device memory in use, via NVML, or None when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return int(out.stdout.strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return None


def test_free_device_data_is_a_noop_on_cpu_datasets():
    """The API must be safe to call unconditionally, so callers do not
    have to branch on device type."""
    X, y = _data()
    ds = lgb.Dataset(X, label=y, params=PARAMS)
    ds.construct()
    assert ds.free_device_data() is ds
    booster = lgb.train(PARAMS, ds, num_boost_round=5)
    assert np.isfinite(booster.predict(X[:100])).all()


def test_free_device_data_before_construct_does_not_raise():
    X, y = _data()
    ds = lgb.Dataset(X, label=y, params=PARAMS)
    assert ds.free_device_data() is ds


def test_subset_matches_training_on_the_filtered_rows():
    """The memory fix must not move a single split: a subset has to stay
    exactly equivalent to building the dataset from those rows."""
    X, y = _data()
    keep = np.arange(ROWS) % 100 != 0  # drop 1%, like an unresolved target tail
    idx = np.where(keep)[0]

    parent = lgb.Dataset(X, label=y, params=PARAMS)
    sub = parent.subset(list(idx))
    from_subset = lgb.train(PARAMS, sub, num_boost_round=20)

    direct = lgb.Dataset(X[keep], label=y[keep], params=PARAMS)
    from_direct = lgb.train(PARAMS, direct, num_boost_round=20)

    Xt = _data(seed=1)[0][:2_000]
    np.testing.assert_allclose(from_subset.predict(Xt), from_direct.predict(Xt), rtol=0, atol=1e-9)


@_REQUIRES_CUDA
def test_loading_a_dataset_does_not_upload_it():
    """Constructing is not using: the columns belong on the card only
    once something trains."""
    X, y = _data()
    before = _gpu_used_bytes()
    ds = lgb.Dataset(X, label=y, params={**PARAMS, "device_type": "cuda"})
    ds.construct()
    after = _gpu_used_bytes()
    assert before is not None
    assert after is not None
    # Allow slack for context/metadata; the dataset itself is ~2MB/1000 rows.
    assert after - before < 8 * 1024 * 1024, f"construct() uploaded {(after - before) / 1e6:.1f}MB before any use"


@_REQUIRES_CUDA
def test_subset_does_not_hold_two_copies_on_the_device():
    """The bug: parent + child resident at once. Peak while training a
    99% subset must stay near ONE dataset, not two."""
    X, y = _data()
    keep = np.arange(ROWS) % 100 != 0
    cuda_params = {**PARAMS, "device_type": "cuda"}

    parent = lgb.Dataset(X, label=y, params=cuda_params)
    parent.construct()
    sub = parent.subset(list(np.where(keep)[0]))
    sub.construct()

    baseline = _gpu_used_bytes()
    lgb.train(cuda_params, sub, num_boost_round=10)
    peak = _gpu_used_bytes()
    one_copy = X.nbytes // 4  # binned bytes are far smaller than the float32 input

    assert peak - baseline < 2 * one_copy, (
        "device memory grew by more than a second copy of the dataset — "
        "the parent is resident alongside the subset again"
    )
