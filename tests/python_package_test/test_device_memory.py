# coding: utf-8
"""Device memory is allocated when it is used, not when data is loaded.

A dataset uploads its columns on first device use. Subsetting a
host-only dataset keeps it host-only, so one dataset is resident where
a parent and child would otherwise both be.

The CPU tests pin the API contract and run everywhere; the CUDA tests
pin the memory behaviour.
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
    """Device memory used by THIS process, or None when unavailable.

    Per-process rather than whole-card: anything else sharing the GPU
    would otherwise make these thresholds flaky.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    mine = os.getpid()
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == mine:
            return int(parts[1]) * 1024 * 1024
    return 0  # no allocation attributed to us yet


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
    """A subset is exactly equivalent to building the dataset from those
    rows -- identical predictions, not merely close ones."""
    X, y = _data()
    keep = np.arange(ROWS) % 100 != 0  # drop 1%, like an unresolved target tail
    idx = np.where(keep)[0]

    parent = lgb.Dataset(X, label=y, params=PARAMS)
    sub = parent.subset(list(idx))
    from_subset = lgb.train(PARAMS, sub, num_boost_round=20)

    direct = lgb.Dataset(X[keep], label=y[keep], params=PARAMS)
    from_direct = lgb.train(PARAMS, direct, num_boost_round=20)

    Xt = _data(seed=1)[0][:2_000]
    np.testing.assert_array_equal(from_subset.predict(Xt), from_direct.predict(Xt))


@_REQUIRES_CUDA
def test_loading_a_dataset_does_not_upload_it():
    """Constructing is not using: columns reach the card only once
    something trains."""
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
def test_subset_of_a_host_only_parent_holds_one_copy():
    """Training a 99% subset costs one resident dataset, not two.

    Sized against a measured single copy rather than a guess: train the
    full dataset first to learn what one costs, then require the subset
    run to stay under 1.5x of it.
    """
    X, y = _data()
    cuda_params = {**PARAMS, "device_type": "cuda"}

    start = _gpu_used_bytes()
    full = lgb.Dataset(X, label=y, params=cuda_params)
    lgb.train(cuda_params, full, num_boost_round=10)
    one_copy = _gpu_used_bytes() - start
    assert one_copy > 0, "could not measure a single resident copy"

    parent = lgb.Dataset(X, label=y, params=cuda_params)
    parent.construct()  # host-only until something uses it on the device
    sub = parent.subset(list(np.where(np.arange(ROWS) % 100 != 0)[0]))
    sub.construct()

    before = _gpu_used_bytes()
    lgb.train(cuda_params, sub, num_boost_round=10)
    grew = _gpu_used_bytes() - before

    assert grew < 1.5 * one_copy, (
        f"subset training added {grew / 1e6:.0f}MB against a {one_copy / 1e6:.0f}MB "
        "single copy — parent and child are both resident"
    )
