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
import sys

import numpy as np
import pytest

import falcata as lgb

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

ROWS, FEATS = 50_000, 40
PARAMS = {"objective": "regression", "num_leaves": 15, "max_bin": 15, "verbosity": -1, "num_threads": 4}


def _isolated(snippet):
    """Run a device-memory measurement in a fresh interpreter, return its number.

    These assertions are deltas, and CUDA does not hand memory back to the
    driver when Python frees a dataset -- it stays counted against the
    process. Measuring in-process therefore makes the numbers depend on
    which tests ran first: they pass alone and fail in a suite. A new
    process starts from a clean card every time.

    The snippet gets this module as `dm` and prints one integer.
    """
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('dm', r'{os.path.abspath(__file__)}')\n"
        "dm = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(dm)\n" + snippet
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, check=False)
    assert out.returncode == 0, f"isolated measurement failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    return int(out.stdout.strip().splitlines()[-1])


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


def _assert_subset_matches_direct(params, **tolerance):
    X, y = _data()
    keep = np.arange(ROWS) % 100 != 0  # drop 1%, like an unresolved target tail
    idx = np.where(keep)[0]

    parent = lgb.Dataset(X, label=y, params=params)
    sub = parent.subset(list(idx))
    from_subset = lgb.train(params, sub, num_boost_round=20)

    direct = lgb.Dataset(X[keep], label=y[keep], params=params)
    from_direct = lgb.train(params, direct, num_boost_round=20)

    Xt = _data(seed=1)[0][:2_000]
    a, b = from_subset.predict(Xt), from_direct.predict(Xt)
    if tolerance:
        np.testing.assert_allclose(a, b, **tolerance)
    else:
        np.testing.assert_array_equal(a, b)


def test_subset_matches_training_on_the_filtered_rows():
    """A subset is exactly equivalent to building the dataset from those
    rows -- identical predictions, not merely close ones.

    device_type is pinned: left unset it resolves to CUDA wherever a GPU
    is present, and the CUDA gain math is not bit-reproducible across two
    differently-shaped runs. The CUDA equivalence is covered below with
    the tolerance that path actually warrants.
    """
    _assert_subset_matches_direct({**PARAMS, "device_type": "cpu"})


@_REQUIRES_CUDA
def test_subset_matches_training_on_the_filtered_rows_cuda():
    """Same equivalence on the device, where the memory change lives.

    Not exact: fp32 gain accumulation makes a subset and a directly built
    dataset diverge in the last bits, which is a deliberate trade in this
    engine, not a defect of the subset path.
    """
    _assert_subset_matches_direct({**PARAMS, "device_type": "cuda"}, rtol=1e-5, atol=1e-6)


@_REQUIRES_CUDA
def test_loading_a_dataset_does_not_upload_it():
    """Constructing is not using: columns reach the card only once
    something trains."""
    grew = _isolated(
        """
X, y = dm._data()
p = {**dm.PARAMS, "device_type": "cuda"}
# Establish the CUDA context first. Creating one costs ~500MB on its own,
# which would swamp the few MB this test is actually looking for.
dm.lgb.train(p, dm.lgb.Dataset(X[:200], label=y[:200], params=p), num_boost_round=1)
before = dm._gpu_used_bytes()
ds = dm.lgb.Dataset(X, label=y, params=p)
ds.construct()
print(dm._gpu_used_bytes() - before)
"""
    )
    # Metadata stays eager and allocation is granular, so allow a few MB;
    # an eager column upload would scale with the data and blow past this.
    assert grew < 8 * 1024 * 1024, f"construct() uploaded {grew / 1e6:.1f}MB before any use"


@_REQUIRES_CUDA
def test_subset_of_a_host_only_parent_holds_one_copy():
    """Training a 99% subset costs one resident dataset, not two.

    Sized against a measured single copy rather than a guess: train the
    full dataset first to learn what one costs, then require the subset
    run to stay under 1.5x of it.
    """
    one_copy = _isolated(
        """
X, y = dm._data()
p = {**dm.PARAMS, "device_type": "cuda"}
start = dm._gpu_used_bytes()
dm.lgb.train(p, dm.lgb.Dataset(X, label=y, params=p), num_boost_round=10)
print(dm._gpu_used_bytes() - start)
"""
    )
    assert one_copy > 0, "could not measure a single resident copy"

    grew = _isolated(
        """
import numpy as np
X, y = dm._data()
p = {**dm.PARAMS, "device_type": "cuda"}
parent = dm.lgb.Dataset(X, label=y, params=p)
parent.construct()  # host-only until something uses it on the device
sub = parent.subset(list(np.where(np.arange(dm.ROWS) % 100 != 0)[0]))
sub.construct()
before = dm._gpu_used_bytes()
dm.lgb.train(p, sub, num_boost_round=10)
print(dm._gpu_used_bytes() - before)
"""
    )

    assert grew < 1.5 * one_copy, (
        f"subset training added {grew / 1e6:.0f}MB against a {one_copy / 1e6:.0f}MB "
        "single copy -- parent and child are both resident"
    )


@_REQUIRES_CUDA
def test_goss_resubsets_on_the_device_every_iteration():
    """GOSS re-subsets each iteration from a parent that IS resident.

    That path takes the device-to-device gather, not the host build: a
    host round trip here would be a full upload per boosting round. The
    parent is trained on first so its columns are resident, which is what
    selects the gather. Correctness is the assertion -- a re-subset that
    silently produced empty or stale columns would not fit the data.
    """
    X, y = _data()
    goss_params = {
        **PARAMS,
        "device_type": "cuda",
        "data_sample_strategy": "goss",
        "top_rate": 0.2,
        "other_rate": 0.1,
    }

    ds = lgb.Dataset(X, label=y, params=goss_params)
    booster = lgb.train(goss_params, ds, num_boost_round=25)

    preds = booster.predict(X)
    assert np.isfinite(preds).all()
    # GOSS on 25 rounds still has to track a signal this clean.
    assert np.corrcoef(preds, y)[0, 1] > 0.5, "GOSS subsetting produced a model that learned nothing"
