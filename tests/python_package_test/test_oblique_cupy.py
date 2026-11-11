"""GPU projections preserve the CPU contract across layouts and special values."""

import os

import numpy as np
import pytest

import falcata as flc

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires a queued CUDA GPU job")


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16])
@pytest.mark.parametrize("layout", ["C", "F", "columns", "reversed"])
def test_cupy_projection_matches_cpu(dtype, layout):
    rng = np.random.default_rng(4)
    fitted = rng.normal(size=(19, 8)).astype(np.float32)
    pool = flc.ObliquePool(num_projections=17, density=3.0, seed=7).fit(fitted)
    host = (fitted * 3).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        host[0, 2] = np.nan
        host[1, pool.feature_idx_[0][0]] = np.inf
        host[2, 1] = -np.inf
        pool.weights_[0][0] = 0.0
    device = cp.asarray(host)
    if layout == "F":
        device = cp.asfortranarray(device)
    elif layout == "columns":
        device = cp.repeat(device, 2, axis=1)[:, ::2]
    elif layout == "reversed":
        device = cp.asarray(host[::-1].copy())[::-1]
    with np.errstate(invalid="ignore"):
        expected = pool.transform(host)
    actual = pool.transform(device)
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-5, atol=2e-6, equal_nan=True)
    repeated = pool.transform(device)
    assert bool(cp.all((actual == repeated) | (cp.isnan(actual) & cp.isnan(repeated))).item())


def test_cupy_projection_metadata_upload_count_is_bounded(monkeypatch):
    data = np.random.default_rng(2).normal(size=(256, 32)).astype(np.float32)
    pool = flc.ObliquePool(num_projections=256, seed=3).fit(data)
    device = cp.asarray(data)
    original = cp.asarray
    uploads = []

    def track(data, *args, **kwargs):
        if isinstance(data, np.ndarray):
            uploads.append(data.nbytes)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", track)
    pool.transform(device)
    assert len(uploads) <= 5


def test_cupy_empty_projection_batch():
    pool = flc.ObliquePool(num_projections=7).fit(np.ones((4, 3), dtype=np.float32))
    result = pool.transform(cp.empty((0, 3), dtype=cp.float64))
    assert result.shape == (0, 10)
    assert result.dtype == cp.float64


@pytest.mark.parametrize("invalid", ["weights", "index", "center"])
def test_cupy_rejects_inconsistent_projection_metadata(invalid):
    host = np.ones((4, 3), dtype=np.float32)
    pool = flc.ObliquePool(num_projections=7).fit(host)
    if invalid == "weights":
        pool.weights_[0] = pool.weights_[0][:-1]
    elif invalid == "index":
        pool.feature_idx_[0][0] = 3
    else:
        pool.center_ = pool.center_[:-1]
    with pytest.raises(ValueError, match="metadata"):
        pool.transform(cp.asarray(host))
