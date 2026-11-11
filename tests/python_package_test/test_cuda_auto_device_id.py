# coding: utf-8
"""With device_type unset, the auto choice must judge the GPU gpu_device_id names, not always device 0.

On origin/master CUDADeviceUsableByDefault() probed device 0 only: gpu_device_id naming a device that does not exist
(or cannot run this build) still resolved to CUDA, and training then died with "invalid device ordinal" instead of
taking the documented CPU fallback.
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _train(params):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = (X[:, 0] + 0.1 * rng.normal(size=2000) > 0).astype(np.float64)
    base = {"objective": "binary", "num_leaves": 7, "deterministic": True, "seed": 1, "verbose": -1, "num_threads": 2}
    booster = flc.train({**base, **params}, flc.Dataset(X, label=y), num_boost_round=3)
    return booster.predict(X)


@_REQUIRES_CUDA
def test_auto_device_falls_back_to_cpu_for_a_missing_gpu_device_id():
    # no host has 64 GPUs: device 63 cannot run anything, so the unset device_type resolves to the CPU
    pred = _train({"gpu_device_id": 63})
    np.testing.assert_array_equal(pred, _train({"device_type": "cpu"}))


@_REQUIRES_CUDA
def test_auto_device_still_picks_an_existing_gpu_device_id():
    pred = _train({"gpu_device_id": 0})
    assert np.all(np.isfinite(pred))
