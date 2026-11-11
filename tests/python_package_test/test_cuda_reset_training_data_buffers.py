# coding: utf-8
"""Replacing the training data of a CUDA booster must size the learner's buffers for the new data.

CUDAHistogramConstructor sized cuda_hist_buffer_ (one partial histogram per grid row for the global-memory construct
of large-bin partitions) from the row count in Init only, and CUDABestSplitFinder sized its global-memory staging
buffers (features with more bins than a block) from the original histogram layout. After Booster.update(train_set=...)
with more rows (or another feature grouping), the kernels wrote past those buffers. Training on the larger data after
the reset must grow the trees the CPU learner grows from the same sequence.
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _data(rows, seed):
    rng = np.random.default_rng(seed)
    # two wide continuous columns (max_bin 1023: large-bin partitions, and more bins than a split-finder block) and
    # a few narrow ones
    X = np.column_stack(
        [
            rng.standard_normal(rows),
            rng.uniform(-3.0, 3.0, rows),
            np.floor(rng.random(rows) * 8),
            np.floor(rng.random(rows) * 5),
        ]
    ).astype(np.float32)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.1 * rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _train_with_reset(device, extra):
    params = {
        "objective": "regression",
        "device_type": device,
        "max_bin": 1023,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "seed": 0,
        "deterministic": True,
        "num_threads": 8,
        "verbose": -1,
        **extra,
    }
    X_small, y_small = _data(3000, 1)
    X_big, y_big = _data(120000, 2)
    small = flc.Dataset(X_small, label=y_small, params=params, free_raw_data=False)
    big = flc.Dataset(X_big, label=y_big, reference=small, params=params, free_raw_data=False)
    booster = flc.Booster(params=params, train_set=small)
    for _ in range(3):
        booster.update()
    for _ in range(8):
        booster.update(train_set=big)
    return booster.predict(X_big)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quantized", [False, True])
def test_reset_training_data_to_more_rows_cuda(quantized):
    extra = {"use_quantized_grad": True, "quant_mode": "fixedpoint"} if quantized else {}
    cuda_pred = _train_with_reset("cuda", extra)
    assert np.all(np.isfinite(cuda_pred))
    if not quantized:
        cpu_pred = _train_with_reset("cpu", extra)
        np.testing.assert_allclose(cuda_pred, cpu_pred, rtol=1e-4, atol=1e-4)
