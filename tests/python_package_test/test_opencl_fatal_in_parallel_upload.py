"""OpenCL (device_type=gpu): a Log::Fatal in the parallel feature upload raises instead of aborting the process.

GPUTreeLearner::AllocateGPUMemory builds the device feature bundles in an OpenMP loop. Resetting the training data
to a dataset whose 4-bit features came out sparse makes that loop Log::Fatal ("GPU tree learner assumes that all bins
are Dense4bitsBin"), and the exception escaping an OpenMP worker thread terminated the process.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "gpu", reason="OpenCL-only regression test")


def test_opencl_reset_to_sparse_bins_raises():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10000, 20))
    X[rng.random(X.shape) < 0.7] = 0.0
    y = X[:, 0] + 0.3 * rng.standard_normal(10000)
    params = {"objective": "regression", "device_type": "gpu", "num_leaves": 7, "num_threads": 8, "verbose": -1}
    train = flc.Dataset(X[:6000], y[:6000], params=params, free_raw_data=False)
    booster = flc.Booster(params, train)
    for _ in range(3):
        booster.update()
    with pytest.raises(flc.basic.FalcataError, match="Dense4bitsBin"):
        booster.update(train_set=flc.Dataset(X[6000:], y[6000:], params=params, reference=train))
