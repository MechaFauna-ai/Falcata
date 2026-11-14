"""CUDA extra_trees on features of two bins: the one threshold is the drawn one.

The numerical split finders draw extra_trees' threshold into a __shared__ int only when the feature has more than
two bins. With two bins the CPU uses threshold 0, the only candidate; the CUDA kernels left the shared value unset,
so the feature's split depended on whatever an earlier kernel left in that slot: often no split, and models that
changed from run to run. With one candidate per feature, extra_trees must train the exhaustive search's model.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


def _model(X, y, extra_trees, max_bin):
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "num_leaves": 7,
        "max_bin": max_bin,
        "learning_rate": 0.3,
        "extra_trees": extra_trees,
        "deterministic": True,
        "quant_mode": "none",
        "verbose": -1,
        "seed": 7,
    }
    booster = flc.train(params, flc.Dataset(X, y, params=params), num_boost_round=5)
    return booster.model_to_string().split("end of trees")[0].replace("extra_trees=1", "extra_trees=0")


@pytest.mark.parametrize("seed", range(4))
def test_cuda_extra_trees_with_two_bins_equals_the_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1000, 3))
    y = X @ rng.standard_normal(3) + 0.3 * (X[:, 1] ** 2) + 0.3 * rng.standard_normal(1000)
    exhaustive = _model(X, y, extra_trees=False, max_bin=2)
    runs = {_model(X, y, extra_trees=True, max_bin=2) for _ in range(3)}
    assert runs == {exhaustive}
