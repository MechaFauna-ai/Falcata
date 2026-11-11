"""CUDA refit computes the CPU learner's leaf outputs: L1, max_delta_step and path smoothing.

The CUDA learner's refit kernel always passed USE_L1=false (lambda_l1 was ignored), and with path_smooth it smoothed
each leaf toward an output built from the leaf and its "sibling", reading a leaf array at the sibling's child code
(an internal node's index, or ~leaf) and passing the row-index array where per-leaf counts belong. A tree loaded from
model text carries no leaf_parent_, so the kernel also read an empty parent array: refit segfaulted.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 5))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * rng.standard_normal(2000)
    return X, y


def _params(device, **extra):
    return {
        "objective": "regression",
        "device_type": device,
        "verbose": -1,
        "min_data_in_leaf": 5,
        "lambda_l2": 1.0,
        "boost_from_average": False,
        "deterministic": True,
        "quant_mode": "none",
        "num_threads": 4,
        **extra,
    }


def _leaf_values(booster):
    frame = booster.trees_to_dataframe()
    return frame.loc[frame["split_feature"].isna(), "value"].to_numpy()


_REGULARIZATION = [
    {"path_smooth": 50.0},
    {"lambda_l1": 5.0},
    {"lambda_l1": 5.0, "path_smooth": 3.0, "max_delta_step": 0.05},
]


@pytest.mark.parametrize("num_leaves", [2, 7, 31])
@pytest.mark.parametrize("extra", _REGULARIZATION, ids=["smooth", "l1", "l1_smooth_mds"])
def test_cuda_refit_on_the_training_data_reproduces_the_trained_leaves(num_leaves, extra):
    X, y = _data()
    params = _params("cuda", num_leaves=num_leaves, **extra)
    model = flc.train(params, flc.Dataset(X, y, params=params), num_boost_round=1)
    refit = model.refit(X, y, decay_rate=0.0)
    np.testing.assert_allclose(_leaf_values(refit), _leaf_values(model), rtol=1e-5, atol=1e-9)
    assert np.all(np.isfinite(refit.predict(X)))


@pytest.mark.parametrize("decay_rate", [0.0, 0.5])
@pytest.mark.parametrize("extra", _REGULARIZATION, ids=["smooth", "l1", "l1_smooth_mds"])
def test_cuda_refit_matches_cpu_refit(decay_rate, extra):
    X, y = _data()
    # one model, refit on both devices: the leaf outputs are the only difference left
    model = flc.train(_params("cpu", num_leaves=15, **extra), flc.Dataset(X, y), num_boost_round=5)
    refits = {
        device: model.refit(
            X[:1200], 0.7 * y[:1200], decay_rate=decay_rate, dataset_params=_params(device, num_leaves=15, **extra)
        )
        for device in ("cpu", "cuda")
    }
    np.testing.assert_allclose(_leaf_values(refits["cuda"]), _leaf_values(refits["cpu"]), rtol=1e-5, atol=1e-9)
