"""CUDA Booster.update(train_set=...) with a dataset whose features were all filtered out.

CUDASingleGPUTreeLearner::ResetTrainingData read share_state_->feature_hist_offsets().back(), which is an empty
vector when the dataset has no usable feature (e.g. every feature dropped by feature_pre_filter because no split
can satisfy min_data_in_leaf): undefined behaviour, a segfault in practice. Init already treated it as zero bins.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


def test_cuda_reset_training_data_without_usable_features():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 3))
    X[rng.random(X.shape) < 0.7] = 0.0
    y = X[:, 0] + 0.3 * rng.standard_normal(200)
    params = {"objective": "regression", "device_type": "cuda", "num_leaves": 7, "min_data_in_leaf": 50, "verbose": -1}
    train = flc.Dataset(X[:120], y[:120], params=params, free_raw_data=False)
    booster = flc.Booster(params, train)
    for _ in range(3):
        booster.update()
    booster.update(train_set=flc.Dataset(X[120:], y[120:], params=params, reference=train))
    booster.update()
    assert all(tree["num_leaves"] == 1 for tree in booster.dump_model()["tree_info"]), "no feature should be usable"
    assert np.all(np.isfinite(booster.predict(X)))
    assert np.all(np.isfinite(booster._Booster__inner_predict(data_idx=0)))


def test_cuda_rescoring_a_model_of_constant_trees():
    # without a usable feature GBDT adds its own constant trees (plain host Trees, not CUDATrees), and every path that
    # rescores existing trees on a CUDA score updater added their constant on the host through the device score
    # pointer (a segfault): update(train_set=...) above, a validation set added after training, rollback_one_iter
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 3))
    X[rng.random(X.shape) < 0.7] = 0.0
    y = X[:, 0] + 0.3 * rng.standard_normal(200)
    params = {"objective": "regression", "device_type": "cuda", "num_leaves": 7, "min_data_in_leaf": 50, "verbose": -1}
    train = flc.Dataset(X[:120], y[:120], params=params, free_raw_data=False)
    booster = flc.Booster(params, train)
    for _ in range(3):
        booster.update()
    booster.add_valid(flc.Dataset(X[120:], y[120:], params=params, reference=train), "valid")
    expected = np.mean((booster.predict(X[120:]) - y[120:]) ** 2)
    assert booster.eval_valid()[0][2] == pytest.approx(expected)
    # an iteration of stumps after the first is not kept (GBDT stops), so count from what the booster reports
    num_iterations = booster.current_iteration()
    booster.rollback_one_iter()
    assert booster.current_iteration() == num_iterations - 1
    np.testing.assert_allclose(booster._Booster__inner_predict(data_idx=0), booster.predict(X[:120]))
