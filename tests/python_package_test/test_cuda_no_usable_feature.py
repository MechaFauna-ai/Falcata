"""CUDA training on a dataset with no usable feature.

With too few rows for min_data_in_leaf, feature_pre_filter drops every feature. GBDT does not call the tree learner
then; RF (sequential rounds and the batched forest trainer) does, and the CUDA learner read its device row data,
which was never built: a segfault in CUDAHistogramConstructor::ColumnsUsedByFeatures
(host_feature_partition_column_index_offsets().back() on an empty vector).
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


@pytest.mark.parametrize("boosting", ["rf", "gbdt"])
@pytest.mark.parametrize("cuda_plan", ["auto", "auto,forest_batched:off"])
def test_cuda_trains_without_a_usable_feature(boosting, cuda_plan):
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    X[rng.random(X.shape) < 0.7] = 0.0
    y = rng.poisson(1.0, 200).astype(float)
    params = {
        "objective": "poisson",
        "boosting": boosting,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "device_type": "cuda",
        "cuda_plan": cuda_plan,
        "num_leaves": 255,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }
    train = flc.Dataset(X[:120], y[:120], params=params)
    valid = flc.Dataset(X[120:], y[120:], reference=train, params=params)
    booster = flc.train(params, train, num_boost_round=10, valid_sets=[valid])
    assert all(tree["num_leaves"] == 1 for tree in booster.dump_model()["tree_info"])
    assert np.all(np.isfinite(booster.predict(X)))
    if boosting == "gbdt":
        # (an RF stump's value after the first round is the learner's root output, which the CUDA learner does not
        # compute for a dataset without features; https://github.com/MechaFauna-ai/Falcata/pull/436 makes every RF
        # stump carry the class constant instead)
        cpu_params = {**params, "device_type": "cpu"}
        cpu = flc.train(cpu_params, flc.Dataset(X[:120], y[:120], params=cpu_params), num_boost_round=10)
        np.testing.assert_allclose(booster.predict(X), cpu.predict(X), rtol=1e-12)
