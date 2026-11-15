# coding: utf-8
"""The auc_mu weight matrix is built only for the auc_mu metric.

Config::Set built the num_class x num_class auc_mu weight matrix on every call: for every Dataset,
Booster, prediction and parameter reset, whether or not auc_mu was used. num_class=65536 asked for
32 GB (std::bad_alloc); the size check of auc_mu_weights also computed num_class * num_class in int.
"""

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError


def test_large_num_class_does_not_build_the_auc_mu_matrix():
    X = np.random.default_rng(0).normal(size=(50, 3))
    # num_class only matters to the objective; the Dataset must not allocate num_class^2 doubles
    flc.Dataset(X, params={"objective": "multiclass", "num_class": 65536, "verbose": -1}).construct()


def test_auc_mu_default_weights_unchanged():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = np.digitize(X[:, 0] + X[:, 1], [-1, 1])
    params = {"objective": "multiclass", "num_class": 3, "metric": "auc_mu", "verbose": -1, "device_type": "cpu"}
    explicit = {**params, "auc_mu_weights": [0, 1, 1, 1, 0, 1, 1, 1, 0]}  # the documented default
    results = []
    for p in (params, explicit):
        evals = {}
        train = flc.Dataset(X, y)
        flc.train(p, train, 5, valid_sets=[train], valid_names=["train"], callbacks=[flc.record_evaluation(evals)])
        results.append(evals["train"]["auc_mu"])
    np.testing.assert_allclose(results[0], results[1])
    assert 0.5 < results[0][-1] <= 1.0


def test_auc_mu_weights_size_check_counts_in_64_bits():
    X = np.random.default_rng(0).normal(size=(50, 3))
    with pytest.raises(FalcataError, match=r"auc_mu_weights must have 4294967296 elements"):
        flc.Dataset(
            X, params={"objective": "multiclass", "num_class": 65536, "auc_mu_weights": [0, 1, 1, 0], "verbose": -1}
        ).construct()
