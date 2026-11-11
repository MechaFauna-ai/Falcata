# coding: utf-8
"""A CSC validation set built against a linear-tree reference gets its raw buffer.

FLC_DatasetCreateFromCSC's reference branch used to skip ResizeRaw, so the
per-row raw writes for linear_tree went into an empty buffer (a segfault or
silent heap corruption). The validation metric must match the dense input.
"""

import numpy as np
import pytest
from scipy import sparse

import falcata as flc


def _valid_l2(make_matrix):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    X[rng.random(X.shape) < 0.3] = 0.0
    y = X @ np.array([1.0, -2.0, 0.5, 3.0, 1.5])
    params = {
        "objective": "regression",
        "linear_tree": True,
        "num_leaves": 4,
        "min_data_in_leaf": 50,
        "deterministic": True,
        "num_threads": 1,
        "verbose": -1,
    }
    train = flc.Dataset(X, label=y, params=params)
    valid = flc.Dataset(make_matrix(X), label=y, reference=train)
    evals = {}
    flc.train(
        params,
        train,
        num_boost_round=5,
        valid_sets=[valid],
        callbacks=[flc.record_evaluation(evals)],
    )
    return evals["valid_0"]["l2"]


@pytest.mark.parametrize("make_matrix", [sparse.csc_matrix, sparse.csr_matrix])
def test_linear_tree_sparse_reference_valid_matches_dense(make_matrix):
    np.testing.assert_allclose(_valid_l2(make_matrix), _valid_l2(np.asarray))
