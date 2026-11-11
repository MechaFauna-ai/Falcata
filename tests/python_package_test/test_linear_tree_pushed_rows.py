# coding: utf-8
"""Rows pushed in batches into a linear-tree dataset keep their raw values.

The FLC_DatasetPushRows* functions used to resize the linear-tree raw buffer
to num_numeric_features + nrow (the batch size) and then write raw values at
start_row + i, so every batch after the first wrote past the end of the
buffer. A Sequence validation set built against a reference (CreateByReference
then PushRows per batch) corrupted the heap.
"""

import numpy as np
import pytest

import falcata as flc


class _BatchedSequence(flc.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


_PARAMS = {
    "objective": "regression",
    "linear_tree": True,
    "num_leaves": 4,
    "min_data_in_leaf": 50,
    "deterministic": True,
    "num_threads": 1,
    "verbose": -1,
}


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0, 1.5])
    return X, y


def _valid_l2(make_valid):
    X, y = _data()
    train = flc.Dataset(X, label=y, params=_PARAMS)
    evals = {}
    flc.train(
        _PARAMS,
        train,
        num_boost_round=5,
        valid_sets=[flc.Dataset(make_valid(X), label=y, reference=train)],
        callbacks=[flc.record_evaluation(evals)],
    )
    return evals["valid_0"]["l2"]


@pytest.mark.parametrize("batch_size", [500, 4096])
def test_linear_tree_sequence_reference_valid_matches_dense(batch_size):
    expected = _valid_l2(np.asarray)
    np.testing.assert_allclose(_valid_l2(lambda X: _BatchedSequence(X, batch_size)), expected)


def test_linear_tree_sequence_train_matches_dense():
    X, y = _data()
    dense = flc.train(_PARAMS, flc.Dataset(X, label=y, params=_PARAMS), num_boost_round=5)
    seq = flc.train(_PARAMS, flc.Dataset(_BatchedSequence(X, 500), label=y, params=_PARAMS), num_boost_round=5)
    np.testing.assert_allclose(seq.predict(X), dense.predict(X))
