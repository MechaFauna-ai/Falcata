# coding: utf-8
"""Dataset construction from a Sequence checks that its batches cover every row.

A negative batch_size made the batch loop empty, and a batch with fewer rows than its slice left the
last rows unpushed: either way the Dataset trained on rows that were never filled in (a different model,
no error).
"""

import numpy as np
import pytest

import falcata as flc


class _Seq(flc.Sequence):
    def __init__(self, X, batch_size, drop_last_row=False):
        self.X = X
        self.batch_size = batch_size
        self.drop_last_row = drop_last_row

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        out = self.X[idx]
        if self.drop_last_row and isinstance(idx, slice):
            out = out[:-1]
        return out


def _data():
    X = np.random.default_rng(0).normal(size=(500, 3))
    return X, X[:, 0]


@pytest.mark.parametrize("batch_size", [-1, 2.5])
def test_sequence_batch_size_must_be_positive(batch_size):
    X, y = _data()
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        flc.Dataset(_Seq(X, batch_size), label=y, params={"verbose": -1}).construct()


def test_sequence_short_batch_is_an_error():
    X, y = _data()
    with pytest.raises(ValueError, match=r"expected 64 rows"):
        flc.Dataset(_Seq(X, 64, drop_last_row=True), label=y, params={"verbose": -1}).construct()


def test_sequence_list_batches_and_default_batch_size():
    X, y = _data()

    class _ListSeq(_Seq):
        def __getitem__(self, idx):
            return self.X[idx].tolist() if isinstance(idx, slice) else self.X[idx]

    params = {"objective": "regression", "verbose": -1, "device_type": "cpu"}
    expected = flc.train(params, flc.Dataset(X, label=y), 3).predict(X)
    for seq in (_Seq(X, None), _ListSeq(X, 7)):
        booster = flc.train(params, flc.Dataset(seq, label=y), 3)
        np.testing.assert_allclose(booster.predict(X), expected)
