"""Sequence input accepts the same numeric dtypes as dense arrays."""

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.int32, np.int64, np.float64])
def test_sequence_numeric_dtype(dtype):
    data = np.arange(80).reshape(40, 2).astype(dtype)

    class ArraySequence(flc.Sequence):
        def __len__(self):
            return len(data)

        def __getitem__(self, index):
            return data[index]

    params = {"min_data_in_bin": 1, "min_data_in_leaf": 1, "num_threads": 2, "verbosity": -1}
    dataset = flc.Dataset(ArraySequence(), label=np.arange(40), params=params).construct()
    booster = flc.train(params, dataset, num_boost_round=2)
    dense = flc.train(params, flc.Dataset(data, label=np.arange(40)), num_boost_round=2)
    np.testing.assert_array_equal(booster.predict(data), dense.predict(data))
    valid = dataset.create_valid(ArraySequence(), label=np.arange(40)).construct()
    assert valid.num_data() == 40
