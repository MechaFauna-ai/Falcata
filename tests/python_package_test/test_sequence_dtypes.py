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


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_sequence_large_integers_match_dense_rounding(dtype):
    data = (2**24 + np.arange(40)).reshape(-1, 1).astype(dtype)

    class ArraySequence(flc.Sequence):
        def __len__(self):
            return len(data)

        def __getitem__(self, index):
            return data[index]

    params = {"device_type": "cpu", "num_threads": 2, "min_data_in_bin": 1, "min_data_in_leaf": 1, "verbosity": -1}
    sequence = flc.train(params, flc.Dataset(ArraySequence(), label=np.arange(40)), num_boost_round=2)
    dense = flc.train(params, flc.Dataset(data, label=np.arange(40)), num_boost_round=2)
    probes = np.arange(float(2**24), float(2**24 + 40), 0.25).reshape(-1, 1)
    np.testing.assert_array_equal(sequence.predict(probes), dense.predict(probes))
