"""Malformed categorical boundaries must fail before allocating threshold storage."""

import struct

import numpy as np
import pytest

import falcata as flc


def _categorical_binary():
    x = np.tile(np.arange(8.0), 20).reshape(-1, 1)
    model = flc.train(
        {
            "objective": "regression",
            "num_threads": 1,
            "num_leaves": 2,
            "min_data_in_leaf": 2,
            "cat_smooth": 0,
            "cat_l2": 0,
            "max_cat_to_onehot": 16,
            "verbosity": -1,
        },
        flc.Dataset(x, label=(x[:, 0] == 1).astype(float), categorical_feature=[0]),
        num_boost_round=1,
    )
    return bytearray(model.model_to_binary(compress_level=0))


def _section(data, ident):
    for i in range(struct.unpack_from("<I", data, 24)[0]):
        entry = 32 + 40 * i
        if struct.unpack_from("<I", data, entry)[0] == ident:
            return struct.unpack_from("<Q", data, entry + 16)[0]
    pytest.fail("section missing")


@pytest.mark.parametrize("boundaries", [[0, -1], [2, 1], [1, 2], [0, 2147483647]])
def test_binary_categorical_boundaries(tmp_path, boundaries):
    data = _categorical_binary()
    struct.pack_into("<ii", data, _section(data, 11), *boundaries)
    path = tmp_path / "invalid.falb"
    path.write_bytes(data)
    with pytest.raises(flc.basic.FalcataError, match="FALB:.*cat_"):
        flc.Booster(model_file=path)


@pytest.mark.parametrize(("field", "fmt", "value"), [(4, "I", 0xFFFFFFFF), (16, "Q", 2**64 - 1)])
def test_binary_tree_array_ranges(tmp_path, field, fmt, value):
    data = _categorical_binary()
    struct.pack_into("<" + fmt, data, _section(data, 2) + field, value)
    path = tmp_path / "invalid.falb"
    path.write_bytes(data)
    with pytest.raises(flc.basic.FalcataError, match="FALB:"):
        flc.Booster(model_file=path)
