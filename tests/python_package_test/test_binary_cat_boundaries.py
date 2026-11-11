"""Malformed categorical boundaries must fail before allocating threshold storage."""

import struct

import falcata as flc
import numpy as np
import pytest


@pytest.mark.parametrize("boundaries", [[0, -1], [2, 1], [1, 2], [0, 2147483647]])
def test_binary_categorical_boundaries(tmp_path, boundaries):
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
    data = bytearray(model.model_to_binary(compress_level=0))
    for i in range(struct.unpack_from("<I", data, 24)[0]):
        entry = 32 + 40 * i
        ident, dtype, codec, shuffle, offset, stored, raw = struct.unpack_from("<IIIIQQQ", data, entry)
        if ident == 11:
            assert stored == 8
            struct.pack_into("<ii", data, offset, *boundaries)
            break
    else:
        pytest.fail("categorical section missing")
    path = tmp_path / "invalid.falb"
    path.write_bytes(data)
    with pytest.raises(flc.basic.FalcataError, match="FALB:.*cat_"):
        flc.Booster(model_file=path)
