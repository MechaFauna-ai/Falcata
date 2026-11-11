"""Unaligned raw binary sections must fail before typed dereferences."""

import struct

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("section_id", [2, 5, 6, 10])
def test_reject_unaligned_binary_section(tmp_path, section_id):
    x = np.random.default_rng(4).normal(size=(80, 2))
    booster = flc.train(
        {"objective": "regression", "num_threads": 1, "min_data_in_leaf": 3, "verbosity": -1},
        flc.Dataset(x, label=x[:, 0]),
        num_boost_round=2,
    )
    data = bytearray(booster.model_to_binary(compress_level=0))
    count = struct.unpack_from("<I", data, 24)[0]
    for i in range(count):
        entry = 32 + 40 * i
        ident, dtype, codec, shuffle, offset, stored, raw = struct.unpack_from("<IIIIQQQ", data, entry)
        if ident == section_id:
            assert codec == 0
            payload = data[offset : offset + stored]
            if len(data) % 2 == 0:
                data.append(0)
            new_offset = len(data)
            data.extend(payload)
            struct.pack_into("<Q", data, entry + 16, new_offset)
            break
    else:
        pytest.fail("Expected section missing")
    path = tmp_path / "unaligned.falb"
    path.write_bytes(data)
    with pytest.raises(flc.basic.FalcataError, match="unaligned"):
        flc.Booster(model_file=path)
