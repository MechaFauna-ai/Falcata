"""model_to_binary must reject a compress_level outside 0-9 with a clear message.

10 or 100 failed deep in zlib as "FALB: zlib compression failed (-2)", and a
negative level was silently treated as 0 (uncompressed).
"""

import numpy as np
import pytest

import falcata as flc


@pytest.fixture(scope="module")
def booster():
    x = np.random.default_rng(0).standard_normal((200, 3))
    return flc.train({"verbose": -1, "num_threads": 1}, flc.Dataset(x, label=x[:, 0]), num_boost_round=2)


@pytest.mark.parametrize("level", [-1, 10, 100])
def test_out_of_range_compress_level_is_rejected(booster, level):
    with pytest.raises(flc.basic.FalcataError, match=r"compress_level must be in \[0, 9\]"):
        booster.model_to_binary(compress_level=level)


@pytest.mark.parametrize("level", [0, 1, 6, 9])
def test_valid_compress_levels_round_trip(booster, level):
    x = np.random.default_rng(1).standard_normal((20, 3))
    restored = flc.Booster(model_bin=booster.model_to_binary(compress_level=level))
    np.testing.assert_array_equal(restored.predict(x), booster.predict(x))
