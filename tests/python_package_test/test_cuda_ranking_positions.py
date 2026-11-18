"""Positions in learning to rank: the CUDA build trains them on the CPU and refuses them on the GPU, clearly.

Metadata::SetPosition used to refuse positions in every CUDA build, so the CUDA wheel could not train position-aware
ranking at all, not even with device_type=cpu. Only the CUDA ranking objectives lack position support; they refuse.
"""

import numpy as np
import pytest

import falcata as flc

from .utils import BuildInfo


def _ranking_data():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 4))
    y = np.clip(np.round(x[:, 0] + 1), 0, 3)
    position = np.tile(np.arange(20), 10)
    return x, y, position


@pytest.mark.parametrize("objective", ["lambdarank", "rank_xendcg"])
def test_positions_train_on_the_cpu(objective):
    x, y, position = _ranking_data()
    params = {"device_type": "cpu", "objective": objective, "verbose": -1, "min_data_in_leaf": 5, "num_leaves": 7}
    booster = flc.train(params, flc.Dataset(x, y, group=[20] * 10, position=position), num_boost_round=3)
    assert np.all(np.isfinite(booster.predict(x)))


@pytest.mark.skipif(not BuildInfo.has_cuda, reason="the CUDA ranking objectives exist only in CUDA builds")
@pytest.mark.parametrize("objective", ["lambdarank", "rank_xendcg"])
def test_positions_are_refused_on_the_gpu(objective):
    x, y, position = _ranking_data()
    params = {"device_type": "cuda", "objective": objective, "verbose": -1, "min_data_in_leaf": 5, "num_leaves": 7}
    with pytest.raises(flc.basic.FalcataError, match="not supported with device_type=cuda"):
        flc.train(params, flc.Dataset(x, y, group=[20] * 10, position=position), num_boost_round=1)
