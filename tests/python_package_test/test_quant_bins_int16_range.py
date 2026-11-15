"""quant_bins must fit the int16 the CUDA discretizer stores each quantized hessian in.

A row's quantized hessian reaches quant_bins (a constant hessian quantizes to exactly quant_bins), and the CUDA
discretizer stores it as int16. quant_bins in [32768, 65534] was accepted: the hessian saturated at 32767 and CUDA
training with a constant-hessian objective (L2 regression) silently grew trees without a single split. The
range check runs in Config for every device, so this is testable on a CPU build.
"""

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("quant_bins", [32768, 40000, 65534])
def test_quant_bins_beyond_int16_is_refused(quant_bins):
    x = np.random.default_rng(0).standard_normal((50, 2))
    with pytest.raises(flc.basic.FalcataError, match=r"quant_bins=\d+ is out of range"):
        flc.train({"verbose": -1, "quant_bins": quant_bins}, flc.Dataset(x, x[:, 0]), 1)


def test_quant_bins_at_int16_max_is_accepted():
    x = np.random.default_rng(0).standard_normal((50, 2))
    flc.train({"verbose": -1, "quant_bins": 32767, "min_data_in_leaf": 5}, flc.Dataset(x, x[:, 0]), 1)
