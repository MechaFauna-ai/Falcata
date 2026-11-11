# coding: utf-8
"""A parameter reset may not switch quantized training on, off, or between modes.

The tree learners choose quantized or float histogram storage, the gradient
discretizer and the split-search functions once, at Init. Booster::ResetConfig
rebuilt none of them (FeatureHistogram::ResetConfig does not even compare
use_quantized_grad), so switching quantization through reset_parameter left
the CUDA learner without a discretizer or with stale quantized buffers. The
reset is now rejected with a clear error.
"""

import os

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _booster(**params):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    p = {"objective": "regression", "verbose": -1, **params}
    booster = flc.Booster(p, flc.Dataset(X, label=X[:, 0], params=p))
    booster.update()
    return booster


def test_unrelated_reset_still_allowed_cpu():
    booster = _booster(device_type="cpu")
    booster.reset_parameter({"learning_rate": 0.05, "num_leaves": 15})
    booster.update()


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("initial", "reset"),
    [
        ({"quant_mode": "none"}, {"use_quantized_grad": True}),
        ({"quant_mode": "stochastic"}, {"quant_mode": "none"}),
        ({"quant_mode": "stochastic"}, {"quant_mode": "fixedpoint"}),
    ],
)
def test_quantization_change_on_reset_raises_cuda(initial, reset):
    booster = _booster(device_type="cuda", **initial)
    with pytest.raises(FalcataError, match="Cannot change quantized training"):
        booster.reset_parameter(reset)
    booster.reset_parameter({"learning_rate": 0.05})
    booster.update()
