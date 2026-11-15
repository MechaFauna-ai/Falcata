# coding: utf-8
"""GOSS skips sampling for the first trunc(1 / learning_rate) iterations, for any learning_rate.

GOSSStrategy::Bagging computed static_cast<int>(1.0f / learning_rate). Below
learning_rate ~4.7e-10 the quotient exceeds INT_MAX and the cast was undefined
behaviour; on x86 it gave INT_MIN, so GOSS subsampled from the first iteration
instead of never (for such a learning rate no iteration count reaches the threshold).

quant_mode=none is pinned for device_type=cuda (an unset device_type on a CUDA build): there deterministic=true
otherwise selects quantized training at a resolution that depends on whether the hessians are constant, and GOSS
declares them non-constant from the start (it rescales the sampled rows' hessians once it samples), so even its
unsampled warm-up trees are quantized differently from plain gbdt. quant_mode=none is the CPU default.
"""

import numpy as np

import falcata as flc


def _model(**params):
    rng = np.random.default_rng(0)
    X = rng.random((2000, 5))
    y = X[:, 0] + rng.normal(0, 0.1, 2000)
    base = {"verbose": -1, "num_threads": 1, "seed": 1, "deterministic": True, "quant_mode": "none"}
    booster = flc.train({**base, **params}, flc.Dataset(X, y), 12)
    return booster.model_to_string().split("parameters:")[0]


def test_goss_with_tiny_learning_rate_never_subsamples():
    assert _model(data_sample_strategy="goss", learning_rate=1e-10) == _model(learning_rate=1e-10)


def test_goss_still_subsamples_after_the_warmup():
    # 1 / 0.3 truncates to 3 warm-up iterations; 12 iterations then subsample and change the model
    assert _model(data_sample_strategy="goss", learning_rate=0.3) != _model(learning_rate=0.3)
