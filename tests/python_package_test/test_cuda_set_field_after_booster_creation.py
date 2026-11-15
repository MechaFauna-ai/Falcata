# coding: utf-8
"""Changing the training Dataset's weight or group after Booster creation takes effect on the CUDA learner.

#533 rebuilt the objective and metrics from the new metadata, which fixed the CPU learner. CUDA objectives read the
Dataset's device copy of the metadata: clearing the weights left the old weights on the device, so the CUDA model kept
training weighted.

With deterministic=true (fixed-point quantized training on CUDA) the learner chose its quantization bin count at
Booster creation from whether the hessians are constant (2048 bins unweighted, 256 weighted), and kept it after the
weights changed: the model differed from one created with the final weights. It now resolves the count again.
"""

import os

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("change", ["group", "add_weight", "clear_weight"])
def test_set_field_after_booster_creation_cuda(change, deterministic):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    # deterministic=true selects fixed-point quantized training, whose bin count follows the hessians
    params = {"verbose": -1, "min_data_in_leaf": 5, "device_type": "cuda", "deterministic": deterministic}
    weight = rng.uniform(0.5, 2.0, size=120)
    if change == "group":
        params["objective"] = "lambdarank"
        y = rng.integers(0, 4, size=120)
        before, after = {"group": [30, 30, 60]}, {"group": [10] * 12}
    else:
        params["objective"] = "regression"
        y = X[:, 0] + rng.normal(size=120)
        before, after = ({}, {"weight": weight}) if change == "add_weight" else ({"weight": weight}, {})

    changed = flc.Dataset(X, y, free_raw_data=False, **before)
    booster = flc.Booster(params, changed)
    if change == "group":
        changed.set_group(after["group"])
    elif change == "add_weight":
        changed.set_weight(after["weight"])
    else:
        changed.set_field("weight", None)
    for _ in range(3):
        booster.update()

    reference = flc.train(params, flc.Dataset(X, y, **after), num_boost_round=3)
    np.testing.assert_allclose(booster.predict(X), reference.predict(X), rtol=1e-5, atol=1e-6)
