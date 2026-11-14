# coding: utf-8
"""A reset that turns bagging off makes the CUDA learner train on every row again.

#521 fixed this for the CPU learner only. The CUDA data partition kept its bagging state: after
reset_parameter({"bagging_freq": 0}) it went on training on the last bag, and read that bag from the sample
strategy's device buffer, which the reset had freed (device use-after-free). The booster scored the trees on every
row, so the training score drifted from the model's predictions.
"""

import os

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

BASE = {
    "objective": "regression",
    "device_type": "cuda",
    "deterministic": True,
    "num_leaves": 7,
    "verbose": -1,
}


def _data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1200, 6))
    return x, x[:, 0] + rng.normal(size=1200) * 0.3


def _train_l2(booster):
    return next(value for _, name, value, _ in booster.eval_train() if name == "l2")


@pytest.mark.parametrize(
    "bagging",
    [
        {"bagging_fraction": 0.5, "bagging_freq": 1},
        {"bagging_fraction": 0.9, "bagging_freq": 1},
        {"bagging_fraction": 0.3, "bagging_freq": 2},
    ],
)
def test_reset_bagging_off_trains_on_all_rows_cuda(bagging):
    x, y = _data()
    booster = flc.train(
        {**BASE, **bagging}, flc.Dataset(x, y, free_raw_data=False), num_boost_round=3, keep_training_booster=True
    )
    initial = booster.model_to_string()
    booster.reset_parameter({"bagging_fraction": 1.0, "bagging_freq": 0})
    for _ in range(4):
        booster.update()
    # the training score matches the model: every tree after the reset was fitted on the rows it is scored on
    np.testing.assert_allclose(_train_l2(booster), np.mean((booster.predict(x) - y) ** 2), rtol=1e-6)
    # and the trees after the reset are those of an unbagged continuation
    expected = flc.train(BASE, flc.Dataset(x, y), num_boost_round=4, init_model=flc.Booster(model_str=initial))
    # CUDA float sums: a continuation from the saved model differs in the last digits
    np.testing.assert_allclose(booster.predict(x), expected.predict(x), rtol=1e-3, atol=1e-4)


def test_reset_bagging_off_then_on_cuda():
    x, y = _data()
    booster = flc.train(
        {**BASE, "bagging_fraction": 0.5, "bagging_freq": 1},
        flc.Dataset(x, y, free_raw_data=False),
        num_boost_round=3,
        keep_training_booster=True,
    )
    booster.reset_parameter({"bagging_fraction": 1.0, "bagging_freq": 0})
    booster.update()
    booster.reset_parameter({"bagging_fraction": 0.5, "bagging_freq": 1})
    for _ in range(3):
        booster.update()
    assert np.isfinite(booster.predict(x)).all()
    assert booster.current_iteration() == 7
