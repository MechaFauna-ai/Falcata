"""A reset that turns bagging off, or switches between subset and index bagging, retrains on the right rows.

The tree learner keeps the last bag (row indices, or a subset dataset for small fractions) until the sampling strategy
hands it a new one. Turning bagging off never did, so later trees kept training on the stale bag while the booster
scored them on every row: the training score drifted from the model's predictions. Switching from a subset bag to an
index bag, or the other way round, read row indices beyond the data (heap-buffer-overflow under ASan).
"""

import numpy as np
import pytest

import falcata as flc

# device_type=cpu: the CUDA learner keeps its own bagging state and is not covered by this fix
BASE = {
    "objective": "regression",
    "device_type": "cpu",
    "deterministic": True,
    "num_threads": 1,
    "force_col_wise": True,
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
        {"bagging_fraction": 0.5, "bagging_freq": 1},  # subset bagging
        {"bagging_fraction": 0.9, "bagging_freq": 1},  # index bagging
        {"bagging_fraction": 0.3, "bagging_freq": 2},
    ],
)
def test_reset_bagging_off_trains_on_all_rows(bagging):
    x, y = _data()
    booster = flc.train(
        {**BASE, **bagging}, flc.Dataset(x, y, free_raw_data=False), num_boost_round=3, keep_training_booster=True
    )
    initial = booster.model_to_string()
    booster.reset_parameter({"bagging_fraction": 1.0, "bagging_freq": 0})
    for _ in range(4):
        booster.update()
    expected = flc.train(BASE, flc.Dataset(x, y), num_boost_round=4, init_model=flc.Booster(model_str=initial))
    np.testing.assert_allclose(booster.predict(x), expected.predict(x), rtol=0, atol=1e-12)
    np.testing.assert_allclose(_train_l2(booster), np.mean((booster.predict(x) - y) ** 2), rtol=1e-6)


@pytest.mark.parametrize(("before", "after"), [(0.3, 0.9), (0.9, 0.3), (0.3, 1.0), (0.9, 1.0)])
def test_reset_bagging_fraction_keeps_training_score_consistent(before, after):
    x, y = _data()
    booster = flc.train(
        {**BASE, "bagging_fraction": before, "bagging_freq": 1},
        flc.Dataset(x, y, free_raw_data=False),
        num_boost_round=3,
        keep_training_booster=True,
    )
    booster.reset_parameter({"bagging_fraction": after, "bagging_freq": 1 if after < 1 else 0})
    for _ in range(4):
        booster.update()
    np.testing.assert_allclose(_train_l2(booster), np.mean((booster.predict(x) - y) ** 2), rtol=1e-6)


@pytest.mark.parametrize(("before", "after"), [(0.3, 1.0), (0.3, 0.9), (0.9, 0.3)])
def test_reset_bagging_by_query_fraction(before, after):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2000, 5))
    y = np.clip(np.round(x[:, 0] + 1), 0, 3)
    params = {**BASE, "objective": "lambdarank", "bagging_by_query": True, "metric": "ndcg", "eval_at": [5]}
    booster = flc.train(
        {**params, "bagging_fraction": before, "bagging_freq": 1},
        flc.Dataset(x, y, group=[20] * 100, free_raw_data=False),
        num_boost_round=3,
        keep_training_booster=True,
    )
    booster.reset_parameter({"bagging_fraction": after, "bagging_freq": 1 if after < 1 else 0})
    for _ in range(3):
        booster.update()
    # the out-of-range reads only show under ASan; a normal build checks the model stays finite
    assert np.isfinite(booster.predict(x)).all()
