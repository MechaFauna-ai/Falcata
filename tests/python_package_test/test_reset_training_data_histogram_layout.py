"""Booster.update(train_set=...) lays the cached histograms out for the new Dataset.

The histogram pool was sized and laid out for the training data the Booster was created with. A new training
Dataset (same bin mappers, through ``reference``) can need another layout, because its feature groups are stored
differently for its own rows. SerialTreeLearner::ResetTrainingData rebuilt the share state for the new layout but
kept the old histogram buffers: histogram construction wrote past them (ASan heap-buffer-overflow; glibc aborts
with "double free or corruption" at teardown) and the trees after the switch were split on misplaced histograms.
"""

import numpy as np
import pytest

import falcata as flc


def _data():
    rng = np.random.default_rng(794032753)
    x = rng.standard_normal((1000, 60))
    x[rng.random(x.shape) < 0.7] = 0.0  # sparse columns
    return x, x @ rng.standard_normal(60) + 0.3 * rng.standard_normal(1000)


@pytest.mark.parametrize("layout", [{"force_col_wise": True}, {"force_row_wise": True}])
def test_update_train_set_matches_a_fresh_learner(layout):
    x, y = _data()
    params = {
        "objective": "regression",
        "device_type": "cpu",
        "num_leaves": 3,
        "min_data_in_leaf": 1,
        "max_bin": 511,
        "learning_rate": 0.3,
        "deterministic": True,
        "num_threads": 1,
        "seed": 1,
        "verbose": -1,
        **layout,
    }
    train = flc.Dataset(x[:600], y[:600], params=params, free_raw_data=False)
    booster = flc.Booster(params, train)
    for _ in range(3):
        booster.update()
    initial = booster.model_to_string()
    switched = flc.Dataset(x[600:800], y[600:800], params=params, reference=train, free_raw_data=False)
    booster.update(train_set=switched)
    for _ in range(3):
        booster.update()

    # the training score follows the model
    l2 = next(value for _, name, value, _ in booster.eval_train() if name == "l2")
    np.testing.assert_allclose(l2, np.mean((booster.predict(x[600:800]) - y[600:800]) ** 2), rtol=1e-6)
    # and the trees grown after the switch are those a learner built for the new Dataset grows
    expected = flc.train(
        params,
        flc.Dataset(x[600:800], y[600:800], params=params, reference=train),
        num_boost_round=4,
        init_model=flc.Booster(model_str=initial),
    )
    np.testing.assert_allclose(booster.predict(x), expected.predict(x), rtol=0, atol=1e-10)
