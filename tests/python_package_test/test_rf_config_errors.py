import numpy as np
import pytest

import falcata as flc

# boosting=rf configurations it cannot train were refused by bare CHECK macros
# ("Check failed: (config->bagging_freq > 0 && ...", "Check failed:
# (train_data->metadata().init_score()) == (nullptr)"), which name neither
# the parameter nor the fix.


def _data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((100, 3)), rng.standard_normal(100)


def test_rf_without_sampling_names_the_parameters():
    X, y = _data()
    with pytest.raises(flc.basic.FalcataError, match="boosting=rf needs each tree to see a sample"):
        flc.train({"objective": "regression", "boosting": "rf", "verbose": -1}, flc.Dataset(X, y), 2)


def test_rf_with_init_score_is_refused_clearly():
    X, y = _data()
    params = {"objective": "regression", "boosting": "rf", "bagging_fraction": 0.5, "bagging_freq": 1, "verbose": -1}
    with pytest.raises(flc.basic.FalcataError, match="boosting=rf does not support init_score"):
        flc.train(params, flc.Dataset(X, y, init_score=np.zeros(100)), 2)


def test_rf_reset_to_no_sampling_is_refused_clearly():
    X, y = _data()
    params = {"objective": "regression", "boosting": "rf", "bagging_fraction": 0.5, "bagging_freq": 1, "verbose": -1}
    bst = flc.train(params, flc.Dataset(X, y), 2, keep_training_booster=True)
    with pytest.raises(flc.basic.FalcataError, match="boosting=rf needs each tree to see a sample"):
        bst.reset_parameter({"bagging_fraction": 1.0})


@pytest.mark.parametrize(
    "sampling",
    [{"bagging_fraction": 0.5, "bagging_freq": 1}, {"feature_fraction": 0.5}, {"data_sample_strategy": "goss"}],
)
def test_rf_valid_sampling_still_trains(sampling):
    X, y = _data()
    bst = flc.train({"objective": "regression", "boosting": "rf", "verbose": -1, **sampling}, flc.Dataset(X, y), 2)
    assert np.all(np.isfinite(bst.predict(X)))
