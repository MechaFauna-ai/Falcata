import numpy as np
import pytest

import falcata as flc

# A dataset whose features are all unusable (here: constant) trains constant
# trees, but any parameter reset failed with "Check failed:
# (train_data->num_features()) > (0)" in HistogramPool::ResetConfig, e.g. a
# learning-rate schedule through the reset_parameter callback.


@pytest.mark.parametrize("col_row", [{}, {"force_col_wise": True}, {"force_row_wise": True}])
def test_reset_parameter_without_usable_features(col_row):
    X = np.full((40, 2), 3.25)
    y = np.arange(40.0)
    params = {"objective": "regression", "verbose": -1, **col_row}
    bst = flc.train(
        params,
        flc.Dataset(X, y),
        num_boost_round=3,
        callbacks=[flc.reset_parameter(learning_rate=[0.1, 0.09, 0.08])],
    )
    np.testing.assert_allclose(bst.predict(X), y.mean())


def test_booster_reset_parameter_without_usable_features():
    X = np.full((40, 2), 3.25)
    y = np.arange(40.0)
    bst = flc.train({"objective": "regression", "verbose": -1}, flc.Dataset(X, y), 1, keep_training_booster=True)
    bst.reset_parameter({"lambda_l2": 1.0, "learning_rate": 0.05})
    bst.update()
    assert np.all(np.isfinite(bst.predict(X)))
