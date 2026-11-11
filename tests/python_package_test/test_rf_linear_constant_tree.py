"""rf + linear_tree must not crash when a class gets a constant tree.

RF scores a class it skips (class_need_train_ false) with a plain constant
Tree, through the linear learner's AddPredictionToScore, which read that
tree's (empty) linear leaf arrays: a null dereference / segfault on tiny
multiclass data. Inherited from LightGBM.
"""

import numpy as np

import falcata as flc


def test_rf_linear_tree_multiclass_tiny_data_trains():
    x = np.array(
        [
            [1.0, 1.72154743, -0.87504344],
            [1.0, -0.77835307, 0.24846097],
            [1.0, 0.23316359, -0.33188526],
            [1.0, -0.84536522, -0.6310183],
            [1.0, 0.54655645, 0.36016879],
        ]
    )
    y = np.array([2.0, 0.0, 0.0, 2.0, 2.0])  # class 1 never occurs
    params = {
        "verbose": -1,
        "num_threads": 1,
        "seed": 758,
        "deterministic": True,
        "num_class": 3,
        "objective": "multiclass",
        "boosting": "rf",
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "num_leaves": 7,
        "min_data_in_leaf": 1,
        "linear_tree": True,
        "extra_trees": True,
    }
    booster = flc.train(params, flc.Dataset(x, label=y), num_boost_round=2)
    pred = booster.predict(x)
    assert pred.shape == (5, 3)
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred.sum(axis=1), 1.0)
