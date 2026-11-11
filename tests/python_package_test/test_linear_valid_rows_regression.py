"""A linear fit needs enough complete rows, rather than feature values."""

import numpy as np

import falcata as flc


def test_refit_linear_fallback_with_two_complete_rows():
    x = np.random.default_rng(8).normal(size=(100, 2))
    y = x[:, 0] + x[:, 1]
    x[0, 0] = np.nan
    booster = flc.train(
        {
            "objective": "regression",
            "linear_tree": True,
            "num_leaves": 4,
            "min_data_in_leaf": 5,
            "num_threads": 2,
            "verbosity": -1,
        },
        flc.Dataset(x, label=y),
        num_boost_round=3,
    )
    leaf_ids = booster.predict(x, pred_leaf=True)[:, -1]
    rows = np.flatnonzero((leaf_ids == 2) & np.isfinite(x).all(axis=1))[:2]
    assert len(rows) == 2
    refit_x = np.full((100, 2), np.nan)
    refit_x[:2] = x[rows]
    refitted = booster.refit(
        refit_x,
        np.arange(100) * 0.1,
        decay_rate=0.0,
        dataset_params={"min_data_in_leaf": 1, "feature_pre_filter": False},
    )
    tree = refitted.dump_model()["tree_info"][-1]["tree_structure"]
    leaf = tree["left_child"]["right_child"]
    assert leaf["leaf_index"] == 2
    assert leaf["leaf_features"] == [0, 1]
    # Two rows cannot fit two coefficients plus an intercept.
    np.testing.assert_array_equal(leaf["leaf_coeff"], [0.0, 0.0])
