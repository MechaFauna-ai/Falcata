"""Linear refit capacity follows the model, including leaves with no rows."""

import numpy as np

import falcata as flc


def test_refit_larger_linear_model_with_small_leaf_limit():
    x = np.random.default_rng(8).normal(size=(100, 2))
    booster = flc.train(
        {
            "objective": "regression",
            "linear_tree": True,
            "num_leaves": 8,
            "min_data_in_leaf": 5,
            "num_threads": 1,
            "verbosity": -1,
        },
        flc.Dataset(x, label=x[:, 0] + x[:, 1]),
        num_boost_round=3,
    )
    # A new training limit does not change the loaded trees' leaf counts.
    booster.params["num_leaves"] = 2
    refit_x = np.tile(x[0], (100, 1))
    refitted = booster.refit(refit_x, np.arange(100) * 0.1, decay_rate=0.0)
    assert np.isfinite(refitted.predict(refit_x)).all()
    assert [t["num_leaves"] for t in refitted.dump_model()["tree_info"]] == [
        t["num_leaves"] for t in booster.dump_model()["tree_info"]
    ]
