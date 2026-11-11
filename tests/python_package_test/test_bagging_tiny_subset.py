import numpy as np
import pytest

import falcata as flc

# bagging_fraction * num_data rounding down to 0 (7 rows at 0.1) made the
# bagging subset Dataset(0) fail with "Check failed: (num_data) > (0)" before
# the first tree, although each bag is drawn row by row and is resized to its
# actual count. GOSS already clamps its subset to one row.


@pytest.mark.parametrize(
    "bagging",
    [
        {"bagging_fraction": 0.1, "bagging_freq": 1},
        {"pos_bagging_fraction": 0.1, "neg_bagging_fraction": 0.1, "bagging_freq": 1},
    ],
)
def test_bagging_fraction_below_one_row(bagging):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((7, 3))
    y = np.array([0, 1, 0, 1, 1, 0, 1], dtype=float)
    params = {"objective": "binary", "min_data_in_leaf": 1, "verbose": -1, **bagging}
    bst = flc.train(params, flc.Dataset(X, y), num_boost_round=5)
    assert np.all(np.isfinite(bst.predict(X)))
