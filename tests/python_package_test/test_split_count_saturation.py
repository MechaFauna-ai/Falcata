import numpy as np
import pytest

import falcata as flc

# Split-side data counts are estimated from hessians (hess * num_data /
# sum_hessian). With large opposite-sign weights sum_hessian is tiny, the
# estimate saturates at INT_MAX (RoundInt), and adding such estimates
# overflowed int: undefined behaviour (UBSan: "signed integer overflow:
# -2147483206 + -714", feature_histogram.hpp). The counts now saturate.
# This test only shows the overflow under an UBSan build; in a normal build it
# checks that training stays finite. Reported by the fz-api fuzzer.


@pytest.mark.parametrize("w_pair", [(1e38, -1e38), (1e30, -1e30), (5.0, -4.999)])
@pytest.mark.parametrize("force", [{"force_col_wise": True}, {"force_row_wise": True}])
def test_opposite_extreme_weights_keep_split_counts_defined(w_pair, force):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + rng.normal(size=60)
    w = np.full(60, 0.5)
    w[3], w[7] = w_pair
    params = {
        "objective": "regression",
        "verbose": -1,
        "min_data_in_leaf": 3,
        "boost_from_average": False,
        "device_type": "cpu",
        **force,
    }
    bst = flc.train(params, flc.Dataset(X, y, weight=w), num_boost_round=2)
    assert np.all(np.isfinite(bst.predict(X)))
