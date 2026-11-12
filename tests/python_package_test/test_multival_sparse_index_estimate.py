import numpy as np
import pytest

import falcata as flc

# The sparse multi-value bin picks its row index type from the sparse rate of
# the bin-construction sample. With bin_construct_sample_cnt=1 that estimate
# is ~0 entries, a 2-byte index is chosen, and the ~150k real entries do not
# fit: master refused to train ("Sparse multi-value bin holds N entries, more
# than its 2-byte row index can count"); before #341 the offsets wrapped.
# The dataset now rebuilds the bin from the exact count.


@pytest.mark.parametrize("extra", [{}, {"force_row_wise": True}, {"max_bin": 2}])
def test_underestimated_sparse_multival_bin_trains(extra):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 100))
    y = X[:, 0] + rng.standard_normal(3000)
    params = {
        "objective": "regression",
        "bin_construct_sample_cnt": 1,
        "feature_pre_filter": False,
        "deterministic": True,
        "num_threads": 2,
        "device_type": "cpu",
        "verbose": -1,
        **extra,
    }
    bst = flc.train(params, flc.Dataset(X, y), num_boost_round=3)
    assert np.all(np.isfinite(bst.predict(X)))
    # same model as with the multi-value bin built dense (no estimate involved)
    dense = flc.train({**params, "force_col_wise": True, "force_row_wise": False}, flc.Dataset(X, y), 3)
    np.testing.assert_allclose(bst.predict(X), dense.predict(X), rtol=1e-9, atol=1e-12)
