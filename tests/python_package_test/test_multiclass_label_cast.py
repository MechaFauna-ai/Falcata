import numpy as np
import pytest

import falcata as flc

# MulticlassSoftmax::Init cast each label to int before range-checking it: a
# label beyond the int range (1e10, or inf, which the Dataset clamps to 1e38)
# was an undefined cast, reported as "found -2147483648" (UBSan:
# float-cast-overflow). The label is range-checked first and reported as is.


@pytest.mark.parametrize(("bad", "shown"), [(1e10, "1e\\+10"), (np.inf, "1e\\+38"), (-5e9, "-5e\\+09")])
def test_multiclass_out_of_int_range_label(bad, shown):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    y = rng.integers(0, 3, 60).astype(float)
    y[7] = bad
    params = {"objective": "multiclass", "num_class": 3, "verbose": -1}
    with pytest.raises(flc.basic.FalcataError, match=rf"Label must be in \[0, 3\), but found {shown} in label"):
        flc.train(params, flc.Dataset(X, y), num_boost_round=2)


def test_multiclass_in_range_labels_still_train():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    y = rng.integers(0, 3, 60).astype(float)
    bst = flc.train({"objective": "multiclass", "num_class": 3, "verbose": -1}, flc.Dataset(X, y), 2)
    assert np.all(np.isfinite(bst.predict(X)))
