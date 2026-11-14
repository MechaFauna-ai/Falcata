import numpy as np
import pytest

import falcata as flc

# Ranking label checks cast the label to int before range-checking it (and
# rank_xendcg computes 2^label through an int with no check): a label beyond
# the int range is undefined behaviour (UBSan), and rank_xendcg labels from
# 1024 on made 2^label inf and the gradients NaN.


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    y = rng.integers(0, 4, 60).astype(float)
    return X, y


@pytest.mark.parametrize("bad", [1e30, 5000.0, -1e30])
def test_rank_xendcg_rejects_out_of_range_labels(bad):
    X, y = _data()
    y[3] = bad
    params = {"objective": "rank_xendcg", "device_type": "cpu", "verbose": -1, "metric": "None"}
    with pytest.raises(flc.basic.FalcataError, match="rank_xendcg: label must be in"):
        flc.train(params, flc.Dataset(X, y, group=[30, 30]), num_boost_round=2)


@pytest.mark.parametrize("bad", [1e30, 1e10])
def test_lambdarank_rejects_huge_labels(bad):
    X, y = _data()
    y[3] = bad
    params = {"objective": "lambdarank", "device_type": "cpu", "verbose": -1}
    with pytest.raises(flc.basic.FalcataError, match="is not less than the number of label mappings"):
        flc.train(params, flc.Dataset(X, y, group=[30, 30]), num_boost_round=2)


def test_rank_xendcg_ordinary_labels_still_train():
    X, y = _data()
    params = {"objective": "rank_xendcg", "device_type": "cpu", "verbose": -1, "metric": "None"}
    bst = flc.train(params, flc.Dataset(X, y, group=[30, 30]), num_boost_round=3)
    assert np.all(np.isfinite(bst.predict(X)))
