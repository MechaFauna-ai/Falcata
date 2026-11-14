"""A NaN numerical threshold must survive the FALB round trip.

A text model may carry threshold=nan (every comparison with it is false, so
the split sends every non-missing value right). The FALB writer keyed its
threshold dictionary with std::less<double>, which is not a strict weak
ordering once a NaN is present: the NaN was merged with an unrelated value
and the FALB copy predicted differently.
"""

import numpy as np
import pytest

import falcata as flc


def _model_with_nan_threshold():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 2))
    booster = flc.train(
        {"verbose": -1, "num_threads": 1, "num_leaves": 4}, flc.Dataset(x, label=x[:, 0] + x[:, 1]), num_boost_round=3
    )
    text = booster.model_to_string()
    start = text.index("Tree=1\n")
    i = text.index("\nthreshold=", start) + 1
    j = text.index("\n", i)
    values = text[i:j].split("=", 1)[1].split(" ")
    values[0] = "nan"
    text = text[:i] + "threshold=" + " ".join(values) + text[j:]
    s = text.index("tree_sizes=")  # the edit changes a block size
    text = text[:s] + text[text.index("\n", s) + 1 :]
    return flc.Booster(model_str=text), x


@pytest.mark.parametrize("kwargs", [{}, {"compress_level": 0}, {"with_stats": True, "with_diagnostics": True}])
def test_nan_threshold_falb_round_trip(kwargs):
    booster, x = _model_with_nan_threshold()
    restored = flc.Booster(model_bin=booster.model_to_binary(**kwargs))
    np.testing.assert_array_equal(restored.predict(x, raw_score=True), booster.predict(x, raw_score=True))
    np.testing.assert_array_equal(restored.predict(x, pred_leaf=True), booster.predict(x, pred_leaf=True))
