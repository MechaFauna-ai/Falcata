"""Predictor scores must reach the native Dataset in class-major order."""

from unittest.mock import Mock

import numpy as np
import pytest

import falcata as flc
from falcata.basic import Dataset, _InnerPredictor


@pytest.mark.parametrize("classes", [1, 3, 7])
@pytest.mark.parametrize("indices", [None, [4, 1, 3]])
def test_init_score_layout(classes, indices):
    scores = np.arange(5 * classes, dtype=np.float64).reshape(5, classes)
    selected = scores if indices is None else scores[indices]
    target = Mock(params=None, _need_slice=False)
    target.num_data.return_value = len(selected)
    predictor = Mock(num_class=classes)
    predictor.predict.return_value = scores
    Dataset._set_init_score_by_predictor(target, predictor, "input.txt", indices)
    actual = target.set_init_score.call_args.args[0]
    np.testing.assert_array_equal(actual, selected.T.ravel())
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous


def test_real_multiclass_predictor_init_scores():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(90, 3))
    y = np.argmax(x, axis=1)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "device_type": "cpu",
        "num_threads": 2,
        "min_data_in_leaf": 2,
        "verbosity": -1,
    }
    booster = flc.train(params, flc.Dataset(x, label=y), num_boost_round=3)
    ds = flc.Dataset(x, label=y, params=params, free_raw_data=False).construct()
    ds._set_predictor(_InnerPredictor.from_booster(booster, pred_parameter={}))
    np.testing.assert_array_equal(ds.get_init_score(), booster.predict(x, raw_score=True))
