"""Predictor scores must reach the native Dataset in class-major order."""

from unittest.mock import Mock

import numpy as np
import pytest

from falcata.basic import Dataset


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
