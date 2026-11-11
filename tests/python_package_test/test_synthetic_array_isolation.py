"""Cached synthetic fixtures must remain independent between callers."""

import numpy as np

from .utils import make_synthetic_regression


def test_synthetic_regression_arrays_are_independent():
    x, y = make_synthetic_regression(n_samples=17, random_state=71029)
    expected_x, expected_y = x.copy(), y.copy()
    x[:] = 999
    y[:] = 999
    actual_x, actual_y = make_synthetic_regression(n_samples=17, random_state=71029)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, expected_y)
