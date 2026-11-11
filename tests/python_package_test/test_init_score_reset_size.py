# coding: utf-8
"""Replacing the init score with one of a different class count resizes it.

Metadata::SetInitScoresFromIterator only resized init_score_ when it was
empty, so a second set_init_score with more classes wrote past the end of the
old buffer and get_init_score read back past it.
"""

import numpy as np

import falcata as flc


def test_set_init_score_grows_and_shrinks():
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.normal(size=(n, 3))
    y = rng.integers(0, 3, size=n).astype(np.float64)
    ds = flc.Dataset(X, label=y, params={"verbose": -1}, free_raw_data=False).construct()

    one = rng.normal(size=n)
    ds.set_init_score(one)
    np.testing.assert_array_equal(ds.get_field("init_score"), one)

    three = rng.normal(size=3 * n)
    ds.set_init_score(three)
    np.testing.assert_array_equal(np.ravel(ds.get_field("init_score"), order="F"), three)

    ds.set_init_score(one)
    np.testing.assert_array_equal(ds.get_field("init_score"), one)
