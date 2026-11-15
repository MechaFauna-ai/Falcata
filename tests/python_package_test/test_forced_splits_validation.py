# coding: utf-8
"""A malformed forced-splits file is rejected instead of silently forcing something else.

json11 reads a missing or non-numeric value as 0, so "feature": "x" (or null, or no
"feature" key) forced a split on feature 0, "feature": 1.7 forced feature 1, and a split
without a numeric "threshold" was dropped without a word.
"""

import json
import re

import numpy as np
import pytest

import falcata as flc


def _train(tmp_path, spec):
    path = tmp_path / "forced.json"
    path.write_text(json.dumps(spec))
    rng = np.random.default_rng(0)
    X = rng.random((500, 3))
    y = X[:, 1] + 0.1 * rng.random(500)
    params = {"verbose": -1, "forcedsplits_filename": str(path), "num_leaves": 4}
    return flc.train(params, flc.Dataset(X, y), 1)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"feature": "x", "threshold": 0.5}, '"feature" must be an integer feature index, got "x"'),
        ({"feature": None, "threshold": 0.5}, '"feature" must be an integer feature index, got null'),
        ({"feature": 1.7, "threshold": 0.5}, '"feature" must be an integer feature index, got 1.7'),
        ({"feature": 0, "threshold": "abc"}, '"threshold" must be a number, got "abc"'),
        ({"feature": 0}, 'a split needs both "feature" and "threshold"'),
        ({"threshold": 0.5}, 'a split needs both "feature" and "threshold"'),
        ({"feature": 0, "threshold": 0.5, "left": {"feature": "q", "threshold": 0.2}}, 'got "q"'),
        ({"feature": 0, "threshold": 0.5, "right": {"feature": 2}}, 'a split needs both "feature" and "threshold"'),
        ({"feature": 0, "threshold": 0.5, "left": [1, 2]}, "every split must be a JSON object"),
    ],
)
def test_malformed_forced_split_is_rejected(tmp_path, spec, message):
    with pytest.raises(flc.basic.FalcataError, match=re.escape(message)):
        _train(tmp_path, spec)


def test_valid_forced_splits_still_apply(tmp_path):
    # an empty child forces nothing below that side of the split
    spec = {"feature": 0, "threshold": 0.5, "left": {"feature": 2, "threshold": 0.3}, "right": {}}
    tree = _train(tmp_path, spec).dump_model()["tree_info"][0]["tree_structure"]
    assert tree["split_feature"] == 0
    assert tree["left_child"]["split_feature"] == 2
