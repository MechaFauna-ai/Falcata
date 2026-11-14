"""dump_model must JSON-escape feature names.

Names may contain a backslash or a control character (only the JSON
structural characters are rejected). They were written into the dump
unescaped, so a tab made dump_model() raise JSONDecodeError and a backslash
either broke the JSON or silently decoded to another name ("a\\b" -> "a<BS>").
"""

import json

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("name", ["tab\tname", "back\\slash", "a\\b", "ctl\x01x", "uni_é_名"])
def test_dump_model_round_trips_feature_names(name):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((200, 2))
    x[:, 1] = rng.integers(0, 4, size=200)
    names = [name, name + "_cat"]
    ds = flc.Dataset(x, label=x[:, 0] + x[:, 1], feature_name=names, categorical_feature=[1])
    booster = flc.train({"verbose": -1, "num_threads": 1}, ds, num_boost_round=3)
    dump = booster.dump_model()
    assert dump["feature_names"] == names
    assert set(dump["feature_infos"]) == set(names)
    assert set(dump["feature_importances"]) <= set(names)
    assert dump["feature_importances"]
    # the dump is plain JSON a strict parser accepts
    json.loads(json.dumps(dump))
