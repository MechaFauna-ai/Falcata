# coding: utf-8
"""String parameters survive a model round trip into Booster.params.

GBDT::GetLoadedParam split each "[name: value]" line at every ':' and wrote string values
into JSON unescaped. A value containing ':' (a Windows path, host:port, a cuda_plan
key:value override) came back empty or cut, and a value containing a backslash (any
Windows path) made the JSON invalid, so Python could not load the model at all. refit
builds its Dataset and Booster from these loaded parameters.
"""

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("path", ["a:b/fs.json", "w\\dir/fs.json", 'q"x/fs.json', "C:\\data\\fs.json"])
def test_string_param_round_trips(tmp_path, path):
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"feature": 0, "threshold": 0.5}')
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = X[:, 0]
    booster = flc.train({"verbose": -1, "forcedsplits_filename": str(target)}, flc.Dataset(X, y), 2)
    loaded = flc.Booster(model_str=booster.model_to_string())
    assert loaded.params["forcedsplits_filename"] == str(target)
    assert loaded.refit(X, y).num_trees() == 2


def test_vector_of_strings_param_round_trips():
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    booster = flc.train({"verbose": -1, "metric": ["l1", "l2"]}, flc.Dataset(X, X[:, 0]), 2)
    assert flc.Booster(model_str=booster.model_to_string()).params["metric"] == ["l1", "l2"]
