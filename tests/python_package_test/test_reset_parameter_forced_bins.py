# coding: utf-8
"""Re-sending an unchanged forcedbins_filename to ResetParameter is accepted.

Booster::CheckDatasetResetConfig raised "Cannot change forced bins" whenever
the key was present, even with the same value; every sibling check only
raises when the value actually changes.
"""

import json

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError


def test_reset_parameter_same_forced_bins(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 4, size=(300, 2))
    y = X[:, 0]
    forced = tmp_path / "forced.json"
    forced.write_text(json.dumps([{"feature": 0, "bin_upper_bound": [1.0, 2.0]}]))
    other = tmp_path / "other.json"
    other.write_text(json.dumps([{"feature": 0, "bin_upper_bound": [3.0]}]))
    params = {"objective": "regression", "forcedbins_filename": str(forced), "verbose": -1}
    booster = flc.Booster(params, flc.Dataset(X, label=y, params=params))
    booster.update()
    booster.reset_parameter({"forcedbins_filename": str(forced), "learning_rate": 0.05})
    with pytest.raises(FalcataError, match="Cannot change forced bins"):
        booster.reset_parameter({"forcedbins_filename": str(other)})
    booster.update()
