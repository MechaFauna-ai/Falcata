"""Malformed feature metadata should fail clearly when dumping a loaded model."""

import re

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("info", ["[0", "[]", "[0:1:2]", ":"])
def test_dump_malformed_feature_info(info):
    x = np.arange(40.0).reshape(-1, 1)
    model = flc.train(
        {"objective": "regression", "num_threads": 1, "min_data_in_leaf": 2, "verbosity": -1},
        flc.Dataset(x, label=x[:, 0]),
        num_boost_round=1,
    )
    text = re.sub(r"^feature_infos=.*$", f"feature_infos={info}", model.model_to_string(), flags=re.MULTILINE)
    with pytest.raises(flc.basic.FalcataError, match="feature_infos"):
        flc.Booster(model_str=text).dump_model()
