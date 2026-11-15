"""A loaded tree whose decision_type holds no valid missing type must be refused.

decision_type keeps the node's MissingType (None 0, Zero 1, NaN 2) in bits 2-3, and the writer never stores 3. A
model carrying 3 used to load: prediction (and the if-else code) read it as None, sending NaN down the zero path,
while dump_model reported missing_type "NaN" with the node's default direction -- so the dump, and everything built
on it (trees_to_dataframe, converters), described a model that predicted differently.
"""

import re

import numpy as np
import pytest

import falcata as flc


@pytest.fixture(scope="module")
def model_str():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 1))
    x[::7, 0] = np.nan
    y = np.where(np.isnan(x[:, 0]), 5.0, x[:, 0])
    booster = flc.train({"verbose": -1, "num_leaves": 2, "min_data_in_leaf": 5}, flc.Dataset(x, y), 1)
    return booster.model_to_string()


def _with_decision_type(model_str, value):
    out, n = re.subn(r"decision_type=\d+", f"decision_type={value}", model_str, count=1)
    assert n == 1
    return out


def test_valid_missing_types_load(model_str):
    decision_type = int(re.search(r"decision_type=(\d+)", model_str).group(1))
    for missing_type in (0, 1, 2):
        flc.Booster(model_str=_with_decision_type(model_str, (decision_type & 3) | (missing_type << 2)))


@pytest.mark.parametrize("default_left", [0, 2])
def test_missing_type_3_is_refused(model_str, default_left):
    with pytest.raises(flc.basic.FalcataError, match="missing type 3"):
        flc.Booster(model_str=_with_decision_type(model_str, 12 | default_left))
