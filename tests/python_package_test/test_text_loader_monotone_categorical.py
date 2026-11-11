# coding: utf-8
"""Monotone constraints on categorical features are rejected for text files too.

DatasetLoader::ConstructFromSampleData raised for a monotone constraint on a
categorical feature, but the text-file bin construction paths
(ConstructBinMappersFromTextData) silently accepted the same configuration.
"""

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError


@pytest.mark.parametrize("two_round", [False, True])
def test_text_file_monotone_categorical_raises(tmp_path, two_round):
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 5, 500), rng.normal(size=500)])
    y = X[:, 1]
    path = tmp_path / "cat.csv"
    np.savetxt(path, np.column_stack([y, X]), delimiter=",")
    params = {"objective": "regression", "monotone_constraints": [1, 0], "two_round": two_round, "verbose": -1}
    with pytest.raises(FalcataError, match="cannot be monotone with respect to categorical"):
        flc.Dataset(str(path), categorical_feature=[0], params=params).construct()
    # the same config from memory already raised
    with pytest.raises(FalcataError, match="cannot be monotone with respect to categorical"):
        flc.Dataset(X, label=y, categorical_feature=[0], params=params).construct()
    # and a constraint on the numerical column alone is fine
    flc.Dataset(str(path), categorical_feature=[0], params=dict(params, monotone_constraints=[0, 1])).construct()
