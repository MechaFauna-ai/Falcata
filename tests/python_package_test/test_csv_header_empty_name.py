# coding: utf-8
"""A CSV header with an empty column name fails clearly.

Common::Split drops empty fields, so an empty name -- pandas' DataFrame.to_csv() writes
one for the index column (",a,b,c") -- moved every later name onto the wrong column and
the load failed on an internal size CHECK ("Check failed: (dataset->num_total_features_)
== (static_cast<int>(feature_names_.size()))").
"""

import numpy as np
import pandas as pd
import pytest

import falcata as flc


def test_pandas_to_csv_with_index_fails_clearly(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((50, 4)), columns=["label", "a", "b", "c"])
    path = tmp_path / "train.csv"
    df.to_csv(path)  # index column with an empty header cell
    with pytest.raises(flc.basic.FalcataError, match="has an empty name for column 0; give every column a name"):
        flc.Dataset(str(path), params={"verbose": -1, "header": True}).construct()


@pytest.mark.parametrize(("header", "column"), [("label,a,,c", 2), ("label,a,b,", 3)])
def test_empty_header_field_fails_clearly(tmp_path, header, column):
    rng = np.random.default_rng(0)
    path = tmp_path / "train.csv"
    rows = "\n".join(",".join(str(v) for v in row) for row in rng.random((50, 4)))
    path.write_text(header + "\n" + rows + "\n")
    with pytest.raises(flc.basic.FalcataError, match=f"has an empty name for column {column}"):
        flc.Dataset(str(path), params={"verbose": -1, "header": True}).construct()


def test_named_header_still_loads(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((50, 4)), columns=["label", "a", "b", "c"])
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    ds = flc.Dataset(str(path), params={"verbose": -1, "header": True}).construct()
    assert ds.get_feature_name() == ["a", "b", "c"]
