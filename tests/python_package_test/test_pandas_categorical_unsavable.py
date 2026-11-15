# coding: utf-8
"""pandas categories the model cannot store fail clearly, and before a model file is written.

The JSON default hook returned unknown objects unchanged, so Timestamp, bytes or Decimal
categories made json report "Circular reference detected" -- also in flc.train's own final
model round trip. save_model wrote the native model first, so a failed save left a file
without its pandas categories.
"""

import decimal

import numpy as np
import pandas as pd
import pytest

import falcata as flc


def _frame(categories):
    rng = np.random.default_rng(0)
    n = 300
    col = pd.Categorical([categories[i % 3] for i in range(n)], categories=list(categories))
    df = pd.DataFrame({"c": col, "x": rng.random(n)})
    return df, (np.arange(n) % 3 == 0) + df["x"]


@pytest.mark.parametrize(
    "categories",
    [
        list(pd.to_datetime(["2020-01-01", "2021-06-01", "2022-03-03"])),
        [b"a", b"b", b"c"],
        [decimal.Decimal("1.5"), decimal.Decimal("2.5"), decimal.Decimal("3.5")],
    ],
)
@pytest.mark.parametrize("suffix", [".txt", ".falb"])
def test_unsavable_categories_fail_clearly_without_writing(tmp_path, categories, suffix):
    df, y = _frame(categories)
    params = {"verbose": -1, "min_data_per_group": 5, "cat_smooth": 1}
    booster = flc.train(params, flc.Dataset(df, y), 3, keep_training_booster=True)
    path = tmp_path / f"model{suffix}"
    with pytest.raises(TypeError, match="Cannot store a pandas category of type"):
        booster.save_model(path)
    assert not path.exists()
    with pytest.raises(TypeError, match="Cannot store a pandas category of type"):
        booster.model_to_string()
    with pytest.raises(TypeError, match="Cannot store a pandas category of type"):
        booster.dump_model()


def test_savable_categories_still_round_trip(tmp_path):
    df, y = _frame(["a", "b", "c"])
    booster = flc.train({"verbose": -1, "min_data_per_group": 5, "cat_smooth": 1}, flc.Dataset(df, y), 3)
    for suffix in (".txt", ".falb"):
        path = tmp_path / f"model{suffix}"
        booster.save_model(path)
        loaded = flc.Booster(model_file=path)
        assert loaded.pandas_categorical == [["a", "b", "c"]]
        np.testing.assert_allclose(loaded.predict(df), booster.predict(df))
