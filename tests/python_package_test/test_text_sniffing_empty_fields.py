"""The CSV/TSV sniffers count a line's fields by its delimiters, empty fields included.

Format detection trimmed every sampled line with Common::Trim, which also strips tabs: a TSV line whose first or
last field is empty lost a tab, the line's column count differed from the others, and the file was refused as
"Unknown format of training data". Label detection for prediction counted Common::Split tokens, which drops empty
ones: when the first line of a label-less file had a missing value, the file was taken to carry a label column and
refused with "The number of features in data (2) is not the same as it was in training data (3)".
"""

import numpy as np
import pytest

import falcata as flc


@pytest.fixture(scope="module")
def booster():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 3))
    x[::4, 0] = np.nan
    x[1::4, 1] = np.nan
    y = np.nan_to_num(x[:, 0], nan=5.0) * 3 + np.nan_to_num(x[:, 1], nan=-2.0) + x[:, 2]
    return flc.train({"verbose": -1, "num_leaves": 8}, flc.Dataset(x, y), 10)


def _cell(v):
    return "" if np.isnan(v) else repr(float(v))


@pytest.mark.parametrize("sep", [",", "\t"])
@pytest.mark.parametrize("first", [[np.nan, 0.5, 1.0], [0.3, np.nan, 1.0], [np.nan, np.nan, 1.0]])
def test_predict_file_with_missing_values_in_its_first_line(tmp_path, booster, sep, first):
    rows = np.array([first, [1.5, 0.2, -1.0], [-2.0, np.nan, 0.5], [0.1, 0.4, 2.0]])
    path = tmp_path / "predict.txt"
    path.write_text("\n".join(sep.join(_cell(v) for v in row) for row in rows) + "\n")
    np.testing.assert_array_equal(booster.predict(str(path)), booster.predict(rows))


def test_tsv_training_file_with_empty_edge_fields_loads(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(200):
        a, b = rng.standard_normal(2)
        # label first; empty first feature on some lines, empty last feature on others (in the sampled head too)
        cells = [repr(float(i % 2)), "" if i % 5 == 1 else repr(float(a)), "" if i % 7 == 2 else repr(float(b))]
        rows.append("\t".join(cells))
    path = tmp_path / "train.tsv"
    path.write_text("\n".join(rows) + "\n")
    ds = flc.Dataset(str(path), params={"verbose": -1}).construct()
    assert ds.num_data() == 200
    assert ds.num_feature() == 2
