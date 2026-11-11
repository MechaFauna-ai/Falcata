# coding: utf-8
"""DatasetLoader validates forced-bins files and label_column.

- GetForcedBins ignored the JSON parse error (failing later in a bare CHECK),
  only upper-bounded the feature index (a negative one indexed out of
  bounds), and removed only adjacent duplicate bounds.
- SetHeader erased feature_names_[label_idx_] with an unchecked numeric
  label_column.
"""

import json

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError


@pytest.fixture
def csv_with_header(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 4, size=(500, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=500)
    path = tmp_path / "data.csv"
    with open(path, "w") as f:
        f.write("y,a,b,c\n")
        np.savetxt(f, np.column_stack([y, X]), delimiter=",")
    return path, X, y


def _num_bins(X, y, forced, tmp_path, name):
    path = tmp_path / name
    path.write_text(json.dumps(forced))
    ds = flc.Dataset(X, label=y, params={"forcedbins_filename": str(path), "max_bin": 255, "verbose": -1})
    ds.construct()
    return ds.feature_num_bin(0)


def test_forced_bins_non_adjacent_duplicates_removed(csv_with_header, tmp_path):
    _, X, y = csv_with_header
    dup = _num_bins(X, y, [{"feature": 0, "bin_upper_bound": [1.0, 2.0, 1.0]}], tmp_path, "dup.json")
    uniq = _num_bins(X, y, [{"feature": 0, "bin_upper_bound": [1.0, 2.0]}], tmp_path, "uniq.json")
    assert dup == uniq


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('[{"feature": 0, "bin_upper_bound": [1.0', "Could not parse forced bins"),
        ('{"feature": 0}', "must contain a JSON array"),
        ('[{"feature": -1, "bin_upper_bound": [1.0]}]', "names feature -1"),
        ('[{"feature": 3, "bin_upper_bound": [1.0]}]', "names feature 3"),
    ],
)
def test_forced_bins_bad_file_raises(csv_with_header, tmp_path, content, match):
    _, X, y = csv_with_header
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(FalcataError, match=match):
        flc.Dataset(X, label=y, params={"forcedbins_filename": str(path), "verbose": -1}).construct()


def test_label_column_out_of_range_raises(csv_with_header):
    path, _, _ = csv_with_header
    with pytest.raises(FalcataError, match="label_column 9 is out of range"):
        flc.Dataset(str(path), params={"header": True, "label_column": "9", "verbose": -1}).construct()
