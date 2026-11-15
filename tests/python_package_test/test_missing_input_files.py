# coding: utf-8
"""A forced-bins or parser-config file the user names must exist.

A missing forcedbins_filename was only a warning ("Could not open ... Will ignore.",
hidden at verbose < 0), and a missing parser_config_file was read as empty: a mistyped
path trained without the forced bins, or parsed the data with the default parser
instead of the configured one. A missing forcedsplits_filename was already an error.
"""

import numpy as np
import pytest

import falcata as flc


def test_missing_forced_bins_file_is_an_error(tmp_path):
    X = np.random.default_rng(0).random((200, 3))
    missing = tmp_path / "no_such_forced_bins.json"
    with pytest.raises(flc.basic.FalcataError, match="Could not open forced bins file"):
        flc.Dataset(X, X[:, 0], params={"verbose": -1, "forcedbins_filename": str(missing)}).construct()


def test_missing_parser_config_file_is_an_error(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "train.csv"
    np.savetxt(data, rng.random((200, 3)), delimiter=",")
    missing = tmp_path / "no_such_parser.json"
    with pytest.raises(flc.basic.FalcataError, match="Could not open parser config file"):
        flc.Dataset(str(data), params={"verbose": -1, "parser_config_file": str(missing)}).construct()


def test_text_file_without_parser_config_still_loads(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "train.csv"
    np.savetxt(data, rng.random((200, 3)), delimiter=",")
    assert flc.Dataset(str(data), params={"verbose": -1}).construct().num_data() == 200
