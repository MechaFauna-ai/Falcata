"""Categorical tree highlighting treats non-finite values as missing."""

import math

import pytest

from falcata.plotting import _determine_direction_for_categorical_split


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_categorical_plot_value_goes_right(value):
    assert _determine_direction_for_categorical_split(value, "0||2") == "right"
