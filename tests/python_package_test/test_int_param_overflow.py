import numpy as np
import pytest

import falcata as flc

# Config::GetInt parsed int parameters with Common::Atoi, which accumulates in
# int: a value beyond the int range (num_class derived from a huge label,
# say) overflowed (undefined behaviour; UBSan: "signed integer overflow:
# 1000000000 * 10") and was used as a garbage value. It is now rejected like
# any other malformed int.


@pytest.mark.parametrize("value", ["10000000000", "-10000000000", "99999999999999999999999"])
def test_out_of_range_int_parameter_is_rejected(value):
    X = np.random.default_rng(0).standard_normal((50, 2))
    y = np.zeros(50)
    with pytest.raises(flc.basic.FalcataError, match="Parameter num_leaves should be of type int"):
        flc.Dataset(X, y, params={"num_leaves": value, "verbose": -1}).construct()


@pytest.mark.parametrize("value", ["2147483647", "-2147483648", "31", " 7"])
def test_int_parameter_limits_still_parse(value):
    X = np.random.default_rng(0).standard_normal((50, 2))
    y = np.zeros(50)
    flc.Dataset(X, y, params={"max_depth": value, "verbose": -1}).construct()
