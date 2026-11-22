"""precise_float_parser=true reads missing values and spacing like the default text parser.

The precise parser handed every token to fast_double_parser and then strtod, and failed when neither took it. An
empty field (the usual way to write a missing value in CSV), NA or null failed with "no conversion to double", and
a space between a value and the delimiter failed with "Input format error". The default parser reads all of these
(the missing ones as NaN), so a file that loaded by default could not be loaded with the precise parser.
"""

import numpy as np
import pytest

import falcata as flc

_ROWS = ["1,0.5,2", "0,,3", "1,NA,1", "0,null,4", "1,1.5 ,2", "0, 2.5,3", "1,nan,5", "0,3.5,1", "1,-,2"]


@pytest.mark.parametrize("sep", [",", "\t"])
def test_precise_parser_reads_missing_tokens_like_default(tmp_path, sep):
    path = tmp_path / "data.txt"
    rows = _ROWS + (["0,4.5,"] if sep == "," else [])  # an empty last field (TSV sniffing trims it: separate issue)
    path.write_text("\n".join(row.replace(",", sep) for row in rows * 20) + "\n")
    # two trainings are compared bit for bit: on CUDA (unset device_type with a usable GPU) non-quantized training
    # repeats only with the deterministic histograms (opt-in since 1.2; CPU ignores the key)
    params = {
        "verbose": -1,
        "min_data_in_bin": 1,
        "min_data_in_leaf": 2,
        "num_leaves": 4,
        "cuda_plan": "auto,nonquant_deterministic:on",
    }
    datasets = {}
    for precise in (False, True):
        ds = flc.Dataset(str(path), params={**params, "precise_float_parser": precise}).construct()
        datasets[precise] = ds
    assert datasets[True].num_data() == datasets[False].num_data() == len(rows) * 20
    np.testing.assert_array_equal(datasets[True].get_label(), datasets[False].get_label())
    boosters = {p: flc.train({**params, "precise_float_parser": p}, ds, 3) for p, ds in datasets.items()}
    probe = np.array([[np.nan, 2.0], [0.5, np.nan], [4.0, 3.0], [1.5, 1.0]])
    np.testing.assert_array_equal(boosters[True].predict(probe), boosters[False].predict(probe))
    # predicting the same file with each parser agrees too
    np.testing.assert_array_equal(
        boosters[False].predict(str(path), precise_float_parser=True),
        boosters[False].predict(str(path), precise_float_parser=False),
    )


def test_precise_parser_still_refuses_unknown_tokens(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1,0.5\n0,abc\n1,1.5\n")
    with pytest.raises(flc.basic.FalcataError, match="Unknown token abc"):
        flc.Dataset(str(path), params={"verbose": -1, "precise_float_parser": True}).construct()
