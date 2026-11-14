"""LibSVM files with a negative or out-of-range feature index are rejected.

A ``-1:value`` token used to index the per-feature sample arrays at -1
(heap-buffer-overflow under ASan), and an index past the int range overflowed
the integer parser (UBSan signed overflow) and wrapped around. An index with
trailing garbage was cut at the garbage by the column counter, so ``1000000000~00:1``
made the loader allocate tens of GB.
"""

import pytest

import falcata as flc


@pytest.mark.parametrize(
    ("token", "message"),
    [
        ("-1:0.5", "LibSVM feature index -1 is negative"),
        ("-7:2", "LibSVM feature index -7 is negative"),
        ("2147483647:1", "is too large"),
        ("99999999999:1", "is too large"),
        # trailing garbage: the column counter used to read this as index 1e9 and size the
        # per-feature arrays for a billion features (tens of GB) before the line parser ran
        ("1000000000~00:1", "LibSVM feature index 1000000000~00 is not an integer"),
        ("3x:1", "LibSVM feature index 3x is not an integer"),
    ],
)
def test_libsvm_bad_feature_index_is_rejected(tmp_path, token, message):
    path = tmp_path / "train.svm"
    lines = [f"{i % 2} 0:{i} 1:{i * 0.5}" for i in range(20)]
    lines[3] = f"1 0:3 {token}"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(flc.basic.FalcataError, match=message):
        flc.Dataset(str(path), params={"verbose": -1}).construct()


def test_libsvm_zero_and_plus_signed_indices_still_parse(tmp_path):
    path = tmp_path / "train.svm"
    path.write_text("".join(f"{i % 2} -0:{i} +1:{i * 0.5}\n" for i in range(20)))
    data = flc.Dataset(str(path), params={"verbose": -1, "min_data_in_bin": 1}).construct()
    assert data.num_feature() == 2
