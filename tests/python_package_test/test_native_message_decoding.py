# coding: utf-8
"""Native errors surface as FalcataError even when their text is not valid UTF-8.

_safe_call decoded FLC_GetLastError() strictly as UTF-8, but the message quotes input:
a data file token in another encoding ("Unknown token abc\\xe9") raised
UnicodeDecodeError instead. So did valid UTF-8 input, because the native side cuts the
message at a fixed byte count (512) and the cut can fall inside a multi-byte character.
"""

import pytest

import falcata as flc


def _rows():
    return [f"{i % 7},{i * 0.5},{(i * 37) % 11}".encode() for i in range(300)]


@pytest.mark.parametrize(
    "bad_token",
    [
        pytest.param(b"abc\xe9", id="latin-1 byte"),
        pytest.param(("é" * 700).encode("utf-8"), id="utf-8 cut by the message limit"),
    ],
)
def test_bad_token_error_is_a_falcata_error(tmp_path, bad_token):
    rows = _rows()
    rows.insert(150, b"1," + bad_token + b",2")
    path = tmp_path / "data.csv"
    path.write_bytes(b"\n".join(rows) + b"\n")
    with pytest.raises(flc.basic.FalcataError, match="Unknown token"):
        flc.Dataset(str(path), params={"verbose": -1}).construct()


def test_decode_native_text_replaces_invalid_bytes():
    assert flc.basic._decode_native_text(b"Unknown token abc\xe9") == "Unknown token abc�"
    assert flc.basic._decode_native_text("é".encode("utf-8")[:1]) == "�"
