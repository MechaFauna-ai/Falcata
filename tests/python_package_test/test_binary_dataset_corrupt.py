"""A corrupt binary dataset file must be refused, never crash the loader.

The .bin loader (Dataset(path) on a file written by save_binary) parsed the
header, metadata, bin mappers and bin data straight from the file, with no
bounds or consistency checks: a corrupt count or index read and wrote out of
bounds, a corrupt size allocated whatever it claimed, and an invalid byte was
loaded into a bool or an enum. Each case runs in a child process so a crash on
an unfixed build fails the test instead of the session.
"""

import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import falcata as flc

_TOKEN_LEN = 40  # len("______LightGBM_Binary_File_Token______\n") padded to 8
_HEADER = _TOKEN_LEN + 8  # the header starts after its size_t length

_CHILD = textwrap.dedent(
    """
    import sys
    import falcata as flc

    try:
        ds = flc.Dataset(sys.argv[1], params={"verbose": -1, "num_threads": 1}).construct()
        flc.train({"verbose": -1, "num_threads": 1}, ds, num_boost_round=1)
    except flc.basic.FalcataError as err:
        print("REFUSED", err)
        sys.exit(0)
    print("LOADED")
    """
)


@pytest.fixture(scope="module")
def binary_file(tmp_path_factory):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 4))
    x[:, 3] = rng.integers(0, 5, size=300)
    path = tmp_path_factory.mktemp("bin") / "train.bin"
    ds = flc.Dataset(x, label=x[:, 0], categorical_feature=[3], params={"verbose": -1})
    ds.save_binary(str(path))
    return path.read_bytes()


def _run(tmp_path, blob):
    path = tmp_path / "corrupt.bin"
    path.write_bytes(blob)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(path)], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, f"child died with {proc.returncode}: {proc.stderr[-1500:]}"
    return proc.stdout


def _with_int(blob, offset, value, fmt="<i"):
    out = bytearray(blob)
    struct.pack_into(fmt, out, offset, value)
    return bytes(out)


def test_valid_file_loads(tmp_path, binary_file):
    assert "LOADED" in _run(tmp_path, binary_file)


# header field offsets: each value is written padded to 8 bytes
_NUM_FEATURES = _HEADER + 8
_NUM_TOTAL_FEATURES = _HEADER + 16
_USE_MISSING = _HEADER + 56
_USED_FEATURE_MAP = _HEADER + 80  # num_total_features ints, padded to 8
_NUM_GROUPS = _USED_FEATURE_MAP + 16  # 4 features -> 16 bytes


@pytest.mark.parametrize(
    ("offset", "value", "fmt"),
    [
        (_NUM_FEATURES, 1 << 30, "<i"),  # more features than the header holds
        (_NUM_TOTAL_FEATURES, -3, "<i"),
        (_USE_MISSING, 7, "<B"),  # not a bool
        (_USED_FEATURE_MAP, 99, "<i"),  # maps to a missing inner feature
        (_NUM_GROUPS, -1, "<i"),
        (_NUM_GROUPS, 1000, "<i"),
        (_TOKEN_LEN, 1 << 60, "<Q"),  # header size far beyond the file
    ],
)
def test_corrupt_header_is_refused(tmp_path, binary_file, offset, value, fmt):
    assert "REFUSED" in _run(tmp_path, _with_int(binary_file, offset, value, fmt))


def test_truncated_file_is_refused(tmp_path, binary_file):
    for cut in (_TOKEN_LEN + 4, _HEADER + 40, len(binary_file) // 2, len(binary_file) - 8):
        assert "REFUSED" in _run(tmp_path, binary_file[:cut])


def test_corrupt_bin_values_are_refused(tmp_path, binary_file):
    # the tail of the file is bin data: bins far beyond any feature's range
    # would index past the histograms during training
    blob = bytearray(binary_file)
    blob[-64:] = b"\xff" * 64
    assert "REFUSED" in _run(tmp_path, bytes(blob))
