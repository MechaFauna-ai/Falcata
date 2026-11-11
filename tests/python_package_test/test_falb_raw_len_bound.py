"""A FALB section header must not make the loader allocate what it claims.

A zlib section stores its decompressed size (raw_len), and the loader
allocated -- and zero-filled -- that many bytes before inflating. Only a
1 TiB cap applied, so a few-KiB file could make the process commit gigabytes
(the fuzzer's case got the process OOM-killed). Deflate expands at most
1032:1, so a claim beyond that is corrupt and is now refused up front.
"""

import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import falcata as flc

_HEADER = 32
_ENTRY = 40
_CODEC_ZLIB = 2  # FALB::kCodecZlib

_CHILD = textwrap.dedent(
    """
    import resource, sys
    import falcata as flc

    blob = open(sys.argv[1], "rb").read()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        flc.Booster(model_bin=blob)
        print("LOADED")
    except flc.basic.FalcataError as err:
        print("REFUSED", err)
    grown_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    print("GROWN_MB", grown_mb)
    """
)


def _zlib_section(blob):
    count = struct.unpack_from("<I", blob, 24)[0]
    for i in range(count):
        off = _HEADER + i * _ENTRY
        _sid, _dtype, codec, _shuffle, _offset, stored_len, _raw_len = struct.unpack_from("<IIIIQQQ", blob, off)
        if codec == _CODEC_ZLIB and stored_len > 0:
            return off, stored_len
    return None, None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ru_maxrss is KiB on Linux")
def test_falb_section_cannot_claim_a_huge_raw_len(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    booster = flc.train({"verbose": -1, "num_threads": 1}, flc.Dataset(x, label=x[:, 0]), num_boost_round=3)
    blob = bytearray(booster.model_to_binary(compress_level=6))
    off, stored_len = _zlib_section(blob)
    assert off is not None, "fixture must contain a zlib section"
    claim = 512 << 20
    assert claim > stored_len * 1032 + 1024
    struct.pack_into("<Q", blob, off + 32, claim)
    path = tmp_path / "huge.falb"
    path.write_bytes(bytes(blob))
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(path)], capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "REFUSED" in proc.stdout, proc.stdout
    grown = float(proc.stdout.split("GROWN_MB")[1])
    assert grown < 128, f"loader committed {grown:.0f} MB for a {len(blob)}-byte file"
