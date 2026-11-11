"""Unaligned binary sections retain their values without undefined typed loads."""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("section_id", [2, 5, 6, 10])
def test_read_unaligned_binary_section(tmp_path, section_id):
    x = np.random.default_rng(4).normal(size=(80, 2))
    booster = flc.train(
        {"objective": "regression", "num_threads": 1, "min_data_in_leaf": 3, "verbosity": -1},
        flc.Dataset(x, label=x[:, 0]),
        num_boost_round=2,
    )
    data = bytearray(booster.model_to_binary(compress_level=0))
    count = struct.unpack_from("<I", data, 24)[0]
    for i in range(count):
        entry = 32 + 40 * i
        ident, dtype, codec, shuffle, offset, stored, raw = struct.unpack_from("<IIIIQQQ", data, entry)
        if ident == section_id:
            assert codec == 0
            payload = data[offset : offset + stored]
            if len(data) % 2 == 0:
                data.append(0)
            new_offset = len(data)
            data.extend(payload)
            struct.pack_into("<Q", data, entry + 16, new_offset)
            break
    else:
        pytest.fail("Expected section missing")
    path = tmp_path / "unaligned.falb"
    path.write_bytes(data)
    restored = flc.Booster(model_file=path)
    np.testing.assert_array_equal(restored.predict(x), booster.predict(x))


@pytest.mark.parametrize("num_leaves", [2, 3])
@pytest.mark.parametrize("compress_level", [0, 6])
def test_binary_mixed_width_stats(tmp_path, num_leaves, compress_level):
    x = np.arange(100, dtype=float).reshape(-1, 1)
    booster = flc.train(
        {"objective": "regression", "num_leaves": num_leaves, "min_data_in_leaf": 2, "num_threads": 1, "verbosity": -1},
        flc.Dataset(x, label=x[:, 0]),
        num_boost_round=1,
    )
    path = tmp_path / "stats.falb"
    booster.save_model(path, with_stats=True, with_diagnostics=True, compress_level=compress_level)
    restored = flc.Booster(model_file=path)
    np.testing.assert_array_equal(restored.predict(x), booster.predict(x))
    assert restored.dump_model()["tree_info"] == booster.dump_model()["tree_info"]


@pytest.mark.skipif(sys.platform == "win32", reason="Requires a GCC-compatible alignment sanitizer")
def test_binary_view_alignment_sanitizer(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("No C++ compiler available")
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "reader"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsanitize=alignment",
            "-fno-sanitize-recover=all",
            f"-I{root / 'include'}",
            str(root / "tests/cpp_tests/test_falb_unaligned_view_regression.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
