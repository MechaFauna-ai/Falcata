"""Regression tests for Numerai benchmark cache identity."""

import json
import os
import sys

import numpy as np
import pytest

BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))
sys.path.insert(0, BENCHMARKS_DIR)

import common  # noqa: E402
import datasets  # noqa: E402

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write_source(path, feature_offset=0):
    n = 230  # enough eras for the production test/embargo split
    target_alt = np.full(n, 0.9, dtype=np.float32)
    target_alt[:3] = np.nan  # exercise a target-dependent row filter
    pq.write_table(
        pa.table(
            {
                "era": [str(i) for i in range(n)],
                "feature_0": (np.arange(n, dtype=np.int16) % 5) + feature_offset,
                "target": np.full(n, 0.1, dtype=np.float32),
                "target_alt": target_alt,
            }
        ),
        path,
    )


@pytest.fixture
def numerai_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(common, "CACHE_DIR", str(cache))
    monkeypatch.setattr(datasets, "CACHE_DIR", str(cache))
    source = tmp_path / "input.parquet"
    _write_source(source)
    monkeypatch.setenv("NUMERAI_PARQUET", str(source))
    monkeypatch.delenv("NUMERAI_TARGET", raising=False)
    return cache / "numerai", source


def test_target_and_source_identity_invalidate_cache(numerai_cache, monkeypatch):
    cache, source = numerai_cache
    datasets.prep_numerai()
    assert common.dataset_ready("numerai")
    assert np.all(np.load(cache / "y.npy") == np.float32(0.1))

    # Changing only the requested label must invalidate and then rebuild labels.
    monkeypatch.setenv("NUMERAI_TARGET", "target_alt")
    assert not common.dataset_ready("numerai")
    datasets.prep_numerai()
    assert common.dataset_ready("numerai")
    assert np.all(np.load(cache / "y.npy") == np.float32(0.9))
    meta = json.loads((cache / "meta.json").read_text())
    assert meta["target"] == "target_alt"
    assert meta["source_identity"]["path"] == str(source.resolve())

    # Cache-only readers use the recorded target; preprocessing with no
    # override must return to the documented default target.
    monkeypatch.delenv("NUMERAI_TARGET")
    monkeypatch.delenv("NUMERAI_PARQUET")
    assert common.dataset_ready("numerai")
    assert not common.dataset_ready("numerai", for_preprocessing=True)
    monkeypatch.setenv("NUMERAI_TARGET", "target")
    assert not common.dataset_ready("numerai")
    monkeypatch.setenv("NUMERAI_TARGET", "target_alt")
    monkeypatch.setenv("NUMERAI_PARQUET", str(source))

    # Same path, different file contents/stat identity must also invalidate.
    old_mtime = source.stat().st_mtime_ns
    _write_source(source, feature_offset=1)
    os.utime(source, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))
    assert not common.dataset_ready("numerai")
    datasets.prep_numerai()
    assert common.dataset_ready("numerai")


def test_legacy_cache_defaults_to_original_target_and_works_without_source(numerai_cache, monkeypatch):
    cache, _ = numerai_cache
    cache.mkdir(parents=True)
    (cache / "meta.json").write_text(json.dumps({"source": "old.parquet", "n_rows": 1}))

    monkeypatch.delenv("NUMERAI_PARQUET")
    assert common.dataset_ready("numerai")
    assert common.dataset_ready("numerai", for_preprocessing=True)
    monkeypatch.setenv("NUMERAI_TARGET", "target_alt")
    assert not common.dataset_ready("numerai")


def test_int8_uses_f32_target_and_rewrites_after_cache_rebuild(numerai_cache, monkeypatch):
    cache, source = numerai_cache
    monkeypatch.setenv("NUMERAI_TARGET", "target_alt")
    datasets.prep_numerai()
    # Omitted target inherits the target recorded by the f32 cache.
    monkeypatch.delenv("NUMERAI_TARGET")
    datasets.prep_numerai_int8()
    old_n_rows = json.loads((cache / "meta.json").read_text())["n_rows"]
    old_values = np.memmap(cache / "X.i8.mem", dtype=np.int8, mode="r", shape=(old_n_rows, 1)).copy()

    # Rebuild f32 from a changed parquet and make int8 follow its exact source/filter.
    old_mtime = source.stat().st_mtime_ns
    _write_source(source, feature_offset=1)
    os.utime(source, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))
    datasets.prep_numerai()
    assert not (cache / "X.i8.mem").exists()
    datasets.prep_numerai_int8()
    new_n_rows = json.loads((cache / "meta.json").read_text())["n_rows"]
    new_values = np.memmap(cache / "X.i8.mem", dtype=np.int8, mode="r", shape=(new_n_rows, 1)).copy()
    assert not np.array_equal(old_values, new_values)
    assert np.all(np.load(cache / "y.npy") == np.float32(0.1))
