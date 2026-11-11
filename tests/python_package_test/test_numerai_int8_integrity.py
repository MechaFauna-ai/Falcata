"""A failed optional int8 conversion must not publish partial cache data."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import datasets  # noqa: E402

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def prepare(tmp_path, monkeypatch, missing=False):
    source = tmp_path / "source.parquet"
    values = [i % 5 for i in range(230)]
    if missing:
        values[17] = None
    pq.write_table(pa.table({"era": [str(i) for i in range(230)], "feature_0": values, "target": [0.1] * 230}), source)
    monkeypatch.setattr(datasets, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NUMERAI_PARQUET", str(source))
    monkeypatch.delenv("NUMERAI_TARGET", raising=False)
    datasets.prep_numerai()
    return tmp_path / "cache" / "numerai"


def test_failed_int8_rebuild_keeps_completed_cache(tmp_path, monkeypatch):
    cache = prepare(tmp_path, monkeypatch)
    datasets.prep_numerai_int8()
    before = (cache / "X.i8.mem").read_bytes()
    original = pq.ParquetFile

    class InterruptedFile:
        def __init__(self, path):
            self.file = original(path)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def iter_batches(self, **kwargs):
            raise RuntimeError("interrupted")

    monkeypatch.setattr(pq, "ParquetFile", InterruptedFile)
    with pytest.raises(RuntimeError, match="interrupted"):
        datasets.prep_numerai_int8()
    assert (cache / "X.i8.mem").read_bytes() == before


def test_int8_rejects_missing_features_without_publishing(tmp_path, monkeypatch):
    cache = prepare(tmp_path, monkeypatch, missing=True)
    with pytest.raises(SystemExit, match="int8"):
        datasets.prep_numerai_int8()
    assert not (cache / "X.i8.mem").exists()
