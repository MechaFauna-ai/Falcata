"""CPU evidence for the explicitly frozen full-scale h60 gate input."""

import hashlib
import importlib
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

GATES = Path(__file__).resolve().parents[1] / "gates"
with patch.object(sys, "path", [str(GATES), *sys.path]):
    saved = importlib.import_module("saved_dataset")
    canonical = importlib.import_module("canonical")
    bench = importlib.import_module("bench_tier")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(saved, "MIN_H60_ROWS", 4)
    monkeypatch.setattr(saved, "H60_FEATURES", 3)
    binary = tmp_path / "honest-b255.dataset"
    token = b"______LightGBM_Binary_File_Token______\n"
    prefix = bytearray(128)
    prefix[: len(token)] = token
    struct.pack_into("N", prefix, 40, 80)
    for i, value in enumerate((4, 3, 3, 0, 255, 200000, 3, 1, 0, 0)):
        struct.pack_into("=i", prefix, 48 + 8 * i, value)
    binary.write_bytes(prefix)
    labels = tmp_path / "labels.parquet"
    y = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    pq.write_table(pa.table({"target_ender_60": y, "era": [1, 1, 2, 2]}), labels)
    manifest = {
        "schema_version": 1,
        "id": "test-frozen-h60",
        "target": "target_ender_60",
        "rows": 4,
        "features": 3,
        "era_range": [1, 2],
        "bin_params": saved.binary_dataset_params(binary),
        "selection": {"rule": "finite-label-source-order", "label_sha256": hashlib.sha256(y.tobytes()).hexdigest()},
        "dataset": {"path": str(binary), "bytes": binary.stat().st_size, "sha256": sha(binary)},
        "labels": {"path": str(labels), "bytes": labels.stat().st_size, "sha256": sha(labels)},
    }
    path = tmp_path / "manifest.json"

    def write():
        path.write_text(json.dumps(manifest))
        monkeypatch.setenv("FALCATA_H60_MANIFEST", str(path))
        monkeypatch.setenv("FALCATA_H60_MANIFEST_SHA256", sha(path))

    write()
    return manifest, path, write, y


def test_frozen_input_rejects_wrong_target_shape_hash_or_unpinned_manifest(frozen, monkeypatch):
    manifest, path, write, y = frozen
    got = saved.frozen_h60()
    np.testing.assert_array_equal(got["y"], y)
    assert got["params"]["max_bin"] == 255
    monkeypatch.delenv("FALCATA_H60_MANIFEST_SHA256")
    with pytest.raises(ValueError, match="manifest"):
        saved.frozen_h60()
    write()
    for key, bad in (("target", "other"), ("rows", 3), ("features", 2)):
        old = manifest[key]
        manifest[key] = bad
        write()
        with pytest.raises(ValueError, match="target|shape|scale"):
            saved.frozen_h60()
        manifest[key] = old
    write()
    Path(manifest["labels"]["path"]).write_bytes(b"mutation")
    with pytest.raises(ValueError, match="identity"):
        saved.frozen_h60()


def test_missing_manifest_is_failure_without_latest_or_legacy_fallback(monkeypatch):
    monkeypatch.delenv("FALCATA_H60_MANIFEST", raising=False)
    assert canonical.run_numerai_treecount()[1] == "FAIL"
    with pytest.raises(ValueError, match="manifest"):
        bench.run_h60_cell("test", {}, 60)


def fake_lgb(frozen, trees=200, late_value=0.001):
    manifest, _, _, y = frozen
    calls = []

    class Dataset:
        def __init__(self, path, params):
            assert path == manifest["dataset"]["path"]
            calls.append(dict(params))

        def construct(self):
            return self

        def num_data(self):
            return 4

        def num_feature(self):
            return 3

        def get_init_score(self):
            return None

        def get_label(self):
            return y.copy()

        def set_label(self, label):
            np.testing.assert_array_equal(label, y)

    def train(params, ds, num_boost_round):
        calls.append(dict(params))
        return SimpleNamespace(
            num_trees=lambda: trees if num_boost_round == 200 else num_boost_round,
            model_to_string=lambda **kw: f"leaf_value={late_value} 0\n",
        )

    return SimpleNamespace(Dataset=Dataset, train=train), calls


@pytest.mark.parametrize(("trees", "value", "want"), [(200, 0.001, "PASS"), (199, 0.001, "FAIL"), (200, 0.0, "FAIL")])
def test_tree_emission_retains_200_and_late_leaf_guard(frozen, monkeypatch, trees, value, want):
    lgb, calls = fake_lgb(frozen, trees, value)
    monkeypatch.setitem(sys.modules, "falcata", lgb)
    assert canonical.run_numerai_treecount()[1] == want
    assert all(p["max_bin"] == 255 for p in calls)
    assert calls[0]["learning_rate"] == 0.0015
    assert calls[0]["min_data_in_leaf"] == 40000
    assert calls[0]["quant_mode"] == "fixedpoint"


def test_h60_uses_same_saved_bins_and_frozen_label_in_all_repeats(frozen, monkeypatch):
    lgb, calls = fake_lgb(frozen)
    monkeypatch.setitem(sys.modules, "falcata", lgb)
    cid, _, params, rounds = bench.CELLS[-1]
    result = bench.run_h60_cell(cid, params, rounds)
    assert result["ok"]
    assert result["input_manifest_sha256"] == os.environ["FALCATA_H60_MANIFEST_SHA256"]
    assert len(calls) == 6
    assert all(p["max_bin"] == 255 for p in calls)
    assert bench.CELLS[-1][2]["max_bin"] == 5


@pytest.mark.parametrize("suffix", [".bin", ".init", ".bin.init"])
def test_frozen_input_rejects_implicit_native_companions(frozen, suffix):
    manifest, _, _, _ = frozen
    companion = Path(manifest["dataset"]["path"] + suffix)
    companion.write_text("unmanifested native input")
    with pytest.raises(ValueError, match="unmanifested.*companion"):
        saved.frozen_h60()


def test_loaded_initial_scores_are_rejected_even_when_labels_match(frozen):
    lgb, _ = fake_lgb(frozen)
    ds = lgb.Dataset(frozen[0]["dataset"]["path"], params={})
    ds.get_init_score = lambda: np.zeros(4)
    with pytest.raises(ValueError, match="initial scores"):
        saved.check_loaded_h60(ds, saved.frozen_h60())
