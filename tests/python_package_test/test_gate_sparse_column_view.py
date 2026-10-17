"""Saved dataset parameters must survive the sparse gate's two training paths."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import falcata as lgb

GATE_PATH = Path(__file__).resolve().parents[1] / "gates" / "sparse_column_view.py"
SPEC = importlib.util.spec_from_file_location("sparse_column_view_gate", GATE_PATH)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.mark.parametrize(
    "bin_params",
    [
        {"max_bin": 5},
        {"max_bin": 255},
        {"max_bin": 5, "min_data_in_bin": 7, "bin_construct_sample_cnt": 1024, "zero_as_missing": True},
        {"max_bin": 255, "use_missing": False},
    ],
)
def test_binary_params_reach_correctness_and_throughput(tmp_path, monkeypatch, bin_params):
    """Real CPU serialization/loading; stub training, never allocate a CUDA context."""
    rng = np.random.default_rng(7)
    values = rng.integers(0, 5, size=(4000, 8)).astype(np.float32)
    params = {"device_type": "cpu", "verbosity": -1, "num_threads": 2, **bin_params}
    path = tmp_path / "arbitrary-name.data"
    original = lgb.Dataset(values, label=rng.normal(size=len(values)), params=params).construct()
    original.save_binary(path)
    # The binary itself is authoritative, not its filename or an optional sidecar.
    path.with_suffix(".data.json").write_text('{"max_bin": 999}')
    loaded = []
    trained = []

    def dataset(filename, params):
        result = lgb.Dataset(filename, params={**params, "device_type": "cpu", "num_threads": 2})
        loaded.append(result)
        return result

    def train(params, ds, num_boost_round):
        assert num_boost_round == gate.ROUNDS
        assert ds.num_data() == len(values)
        trained.append(dict(params))
        return SimpleNamespace(model_to_string=lambda: "\nTree=0\nleaf_count=2000 2000\nleaf_weight=2000 2000\n")

    def timed(params, ds, _lgb):
        assert ds is loaded[0]
        trained.append(dict(params))
        return 1.0 if "cuda_plan" in params else 1.06

    monkeypatch.setattr(gate, "timed_train", timed)
    assert gate.run_dataset(path, SimpleNamespace(Dataset=dataset, train=train))
    assert len(trained) == 3  # correctness, packed, gather
    for actual in trained:
        for key, value in bin_params.items():
            assert actual[key] == value
    assert gate.PARAMS["max_bin"] == 5  # synthetic inputs keep their original construction


def test_saved_params_do_not_relax_leaf_invariants(tmp_path, monkeypatch):
    booster = SimpleNamespace(model_to_string=lambda: "\nTree=0\nleaf_count=999 2000\nleaf_weight=999 2001\n")
    assert not gate.check("bad leaves", booster, min_data_in_leaf=1000)


def test_saved_params_do_not_relax_packed_read_floor(monkeypatch):
    ds = SimpleNamespace(get_params=lambda: {"max_bin": 255})
    monkeypatch.setattr(gate, "timed_train", lambda params, *_: 1.0 if "cuda_plan" in params else 1.04)
    assert not gate.check_packed_read_speedup(Path("arbitrary-name.data"), ds, None)


@pytest.mark.parametrize("contents", [b"", b"not a binary dataset", b"______LightGBM_Binary_File_Token______\n"])
def test_bad_binary_header_fails_before_training(tmp_path, contents):
    path = tmp_path / "bad.data"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="binary|header"):
        gate.binary_dataset_params(path)
