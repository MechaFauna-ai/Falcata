"""The HDF example closes input handles on success and failure."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize("fail", [False, True])
def test_multiple_hdf_files_are_closed(tmp_path, monkeypatch, fail):
    h5py = pytest.importorskip("h5py")
    path = Path(__file__).resolve().parents[2] / "examples" / "python-guide" / "dataset_from_multi_hdf5.py"
    spec = importlib.util.spec_from_file_location("hdf_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = [tmp_path / f"input{i}.h5" for i in range(2)]
    for file in inputs:
        with h5py.File(file, "w") as handle:
            handle.create_dataset("X", data=np.arange(80, dtype=float).reshape(40, 2))
            handle.create_dataset("Y", data=np.arange(40, dtype=float))
    opened = []
    original = h5py.File

    def opening(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(module.h5py, "File", opening)
    monkeypatch.chdir(tmp_path)
    if fail:
        monkeypatch.setattr(module.flc, "Dataset", Mock(side_effect=RuntimeError("construction failed")))
        with pytest.raises(RuntimeError, match="construction failed"):
            module.create_dataset_from_multiple_hdf(inputs, batch_size=8)
    else:
        module.create_dataset_from_multiple_hdf(inputs, batch_size=8)
        assert (tmp_path / "regression.train.from_hdf.bin").exists()
    assert len(opened) == 2
    assert all(not handle.id.valid for handle in opened)
