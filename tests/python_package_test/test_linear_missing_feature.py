"""Linear scoring must skip split features filtered out of the new Dataset."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB


def test_linear_missing_feature(tmp_path):
    if sys.platform != "linux" or os.getenv("TASK") in {"cuda", "gpu", "mpi"}:
        pytest.skip("Native bounds-check harness requires a Linux CPU build")
    # Deleting a null device builder is a no-op on CUDA builds; CPU builds reject
    # this API before touching the handle. Probe the loaded library without GPU work.
    if _LIB.FLC_DatasetDeviceBuilderFree(None) == 0:
        pytest.skip("Private-header harness does not support CUDA library layouts")
    if b"requires a CUDA build" not in _LIB.FLC_GetLastError():
        pytest.skip("Cannot identify the loaded library as a CPU build")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "linear_missing_feature"
    library = Path(_LIB._name).resolve()
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-D_GLIBCXX_ASSERTIONS",
            f"-I{root}",
            f"-I{root / 'include'}",
            f"-I{root / 'external_libs/fast_double_parser/include'}",
            f"-I{root / 'external_libs/fmt/include'}",
            str(root / "tests/train_regression/linear_missing_feature.cpp"),
            str(library),
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_refit_with_filtered_split_and_retained_nan_feature():
    x = np.random.default_rng(3).normal(size=(200, 2))
    y = 2 * x[:, 0] + x[:, 1]
    model = flc.train(
        {
            "objective": "regression",
            "linear_tree": True,
            "min_data_in_leaf": 5,
            "num_leaves": 2,
            "num_threads": 1,
            "verbosity": -1,
        },
        flc.Dataset(x, label=y),
        num_boost_round=1,
    )
    assert model.dump_model()["tree_info"][0]["tree_structure"]["split_feature"] == 0
    refit_x = x.copy()
    refit_x[:, 0] = 100.0
    refit_x[0, 1] = np.nan
    refitted = model.refit(refit_x, y, decay_rate=0.5, dataset_params={"feature_pre_filter": True})
    assert np.isfinite(refitted.predict(refit_x)).all()
