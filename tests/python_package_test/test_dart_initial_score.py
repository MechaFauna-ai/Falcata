"""DART scores can be retrieved before the first iteration."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from falcata.basic import _LIB


def test_dart_initial_score(tmp_path):
    if os.getenv("FALCATA_RUN_CPU_CPP_REGRESSIONS") != "1":
        pytest.skip("Internal C++ test requires an explicitly selected CPU-only library")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "dart_initial_score"
    library = Path(_LIB._name).resolve()
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsanitize=bool",
            "-fno-sanitize-recover=all",
            f"-I{root}",
            f"-I{root / 'include'}",
            f"-I{root / 'external_libs/fast_double_parser/include'}",
            f"-I{root / 'external_libs/fmt/include'}",
            str(root / "tests/train_regression/dart_initial_score.cpp"),
            str(library),
            f"-Wl,-rpath,{library.parent}",
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_dart_custom_objective_before_first_iteration():
    import falcata as flc
    import numpy as np

    x = np.random.default_rng(42).normal(size=(60, 2))
    y = x[:, 0]

    def objective(prediction, data):
        assert np.isfinite(prediction).all()
        return prediction - data.get_label(), np.ones_like(prediction)

    def train():
        return flc.train(
            {
                "boosting": "dart",
                "objective": objective,
                "num_threads": 1,
                "min_data_in_leaf": 2,
                "verbosity": -1,
                "drop_seed": 7,
            },
            flc.Dataset(x, label=y),
            num_boost_round=5,
        ).predict(x)

    np.testing.assert_array_equal(train(), train())
