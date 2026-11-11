"""Malformed tree-size tables must fail before parallel tree parsing."""

import os
import re
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("sizes", ["999999999999", "18446744073709551615 1", "0", "1 999999999999"])
def test_invalid_tree_sizes(tmp_path, sizes):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(80, 3))
    model = flc.train(
        {"objective": "regression", "num_threads": 2, "min_data_in_leaf": 3, "verbosity": -1},
        flc.Dataset(x, label=x[:, 0]),
        num_boost_round=2,
    )
    text = re.sub(r"^tree_sizes=.*$", f"tree_sizes={sizes}", model.model_to_string(), flags=re.MULTILINE)
    path = tmp_path / "corrupt.txt"
    path.write_text(text)
    result = subprocess.run(
        [sys.executable, "-c", "import falcata,sys; falcata.Booster(model_file=sys.argv[1])", str(path)],
        capture_output=True,
        check=False,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    assert "Model format error: invalid tree_sizes" in result.stderr
