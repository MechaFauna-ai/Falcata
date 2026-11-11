"""Model conversion reports its result and requested capability loss."""

import numpy as np
import pytest

import falcata as flc
from falcata.__main__ import main


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--drop-stats"],
        ["--drop-diagnostics"],
        ["--drop-stats", "--drop-diagnostics"],
        ["--f32-leaves"],
        ["--format", "txt"],
    ],
)
def test_conversion_reports_summary(tmp_path, capsys, flags):
    x = np.arange(120, dtype=float).reshape(60, 2)
    model = flc.train(
        {"objective": "regression", "device_type": "cpu", "num_threads": 2, "verbosity": -1},
        flc.Dataset(x, label=x[:, 0]),
        num_boost_round=3,
    )
    source, destination = tmp_path / "source.txt", tmp_path / "model.falb"
    model.save_model(source)
    capsys.readouterr()
    assert main(["convert", str(source), str(destination), *flags]) == 0
    output = capsys.readouterr()
    assert str(destination) in output.out
    assert "->" in output.out
    assert ("pred_contrib" in output.err) == ("--drop-stats" in flags)
    assert ("feature_importance('gain')" in output.err) == ("--drop-diagnostics" in flags)
    assert ("lossy" in output.err) == ("--f32-leaves" in flags)
    assert ("lossless" in output.out) == (not flags)
    assert destination.read_bytes().startswith(b"FALB") == ("txt" not in flags)
