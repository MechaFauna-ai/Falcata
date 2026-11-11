"""Model conversion reports its result and requested capability loss."""

import numpy as np
import pytest

import falcata as flc
from falcata.__main__ import main


@pytest.mark.parametrize("drop", [False, True])
def test_conversion_reports_summary(tmp_path, capsys, drop):
    x = np.arange(120, dtype=float).reshape(60, 2)
    model = flc.train(
        {"objective": "regression", "device_type": "cpu", "num_threads": 2, "verbosity": -1},
        flc.Dataset(x, label=x[:, 0]),
        num_boost_round=3,
    )
    source, destination = tmp_path / "source.txt", tmp_path / "model.falb"
    model.save_model(source)
    capsys.readouterr()
    flags = ["--drop-stats", "--drop-diagnostics"] if drop else []
    assert main(["convert", str(source), str(destination), *flags]) == 0
    output = capsys.readouterr()
    assert str(destination) in output.out
    assert "->" in output.out
    if drop:
        assert "pred_contrib" in output.err
        assert "feature_importance('gain')" in output.err
    else:
        assert "lossless" in output.out
        assert output.err == ""
    assert destination.read_bytes().startswith(b"FALB")
