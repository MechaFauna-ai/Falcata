"""FALB-only conversion options cannot be silently ignored for text output."""

import numpy as np
import pytest

import falcata as flc
from falcata.__main__ import main


@pytest.mark.parametrize("option", ["--drop-stats", "--drop-diagnostics", "--f32-leaves", "--compress-level"])
@pytest.mark.parametrize("format_option", [[], ["--format", "txt"]])
def test_text_convert_rejects_falb_options(tmp_path, option, format_option):
    args = ["convert", str(tmp_path / "missing.txt"), str(tmp_path / "output.txt"), *format_option, option]
    if option == "--compress-level":
        args.append("6")
    with pytest.raises(ValueError, match="FALB"):
        main(args)


@pytest.mark.parametrize(
    ("extension", "options"),
    [("txt", []), ("falb", []), ("falb", ["--compress-level", "0", "--drop-diagnostics"])],
)
def test_default_conversion_roundtrip(tmp_path, extension, options):
    data = np.arange(80, dtype=np.float64).reshape(40, 2)
    booster = flc.train(
        {"device_type": "cpu", "num_threads": 2, "verbosity": -1, "min_data_in_leaf": 1},
        flc.Dataset(data, label=np.arange(40)),
        num_boost_round=2,
    )
    source = tmp_path / "source.txt"
    output = tmp_path / f"output.{extension}"
    booster.save_model(source)
    assert main(["convert", str(source), str(output), *options]) == 0
    restored = flc.Booster(model_file=output)
    np.testing.assert_array_equal(restored.predict(data), booster.predict(data))
    if "--drop-diagnostics" not in options:
        reference = flc.Booster(model_file=source)
        np.testing.assert_array_equal(restored.feature_importance("gain"), reference.feature_importance("gain"))
