"""FALB-only conversion options cannot be silently ignored for text output."""

import pytest

from falcata.__main__ import main


@pytest.mark.parametrize("option", ["--drop-stats", "--drop-diagnostics", "--f32-leaves", "--compress-level"])
@pytest.mark.parametrize("format_option", [[], ["--format", "txt"]])
def test_text_convert_rejects_falb_options(tmp_path, option, format_option):
    args = ["convert", str(tmp_path / "missing.txt"), str(tmp_path / "output.txt"), *format_option, option]
    if option == "--compress-level":
        args.append("6")
    with pytest.raises(ValueError, match="FALB"):
        main(args)
