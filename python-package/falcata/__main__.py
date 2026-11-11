"""``python -m falcata`` -- small model-artifact utilities.

convert
    Rewrite a model between the LightGBM text format and Falcata's binary
    FALB format. Both directions are lossless for the model itself; text is
    what stock LightGBM and treelite/FIL can read, FALB is ~10x smaller.
"""

import argparse
import sys
from pathlib import Path


def _cmd_convert(args: argparse.Namespace) -> int:
    import falcata as flc  # noqa: PLC0415 -- lazy: keep `python -m falcata --help` fast

    src = Path(args.input)
    dst = Path(args.output)
    fmt = args.format
    if fmt == "auto":
        stripped = [x.lower() for x in dst.suffixes if x.lower() != ".gz"]
        fmt = "falb" if stripped and stripped[-1] == ".falb" else "txt"
    if fmt != "falb" and (
        args.drop_stats or args.drop_diagnostics or args.f32_leaves or args.compress_level is not None
    ):
        raise ValueError("FALB conversion options require FALB output")
    booster = flc.Booster(model_file=str(src))
    kwargs = {}
    if fmt == "falb":
        # Converting is transforming an EXISTING artifact, so it is lossless by
        # default: per-node counts/weights and gains are kept unless the caller
        # explicitly drops them. (Dropping them is the bigger space win --
        # roughly 3x on a real model -- but it silently breaks pred_contrib and
        # feature_importance("gain") later, so it has to be asked for.)
        kwargs = {
            "with_stats": not args.drop_stats,
            "with_diagnostics": not args.drop_diagnostics,
            "f32_leaves": args.f32_leaves,
            "compress_level": 6 if args.compress_level is None else args.compress_level,
        }
    booster.save_model(str(dst), num_iteration=-1, format=fmt, **kwargs)
    src_size, dst_size = src.stat().st_size, dst.stat().st_size
    ratio = src_size / dst_size if dst_size else float("nan")
    print(f"{src} ({src_size:,} B) -> {dst} ({dst_size:,} B, {ratio:.1f}x)")  # noqa: T201
    if fmt == "falb":
        dropped = []
        unavailable = []
        if args.drop_stats:
            dropped.append("stats")
            unavailable.append("pred_contrib")
        if args.drop_diagnostics:
            dropped.append("diagnostics")
            unavailable.append("feature_importance('gain')")
        if dropped:
            print(  # noqa: T201
                f"  dropped {' and '.join(dropped)}: {' / '.join(unavailable)} unavailable",
                file=sys.stderr,
            )
        if args.f32_leaves:
            print("  float32 leaves: lossy model conversion", file=sys.stderr)  # noqa: T201
        elif not dropped:
            print("  lossless model conversion")  # noqa: T201
    return 0


def main(argv: "list[str] | None" = None) -> int:
    """Entry point for the ``python -m falcata`` command line."""
    ap = argparse.ArgumentParser(prog="falcata")
    sub = ap.add_subparsers(dest="command", required=True)

    cv = sub.add_parser("convert", help="convert a model between txt and falb")
    cv.add_argument("input", help="model file to read (format detected by magic)")
    cv.add_argument("output", help="model file to write")
    cv.add_argument(
        "--format",
        choices=("auto", "txt", "falb"),
        default="auto",
        help="output format; auto picks by the output extension",
    )
    cv.add_argument(
        "--drop-stats", action="store_true", help="discard per-node counts/weights (much smaller; breaks pred_contrib)"
    )
    cv.add_argument(
        "--drop-diagnostics", action="store_true", help="discard split_gain/internal_value (breaks gain importance)"
    )
    cv.add_argument("--f32-leaves", action="store_true", help="store leaf values as float32 (LOSSY, ~35%% smaller)")
    cv.add_argument(
        "--compress-level",
        type=int,
        default=None,
        help="FALB zlib level 0-9 (default 6); 0 keeps sections raw and mmap-able",
    )
    cv.set_defaults(func=_cmd_convert)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
