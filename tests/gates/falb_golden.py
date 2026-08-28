"""FALB writer-byte golden gate.

``falb_compat.py`` promises that the committed fixtures keep LOADING and
predicting; this gate locks the WRITER: re-serializing each committed fixture
must produce byte-for-byte the locked output. Any writer change -- intended or
accidental -- shows up as an md5 mismatch here instead of silently shipping a
new binary layout.

Machine-independent by construction: the inputs are committed fixture files
(never freshly trained models, whose bytes vary across machines), and every
locked blob uses ``compress_level=0`` so no zlib implementation is involved
(the system zlib is not pinned; raw sections are).

  python tests/gates/falb_golden.py           # check (the gate)
  python tests/gates/falb_golden.py --lock    # (re)write falb_golden.json:
                                              # only for an understood,
                                              # intentional writer change
"""

import hashlib
import json
import sys
from pathlib import Path

import falcata as flc

FIXTURE_DIR = Path(__file__).resolve().parent / "falb_fixtures"
GOLDEN_PATH = Path(__file__).resolve().parent / "falb_golden.json"


def writer_blobs(fixture_dir):
    """name -> writer bytes, for every locked writer invocation of one fixture.

    Two loaders x two writer surfaces: the lean default sections (from the
    binary fixture) and the full section set including stats and diagnostics
    (from the text fixture, which always carries them).
    """
    from_falb = flc.Booster(model_file=str(fixture_dir / "model.falb"))
    from_text = flc.Booster(model_file=str(fixture_dir / "model.txt"))
    return {
        "falb-load/lean": from_falb.model_to_binary(compress_level=0),
        "text-load/full": from_text.model_to_binary(with_stats=True, with_diagnostics=True, compress_level=0),
    }


def fixture_dirs():
    return sorted(p.parent for p in FIXTURE_DIR.glob("*/meta.json"))


def lock():
    golden = {}
    for d in fixture_dirs():
        golden[d.name] = {variant: hashlib.md5(blob).hexdigest() for variant, blob in writer_blobs(d).items()}
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print(f"locked {len(golden)} fixture(s) -> {GOLDEN_PATH.name}")


def check():
    if not GOLDEN_PATH.exists():
        print("FAIL: falb_golden.json missing -- run --lock once and commit it")
        return 1
    golden = json.loads(GOLDEN_PATH.read_text())
    bad = 0
    for d in fixture_dirs():
        want = golden.get(d.name)
        if want is None:
            print(f"  FAIL  {d.name}: fixture has no golden entry -- run --lock and review")
            bad += 1
            continue
        got = {variant: hashlib.md5(blob).hexdigest() for variant, blob in writer_blobs(d).items()}
        for variant in sorted(set(want) | set(got)):
            ok = want.get(variant) == got.get(variant)
            print(f"  {'PASS' if ok else 'FAIL'}  {d.name}/{variant} md5={got.get(variant)}")
            bad += 0 if ok else 1
    print(f"{'FALB WRITER GOLDEN GATE PASS' if bad == 0 else f'FALB WRITER GOLDEN GATE: {bad} FAILURE(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--lock" in sys.argv:
        lock()
    sys.exit(check())
