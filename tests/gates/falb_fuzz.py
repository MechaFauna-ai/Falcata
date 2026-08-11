"""FALB parser fuzz gate: a model file is untrusted input.

Structure-aware mutations of valid FALB files -- truncations, byte flips, and
header-focused overwrites -- fed to the loader in child processes. Every
mutation must either load successfully or raise a clean error; a child killed
by a signal (segfault, abort) fails the gate with the exact repro.

  python tests/gates/falb_fuzz.py [--seconds N] [--seed N]
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import falcata as flc

BATCH = 40  # mutations per child process

_WORKER = r"""
import sys
import falcata as flc

for path in sys.argv[1:]:
    try:
        flc.Booster(model_file=path)
    except Exception:
        pass  # clean refusal is a correct outcome
    print("OK", path, flush=True)
"""


def build_seed_files(tmp):
    """A few FALB files with different sections present (cats, multiclass)."""
    files = []
    for name, params, cats in [
        ("plain", {"objective": "regression", "num_leaves": 31}, 0),
        ("cat", {"objective": "regression", "num_leaves": 31}, 2),
        ("multi", {"objective": "multiclass", "num_class": 3, "num_leaves": 15}, 0),
    ]:
        rng = np.random.default_rng(7)
        x = rng.standard_normal((2000, 6))
        for j in range(cats):
            x[:, j] = rng.integers(0, 30, size=2000)
        y = x @ rng.standard_normal(6)
        if name == "multi":
            y = (x @ rng.standard_normal((6, 3))).argmax(1).astype(float)
        p = dict(params, verbose=-1, seed=42, num_threads=1)
        ds = flc.Dataset(x, label=y, params=p, categorical_feature=list(range(cats)) or "auto")
        bst = flc.train(p, ds, num_boost_round=8)
        f = tmp / f"{name}.falb"
        bst.save_model(str(f), format="falb")
        files.append(f.read_bytes())
    return files


def mutate(rng, blob):
    """One structure-aware mutation of a valid FALB byte string."""
    b = bytearray(blob)
    kind = rng.integers(0, 4)
    if kind == 0:  # truncate anywhere (including inside the header)
        return bytes(b[: rng.integers(1, len(b))])
    if kind == 1:  # header assault: the first 64 bytes hold magic/version/offsets
        for _ in range(rng.integers(1, 6)):
            b[rng.integers(0, min(64, len(b)))] = rng.integers(0, 256)
        return bytes(b)
    if kind == 2:  # scattered bit flips in the body
        for _ in range(rng.integers(1, 9)):
            i = rng.integers(0, len(b))
            b[i] ^= 1 << rng.integers(0, 8)
        return bytes(b)
    # kind == 3: splice -- copy a random chunk over another region (breaks
    # section sizes/offsets while keeping plausible content)
    n = int(rng.integers(4, 256))
    src = int(rng.integers(0, max(1, len(b) - n)))
    dst = int(rng.integers(0, max(1, len(b) - n)))
    b[dst : dst + n] = b[src : src + n]
    return bytes(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        seeds = build_seed_files(tmp)
        deadline = time.monotonic() + args.seconds
        tried = 0
        while time.monotonic() < deadline:
            paths = []
            recipes = []
            for i in range(BATCH):
                blob = seeds[int(rng.integers(0, len(seeds)))]
                mseed = int(rng.integers(0, 2**31))
                mutated = mutate(np.random.default_rng(mseed), blob)
                f = tmp / f"m{tried + i}.falb"
                f.write_bytes(mutated)
                paths.append(str(f))
                recipes.append(mseed)
            proc = subprocess.run(
                [sys.executable, "-c", _WORKER, *paths],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if proc.returncode != 0:
                done = proc.stdout.count("OK ")
                crash_idx = tried + done
                print(
                    f"FAIL: loader killed (exit {proc.returncode}) on mutation #{crash_idx} "
                    f"(mutation seed {recipes[done]}, fuzz seed {args.seed})"
                )
                repro = Path(__file__).resolve().parent / "falb_fuzz_crash.falb"
                shutil.copy(paths[done], repro)
                print(f"repro file kept at: {repro}")
                return 1
            for f in paths:
                Path(f).unlink(missing_ok=True)
            tried += BATCH
        print(f"FALB FUZZ GATE PASS ({tried} mutations, no loader crash)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
