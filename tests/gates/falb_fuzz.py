"""FALB parser fuzz gate: a model file is untrusted input.

Structure-aware mutations of valid FALB files -- truncations, byte flips,
header-focused overwrites, boundary values in section-table fields and in
uncompressed section payloads -- fed to the loader in child processes, which
then predict (normal, raw, leaf, contrib) and re-serialize every model that
loads. Every mutation must either load and run or raise a clean error; a child
killed by a signal (segfault, abort) fails the gate with the exact repro.

  python tests/gates/falb_fuzz.py [--seconds N] [--seed N]
"""

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import falcata as flc

BATCH = 40  # mutations per child process

#: past fuzz findings, committed verbatim: every file here once crashed or
#: hung the loader, and each run replays them first so a fixed bug stays fixed
CRASH_DIR = Path(__file__).resolve().parent / "falb_fixtures" / "crashes"

_WORKER = r"""
import sys
import numpy as np
import falcata as flc

# a loaded model must also be USABLE without crashing: most loader defects
# (cyclic or out-of-range children, bad bitset indices, inconsistent output
# metadata) only show once something walks the trees
X = np.array([[0.0, np.nan, -1.5, 3.0, 1e30, 7.0, 2.0, -0.0]] * 2)
for path in sys.argv[1:]:
    try:
        booster = flc.Booster(model_file=path)
        x = X[:, : booster.num_feature()] if booster.num_feature() <= X.shape[1] else np.zeros((2, booster.num_feature()))
        for kw in ({}, {"raw_score": True}, {"pred_leaf": True}, {"pred_contrib": True}):
            try:
                booster.predict(x, **kw)
            except Exception:
                pass
        for use in (booster.dump_model, booster.model_to_string, booster.model_to_binary):
            try:
                use()
            except Exception:
                pass
    except Exception:
        pass  # clean refusal is a correct outcome
    print("OK", path, flush=True)
"""

#: a 40-byte section table entry: (field offset, struct format) for dtype,
#: codec, shuffle, offset, stored_len, raw_len (see FALB::SectionEntry)
_ENTRY_FIELDS = [(4, "<I"), (8, "<I"), (12, "<I"), (16, "<Q"), (24, "<Q"), (32, "<Q")]
_BOUNDARY = [0, 1, 2, 7, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1, 2**64 - 1]


def build_seed_files(tmp):
    """A few FALB files with different sections present (cats, multiclass)."""
    files = []
    for name, params, cats in [
        ("plain", {"objective": "regression", "num_leaves": 31}, 0),
        ("cat", {"objective": "regression", "num_leaves": 31}, 2),
        ("multi", {"objective": "multiclass", "num_class": 3, "num_leaves": 15}, 0),
        # constant (1-leaf) trees beside grown ones, saved WITH the sections
        # the loader addresses by COUNT rather than by a stored offset: a
        # zero-node tree makes every one of those blocks a degenerate width
        ("constant", {"objective": "multiclass", "num_class": 3, "num_leaves": 15}, 0),
        # uncompressed sections: payload words can then be hit directly
        # (kind 5), and f32 leaves exercise the other leaf dtype
        ("raw-cat", {"objective": "regression", "num_leaves": 31}, 2),
        ("raw-f32", {"objective": "binary", "num_leaves": 31}, 0),
    ]:
        rng = np.random.default_rng(7)
        x = rng.standard_normal((2000, 6))
        for j in range(cats):
            x[:, j] = rng.integers(0, 30, size=2000)
        y = x @ rng.standard_normal(6)
        if name == "multi":
            y = (x @ rng.standard_normal((6, 3))).argmax(1).astype(float)
        if name == "constant":
            # class 2 never occurs, so its per-class trees find no gain and
            # come back constant every round while classes 0 and 1 grow
            y = (x[:, 1] > 0).astype(float)
        if name == "raw-f32":
            y = (y > np.median(y)).astype(float)
        # explicit cpu: the fuzzer tests the PARSER, and cpu-trained seed
        # files keep every mutated load off the GPU -- immune to whatever
        # state the preceding gates left the device in
        p = dict(params, verbose=-1, seed=42, num_threads=1, device_type="cpu")
        ds = flc.Dataset(x, label=y, params=p, categorical_feature=list(range(cats)) or "auto")
        bst = flc.train(p, ds, num_boost_round=8)
        f = tmp / f"{name}.falb"
        full = name in ("constant", "raw-cat")
        extra = {"compress_level": 0} if name.startswith("raw") else {}
        if name == "raw-f32":
            extra["f32_leaves"] = True
        bst.save_model(str(f), format="falb", with_stats=full, with_diagnostics=full, **extra)
        files.append(f.read_bytes())
    return files


def mutate(rng, blob):
    """One structure-aware mutation of a valid FALB byte string."""
    b = bytearray(blob)
    kind = rng.integers(0, 6)
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
    count = struct.unpack_from("<I", b, 24)[0] if len(b) >= 32 else 0
    count = min(count, max(0, (len(b) - 32) // 40))
    if kind == 4 and count:  # one section-table field set to a boundary value
        entry = 32 + 40 * int(rng.integers(0, count))
        off, fmt = _ENTRY_FIELDS[int(rng.integers(0, len(_ENTRY_FIELDS)))]
        width = 8 if fmt == "<Q" else 4
        # index the list: rng.choice would coerce 2**63-1 / 2**64-1 to float64
        values = [*_BOUNDARY, len(b), len(b) - 1, int(rng.integers(0, 2**31))]
        value = values[int(rng.integers(0, len(values)))] % (2 ** (8 * width))
        struct.pack_into(fmt, b, entry + off, value)
        return bytes(b)
    if kind == 5 and count:  # a word inside an uncompressed section payload
        entry = 32 + 40 * int(rng.integers(0, count))
        codec, _shuffle, start, stored = struct.unpack_from("<IIQQ", b, entry + 8)
        if codec == 0 and stored >= 4 and start + stored <= len(b):
            pos = start + 4 * int(rng.integers(0, stored // 4))
            values = [0, 1, 2**31 - 1, 2**31, 2**32 - 1, 255, 65535, int(rng.integers(0, 2**32))]
            value = values[int(rng.integers(0, len(values)))]
            struct.pack_into("<I", b, pos, value)
            return bytes(b)
    # kind == 3 (and the fallback): splice -- copy a random chunk over another
    # region (breaks section sizes/offsets while keeping plausible content)
    n = int(rng.integers(4, 256))
    src = int(rng.integers(0, max(1, len(b) - n)))
    dst = int(rng.integers(0, max(1, len(b) - n)))
    b[dst : dst + n] = b[src : src + n]
    return bytes(b)


def replay_crash_corpus():
    """Each committed finding must load or refuse cleanly, in bounded time."""
    files = sorted(CRASH_DIR.glob("*.falb"))
    for f in files:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _WORKER, str(f)],
                capture_output=True,
                timeout=20,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL: loader HUNG on committed crash fixture {f.name}")
            return 1
        if proc.returncode != 0:
            print(f"FAIL: loader killed (exit {proc.returncode}) on committed crash fixture {f.name}")
            return 1
    print(f"crash corpus replay: {len(files)} fixture(s) load or refuse cleanly")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rc = replay_crash_corpus()
    if rc != 0:
        return rc

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
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _WORKER, *paths],
                    capture_output=True,
                    text=True,
                    errors="replace",  # a refusal may quote raw bytes of the corrupt file
                    timeout=300,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                # a hang is a finding just like a crash: bisect the batch with
                # a per-file timeout to name the culprit
                done = (e.stdout or b"").decode(errors="replace").count("OK ")
                for j, path in enumerate(paths):
                    try:
                        single = subprocess.run(
                            [sys.executable, "-c", _WORKER, path],
                            capture_output=True,
                            timeout=20,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        repro = Path(__file__).resolve().parent / "falb_fuzz_crash.falb"
                        shutil.copy(path, repro)
                        print(
                            f"FAIL: loader HUNG on mutation #{tried + j} "
                            f"(mutation seed {recipes[j]}, fuzz seed {args.seed})"
                        )
                        print(f"repro file kept at: {repro}")
                        return 1
                    if single.returncode != 0:
                        repro = Path(__file__).resolve().parent / "falb_fuzz_crash.falb"
                        shutil.copy(path, repro)
                        print(
                            f"FAIL: loader killed (exit {single.returncode}) on mutation #{tried + j} "
                            f"(mutation seed {recipes[j]}, fuzz seed {args.seed})"
                        )
                        print(f"repro file kept at: {repro}")
                        return 1
                print(
                    f"FAIL: batch timed out after {done} loads but no single file "
                    f"hangs alone (fuzz seed {args.seed}, batch start #{tried}) -- "
                    "suspect environment slowness, rerun"
                )
                return 1
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
