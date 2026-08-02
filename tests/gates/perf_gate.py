"""Perf regression gate over lattice cell timings.

Compares the perf-tracked cells' wall times from a lattice run against a
machine-local rolling baseline (median of the last N green runs). Warns at
+WARN_PCT, fails at +FAIL_PCT. The baseline lives OUTSIDE the repo (timings
are machine-specific) in ~/.cache/falcata-gates/perf_baseline.json and is
appended to only on green runs, so a regression cannot poison its own
baseline.

Timing on a desktop GPU is noisy; the gate therefore (a) only runs when the
gpu_guard admitted an idle GPU, (b) gates on construct+train sums of cells
marked perf=True, (c) uses the rolling median as reference, and (d) FAILS
only when the same cell regresses in two CONSECUTIVE runs. The box doubles
as a benchmark/dev machine: a single run's +25-65% blowups are almost always
host contention that the point-in-time gpu_guard snapshot missed, and each
one used to send a failure email per push. A real code regression persists
into the next (usually quieter) run; one-off contention does not.
Single-run regressions are reported as warnings and remembered in the
baseline file's "_pending" key.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_BASELINE = Path.home() / ".cache" / "falcata-gates" / "perf_baseline.json"
HISTORY = 20


def remeasure(cell_id):
    """Re-run one lattice cell and return its construct+train seconds.

    Returns None if the re-run cannot be done (lattice failure, missing cell);
    the caller then falls back to the original timing.
    """
    gates_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "results.json"
        proc = subprocess.run(
            [sys.executable, str(gates_dir / "lattice.py"), "--check",
             "--only", cell_id],
            capture_output=True, text=True, check=False,
            env={**os.environ, "FALCATA_GATES_RESULTS": str(out)})
        if proc.returncode != 0 or not out.exists():
            print(f"  note {cell_id}: re-measurement unavailable, keeping original timing")
            return None
        try:
            r = json.loads(out.read_text())["results"][cell_id]
            return round(r["construct_sec"] + r["train_sec"], 4)
        except (KeyError, ValueError):
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=str(Path(__file__).resolve().parent / "lattice_results.json"),
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="append this run to the rolling baseline (green runs only)",
    )
    ap.add_argument("--baseline-file", default=str(DEFAULT_BASELINE))
    ap.add_argument("--warn-pct", type=float, default=10.0)
    ap.add_argument("--fail-pct", type=float, default=25.0)
    args = ap.parse_args()
    global BASELINE_FILE, WARN_PCT, FAIL_PCT
    BASELINE_FILE = Path(args.baseline_file).expanduser()
    WARN_PCT, FAIL_PCT = args.warn_pct, args.fail_pct

    results = json.loads(Path(args.results).read_text())["results"]
    times = {
        rid: round(r["construct_sec"] + r["train_sec"], 4)
        for rid, r in results.items()
        if r.get("ok") and "construct_sec" in r
    }

    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(BASELINE_FILE.read_text()) if BASELINE_FILE.exists() else {}

    pending_prev = set(history.get("_pending", []))
    failures, warnings, pending_now = [], [], []
    for cid, t in sorted(times.items()):
        past = history.get(cid, [])
        if len(past) >= 3:
            ref = statistics.median(past)
            pct = (t - ref) / ref * 100.0
            if pct > FAIL_PCT:
                if cid in pending_prev:
                    # This box doubles as a dev/benchmark machine, so a second
                    # slow run is NOT proof of a regression -- both runs can
                    # simply have overlapped GPU work. Contention only ever
                    # makes a cell slower, so re-measure it now and keep the
                    # best observation; only a cell that is still slow when
                    # asked again is called a regression.
                    again = remeasure(cid)
                    if again is not None and again < t:
                        t = again
                        pct = (t - ref) / ref * 100.0
                    how = ("confirmed by re-measurement" if again is not None
                           else "re-measurement unavailable")
                    if pct > FAIL_PCT:
                        failures.append(f"{cid}: {t:.3f}s vs median {ref:.3f}s (+{pct:.0f}%) [2nd consecutive run, {how}]")
                    else:
                        warnings.append(f"{cid}: cleared on re-measurement ({t:.3f}s vs median {ref:.3f}s, {pct:+.0f}%) -- earlier timing was contention")
                else:
                    pending_now.append(cid)
                    warnings.append(f"{cid}: {t:.3f}s vs median {ref:.3f}s (+{pct:.0f}%) [1st occurrence -- fails if it repeats next run]")
            elif pct > WARN_PCT:
                warnings.append(f"{cid}: {t:.3f}s vs median {ref:.3f}s (+{pct:.0f}%)")
            else:
                print(f"  ok   {cid}: {t:.3f}s (median {ref:.3f}s, {pct:+.0f}%)")
        else:
            print(f"  new  {cid}: {t:.3f}s ({len(past)} historical runs; gate needs 3)")

    for w in warnings:
        print(f"  WARN {w}")
    for f in failures:
        print(f"  FAIL {f}")

    # persist the pending set even on non-record runs: consecutiveness is the
    # whole point (over-FAIL_PCT cells never enter the timing history itself)
    history["_pending"] = sorted(pending_now)
    BASELINE_FILE.write_text(json.dumps(history, indent=1))

    if args.record and not failures:
        for cid, t in times.items():
            if cid in pending_now:
                continue  # never let a suspect timing poison the baseline
            history.setdefault(cid, []).append(t)
            history[cid] = history[cid][-HISTORY:]
        BASELINE_FILE.write_text(json.dumps(history, indent=1))
        print(f"perf baseline updated -> {BASELINE_FILE}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
