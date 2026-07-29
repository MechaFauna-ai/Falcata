"""Perf regression gate over lattice cell timings.

Compares the perf-tracked cells' wall times from a lattice run against a
machine-local rolling baseline (median of the last N green runs). Warns at
+WARN_PCT, fails at +FAIL_PCT. The baseline lives OUTSIDE the repo (timings
are machine-specific) in ~/.cache/exaboost-gates/perf_baseline.json and is
appended to only on green runs, so a regression cannot poison its own
baseline.

Timing on a desktop GPU is noisy; the gate therefore (a) only runs when the
gpu_guard admitted an idle GPU, (b) gates on construct+train sums of cells
marked perf=True, (c) uses the rolling median as reference.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

DEFAULT_BASELINE = Path.home() / ".cache" / "exaboost-gates" / "perf_baseline.json"
HISTORY = 20


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

    failures, warnings = [], []
    for cid, t in sorted(times.items()):
        past = history.get(cid, [])
        if len(past) >= 3:
            ref = statistics.median(past)
            pct = (t - ref) / ref * 100.0
            if pct > FAIL_PCT:
                failures.append(f"{cid}: {t:.3f}s vs median {ref:.3f}s (+{pct:.0f}%)")
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

    if args.record and not failures:
        for cid, t in times.items():
            history.setdefault(cid, []).append(t)
            history[cid] = history[cid][-HISTORY:]
        BASELINE_FILE.write_text(json.dumps(history, indent=1))
        print(f"perf baseline updated -> {BASELINE_FILE}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
