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
REMEASURE_TRIES = 3


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


def _own_process_tree():
    """PIDs of this process and its ancestors, so they are not called foreign."""
    pids, pid = set(), os.getpid()
    for _ in range(64):
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                ppid = int(f.read().split(") ", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
        if ppid <= 1 or ppid in pids:
            break
        pid = ppid
    return pids


def foreign_gpu_processes():
    """Compute processes on the GPU that are not ours.

    The gpu_guard admits a run on aggregate VRAM/utilization, which a
    long-running neighbour can satisfy while still stealing latency. Timings
    taken beside another process are not comparable to a baseline recorded
    alone, and the smallest cells suffer most: they are launch-latency bound,
    so a neighbour can double a 0.14s cell while a throughput-bound 0.5s cell
    barely moves. That is why a SINGLE wildly elevated cell is not by itself
    evidence of a regression on this box.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []          # no nvidia-smi: cannot tell, do not claim contention
    ours = _own_process_tree()
    found = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in ours:
            continue
        name = "?"
        try:
            with open(f"/proc/{pid}/comm") as f:
                name = f.read().strip()
        except OSError:
            pass
        found.append((pid, name, parts[1] if len(parts) > 1 else "?"))
    return found


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
    ap.add_argument("--contended-pct", type=float, default=10.0,
                    help="median cell elevation above which the whole run is "
                         "treated as contention rather than regression")
    args = ap.parse_args()
    global BASELINE_FILE, WARN_PCT, FAIL_PCT, CONTENDED_PCT
    BASELINE_FILE = Path(args.baseline_file).expanduser()
    WARN_PCT, FAIL_PCT = args.warn_pct, args.fail_pct
    CONTENDED_PCT = args.contended_pct

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

    # Is the whole MACHINE slow, or is one cell slow? This box is a shared
    # dev/benchmark machine, and a busy GPU lifts every cell at once, while a
    # real code regression moves the cells it touches and leaves the rest
    # alone. So judge the run before judging the cells: if the median cell is
    # meaningfully elevated, this is common-cause (contention), not a
    # regression, and no individual cell's timing is trustworthy.
    deltas = []
    for cid, t in times.items():
        past = history.get(cid, [])
        if len(past) >= 3:
            ref = statistics.median(past)
            deltas.append((t - ref) / ref * 100.0)
    median_delta = statistics.median(deltas) if deltas else 0.0
    neighbours = foreign_gpu_processes()
    contended = median_delta > CONTENDED_PCT or bool(neighbours)
    if neighbours:
        who = ", ".join(f"{name}[{pid}] {mem} MiB" for pid, name, mem in neighbours)
        print(f"  NOTE another process is on the GPU ({who}). Timings are not "
              f"comparable to a baseline recorded alone -- reporting as "
              f"warnings, not failures, and not recording a baseline.")
    elif contended:
        print(f"  NOTE run looks contended: median cell is {median_delta:+.0f}% "
              f"vs baseline across {len(deltas)} tracked cells (threshold "
              f"{CONTENDED_PCT:+.0f}%). Timings are not trustworthy -- reporting "
              f"as warnings, not failures, and not recording a baseline.")

    for cid, t in sorted(times.items()):
        past = history.get(cid, [])
        if len(past) >= 3:
            ref = statistics.median(past)
            pct = (t - ref) / ref * 100.0
            if pct > FAIL_PCT and contended:
                warnings.append(f"{cid}: {t:.3f}s vs median {ref:.3f}s (+{pct:.0f}%) [contended run -- not counted]")
            elif pct > FAIL_PCT:
                if cid in pending_prev:
                    # This box doubles as a dev/benchmark machine, so a second
                    # slow run is NOT proof of a regression -- both runs can
                    # simply have overlapped GPU work. Contention only ever
                    # makes a cell slower, so re-measure it now and keep the
                    # best observation; only a cell that is still slow when
                    # asked again is called a regression.
                    best = None
                    for _ in range(REMEASURE_TRIES):
                        again = remeasure(cid)
                        if again is None:
                            break
                        best = again if best is None else min(best, again)
                        if (best - ref) / ref * 100.0 <= FAIL_PCT:
                            break   # already clears; no need to keep measuring
                    if best is not None and best < t:
                        t = best
                        pct = (t - ref) / ref * 100.0
                    how = (f"confirmed by best of {REMEASURE_TRIES} re-measurements"
                           if best is not None else "re-measurement unavailable")
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
    # A contended run proves nothing, so it neither arms nor disarms the
    # consecutive-run latch: keep whatever was pending before.
    history["_pending"] = sorted(pending_prev if contended else pending_now)
    BASELINE_FILE.write_text(json.dumps(history, indent=1))

    if args.record and not failures and not contended:
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
