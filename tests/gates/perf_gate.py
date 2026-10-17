"""Perf regression gate over cell construct+train timings.

Only strict gpuq jobs may evaluate or record timings: invoke with --gpuq-strict
inside a gpuq submit --exclusive job with an explicit bounded timeout. The daemon
must supply GPUQ_JOB_ID and a readable GPUQ_CONTENTION_LOG, including an empty log
for a clean job. Missing/unreadable context is an error, never an idle verdict.
There is no standalone bypass or second GPU-idle detector. Desktop contexts and
broad slowdowns do not override the scheduler's contention evidence.

The baseline is machine-local: --baseline-file overrides FALCATA_PERF_BASELINE,
which defaults to ~/.cache/falcata-gates/perf_baseline.json. Warn/fail thresholds,
two consecutive regressions and best-of-three remeasurement govern clean jobs.
Contended jobs cannot change timing history or the per-cell pending latch. The
log is reread after remeasurements and immediately before saving. A warning on
remeasurement does not itself establish contention.

A child cannot certify the remainder of its queue job. Nightly recovery should
point FALCATA_PERF_BASELINE at staged history; its parent must publish only after
the daemon confirms that the completed strict job remained clean.
"""

import argparse
import json
import os
import stat
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_BASELINE = Path.home() / ".cache" / "falcata-gates" / "perf_baseline.json"
HISTORY = 20
REMEASURE_TRIES = 3


class GPUQContextError(RuntimeError):
    """The scheduler's strict-job evidence cannot be trusted or read."""


class StrictGPUQ:
    def __init__(self):
        self.job = os.environ.get("GPUQ_JOB_ID", "")
        path = os.environ.get("GPUQ_CONTENTION_LOG", "")
        if not self.job.isascii() or not self.job.isdigit() or int(self.job) <= 0 or not path:
            raise GPUQContextError("requires GPUQ_JOB_ID and GPUQ_CONTENTION_LOG from a strict gpuq job")
        self.path = Path(path)
        self.identity = None
        self.text = ""

    def contended(self):
        try:
            if not self.path.is_file():
                raise GPUQContextError(f"contention log is missing or not a regular file: {self.path}")
            with self.path.open() as log:
                info = os.fstat(log.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise GPUQContextError(f"contention log is not a regular file: {self.path}")
                text = log.read()
        except (OSError, UnicodeError) as e:
            raise GPUQContextError(f"cannot read contention log {self.path}: {e}") from e
        identity = (info.st_dev, info.st_ino)
        if (self.identity is not None and identity != self.identity) or not text.startswith(self.text):
            raise GPUQContextError(f"contention log was replaced or truncated: {self.path}")
        self.identity, self.text = identity, text
        return bool(text.strip())

    def skip(self):
        print(f"  WARN strict gpuq job {self.job} is contended ({self.path}); timing history and pending unchanged")
        return 0


def remeasure(cell_id):
    """Re-run one lattice cell and return its construct+train seconds.

    Returns None if the re-run cannot be done (lattice failure, missing cell);
    the caller then falls back to the original timing.
    """
    gates_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "results.json"
        proc = subprocess.run(
            [sys.executable, str(gates_dir / "lattice.py"), "--check", "--only", cell_id],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "FALCATA_GATES_RESULTS": str(out)},
        )
        if proc.returncode != 0 or not out.exists():
            print(f"  note {cell_id}: re-measurement unavailable, keeping original timing")
            return None
        try:
            r = json.loads(out.read_text())["results"][cell_id]
            return round(r["construct_sec"] + r["train_sec"], 4)
        except (KeyError, ValueError):
            return None


def evaluate(args, queue):
    baseline_file = Path(args.baseline_file).expanduser()
    print(
        f"perf policy: strict-gpuq job={queue.job} contention_log={queue.path} baseline={baseline_file} "
        f"warn={args.warn_pct:g}% fail={args.fail_pct:g}% repeat=2 remeasure={REMEASURE_TRIES} record={args.record}"
    )
    if queue.contended():
        return queue.skip()

    results = json.loads(Path(args.results).read_text())["results"]
    times = {
        rid: round(r["construct_sec"] + r["train_sec"], 4)
        for rid, r in results.items()
        if r.get("ok") and "construct_sec" in r
    }
    history = json.loads(baseline_file.read_text()) if baseline_file.exists() else {}
    pending_prev = set(history.get("_pending", []))
    failures, warnings, pending_now = [], [], []

    for cid, t in sorted(times.items()):
        past = history.get(cid, [])
        if len(past) >= 3:
            ref = statistics.median(past)
            pct = (t - ref) / ref * 100.0
            if pct > args.fail_pct:
                if cid in pending_prev:
                    best = None
                    for _ in range(REMEASURE_TRIES):
                        if queue.contended():
                            return queue.skip()
                        again = remeasure(cid)
                        if queue.contended():
                            return queue.skip()
                        if again is None:
                            break
                        best = again if best is None else min(best, again)
                        if (best - ref) / ref * 100.0 <= args.fail_pct:
                            break
                    if best is not None and best < t:
                        t = times[cid] = best
                        pct = (t - ref) / ref * 100.0
                    how = (
                        f"confirmed by best of {REMEASURE_TRIES} re-measurements"
                        if best is not None
                        else "re-measurement unavailable"
                    )
                    if pct > args.fail_pct:
                        failures.append(
                            f"{cid}: {t:.3f}s vs median {ref:.3f}s (+{pct:.0f}%) [2nd consecutive run, {how}]"
                        )
                    else:
                        warnings.append(
                            f"{cid}: cleared on re-measurement ({t:.3f}s vs median {ref:.3f}s, {pct:+.0f}%)"
                        )
                else:
                    pending_now.append(cid)
                    warnings.append(
                        f"{cid}: {t:.3f}s vs median {ref:.3f}s (+{pct:.0f}%) [1st occurrence -- fails if it repeats next run]"
                    )
            elif pct > args.warn_pct:
                warnings.append(f"{cid}: {t:.3f}s vs median {ref:.3f}s (+{pct:.0f}%)")
            else:
                print(f"  ok   {cid}: {t:.3f}s (median {ref:.3f}s, {pct:+.0f}%)")
        else:
            print(f"  new  {cid}: {t:.3f}s ({len(past)} historical runs; gate needs 3)")

    # Preserve consecutiveness even without --record. Suspect timings never
    # enter history, and any failure prevents the entire run from recording.
    history["_pending"] = sorted(pending_now)
    recording = args.record and not failures
    if recording:
        for cid, t in times.items():
            if cid in pending_now:
                continue
            history.setdefault(cid, []).append(t)
            history[cid] = history[cid][-HISTORY:]
    payload = json.dumps(history, indent=1)
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    if queue.contended():
        return queue.skip()
    baseline_file.write_text(payload)

    for w in warnings:
        print(f"  WARN {w}")
    for f in failures:
        print(f"  FAIL {f}")
    if recording:
        print(f"perf baseline updated -> {baseline_file}")
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "lattice_results.json"))
    ap.add_argument("--record", action="store_true", help="append clean green timings to the rolling baseline")
    ap.add_argument("--baseline-file", default=os.environ.get("FALCATA_PERF_BASELINE") or str(DEFAULT_BASELINE))
    ap.add_argument("--warn-pct", type=float, default=10.0)
    ap.add_argument("--fail-pct", type=float, default=25.0)
    ap.add_argument(
        "--gpuq-strict", action="store_true", help="declare strict gpuq submission; scheduler context is mandatory"
    )
    args = ap.parse_args(argv)
    if not args.gpuq_strict:
        ap.error("use --gpuq-strict inside gpuq submit --exclusive with an explicit bounded timeout; no bypass")
    try:
        return evaluate(args, StrictGPUQ())
    except GPUQContextError as e:
        print(f"  FAIL strict gpuq context: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
