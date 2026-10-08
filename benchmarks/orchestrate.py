"""Resumable benchmark matrix runner.

Executes bench.py sequentially, one process per (library, dataset) that runs
every pending cell of the pair -- the data is mapped once and each regime's
Dataset is constructed once for all of its kinds; each run still has the GPU
to itself -- skipping cells already recorded in ``<workspace>/results/runs.jsonl``, so it
is safe to interrupt and relaunch at any time. Datasets whose preprocessed
cache is missing are skipped with a note (run datasets.py first).

Usage::

    python benchmarks/orchestrate.py                 # everything available
    python benchmarks/orchestrate.py --only higgs,epsilon
    python benchmarks/orchestrate.py --libraries falcata-stoch,falcata-fixed --results results/runs_rerun.jsonl
    python benchmarks/orchestrate.py --dry-run

``--libraries`` restricts the matrix to a comma-separated subset of the library
arms; ``--results`` writes to (and resumes from) another results file than
``<workspace>/results/runs.jsonl``, so a rerun of one library leaves the
baseline file untouched.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from common import (
    ALL_LIBRARIES,
    RUNS_JSONL,
    dataset_ready,
    library_runs_cell,
    regimes_for,
    venv_python,
)

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")

#: small/fast first for early signal; the huge ones last
# small/fast first for early signal; the two airline variants last
DATASET_ORDER = [
    "fraud",
    "covtype",
    "year",
    "higgs",
    "epsilon",
    "numerai",
    "airline",
    "airline-cat",
]
KINDS = ["warmup", "timed1", "timed2", "timed3", "curve"]
#: multi-hour 30k-tree runs measure ±1% across repeats and warm their own
#: caches during construct — ONE timed run per library, no warmup, no curve
#: (fast cells keep the full warmup+3+curve protocol)
# 30k-round regimes: repeats are unaffordable, so one timed run, no warmup/curve
REGIME_KINDS = {"numerai-deep": ["timed1"], "numerai-leaf": ["timed1"]}
REGIME_TIMEOUT_S = {"numerai-deep": 18000, "numerai-leaf": 18000}
TIMEOUT_S = {
    "fraud": 1800,
    "covtype": 1800,
    "year": 1800,
    "higgs": 7200,
    "epsilon": 7200,
    "numerai": 43200,
    "airline": 14400,
}


def load_done(results):
    """(library, dataset, regime, kind) -> status of every cell recorded in ``results``."""
    done = {}
    if os.path.exists(results):
        with open(results) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(r["library"], r["dataset"], r["regime"], r["kind"])] = r["status"]
    return done


def summarize(statuses):
    """'4 ok, 1 insane' style count of a group's cell statuses."""
    counts = {}
    for st in statuses:
        counts[st] = counts.get(st, 0) + 1
    return ", ".join(f"{n} {st}" for st, n in counts.items())


def record(results, status, lib, ds, reg, kind, **extra):
    with open(results, "a") as f:
        f.write(
            json.dumps(
                {
                    "library": lib,
                    "dataset": ds,
                    "regime": reg,
                    "kind": kind,
                    "status": status,
                    **extra,
                }
            )
            + "\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated dataset subset")
    ap.add_argument("--libraries", default=None, help="comma-separated library subset (default: every arm)")
    ap.add_argument("--results", default=None, help=f"results file to append to and resume from (default {RUNS_JSONL})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    datasets = args.only.split(",") if args.only else DATASET_ORDER
    libraries = ALL_LIBRARIES
    if args.libraries:
        libraries = args.libraries.split(",")
        unknown = [lib for lib in libraries if lib not in ALL_LIBRARIES]
        if unknown:
            sys.exit(f"unknown libraries {unknown}; known: {ALL_LIBRARIES}")
    results = os.path.abspath(args.results) if args.results else RUNS_JSONL

    cells = []
    for ds in datasets:
        if not dataset_ready(ds):
            print(
                f"NOTE: dataset '{ds}' not prepared, skipping (run datasets.py)",
                flush=True,
            )
            continue
        for reg in regimes_for(ds):
            for lib in libraries:
                if not library_runs_cell(lib, ds, reg):
                    continue
                for kind in REGIME_KINDS.get(reg, KINDS):
                    cells.append((lib, ds, reg, kind))

    done = load_done(results)
    todo = [c for c in cells if c not in done]
    print(
        f"matrix: {len(cells)} cells, {len(cells) - len(todo)} done, {len(todo)} to run",
        flush=True,
    )
    if args.dry_run:
        for c in todo:
            print(c)
        return

    os.makedirs(os.path.dirname(results), exist_ok=True)
    # one bench.py process per (library, dataset): the data is mapped once and
    # each regime's Dataset is constructed once for all of its kinds, instead of
    # once per cell. Cells keep their own records, so a crash mid-group loses
    # nothing already recorded and a relaunch resumes at the first missing cell.
    by_pair = {}
    for lib, ds, reg, kind in todo:
        by_pair.setdefault((lib, ds), []).append((reg, kind))
    groups = list(by_pair.items())
    n_run = 0
    for (lib, ds), group_cells in groups:
        done = load_done(results)
        pending = [(reg, kind) for reg, kind in group_cells if (lib, ds, reg, kind) not in done]
        if not pending:
            continue
        cmd = [
            venv_python(lib),
            BENCH,
            "--library",
            lib,
            "--dataset",
            ds,
            "--cells",
            ",".join(f"{reg}:{kind}" for reg, kind in pending),
            "--out",
            results,
        ]
        timeout = sum(REGIME_TIMEOUT_S.get(reg, TIMEOUT_S.get(ds, 7200)) for reg, _ in pending)
        t0 = time.time()
        print(
            f"[{n_run + 1}-{n_run + len(pending)}/{len(todo)}] RUN {lib}/{ds} "
            f"({len(pending)} cells: {' '.join(f'{r}:{k}' for r, k in pending)})",
            flush=True,
        )
        n_run += len(pending)
        try:
            # the child prints one JSON line per finished cell straight to this log
            p = subprocess.run(
                cmd,
                check=False,
                timeout=timeout,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                },
                stderr=subprocess.PIPE,
                text=True,
            )
            status = "ok" if p.returncode == 0 else "failed"
            stderr_tail = p.stderr[-2000:]
        except subprocess.TimeoutExpired as e:
            status = "timeout"
            stderr_tail = (e.stderr or b"")[-2000:] if isinstance(e.stderr, bytes) else str(e.stderr or "")[-2000:]
        done = load_done(results)
        missing = [(reg, kind) for reg, kind in pending if (lib, ds, reg, kind) not in done]
        if missing:
            # the child died (segfault, OOM kill, timeout) on the first missing
            # cell before writing its record; record that one so a resume does
            # not retry it forever, and leave the rest for the relaunch
            reg, kind = missing[0]
            record(
                results,
                status if status == "timeout" else "failed",
                lib,
                ds,
                reg,
                kind,
                error=f"process {status}: {stderr_tail[-500:]}",
            )
            print(
                f"    cell {reg}:{kind} left no record ({status}); {len(missing) - 1} cells of this group remain for a relaunch",
                flush=True,
            )
        if status != "ok" and stderr_tail.strip():
            sys.stderr.write(stderr_tail + "\n")
        done = load_done(results)
        statuses = [done.get((lib, ds, reg, kind), "missing") for reg, kind in pending]
        print(f"    -> {status} ({time.time() - t0:.0f}s): {summarize(statuses)}", flush=True)


if __name__ == "__main__":
    main()
