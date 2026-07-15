"""Resumable benchmark matrix runner.

Executes bench.py cells sequentially (each run gets the GPU exclusively),
skipping cells already recorded in ``<workspace>/results/runs.jsonl``, so it
is safe to interrupt and relaunch at any time. Datasets whose preprocessed
cache is missing are skipped with a note (run datasets.py first).

Usage::

    python benchmarks/orchestrate.py                 # everything available
    python benchmarks/orchestrate.py --only higgs,epsilon
    python benchmarks/orchestrate.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time

from common import (
    LIBRARIES,
    RUNS_JSONL,
    dataset_ready,
    library_runs_cell,
    regimes_for,
    venv_python,
)

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")

#: small/fast first for early signal; the huge ones last
DATASET_ORDER = ["fraud", "covtype", "year", "higgs", "epsilon", "numerai", "airline"]
KINDS = ["warmup", "timed1", "timed2", "timed3", "curve"]
#: multi-hour 30k-tree runs measure ±1% across repeats — one timed run suffices
#: (warmup stays as the page-cache/sanity run; fast cells keep the full protocol)
REGIME_KINDS = {"numerai-deep": ["warmup", "timed1"]}
TIMEOUT_S = {
    "fraud": 1800,
    "covtype": 1800,
    "year": 1800,
    "higgs": 7200,
    "epsilon": 7200,
    "numerai": 43200,
    "airline": 14400,
}


def load_done():
    done = {}
    if os.path.exists(RUNS_JSONL):
        with open(RUNS_JSONL) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(r["library"], r["dataset"], r["regime"], r["kind"])] = r["status"]
    return done


def record(status, lib, ds, reg, kind, **extra):
    with open(RUNS_JSONL, "a") as f:
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    datasets = args.only.split(",") if args.only else DATASET_ORDER

    cells = []
    for ds in datasets:
        if not dataset_ready(ds):
            print(
                f"NOTE: dataset '{ds}' not prepared, skipping (run datasets.py)",
                flush=True,
            )
            continue
        for reg in regimes_for(ds):
            for lib in LIBRARIES:
                if not library_runs_cell(lib, ds, reg):
                    continue
                for kind in REGIME_KINDS.get(reg, KINDS):
                    cells.append((lib, ds, reg, kind))

    done = load_done()
    todo = [c for c in cells if c not in done]
    print(
        f"matrix: {len(cells)} cells, {len(cells) - len(todo)} done, {len(todo)} to run",
        flush=True,
    )
    if args.dry_run:
        for c in todo:
            print(c)
        return

    os.makedirs(os.path.dirname(RUNS_JSONL), exist_ok=True)
    for i, (lib, ds, reg, kind) in enumerate(todo):
        done = load_done()
        if (lib, ds, reg, kind) in done:
            continue
        # if the warmup for this combo failed, don't waste time on the rest
        if kind != "warmup" and done.get((lib, ds, reg, "warmup")) == "failed":
            print(f"SKIP {lib}/{ds}/{reg}/{kind} (warmup failed)", flush=True)
            record("skipped_warmup_failed", lib, ds, reg, kind)
            continue
        cmd = [
            venv_python(lib),
            BENCH,
            "--library",
            lib,
            "--dataset",
            ds,
            "--regime",
            reg,
            "--kind",
            kind,
        ]
        t0 = time.time()
        print(f"[{i + 1}/{len(todo)}] RUN {lib}/{ds}/{reg}/{kind}", flush=True)
        try:
            p = subprocess.run(
                cmd,
                check=False,
                timeout=TIMEOUT_S.get(ds, 7200),
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                },
                capture_output=True,
                text=True,
            )
            status = "ok" if p.returncode == 0 else "failed"
            if p.returncode != 0 and (lib, ds, reg, kind) not in load_done():
                # the child died before writing its record (e.g. segfault);
                # record the failure so resume doesn't retry it forever
                record(
                    "failed",
                    lib,
                    ds,
                    reg,
                    kind,
                    error=f"exit code {p.returncode}: {p.stderr[-500:]}",
                )
            if p.returncode != 0:
                sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:] + "\n")
        except subprocess.TimeoutExpired:
            status = "timeout"
            record("timeout", lib, ds, reg, kind)
        print(f"    -> {status} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
