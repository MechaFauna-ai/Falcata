"""Pre-merge speed gate: is a pull request faster, slower, or the same as the commit it branches from?

The perf gate (perf_gate.py) compares one nightly run on master with a rolling median of earlier nights. That
catches large regressions after they have merged; it cannot resolve a few percent (cells drift 4-8% between nights
on this box), and it judges master, not the change. This gate measures the change itself before it merges:

* Two builds, BASE (the merge base of the pull request) and HEAD, run the same fixed set of cells (speed_cells.py:
  the bench tier's shapes, v6-like tree fits and the production h60 shape) in the SAME job, interleaved: pair i runs
  base then head (head then base on odd pairs), each run in a fresh process. Drift, warm-up and thermal state fall on
  both arms alike, which is what lets a 3% difference show (the kernel-evo A/B design that found the #47-#49 gains).
* Every run records stage times (construct, booster create, trees, predict), steady ms per iteration, host and
  device memory peaks, and the model md5.
* Per cell and stage the paired log ratios head/base give an estimate (geometric mean ratio) with a 95% t interval
  and a one-sided p-value for "head is slower"; p-values are Holm-adjusted over every (cell, stage) test of the run.
* Verdict per test: FAIL when the estimate is at least --fail-pct slower AND the adjusted p is below --alpha;
  INCONCLUSIVE when the estimate passes the threshold without significance (the run adds pairs to that cell, up to
  --max-pairs, before it settles); FASTER / SLOWER-below-threshold / SAME otherwise. Memory: FAIL when the median
  peak grows by more than --mem-fail-pct and more than 0.1 GB. A changed model md5 is reported, never failed (the
  lattice decides correctness; a reviewer confirms the change is intended).
* The result is written whole before any statistics (results JSON), then the verdict (markdown for the pull
  request, exit code), then one line per run in a per-commit history (--history), so trends across master commits
  stay visible.

The gate refuses to judge a contended card: inside a strict gpuq job a non-empty $GPUQ_CONTENTION_LOG, or a foreign
compute process seen before a run, ends the run with exit 3 ("contended, rerun"), never with a verdict.

    speed_gate.py run --base PY[:PYTHONPATH] --head PY[:PYTHONPATH] --out results.json [--cells 'gates/*,v6/*']
    speed_gate.py compare results.json [--markdown out.md] [--history ~/.cache/falcata-gates/speed_history.jsonl]
    speed_gate.py child --cell ID            (one timed run; used by run)
    speed_gate.py list                       (the cells and whether their data is present)

Exit codes: 0 pass, 1 fail, 2 configuration error, 3 contended or incomplete (no verdict).
"""

import argparse
import fnmatch
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

GATES_DIR = Path(__file__).resolve().parent
STAGES = ("construct_s", "create_s", "trees_s", "predict_s")
MEMORY = ("host_gb_peak", "dev_gb_peak")
DEFAULT_HISTORY = Path.home() / ".cache" / "falcata-gates" / "speed_history.jsonl"


# ----------------------------------------------------------------------------------------------- statistics


def _t_quantile(p, df):
    from scipy import stats

    return float(stats.t.ppf(p, df))


def _t_sf(t, df):
    """P(T > t) for Student t with df degrees of freedom."""
    from scipy import stats

    return float(stats.t.sf(t, df))


def paired_ratio(base, head):
    """Paired log-ratio statistics of head/base: estimate (geometric mean ratio), 95% two-sided t interval,
    one-sided p for 'head slower' (H0: mean log ratio <= 0) and for 'head faster', n. Pairs with a non-positive time
    are dropped and counted."""
    pairs = [(b, h) for b, h in zip(base, head) if b and h and b > 0 and h > 0]
    n = len(pairs)
    out = {"n": n, "dropped": len(list(zip(base, head))) - n}
    if n < 2:
        out.update(ratio=None, lo=None, hi=None, p_slower=None, p_faster=None, reason="fewer than 2 valid pairs")
        return out
    logs = [math.log(h / b) for b, h in pairs]
    mean = sum(logs) / n
    var = sum((x - mean) ** 2 for x in logs) / (n - 1)
    se = math.sqrt(var / n)
    q = _t_quantile(0.975, n - 1)
    if se == 0:
        p_slower = 0.0 if mean > 0 else 1.0
        p_faster = 0.0 if mean < 0 else 1.0
    else:
        p_slower = _t_sf(mean / se, n - 1)
        p_faster = _t_sf(-mean / se, n - 1)
    out.update(
        ratio=math.exp(mean),
        lo=math.exp(mean - q * se),
        hi=math.exp(mean + q * se),
        p_slower=p_slower,
        p_faster=p_faster,
        cv_pct=100 * math.sqrt(var),
    )
    return out


def holm(pvalues):
    """Holm step-down adjusted p-values (dict key -> p; None kept as None)."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m = len(items)
    adj, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    for k, p in pvalues.items():
        if p is None:
            adj[k] = None
    return adj


def judge(stat, adj_p_slower, adj_p_faster, fail_pct, alpha, warn_pct=1.0):
    """Verdict for one (cell, stage) test."""
    r = stat.get("ratio")
    if r is None:
        return "NO-DATA"
    thr = 1 + fail_pct / 100
    if r >= thr and adj_p_slower is not None and adj_p_slower < alpha:
        return "FAIL"
    if r >= thr:
        return "INCONCLUSIVE"
    if adj_p_faster is not None and adj_p_faster < alpha and r <= 1 - warn_pct / 100:
        return "FASTER"
    if adj_p_slower is not None and adj_p_slower < alpha and r >= 1 + warn_pct / 100:
        return "SLOWER"
    return "SAME"


def needs_more(stat, fail_pct):
    """True while a cell's interval straddles the fail threshold on the slow side (more pairs could decide it)."""
    if stat.get("ratio") is None:
        return True
    thr = 1 + fail_pct / 100
    return stat["lo"] < thr < stat["hi"] and stat["ratio"] > 1.0


# ----------------------------------------------------------------------------------------------- one run (child)


def _vmhwm_gb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


class _DeviceMemory:
    """This process's device memory via NVML (nvidia-ml-py), sampled at 50 Hz; None when NVML is missing."""

    def __init__(self):
        self.peak = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001 - reported as no device memory figure, never as zero
            self.nv = None
            return
        self.peak, self.stop = 0.0, False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        pid = os.getpid()
        while not self.stop:
            try:
                for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h):
                    if p.pid == pid:
                        self.peak = max(self.peak, (p.usedGpuMemory or 0) / 2**30)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.02)

    def close(self):
        if self.nv is not None:
            self.stop = True
            self.t.join(timeout=1)
        return None if self.peak is None else round(self.peak, 3)


def model_md5(bst):
    body = bst.model_to_string().split("\nparameters:")[0]
    keep = [line for line in body.splitlines() if not line.startswith(("[cuda_plan:", "version="))]
    return hashlib.md5("\n".join(keep).encode()).hexdigest()


def child(cell_id, rounds=None):
    import numpy as np

    sys.path.insert(0, str(GATES_DIR))
    from speed_cells import cells

    cell = cells()[cell_id]
    rec = {"cell": cell_id}
    dm = _DeviceMemory()
    import falcata as flc

    rec["falcata"] = flc.__version__
    rec["lib"] = str(Path(flc.__file__).resolve().parent / "lib" / "lib_falcata.so")
    try:
        t0 = time.perf_counter()
        data = cell.load()
        rec["load_s"] = time.perf_counter() - t0
        params = dict(cell.params)
        n_iter = rounds or cell.rounds
        t0 = time.perf_counter()
        ds, va = cell.construct(flc, data, params)
        rec["construct_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        bst = flc.Booster({**params, "metric": "None"}, ds)
        if va is not None:
            bst.add_valid(va, "te")
        rec["create_s"] = time.perf_counter() - t0
        stamps = []
        t0 = time.perf_counter()
        for _ in range(n_iter):
            bst.update()
            stamps.append(time.perf_counter())
        rec["trees_s"] = stamps[-1] - t0
        d = np.diff([t0, *stamps])
        rec["iter_ms"] = 1000 * float(np.median(d[len(d) // 10 :]))
        rec["num_trees"] = bst.num_trees()
        want = n_iter * int(params.get("num_class", 1))
        if rec["num_trees"] != want:
            raise RuntimeError(f"expected {want} trees, got {rec['num_trees']}")
        rec["md5"] = model_md5(bst)
        Xt = cell.test(data)
        if Xt is not None:
            t0 = time.perf_counter()
            bst.predict(Xt)
            rec["predict_s"] = time.perf_counter() - t0
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 - the record carries the failure; the parent decides
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["host_gb_peak"] = _vmhwm_gb()
    rec["dev_gb_peak"] = dm.close()
    print("SPEED " + json.dumps(rec), flush=True)
    return 0 if rec["ok"] else 1


# ----------------------------------------------------------------------------------------------- the paired run


def _foreign_gpu_processes():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [int(x) for x in out if x.strip().isdigit()]


def _contended():
    log = os.environ.get("GPUQ_CONTENTION_LOG")
    if log and Path(log).is_file() and Path(log).read_text().strip():
        return f"gpuq contention log {log}: {Path(log).read_text().strip()[:200]}"
    foreign = _foreign_gpu_processes()
    if foreign:
        return f"foreign GPU processes {foreign}"
    return None


def _arm(spec):
    py, _, pp = spec.partition(":")
    return {"python": py, "pythonpath": pp}


def _one(arm, cell, timeout, rounds=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = ":".join(x for x in (arm["pythonpath"], str(GATES_DIR)) if x)
    env.setdefault("FALCATA_FIL", "0")
    cmd = [arm["python"], str(Path(__file__).resolve()), "child", "--cell", cell]
    if rounds:
        cmd += ["--rounds", str(rounds)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=timeout, cwd="/tmp")
    except subprocess.TimeoutExpired:
        return {"cell": cell, "ok": False, "error": f"timeout after {timeout}s"}
    line = [x for x in p.stdout.splitlines() if x.startswith("SPEED ")]
    rec = json.loads(line[-1][6:]) if line else {"cell": cell, "ok": False, "error": (p.stderr or p.stdout)[-2000:]}
    rec["wall_s"] = time.time() - t0
    return rec


def _machine():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "")
    return {"host": platform.node(), "gpu": q, "cpu": cpu}


def run(args):
    sys.path.insert(0, str(GATES_DIR))
    from speed_cells import cells

    allc = cells()
    pats = args.cells.split(",")
    chosen = [c for c in allc if any(fnmatch.fnmatch(c, p) for p in pats)]
    missing = {c: allc[c].missing() for c in chosen if allc[c].missing()}
    chosen = [c for c in chosen if c not in missing]
    arms = {"base": _arm(args.base), "head": _arm(args.head)}
    res = {
        "schema": 1,
        "started": time.time(),
        "machine": _machine(),
        "arms": arms,
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "pr": args.pr,
        "cells": chosen,
        "skipped": missing,
        "runs": [],
        "fail_pct": args.fail_pct,
        "mem_fail_pct": args.mem_fail_pct,
        "alpha": args.alpha,
    }
    out = Path(args.out)

    def save():
        out.write_text(json.dumps(res, indent=1))

    deadline = time.time() + args.budget_s
    pairs = dict.fromkeys(chosen, 0)
    open_cells = list(chosen)
    while open_cells:
        for c in list(open_cells):
            i = pairs[c]
            order = ("base", "head") if i % 2 == 0 else ("head", "base")
            for arm in order:
                why = _contended()
                if why:
                    res["contended"] = why
                    save()
                    print(f"CONTENDED: {why}; no verdict", file=sys.stderr)
                    return 3
                r = _one(arms[arm], c, args.timeout)
                r.update(arm=arm, pair=i, t=time.time())
                res["runs"].append(r)
                save()
                print(
                    json.dumps(
                        {
                            k: r.get(k)
                            for k in (
                                "cell",
                                "arm",
                                "pair",
                                "ok",
                                "construct_s",
                                "trees_s",
                                "predict_s",
                                "md5",
                                "error",
                            )
                        }
                    ),
                    flush=True,
                )
            pairs[c] += 1
            if pairs[c] < args.min_pairs:
                continue
            stats = cell_stats(res["runs"], c)
            noisy = any(needs_more(s, args.fail_pct) for s in stats.values() if s.get("n"))
            if not noisy or pairs[c] >= args.max_pairs or time.time() > deadline:
                open_cells.remove(c)
    res["finished"] = time.time()
    if time.time() > deadline:
        res["budget_exhausted"] = True
    save()
    return 0


def cell_stats(runs, cell):
    by = {}
    for r in runs:
        if r["cell"] == cell and r.get("ok"):
            by.setdefault(r["pair"], {})[r["arm"]] = r
    pairs = [p for _, p in sorted(by.items()) if "base" in p and "head" in p]
    out = {}
    for st in (*STAGES, "iter_ms"):
        b = [p["base"].get(st) for p in pairs]
        h = [p["head"].get(st) for p in pairs]
        if any(x is None for x in b + h) or not pairs:
            continue
        s = paired_ratio(b, h)
        s["base_median"] = _median(b)
        s["head_median"] = _median(h)
        out[st] = s
    return out


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# ----------------------------------------------------------------------------------------------- verdict


def compare(args):
    res = json.loads(Path(args.results).read_text())
    if res.get("contended"):
        text = f"## Speed gate: **NO VERDICT**\n\nThe card was contended ({res['contended']}); the run is repeated.\n"
        if args.markdown:
            Path(args.markdown).write_text(text)
        print(text)
        return 3
    fail_pct = args.fail_pct if args.fail_pct is not None else res["fail_pct"]
    mem_pct = args.mem_fail_pct if args.mem_fail_pct is not None else res["mem_fail_pct"]
    alpha = args.alpha if args.alpha is not None else res["alpha"]
    errors = [r for r in res["runs"] if not r.get("ok")]
    stats = {c: cell_stats(res["runs"], c) for c in res["cells"]}
    ps = {(c, st): s.get("p_slower") for c, d in stats.items() for st, s in d.items() if st in STAGES}
    pf = {(c, st): s.get("p_faster") for c, d in stats.items() for st, s in d.items() if st in STAGES}
    adj_s, adj_f = holm(ps), holm(pf)
    rows, fails, inconclusive = [], [], []
    for c, d in stats.items():
        for st in STAGES:
            if st not in d:
                continue
            s = d[st]
            v = judge(s, adj_s.get((c, st)), adj_f.get((c, st)), fail_pct, alpha)
            # stages shorter than 20 ms carry timer noise only: reported, never failed
            if v in ("FAIL", "INCONCLUSIVE") and (s["base_median"] or 0) < 0.02:
                v = "SAME (below 20 ms)"
            rows.append((c, st, s, v, adj_s.get((c, st))))
            if v == "FAIL":
                fails.append(
                    f"{c} {st}: {100 * (s['ratio'] - 1):+.1f}% (95% CI {100 * (s['lo'] - 1):+.1f}% to "
                    f"{100 * (s['hi'] - 1):+.1f}%, n={s['n']}, Holm p={adj_s[(c, st)]:.2g})"
                )
            elif v == "INCONCLUSIVE":
                inconclusive.append(
                    f"{c} {st}: {100 * (s['ratio'] - 1):+.1f}% (95% CI {100 * (s['lo'] - 1):+.1f}% "
                    f"to {100 * (s['hi'] - 1):+.1f}%, n={s['n']})"
                )
    mem_rows = []
    for c in res["cells"]:
        for m in MEMORY:
            b = _median([r.get(m) for r in res["runs"] if r["cell"] == c and r["arm"] == "base" and r.get("ok")])
            h = _median([r.get(m) for r in res["runs"] if r["cell"] == c and r["arm"] == "head" and r.get("ok")])
            if b is None or h is None:
                continue
            grow = (h - b) / b * 100 if b > 0 else 0.0
            bad = grow > mem_pct and h - b > 0.1
            mem_rows.append((c, m, b, h, grow, bad))
            if bad:
                fails.append(f"{c} {m}: {b:.2f} -> {h:.2f} GB ({grow:+.1f}%)")
    md5 = {}
    for c in res["cells"]:
        for arm in ("base", "head"):
            md5.setdefault(c, {})[arm] = sorted(
                {r["md5"] for r in res["runs"] if r["cell"] == c and r["arm"] == arm and r.get("md5")}
            )
    changed = [c for c, d in md5.items() if d.get("base") and d.get("head") and d["base"] != d["head"]]
    unstable = [c for c, d in md5.items() if len(d.get("base", [])) > 1 or len(d.get("head", [])) > 1]
    if errors:
        fails.append(
            f"{len(errors)} run(s) failed, first: {errors[0]['cell']} ({errors[0].get('arm')}): "
            f"{str(errors[0].get('error'))[:300]}"
        )
    verdict = "FAIL" if fails else ("INCONCLUSIVE" if inconclusive else "PASS")
    md = _markdown(res, rows, mem_rows, changed, unstable, fails, inconclusive, verdict, fail_pct, mem_pct, alpha)
    if args.markdown:
        Path(args.markdown).write_text(md)
    print(md)
    if args.history:
        _append_history(Path(args.history), res, stats, verdict)
    if verdict == "FAIL":
        return 1
    return 3 if verdict == "INCONCLUSIVE" and args.inconclusive_is_error else 0


def _pct(x):
    return "" if x is None else f"{100 * (x - 1):+.1f}%"


def _markdown(res, rows, mem_rows, changed, unstable, fails, inconclusive, verdict, fail_pct, mem_pct, alpha):
    m = res["machine"]
    lines = [
        f"## Speed gate: **{verdict}**",
        "",
        f"Head `{(res.get('head_sha') or '?')[:10]}` against base `{(res.get('base_sha') or '?')[:10]}` on "
        f"{m['gpu']} / {m['cpu']}. Interleaved pairs in fresh processes; ratio = head / base (geometric mean of "
        f"the pairs; negative is faster), 95% t interval; fail = at least {fail_pct:g}% slower with Holm-adjusted "
        f"one-sided p < {alpha:g}, or a memory peak {mem_pct:g}% and 0.1 GB higher.",
        "",
    ]
    if fails:
        lines += ["**Failures**", *[f"- {f}" for f in fails], ""]
    if inconclusive:
        lines += [
            "**Inconclusive** (slower by the threshold or more, not significant at the pair limit)",
            *[f"- {f}" for f in inconclusive],
            "",
        ]
    if res.get("skipped"):
        lines += [
            "**Cells not run (data missing on this machine):** "
            + ", ".join(f"{c} ({why})" for c, why in res["skipped"].items()),
            "",
        ]
    lines += [
        "| cell | stage | base median | head median | ratio | 95% CI | n | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c, st, s, v, _ in rows:
        lines.append(
            f"| {c} | {st[:-2]} | {s['base_median']:.3f} s | {s['head_median']:.3f} s | {_pct(s['ratio'])} | "
            f"{_pct(s['lo'])} to {_pct(s['hi'])} | {s['n']} | {v} |"
        )
    lines += ["", "| cell | memory | base GB | head GB | change |", "|---|---|---|---|---|"]
    for c, mm, b, h, g, bad in mem_rows:
        lines.append(f"| {c} | {mm} | {b:.2f} | {h:.2f} | {g:+.1f}%{' FAIL' if bad else ''} |")
    lines += [
        "",
        "Model md5 changed (behaviour change; the lattice judges correctness, the reviewer confirms intent): "
        + (", ".join(changed) if changed else "none"),
    ]
    if unstable:
        lines.append("Model md5 differs between repeats of one build (non-deterministic cell): " + ", ".join(unstable))
    return "\n".join(lines) + "\n"


def _append_history(path, res, stats, verdict):
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "t": time.time(),
        "pr": res.get("pr"),
        "base_sha": res.get("base_sha"),
        "head_sha": res.get("head_sha"),
        "machine": res["machine"],
        "verdict": verdict,
        "cells": {
            c: {
                st: {k: s.get(k) for k in ("base_median", "head_median", "ratio", "lo", "hi", "n")}
                for st, s in d.items()
            }
            for c, d in stats.items()
        },
    }
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("child")
    c.add_argument("--cell", required=True)
    c.add_argument("--rounds", type=int)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True, help="python[:PYTHONPATH] of the base build")
    r.add_argument("--head", required=True, help="python[:PYTHONPATH] of the head build")
    r.add_argument("--base-sha")
    r.add_argument("--head-sha")
    r.add_argument("--pr")
    r.add_argument("--out", required=True)
    r.add_argument("--cells", default="gates/*,v6like/*,h60/*")
    r.add_argument("--min-pairs", type=int, default=5)
    r.add_argument("--max-pairs", type=int, default=12)
    r.add_argument("--fail-pct", type=float, default=3.0)
    r.add_argument("--mem-fail-pct", type=float, default=5.0)
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--timeout", type=int, default=600, help="seconds per run")
    r.add_argument("--budget-s", type=int, default=1500, help="no new pairs after this many seconds")
    k = sub.add_parser("compare")
    k.add_argument("results")
    k.add_argument("--markdown")
    k.add_argument("--history", help=f"append one line per gate run (e.g. {DEFAULT_HISTORY})")
    k.add_argument("--fail-pct", type=float)
    k.add_argument("--mem-fail-pct", type=float)
    k.add_argument("--alpha", type=float)
    k.add_argument("--inconclusive-is-error", action="store_true")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "child":
        return child(a.cell, a.rounds)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "compare":
        return compare(a)
    sys.path.insert(0, str(GATES_DIR))
    from speed_cells import cells

    for cid, cell in cells().items():
        print(f"{cid:32s} {cell.rounds:5d} iterations  {cell.missing() or 'data present'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
