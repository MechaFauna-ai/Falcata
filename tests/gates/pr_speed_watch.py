"""Pull-based trigger of the pre-merge speed gate: finds pull requests that need a speed verdict, builds both sides,
queues the measurement on the GPU queue and reports the verdict on the pull request as the commit status
`falcata/speed` plus one comment that it keeps up to date.

Why pull-based: this repository is public, so a self-hosted runner that GitHub dispatches jobs to would let a fork's
pull request run code on the GPU host (the reason the GPU workflows were retired for a local cron, b9187878). This
script runs on the GPU host from a timer, asks GitHub what is open, and only ever builds pull requests whose head
branch lives in this repository (people with write access); forks are never built. The gate scripts themselves
always run from the base checkout, so a pull request cannot change the rules it is judged by.

    pr_speed_watch.py poll   [--dry-run]     one pass: queue what needs a verdict (timer, e.g. every 20 minutes)
    pr_speed_watch.py report --key K         gpuq --on-finish hook: compare, post status and comment, record history

Rules of one pass:
* open pull requests into master, not drafts (a draft with the label `speed-gate` is included), head in this repo;
* the head commit has no `falcata/speed` status yet (the label `speed-gate-rerun` forces a new run; remove it after);
* a pull request that changes nothing the library is built from (src/, include/, python-package/, cmake/,
  CMakeLists.txt, external_libs/, VERSION.txt) gets `success` "no library change" without a run;
* master itself: each new master head is measured against its first parent, so the history holds one line per
  merged commit (status on the master commit, no comment).
The measurement compares the head with the merge base of the pull request, both built by ci_build.sh into
per-commit caches (ccache makes the second build of a commit cheap), in one strict --exclusive gpuq job (a quiet
card; a contended job gives no verdict and is requeued by the next pass).

Configuration (environment): FALCATA_SPEED_REPO (owner/name, default MechaFauna-ai/Falcata), FALCATA_SPEED_HOME
(default ~/.cache/falcata-gates/speed: checkouts, builds, results, state), GPUQ (the gpuq executable, default
`gpuq`), FALCATA_SPEED_PRIORITY (default 50), plus the data variables of speed_cells.py (FALCATA_BENCH_CACHE,
FALCATA_SPEED_DATA, FALCATA_H60_MANIFEST, FALCATA_H60_MANIFEST_SHA256), which are passed to the job.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = os.environ.get("FALCATA_SPEED_REPO", "MechaFauna-ai/Falcata")
HOME = Path(os.environ.get("FALCATA_SPEED_HOME", Path.home() / ".cache" / "falcata-gates" / "speed"))
GPUQ = os.environ.get("GPUQ", "gpuq")
PRIORITY = os.environ.get("FALCATA_SPEED_PRIORITY", "50")
CONTEXT = "falcata/speed"
MARKER = "<!-- falcata-speed-gate -->"
LIBRARY_PATHS = ("src/", "include/", "python-package/", "cmake/", "CMakeLists.txt", "external_libs", "VERSION.txt")
PASS_ENV = (
    "FALCATA_BENCH_CACHE",
    "FALCATA_SPEED_DATA",
    "FALCATA_H60_MANIFEST",
    "FALCATA_H60_MANIFEST_SHA256",
    "FALCATA_CUDA_HOME",
)
GATES = Path(__file__).resolve().parent


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def gh_json(*args):
    return json.loads(sh(["gh", *args]))


def set_status(sha, state, description, dry=False):
    description = description[:139]
    print(f"status {sha[:10]} {state}: {description}")
    if not dry:
        sh(
            [
                "gh",
                "api",
                "-X",
                "POST",
                f"repos/{REPO}/statuses/{sha}",
                "-f",
                f"state={state}",
                "-f",
                f"context={CONTEXT}",
                "-f",
                f"description={description}",
            ]
        )


def has_status(sha):
    st = gh_json("api", f"repos/{REPO}/commits/{sha}/statuses")
    return any(s.get("context") == CONTEXT for s in st)


def mirror():
    """A bare-ish mirror clone that every build checks out from (worktrees per commit)."""
    m = HOME / "mirror"
    if not m.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--quiet", f"https://github.com/{REPO}.git", str(m)])
    sh(
        [
            "git",
            "-C",
            str(m),
            "fetch",
            "--quiet",
            "origin",
            "+refs/heads/*:refs/remotes/origin/*",
            "+refs/pull/*/head:refs/remotes/pr/*",
        ]
    )
    return m


def checkout(m, sha):
    d = HOME / "src" / sha
    if not d.exists():
        sh(["git", "-C", str(m), "worktree", "add", "--quiet", "--detach", str(d), sha])
        sh(["git", "-C", str(d), "submodule", "update", "--init", "--quiet"])
    return d


def build(m, sha):
    """The commit's wheel in its own venv (ci_build.sh of the commit itself: the build a user gets)."""
    d = checkout(m, sha)
    venv = HOME / "builds" / sha
    py = venv / "bin" / "python"
    ok = venv / ".built"
    if not ok.exists():
        env = {**os.environ, "GATES_VENV": str(venv), "FALCATA_DIST_DIR": str(HOME / "dist" / sha)}
        log = HOME / "logs" / f"build-{sha[:12]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            r = subprocess.run(
                ["bash", "tests/gates/ci_build.sh"], check=False, cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT
            )
        if r.returncode != 0:
            raise RuntimeError(f"build of {sha[:10]} failed (log {log})")
        sh([str(py), "-m", "pip", "install", "--quiet", "nvidia-ml-py"])
        ok.write_text(time.strftime("%FT%TZ", time.gmtime()))
    return py


def library_changed(m, base, head):
    files = sh(["git", "-C", str(m), "diff", "--name-only", f"{base}...{head}"]).split()
    return [f for f in files if f.startswith(LIBRARY_PATHS)]


def queue(m, kind, number, base, head, dry):
    key = f"falcata-speed-{head[:12]}"
    job = HOME / "jobs" / key
    job.mkdir(parents=True, exist_ok=True)
    meta = {"kind": kind, "pr": number, "base": base, "head": head, "queued": time.time()}
    (job / "meta.json").write_text(json.dumps(meta))
    if dry:
        print(f"would queue {key}: {kind} {number} base {base[:10]} head {head[:10]}")
        return
    pyb, pyh = build(m, base), build(m, head)
    gates = checkout(m, sh(["git", "-C", str(m), "rev-parse", "origin/master"]).strip()) / "tests" / "gates"
    run = [
        str(pyb),
        str(gates / "speed_gate.py"),
        "run",
        "--base",
        str(pyb),
        "--head",
        str(pyh),
        "--base-sha",
        base,
        "--head-sha",
        head,
        "--pr",
        str(number),
        "--out",
        str(job / "results.json"),
    ]
    finish = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} report --key {key}"
    env = [f"--env={k}={os.environ[k]}" for k in PASS_ENV if k in os.environ]
    sh(
        [
            GPUQ,
            "submit",
            "--key",
            key,
            "--exclusive",
            "--priority",
            PRIORITY,
            "--timeout",
            "1800s",
            "--needs-vram",
            "20000",
            "--max-attempts",
            "1",
            "--retries",
            "0",
            "--description",
            f"Falcata speed gate: {kind} {number}, head {head[:10]} vs base {base[:10]}",
            "--cwd",
            str(job),
            *env,
            "--on-finish",
            finish,
            "--",
            *run,
        ]
    )
    set_status(head, "pending", f"speed gate queued: head vs {base[:10]}")


def poll(args):
    m = mirror()
    prs = gh_json(
        "pr",
        "list",
        "--repo",
        REPO,
        "--state",
        "open",
        "--base",
        "master",
        "--limit",
        "100",
        "--json",
        "number,headRefOid,isDraft,isCrossRepository,labels,title",
    )
    for pr in prs:
        labels = {x["name"] for x in pr["labels"]}
        head = pr["headRefOid"]
        if pr["isCrossRepository"]:
            continue  # forks are never built on this host
        if pr["isDraft"] and "speed-gate" not in labels:
            continue
        if has_status(head) and "speed-gate-rerun" not in labels:
            continue
        base = sh(["git", "-C", str(m), "merge-base", "origin/master", head]).strip()
        if not library_changed(m, base, head):
            set_status(head, "success", "no library change: nothing to measure", args.dry_run)
            continue
        queue(m, "pr", pr["number"], base, head, args.dry_run)
    master = sh(["git", "-C", str(m), "rev-parse", "origin/master"]).strip()
    if not has_status(master):
        parent = sh(["git", "-C", str(m), "rev-parse", f"{master}^1"]).strip()
        if library_changed(m, parent, master):
            queue(m, "master", "master", parent, master, args.dry_run)
        else:
            set_status(master, "success", "no library change: nothing to measure", args.dry_run)
    return 0


def report(args):
    job = HOME / "jobs" / args.key
    meta = json.loads((job / "meta.json").read_text())
    state = os.environ.get("GPUQ_JOB_STATE", "")
    code = os.environ.get("GPUQ_JOB_EXIT", "")
    res = job / "results.json"
    if state != "done" or code not in ("0",) or not res.exists():
        if code == "3":
            set_status(meta["head"], "error", "card contended: no verdict; the next pass requeues it")
        else:
            set_status(meta["head"], "error", f"gate job {state or '?'} (exit {code or '?'}): see the queue log")
        return 0
    md = job / "summary.md"
    hist = Path(os.environ.get("FALCATA_SPEED_HISTORY", HOME / "history.jsonl"))
    r = subprocess.run(
        [
            sys.executable,
            str(GATES / "speed_gate.py"),
            "compare",
            str(res),
            "--markdown",
            str(md),
            "--history",
            str(hist),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    text = md.read_text() if md.exists() else r.stdout
    first = next((line[2:] for line in text.splitlines() if line.startswith("- ")), "")
    if r.returncode == 0:
        set_status(meta["head"], "success", "no regression beyond the threshold" + (f" ({first})" if first else ""))
    elif r.returncode == 1:
        set_status(meta["head"], "failure", first or "regression beyond the threshold")
    else:
        set_status(meta["head"], "error", "no verdict (contended or inconclusive): " + first)
    if meta["kind"] == "pr":
        body = MARKER + "\n" + text
        comments = gh_json("api", f"repos/{REPO}/issues/{meta['pr']}/comments")
        mine = [c for c in comments if MARKER in (c.get("body") or "")]
        if mine:
            sh(["gh", "api", "-X", "PATCH", f"repos/{REPO}/issues/comments/{mine[-1]['id']}", "-f", f"body={body}"])
        else:
            sh(["gh", "api", "-X", "POST", f"repos/{REPO}/issues/{meta['pr']}/comments", "-f", f"body={body}"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poll")
    p.add_argument("--dry-run", action="store_true", help="print what would be queued; post nothing, build nothing")
    r = sub.add_parser("report")
    r.add_argument("--key", required=True)
    a = ap.parse_args(argv)
    return poll(a) if a.cmd == "poll" else report(a)


if __name__ == "__main__":
    sys.exit(main())
