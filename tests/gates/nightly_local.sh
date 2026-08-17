#!/bin/bash
# Nightly GPU gate suite, run by cron on the developer's own machine.
#
# This replaces the "GPU Gates Nightly" GitHub workflow. A self-hosted runner
# LISTENS for jobs GitHub sends it, which is why GitHub advises against pairing
# one with a public repository: a fork's pull request can try to execute on the
# host. Cron inverts the direction -- this machine pulls the repo and runs the
# gates itself, so there is no inbound execution path at all.
#
# Normally enqueued by gpuq rather than run directly, so it waits for the GPU
# behind whatever else is queued (see the crontab entry for falcata-nightly).
# Running it by hand works too -- it falls back to gpu_guard.py per step.
#
# Logs: ~/.cache/falcata-gates/nightly-YYYY-MM-DD.log (30 kept)
# Exit: 0 all gates passed, non-zero otherwise (the log names the failed step).
set -u -o pipefail

REPO="${FALCATA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOGDIR="$HOME/.cache/falcata-gates"
LOG="$LOGDIR/nightly-$(date +%F).log"
VENV="$REPO/.gates-venv/bin/python"
mkdir -p "$LOGDIR"

# Machine-local data locations. The gate scripts default to repo-relative paths
# so they stay runnable on any checkout; this box keeps the real caches outside
# the repo, so point them at the actual copies here rather than in tracked code.
export FALCATA_BENCH_CACHE="${FALCATA_BENCH_CACHE:-$HOME/Documents/exaboost-bench/data/cache}"
export FALCATA_NUMERAI_V53="${FALCATA_NUMERAI_V53:-$HOME/Documents/numerai/data/1226_int8nan.dataset}"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
# The build is the one hard stop: every later gate tests the artifact it
# produces, so continuing past it would just report noise.
abort() { log "ABORT: $*"; echo "ABORTED: $*" > "$LOGDIR/last-status"; exit 1; }
# Everything else records the failure and carries on. Stopping at the first one
# meant a single known-failing fuzz spec hid the canonical md5 locks, the bench
# tier and the perf gate every single night -- the gates most likely to catch a
# real regression were the ones never reached.
FAILED_STEPS=""
fail() { log "FAIL: $*"; FAILED_STEPS="$FAILED_STEPS
  - $*"; }

exec 2>>"$LOG"
log "=== nightly gates start ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null))"

cd "$REPO" || fail "repo not found at $REPO"

# Only test what is pushed: a dirty tree means a human is mid-edit, and a
# nightly verdict on uncommitted work is not reproducible.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "SKIP: working tree dirty (uncommitted changes) -- nothing to verify"
  echo "SKIPPED: dirty tree" > "$LOGDIR/last-status"
  # Exit 0 (this is not a failure) but poison the token: nothing was verified,
  # so the next submit must not be suppressed as "already covered".
  if [ -n "${GPUQ_JOB_ID:-}" ] && command -v gpuq >/dev/null 2>&1; then
    gpuq token "skipped-dirty-$(date +%s)" || true
  fi
  exit 0
fi
if git fetch --quiet origin; then
  git merge --ff-only --quiet origin/master 2>>"$LOG" || log "WARN: no fast-forward to origin/master; testing local HEAD"
else
  log "WARN: git fetch failed; testing local HEAD"
fi
log "testing $(git rev-parse --short HEAD)"
# The queue recorded a token at dispatch; this is the exact commit we ended up
# on after the fast-forward above, so it supersedes it.
if [ -n "${GPUQ_JOB_ID:-}" ] && command -v gpuq >/dev/null 2>&1; then
  gpuq token "$(git rev-parse HEAD)" || true
fi

step() {  # step <name> <gpu_free_mb|-> <command...>
  local name="$1" need="$2"; shift 2
  # Under gpuq the scheduler already holds the GPU for us; the per-step guard
  # would only be waiting on ourselves. Keep it for direct manual runs.
  if [ "$need" != "-" ] && [ -z "${GPUQ_JOB_ID:-}" ]; then
    if ! python3 tests/gates/gpu_guard.py --require-free-mb "$need" --max-util 10 --timeout-min 120 >>"$LOG" 2>&1; then
      fail "$name (GPU never freed up)"
      return 0
    fi
  fi
  log "-> $name"
  "$@" >>"$LOG" 2>&1 || fail "$name"
}

log "-> build CUDA wheel"
bash tests/gates/ci_build.sh >>"$LOG" 2>&1 || abort "build CUDA wheel"
step "lattice (FALCATA_VERIFY=1)"  12000 env FALCATA_VERIFY=1 "$VENV" tests/gates/lattice.py --check
step "selective equivalence"       -     "$VENV" tests/gates/selective_equivalence.py --seeds 32 --rounds 30
step "fuzz (45 min, corpus first)" -     "$VENV" tests/gates/fuzz.py --minutes 45
step "validation scoring"          -     "$VENV" tests/gates/valid_metric.py
step "FALB python plumbing"        -     "$VENV" tests/gates/falb_python.py
step "FALB binary roundtrip"       -     env FALCATA_LIB="$REPO/lib_falcata.so" "$VENV" tests/gates/falb_roundtrip.py
step "FALB frozen-fixture compat"  -     "$VENV" tests/gates/falb_compat.py
step "FALB parser fuzz"            -     "$VENV" tests/gates/falb_fuzz.py --seconds 120
# The Python suite on CUDA. GH CI runs the same tests on CPU; hosted runners
# have no GPU, so this is the only place the CUDA half executes. --forked keeps
# failures independent: a CUDA context error is sticky, and in one process a
# single bug turns into a cascade that hides everything after it.
step "python suite (CUDA)"         -     env TASK=cuda "$VENV" -m pytest tests/python_package_test tests/c_api_test \
  -q --forked --timeout=900 -p no:cacheprovider
# The C++ (GoogleTest) suite and the distributed CLI tests, against a CUDA
# build. GH CI runs both on a CPU build; this is the shipped configuration, and
# the only place the test binary's CUDA device link is exercised at all.
step "cpp + distributed (CUDA)"    -     env FALCATA_GATES_PYTHON="$VENV" bash tests/gates/cpp_suite.sh
step "import parity: xgboost"      -     "$VENV" tests/gates/import_xgboost.py
step "import parity: catboost"     -     "$VENV" tests/gates/import_catboost.py
step "fixedpoint tree-emission"    20000 "$VENV" tests/gates/canonical.py numerai-treecount
step "canonical md5 locks"         -     "$VENV" tests/gates/canonical.py all
# Runs against every binary dataset this machine has, because the defect it
# guards (a sampled column served from the wrong bin encoding) only appears on
# data carrying sparse-encoded columns -- no synthetic shape reproduces it.
step "sparse column view"          20000 "$VENV" tests/gates/sparse_column_view.py
step "bench tier"                  12000 "$VENV" tests/gates/bench_tier.py --out tests/gates/bench_results.json
step "perf gate"                   -     "$VENV" tests/gates/perf_gate.py \
  --results tests/gates/bench_results.json \
  --baseline-file "$LOGDIR/bench_baseline.json" --warn-pct 4 --fail-pct 8 --record

if [ -n "$FAILED_STEPS" ]; then
  log "=== GATES FAILED:$FAILED_STEPS"
  { echo "FAILED $(git rev-parse --short HEAD)"; echo "$FAILED_STEPS"; } > "$LOGDIR/last-status"
  find "$LOGDIR" -name 'nightly-*.log' -mtime +30 -delete 2>/dev/null
  exit 1
fi
log "=== ALL GATES PASSED"
echo "PASSED $(git rev-parse --short HEAD)" > "$LOGDIR/last-status"
find "$LOGDIR" -name 'nightly-*.log' -mtime +30 -delete 2>/dev/null
exit 0
