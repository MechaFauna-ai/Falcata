#!/bin/bash
# Nightly GPU gate suite, run by cron on the developer's own machine.
#
# This replaces the "GPU Gates Nightly" GitHub workflow. A self-hosted runner
# LISTENS for jobs GitHub sends it, which is why GitHub advises against pairing
# one with a public repository: a fork's pull request can try to execute on the
# host. Cron inverts the direction -- this machine pulls the repo and runs the
# gates itself, so there is no inbound execution path at all.
#
# Install (runs 03:00 daily):
#   crontab -l 2>/dev/null | grep -v nightly_local.sh > /tmp/ct
#   echo "0 3 * * * $HOME/Documents/lightgbm-fork/tests/gates/nightly_local.sh" >> /tmp/ct
#   crontab /tmp/ct
#
# Logs: ~/.cache/falcata-gates/nightly-YYYY-MM-DD.log (30 kept)
# Exit: 0 all gates passed, non-zero otherwise (the log names the failed step).
set -u -o pipefail

REPO="${FALCATA_REPO:-$HOME/Documents/lightgbm-fork}"
LOGDIR="$HOME/.cache/falcata-gates"
LOG="$LOGDIR/nightly-$(date +%F).log"
VENV="$REPO/.gates-venv/bin/python"
mkdir -p "$LOGDIR"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
fail() { log "FAIL: $*"; echo "FAILED: $*" >> "$LOGDIR/last-status"; exit 1; }

exec 2>>"$LOG"
log "=== nightly gates start ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null))"

cd "$REPO" || fail "repo not found at $REPO"

# Only test what is pushed: a dirty tree means a human is mid-edit, and a
# nightly verdict on uncommitted work is not reproducible.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "SKIP: working tree dirty (uncommitted changes) -- nothing to verify"
  echo "SKIPPED: dirty tree" > "$LOGDIR/last-status"
  exit 0
fi
if git fetch --quiet origin; then
  git merge --ff-only --quiet origin/master 2>>"$LOG" || log "WARN: no fast-forward to origin/master; testing local HEAD"
else
  log "WARN: git fetch failed; testing local HEAD"
fi
log "testing $(git rev-parse --short HEAD)"

step() {  # step <name> <gpu_free_mb|-> <command...>
  local name="$1" need="$2"; shift 2
  if [ "$need" != "-" ]; then
    python3 tests/gates/gpu_guard.py --require-free-mb "$need" --max-util 10 --timeout-min 120 >>"$LOG" 2>&1 \
      || fail "$name (GPU never freed up)"
  fi
  log "-> $name"
  "$@" >>"$LOG" 2>&1 || fail "$name"
}

step "build CUDA wheel"            -     bash tests/gates/ci_build.sh
step "lattice (FALCATA_VERIFY=1)"  12000 env FALCATA_VERIFY=1 "$VENV" tests/gates/lattice.py --check
step "selective equivalence"       -     "$VENV" tests/gates/selective_equivalence.py --seeds 32 --rounds 30
step "fuzz (45 min, corpus first)" -     "$VENV" tests/gates/fuzz.py --minutes 45
step "fixedpoint tree-emission"    20000 "$VENV" tests/gates/canonical.py numerai-treecount
step "canonical md5 locks"         -     "$VENV" tests/gates/canonical.py all
step "bench tier"                  12000 "$VENV" tests/gates/bench_tier.py --out tests/gates/bench_results.json
step "perf gate"                   -     "$VENV" tests/gates/perf_gate.py \
  --results tests/gates/bench_results.json \
  --baseline-file "$LOGDIR/bench_baseline.json" --warn-pct 4 --fail-pct 8 --record

log "=== ALL GATES PASSED"
echo "PASSED $(git rev-parse --short HEAD)" > "$LOGDIR/last-status"
find "$LOGDIR" -name 'nightly-*.log' -mtime +30 -delete 2>/dev/null
exit 0
