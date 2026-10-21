#!/usr/bin/env bash
# Everything that must be true BEFORE building release artifacts, checked
# mechanically. Exits non-zero on the first unmet condition and names it.
#
#   tools/release-preflight.sh             # check and report
#   tools/release-preflight.sh --wait      # additionally wait for in-flight CI
#   tools/release-preflight.sh --released  # re-check after the PyPI upload
#
# Release order (the script runs twice):
#   1. tools/release-preflight.sh             version must NOT be on PyPI yet
#   2. wheel (tools/build-release-wheel.sh) + sdist (build-python.sh sdist)
#   3. upload to TestPyPI and validate there
#   4. upload the identical files to PyPI
#   5. tools/release-preflight.sh --released  version MUST be on PyPI now
#   6. tag v$V and push the tag
#   7. GitHub release
# --released inverts only the PyPI version check; every other check is the
# same, so the tag goes on a commit that is still clean, synced and green.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
WAIT=0
RELEASED=0
for arg in "$@"; do
  case "$arg" in
    --wait) WAIT=1 ;;
    --released) RELEASED=1 ;;
    *) echo "usage: $0 [--wait] [--released]" >&2; exit 2 ;;
  esac
done
FAIL=0
say() { printf '  %-42s %s\n' "$1" "$2"; }
bad() { say "$1" "FAIL: $2"; FAIL=1; }

echo "release preflight on $(git rev-parse --short HEAD)"

# 1. clean tree, on master, in sync with origin
if [ -z "$(git status --porcelain)" ]; then
  say "working tree clean" ok
else
  bad "working tree clean" "uncommitted or untracked files present"
fi
if [ "$(git branch --show-current)" = "master" ]; then
  say "on master" ok
else
  bad "on master" "on branch $(git branch --show-current)"
fi
git fetch --quiet origin
if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)" ]; then
  say "synced with origin" ok
else
  bad "synced with origin" "HEAD != origin/master"
fi

# 2. version consistency, and the target version must be new on prod PyPI
#    (or, with --released, must already be there)
V=$(tr -d '[:space:]' < VERSION.txt)
PV=$(grep -m1 '^version = ' python-package/pyproject.toml | cut -d'"' -f2)
if [ "$V" = "$PV" ]; then
  say "VERSION.txt == pyproject ($V)" ok
else
  bad "VERSION.txt == pyproject" "$V vs $PV"
fi
if curl -sf "https://pypi.org/pypi/falcata/$V/json" > /dev/null 2>&1; then
  ON_PYPI=1
else
  ON_PYPI=0
fi
if [ "$RELEASED" = 1 ]; then
  if [ "$ON_PYPI" = 1 ]; then
    say "version $V on PyPI" ok
  else
    bad "version $V on PyPI" "not found on prod (upload before tagging)"
  fi
elif [ "$ON_PYPI" = 1 ]; then
  bad "version $V unreleased on PyPI" "already exists on prod (after upload use --released)"
else
  say "version $V unreleased on PyPI" ok
fi

# 3. the pinned lint over the full tree (identical to CI's lint job)
if pre-commit run --all-files > /tmp/preflight-lint.log 2>&1; then
  say "pre-commit --all-files" ok
else
  bad "pre-commit --all-files" "see /tmp/preflight-lint.log"
fi

# 4. hosted CI green on this exact commit
ci() {
  gh run list --branch master --limit 8 \
    --json headSha,status,conclusion,name \
    --jq "[.[] | select(.headSha==\"$(git rev-parse HEAD)\")]"
}
S=$(ci)
if [ "$WAIT" = 1 ]; then
  while [ "$(echo "$S" | jq '[.[] | select(.status!="completed")] | length')" != "0" ] \
      || [ "$(echo "$S" | jq length)" -lt 4 ]; do
    sleep 30
    S=$(ci)
  done
fi
GREEN=$(echo "$S" | jq '[.[] | select(.conclusion=="success")] | length')
TOTAL=$(echo "$S" | jq length)
if [ "$TOTAL" -ge 4 ] && [ "$GREEN" = "$TOTAL" ]; then
  say "hosted CI on HEAD ($GREEN/$TOTAL)" ok
else
  bad "hosted CI on HEAD" "$GREEN/$TOTAL green (use --wait for in-flight runs)"
fi

# 5. the nightly gate suite has passed this exact commit
NIGHTLY=$(cat "$HOME/.cache/falcata-gates/last-status" 2>/dev/null || echo none)
if [ "$NIGHTLY" = "PASSED $(git rev-parse --short HEAD)" ]; then
  say "nightly gates on HEAD" ok
else
  bad "nightly gates on HEAD" "last-status: '$NIGHTLY' (enqueue via gpuq, key falcata-nightly)"
fi

if [ "$FAIL" = 0 ] && [ "$RELEASED" = 1 ]; then
  echo "PREFLIGHT GREEN (released): tag v$V on this commit, push the tag,"
  echo "then publish the GitHub release."
elif [ "$FAIL" = 0 ]; then
  echo "PREFLIGHT GREEN: build with tools/build-release-wheel.sh + build-python.sh sdist,"
  echo "validate on TestPyPI per the release policy, then upload the identical files to prod."
else
  echo "PREFLIGHT FAILED: fix the items above before building release artifacts."
fi
exit "$FAIL"
