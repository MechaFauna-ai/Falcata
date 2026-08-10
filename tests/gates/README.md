# Falcata regression gates

Per-commit and nightly GPU regression testing for the three failure classes
this project keeps hitting: silently invalid trees, quality regressions, and
perf regressions -- each surfacing only for specific (config x dataset-shape)
combinations.

These gates need a CUDA GPU, so they do NOT run in GitHub Actions: this is a
public repository, and a self-hosted runner would let a fork's pull request
execute code on the GPU host. They run locally instead -- the machine pulls the
repo and tests it, so nothing inbound can trigger execution.

## Per-commit (~10 min, run locally)

    bash tests/gates/ci_build.sh                       # CUDA wheel into .gates-venv
    .gates-venv/bin/python tests/gates/lattice.py --check

| Detector | Mechanism |
|---|---|
| invalid trees / silent behavior change | **md5 lattice**: ~40 fingerprinted quant cells (bit-deterministic) across 9 path-engineered dataset profiles; baselines in `md5_lattice.json` |
| plan decision changes the model | **equality cells**: every `cuda_plan` key flipped vs its base cell must be bit-identical |
| broken models | validity asserts in every cell: round-trip, finite preds, tree count |
| quality regression | metric recorded per cell; nondeterministic (non-quant / fp32) cells gate on metric floor with 2% tolerance |
| perf regression | `perf_gate.py`: perf-tracked cell times vs machine-local rolling median (warn +10%, fail +25%), recorded only on green runs |

**Re-baselining:** an intended behavior change (e.g. a tie-break fix) fails the
lattice by design. Re-baseline explicitly -- `lattice.py --update <cell-id> ...`
(or `--baseline` for everything) -- and commit the `md5_lattice.json` diff; the
diff review shows exactly which cells moved and is itself the review artifact.

## Nightly (`nightly_local.sh`, cron)

    crontab -l 2>/dev/null | grep -v nightly_local.sh > /tmp/ct
    echo "0 3 * * * $HOME/Documents/lightgbm-fork/tests/gates/nightly_local.sh" >> /tmp/ct
    crontab /tmp/ct

Waits for a free GPU, fast-forwards to `origin/master` (skips entirely if the
tree is dirty), runs every tier below, and writes
`~/.cache/falcata-gates/nightly-<date>.log` plus a one-line `last-status`.

- `FALCATA_VERIFY=1` lattice sweep: ingestion fast paths self-verify
  byte-for-byte against their reference implementations.
- `fuzz.py`: seeded random (shape, dtype, NaN/sparsity) x (tree shape, quant
  mode, plan flips, objective) specs; checks validity, quant-mode determinism
  (train twice -> identical md5), and tolerant CPU-vs-CUDA metric parity.
  Failures print a `--spec` repro; promote real bugs into `fuzz_corpus.json`
  so they rerun first, forever.

## Shared GPU

The runner box's GPU is shared with training jobs. Every GPU step starts with
`gpu_guard.py`, which waits for a sustained idle window (default: >=8 GB free,
<=20% util for 15 s) and exits distinctly ("GPU busy", exit 3) on timeout --
rerun the workflow later; nothing runs on a busy GPU.

## Local usage

```bash
bash tests/gates/ci_build.sh          # venv + CUDA wheel (ccache if present)
.gates-venv/bin/python tests/gates/lattice.py --list     # show the matrix
python3 tests/gates/gpu_guard.py && \
  .gates-venv/bin/python tests/gates/lattice.py --check  # run the gates
```
