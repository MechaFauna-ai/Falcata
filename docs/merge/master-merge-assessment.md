# hybrid-level-growth -> master: merge assessment

**Audit scope:** prepare the eventual `hybrid-level-growth` -> `master` merge for
Felix's review. Read-only audit; no code changed, nothing built, no PR opened, no
merge performed. A scratch worktree used for the trial merge was removed afterward.

- Branch tip: `bf922f90`
- master (`origin/master`): `896bd508`
- Merge-base: `ce3adf39` ("Merge master (dtype-test fix, CI) before landing PR #34")

## TL;DR / GO-NO-GO

**GO, with conditions.** The delta is small and self-contained (13 commits, 12 files,
+782/-143), the trial merge into current master is **conflict-free**, no
default-changing new gates are introduced, and the risky new path (quant graph loop)
is **opt-in / OFF by default**. Conditions before landing are listed at the bottom.

## Important framing correction

The task brief describes a very large body of work (4-bit packing, fp32 modes,
int8/uint8/uint16/fp16/pandas/CuPy ingestion, GPU-native construction + EFB precheck,
FIL predict, the full graph L1/L1.5/L2 arc, multiclass fix, quant graph opt-in). That
body **already landed on master** via **PR #34** (`86493a9c`, "Merge pull request #34
from BelixRogner/hybrid-level-growth"). The merge-base `ce3adf39` is precisely the
pre-#34 point. Since then master also absorbed PR #36 (benchmark suite), PR #31
(Windows MSVC CUDA build fixes), and PR #14 (lambdarank deterministic).

This assessment therefore covers **only the 13 commits the branch accumulated in
parallel after `ce3adf39`** — i.e., what a `hybrid-level-growth -> master` merge would
*newly* introduce today. Those 13 commits are a focused increment on top of the
already-landed #34 work: quant-mode graph support, graph A2 frozen grids, a multiclass
graph crash fix, and a quant packed-histogram overflow fix, plus roadmap docs.

## 1. Commit groups (13 commits, origin/master..hybrid-level-growth)

| Group | Commits | Files | What it does |
|---|---|---|---|
| **Quant graph support** | `a2279763`, `5c61a0ed` | best_split_finder.{cu,hpp}, data_partition.cu, histogram_constructor.{cpp,cu,hpp}, hybrid_graph.{cu,hpp}, tree_learner.cpp | Extends the device-driven level loop (graphs L1.5/A2) to quantized training: device-derived per-leaf hist bit-widths, guarded quant body kernels, capture wiring. Graph prefix reaches quant training but is **opt-in** (`EXABOOST_GRAPH_QUANT=1`) pending controller-latency work. |
| **Graph A2 (frozen grids)** | `199dc6fa` | cuda_tree.{hpp,cu}, best_split_finder.{cu,hpp}, data_partition.cu, histogram_constructor.{cu,hpp}, hybrid_graph.{cu,hpp}, tree_learner.cpp | Freezes body-kernel grids at pow2 buckets so the captured graph's launch dims are stable; idle blocks self-guard on device-read live counts. Largest single-commit surface. |
| **Multiclass graph fix** | `84db39cd` | tree_learner.cpp (15 lines) | Upload the graph exec before the first launch; re-enables multiclass on the graph path (root-causes and lifts the earlier `6bc8779e` gate that landed in #34). |
| **Quant overflow fix** | `3afe7c62` | histogram_constructor.{cpp,hpp} | Fixes a packed shared-histogram overflow that collapsed quantized training at `num_grad_quant_bins >= 16`. Correctness fix. |
| **Roadmap docs** | `bf922f90`, `df2ee7f3`, `724aff1b`, `88d68d53`, `71242126`, `779ff3c5`, `20379d9d`, `c8b5aaf8` | ROADMAP.md only | Investigation notes; no code. |

Delta: **12 files, +782 / -143** (`git diff --stat origin/master...hybrid-level-growth`).
Code churn concentrates in `cuda_histogram_constructor.cu` (+311) and
`cuda_hybrid_graph.hpp` (+105).

## 2. Conflict assessment vs current master

Method: `git worktree add --detach <scratch> hybrid-level-growth`, then
`git merge --no-commit --no-ff origin/master` inside it. Worktree removed afterward.

**Result: clean merge, zero conflicts.**

- `git status` after the trial merge shows **no unmerged (`U`) paths**.
- Exactly one file needed textual auto-merge: `cuda_single_gpu_tree_learner.cpp`
  (a 9-line hunk). The auto-merge is benign: it is the PR #31 MSVC fix that replaces
  the `1UL` literal with `static_cast<size_t>(1)` in `AllocateBitset()` (LLP64
  `unsigned long` is 32-bit; `size_t` is 64-bit). No logic change, no interaction with
  branch code.
- All other staged changes are pure master-side additions being pulled in
  (lambdarank determinism in `cuda_rank_objective.*`, benchmark suite, `meta.h`,
  `CMakeLists.txt`, `test_dual.py`), which the branch does not touch.

**Conflict list: none. Resolution guidance: none required** — the merge is
mechanically clean. (The two-dot `git diff --stat origin/master hybrid-level-growth`
prints +798/-747 and shows `test_dual.py -66`; that is master-only work the branch
lacks, not a branch change — do not read it as a delete.)

## 3. Risk audit (read-only)

### 3a. Default-behavior changes vs upstream/master

The only runtime env gate *introduced by these 13 commits* is `EXABOOST_GRAPH_QUANT`,
and it defaults **OFF**. The `#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED` token that also
appears is a compile guard, not a runtime behavior switch.

All the "on by default" gates (`EXABOOST_HYBRID_GROWTH`, `_GRAPH_LEVEL_LOOP`,
`_HYBRID_BATCH_*`, `_ONE_SYNC`, `_SELECTIVE`) already shipped in master via PR #34 —
this merge does **not** change their defaults. For completeness the full env-default
table is below; the "New here?" column marks what this merge actually changes.

**Severity ranking of findings:**

- **[LOW] Quant graph reaches training by default when `GRAPH_LEVEL_LOOP` is on, but
  quant path is guarded OFF.** `a2279763` wires quant training into
  `TrainLevelWisePrefixGraph`, but `HybridGraphPrefixUsable()` returns `false` for
  `config_->use_quantized_grad` unless `EXABOOST_GRAPH_QUANT=1`. So default behavior
  for quant users is unchanged (classic two-sync loop). The commit body asserts "every
  host-launched path is bit-for-bit unchanged" and "`EXABOOST_GRAPH_LEVEL_LOOP=0`
  restores today's behavior exactly," backed by an extensive md5 A/B lock matrix.
  Residual risk is only for users who explicitly opt in.

- **[LOW] `84db39cd` re-enables multiclass on the graph path** (lifting the `6bc8779e`
  gate from #34). This *does* change default behavior for multiclass + graph-loop
  users. Mitigation: root cause documented (exec must be uploaded before first
  launch); roadmap notes a multiclass A/B + 15-run soak. Still, multiclass on the
  non-quant graph path is now default-ON — flag for reviewer attention as the single
  default-behavior change with the widest reach.

- **[LOW] `3afe7c62` is a pure correctness fix** (packed shared-hist overflow at
  `num_grad_quant_bins >= 16`). Strictly improves behavior; only reaches quant users.

- **[INFO] `199dc6fa` (frozen pow2 grids)** is the largest surface and touches the
  hot construct/find/subtract kernels. It is behavior-preserving by design (idle
  blocks self-guard on device-read live counts) and md5-locked, but its size warrants
  the closest reviewer read of the set.

### 3b. TODO/FIXME/debug/instrumentation left in

**None in the branch-only delta.** Grep over `git diff origin/master...hybrid-level-growth`
added lines for `TODO|FIXME|XXX|HACK|DEBUG|printf|std::cout|fprintf` returned zero
hits. (The many `EXABOOST_HYBRID_DEBUG`/`_DIAG` diagnostic gates already exist on
master and are untouched here.)

### 3c. Files with unusually large diffs (reviewer attention)

1. `src/treelearner/cuda/cuda_histogram_constructor.cu` (+311) — quant body kernels +
   frozen-grid sizing. Highest-priority read.
2. `src/treelearner/cuda/cuda_hybrid_graph.hpp` (+105) — capture/state wiring.
3. `src/treelearner/cuda/cuda_best_split_finder.cu` (+93) — quant find kernel + A2 guards.
4. `src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp` (+89 three-dot) — gate
   resolution, `HybridGraphPrefixUsable()`, quant graph entry.

### 3d. md5-lock story (which behaviors are bit-locked)

- **Bit-locked (md5 A/B graph ON vs OFF identical):** the entire graph level loop,
  including this increment's quant extension. Commit `a2279763` records a lock matrix:
  covtype 1023/10 quant GROWTH x GRAPH, numerai int8 quant, fraud/year 63/6 quant,
  covtype 63/6 quant bagging, multiclass covtype 63/6 quant A/B + 15-run soak,
  `num_grad_quant_bins=16` higgs 4M (AUC .83013, md5 A/B identical), compute-sanitizer
  racecheck clean on fraud quant graph ON. Quant integer histograms keep md5 locks
  valid because accumulation is order-independent by construction.
- **Intentionally nondeterministic (not md5-locked):** non-quant CUDA float-atomic
  histogram accumulation and fp32 gain/hist modes (quality-gated, not bit-identical —
  ROADMAP explicitly deprioritizes a deterministic non-quant CUDA mode). These modes
  are OFF by default and unchanged by this merge.

### 3e. Test coverage added vs feature surface

**No automated tests are added by these 13 commits** (`git diff --stat
origin/master...hybrid-level-growth -- tests` is empty). Correctness rests entirely on
the **manual md5 A/B lock methodology** run out-of-tree (documented in commit bodies
and ROADMAP), not on CI-enforced assertions. This is the principal audit gap:
the quant-graph and multiclass-graph paths that this merge newly activates have strong
manual verification but **no regression guard in the test suite**. Recommend Felix
weigh whether an md5 lock harness should be CI-wired before or shortly after landing.

## 4. Env-default safety table

`Default` = behavior when the variable is unset. `New here?` = introduced/changed by
these 13 commits (vs already present on master from PR #34).

| Env var | Default | Safe? | New here? | Notes |
|---|---|---|---|---|
| `EXABOOST_GRAPH_QUANT` | **OFF** | Yes | **NEW** | Opt-in quant graph loop; net loss on cheap/large-level quant shapes, so intentionally off pending controller-latency work. |
| `EXABOOST_GRAPH_LEVEL_LOOP` | ON | Yes | changed reach | On master already. Heavily guarded: driver >= 12040, depth-limited exact regime, split bounds. Quant sub-path gated by `GRAPH_QUANT`. Multiclass sub-path newly ungated (`84db39cd`). `=0` restores classic loop bit-exactly. |
| `EXABOOST_HYBRID_GROWTH` | ON | Yes | no (in #34) | Master default; `=0` falls back to classic leaf-wise. |
| `EXABOOST_HYBRID_BATCH_KERNELS` | ON | Yes | no | Master default. |
| `EXABOOST_HYBRID_BATCH_APPLY` | ON | Yes | no | Master default. |
| `EXABOOST_HYBRID_ONE_SYNC` | ON | Yes | no | Master default. |
| `EXABOOST_HYBRID_SELECTIVE` | ON | Yes | no | Master default. |
| `EXABOOST_FP32_GAIN` | OFF | Yes | no | Quality-gated, opt-in. |
| `EXABOOST_FP32_HIST` | OFF | Yes | no | Quality-gated, opt-in. |
| `EXABOOST_ROWDATA_4BIT` | OFF | Yes | no | Opt-in; VERIFY variant. |
| `EXABOOST_FAST_ROWDATA` | OFF | Yes | no | Opt-in; VERIFY variant. |
| `EXABOOST_GPU_CONSTRUCT` | OFF | Yes | no | Opt-in; VERIFY variant. |
| `EXABOOST_EFB_PRECHECK` | OFF | Yes | no | Opt-in; VERIFY variant. |
| `EXABOOST_HYBRID_DEBUG` / `_DIAG` | OFF | Yes | no | Diagnostics; steer host path only. |

The kill-switch discipline is intact: everything this merge newly introduces defaults
to preserving prior behavior, and every optimization has an explicit off switch.

## 5. CI status

CI runs on `BelixRogner/Falcata` for the branch (latest, 2026-07-15):

- **Green:** C++, Build, R-package, SWIG, Static Analysis, Optional checks.
- **Red:** Python-package. Failing legs are **windows-2022 (MSVC)** and
  **macos-15-intel** on the "Setup and run tests" / "Upload artifacts" steps; **all
  Linux legs (manylinux_2_28 latest/oldest, arm bdist) pass.** Raw failed-step logs
  were not retrievable at audit time (retention/permission), so the exact error text
  could not be captured. The Linux-green / Windows+macOS-red shape and step location
  match the known Windows conda infra flake (pyexpat) noted in prior runs, not a
  compile or logic failure in the CUDA code (C++/Build/Static-Analysis are all green).

**Reviewer action:** confirm the Python-package red is the known infra flake before
landing (re-run the leg, or eyeball one Windows log for pyexpat/conda) rather than
assuming it. This is listed as a GO condition below.

## GO / NO-GO recommendation

**GO, conditional.** Rationale:
- Clean, conflict-free merge into current master (PRs #14/#31/#36 already absorbed).
- Small, coherent 13-commit increment; no leftover instrumentation.
- No new default-behavior gate changes except the deliberately-guarded quant graph
  (opt-in, OFF) and multiclass re-enable (root-caused, md5+soak verified).
- Strong manual md5/racecheck verification on the newly-activated paths.

**Conditions before Felix lands it:**
1. Confirm the Python-package CI red is the known Windows/macOS conda infra flake, not
   a real failure (re-run or inspect one log).
2. Accept the test-coverage gap: the quant-graph / multiclass-graph paths have manual
   md5 locks but no CI regression guard. Optionally file a follow-up to wire an md5
   lock harness into CI.
3. Give `199dc6fa` (frozen grids, +311 in the construct kernel) a focused correctness
   read as the largest-surface change.
4. Sanity-check that `EXABOOST_GRAPH_QUANT` remains OFF by default in the merged tree
   (it is on the branch) so quant users' default behavior is unchanged.
