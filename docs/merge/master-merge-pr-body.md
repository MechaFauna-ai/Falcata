# hybrid-level-growth -> master

> Draft PR description, prepared for review. Not yet opened. Merges cleanly into
> current master (conflict-free trial merge; PRs #14/#31/#36 already absorbed).

## What & why

Falcata's CUDA single-GPU tree learner grows trees one split at a time, so per-tree
time is dominated by kernel-launch and host-sync latency rather than GPU compute. This
line of work replaces that with **hybrid level-batched growth** — a depth-wise prefix
that batches every leaf on a level into one launch per kernel family, then falls back
to leaf-wise for the tail — and drives the whole level loop **on-device via CUDA
graphs**, collapsing per-level host round-trips. Every fast path has a kill switch that
restores the classic behavior bit-for-bit.

This PR brings `hybrid-level-growth` up to master. The bulk of the feature surface
(hybrid growth, 4-bit packing, fp32 gain/hist modes, int8/uint8/uint16/fp16/pandas/
CuPy ingestion, GPU-native construction + EFB precheck, FIL predict, the graph
L1/L1.5/L2 arc) landed earlier via PR #34; **this increment adds quant-mode graph
support (opt-in), graph A2 frozen grids, a multiclass graph re-enable, and a quant
packed-histogram overflow fix.** See "This increment" below for the exact delta.

## Feature summary (cumulative branch capability)

- **Hybrid depth-wise-prefix + leaf-wise-tail growth** — level-batched construct /
  fix / subtract / find / sync, then leaf-wise for the tail. Exactly leaf-wise-
  equivalent in the budget-limited (grow-then-prune) regime.
- **Graph level loop (L1/L1.5/A2)** — the device runs the level loop end to end; the
  host reads back once per graph. A2 freezes body-kernel grids at pow2 buckets and
  self-guards idle blocks on device-read live counts.
- **Quant graph support** *(this increment)* — quantized training reaches the device
  level loop with device-derived per-leaf histogram bit-widths and guarded quant body
  kernels. Bit-exact vs the host loop; opt-in via `EXABOOST_GRAPH_QUANT=1` pending
  controller-latency tuning.
- **4-bit bin packing, fp32 gain / fp32-atomic histogram modes.**
- **Ingestion** — int8 / uint8 / uint16 / fp16 / pandas small-int / CuPy paths;
  GPU-native dataset construction with an EFB density precheck.
- **FIL predict integration.**

### Headline measured numbers

Measured on the branch's benchmark suite (medians; see ROADMAP.md for full matrices
and exact shapes):

- **numerai train** ~86 -> ~59s (2000-tree, cumulative hybrid + packing wins;
  ROADMAP records 87.6 -> 66.3s from 4-bit packing alone, further with fp32-hist).
- **Dataset construction** numerai f32 ~37.5 -> ~7.2-7.8s (GPU-native construct),
  int8 15.2 -> 4.5-4.9s, CuPy int8 3.4-3.8s.
- **Graph level loop** same-session A/B vs host loop: **fraud 63/6 -11.9%**, fraud
  1023/10 -10.9%, covtype 63/6 -8.8%, year -5.8%, numerai parity (construct-bound).
  Launch API 132 -> ~30 us/tree.
- **Deep regime** (fp32-hist + related flags): epsilon deep -36%, year -18%,
  covtype -16%, fraud deep -14%, higgs deep -12%. Competitive multiples vs XGBoost on
  the deep scoreboard (~4x on the deep configs; see benchmark suite).
- **Quant overflow fix** *(this increment)* — restores quantized training at
  `num_grad_quant_bins >= 16` (previously collapsed); higgs 4M AUC .83013, md5 A/B
  identical.

## Kill switches (default-preserving)

Every optimization is gated; unset = prior behavior. The only gate **new in this
increment** is `EXABOOST_GRAPH_QUANT` (OFF by default).

| Env var | Default | Effect when unset |
|---|---|---|
| `EXABOOST_HYBRID_GROWTH` | ON | `=0` -> classic one-split-at-a-time leaf-wise loop |
| `EXABOOST_GRAPH_LEVEL_LOOP` | ON | `=0` -> classic two-sync host loop (bit-exact) |
| `EXABOOST_GRAPH_QUANT` | **OFF** | quant training stays on the host loop until opted in |
| `EXABOOST_HYBRID_BATCH_KERNELS` / `_BATCH_APPLY` / `_ONE_SYNC` / `_SELECTIVE` | ON | `=0` -> per-pair / per-split / two-sync fallbacks |
| `EXABOOST_FP32_GAIN` / `EXABOOST_FP32_HIST` | OFF | double-precision path (quality-gated, opt-in) |
| `EXABOOST_ROWDATA_4BIT` / `_FAST_ROWDATA` / `_GPU_CONSTRUCT` / `_EFB_PRECHECK` | OFF | classic ingestion / construction; each has a `_VERIFY` twin |

The graph loop is additionally guarded at runtime: CUDA driver >= 12040, depth-limited
exact regime only, and per-level split bounds. Any fallback (env off, unsupported
driver, capture failure) reverts to the classic loop.

## Correctness: the md5-gate methodology

The graph path (including this increment's quant extension) is validated by an
**md5 A/B lock**: run graph ON vs OFF on a fixed config and diff the md5 of the model
output. The two must be **bit-identical**. Because quant-mode histograms accumulate as
integers, the md5 locks hold under any reduction order.

Verified lock matrix for this increment (`a2279763`): covtype 1023/10 quant
GROWTH x GRAPH, numerai int8 quant, fraud/year 63/6 quant, covtype 63/6 quant bagging
0.8/1, multiclass covtype 63/6 quant A/B + 15-run soak, `num_grad_quant_bins=16`
higgs 4M (md5 A/B identical), plus compute-sanitizer racecheck clean on fraud quant
graph ON. Non-quant float-atomic and fp32 modes are intentionally not bit-locked
(quality-gated) and remain OFF by default.

## This increment (delta vs current master)

13 commits, 12 files, **+782 / -143** (`git diff --stat origin/master...hybrid-level-growth`).

- **Quant graph support** (`a2279763`, `5c61a0ed`) — device hist-bits, guarded quant
  body kernels; graph prefix reaches quant training, opt-in via `EXABOOST_GRAPH_QUANT`.
- **Graph A2 frozen grids** (`199dc6fa`) — pow2 body-kernel grid buckets; idle-block
  guards on device-read live counts. Largest surface (+311 in the construct kernel).
- **Multiclass graph re-enable** (`84db39cd`) — upload exec before first launch; lifts
  the earlier multiclass graph gate.
- **Quant overflow fix** (`3afe7c62`) — packed shared-histogram overflow at
  `num_grad_quant_bins >= 16`.
- **ROADMAP docs** (8 commits) — investigation notes, no code.

## Merge & CI

- **Trial merge into current master: conflict-free.** One benign auto-merge in
  `cuda_single_gpu_tree_learner.cpp` (PR #31's MSVC `1UL` -> `size_t` fix).
- **CI:** C++, Build, R-package, SWIG, Static Analysis green. Python-package red on
  Windows (MSVC) and macOS legs only — all Linux legs pass; this matches the known
  Windows/macOS conda infra flake (confirm before merge).

## Reviewer notes

- Focus on `cuda_histogram_constructor.cu` (+311) and the quant body kernels.
- Confirm `EXABOOST_GRAPH_QUANT` stays OFF by default (quant users' default behavior
  unchanged).
- No automated tests are added by this increment; correctness rests on the out-of-tree
  md5 lock matrix above. A CI-wired md5 harness is a reasonable follow-up.
