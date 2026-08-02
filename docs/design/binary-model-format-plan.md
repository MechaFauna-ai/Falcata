# Falcata binary model format (FALB) — design plan

**Status:** adopted; v1 scope fixed 2026-08-02 · **Origin:** numerai prod-artifact analysis

> **v1 decisions (2026-08-02 review #2).** M1 optimizes for load time and size:
> core sections + categoricals, raw/mmap-able by default, structural stats
> behind a flag (see §3.1 for why the original threshold design had to
> change). Pickle flips to FALB in M2 with an escape hatch. Importers (M4/M5)
> are independent of the format and can proceed on their own schedule.

## 1. Motivation (measured, real production model)

> **M1 + compression, measured on a real artifact (2026-08-02).** numerai
> production model `target_jasper_60_11_...`: 45,000 trees, 3555 features,
> ~100 leaves/tree.
>
> | representation | size | vs text | exact? |
> |---|---|---|---|
> | model text | 471.6 MB | 1.0x | - |
> | gzip -6 text | 148.1 MB | 3.2x | yes |
> | FALB raw (mmap-able) | 65.5 MB | 7.2x | **bit-exact** |
> | **FALB zlib-6 (default)** | **45.6 MB** | **10.3x** | **bit-exact** |
> | FALB zlib-9 | 45.4 MB | 10.4x | bit-exact |
> | FALB zlib-6 + f32 leaves (opt-in) | 29.6 MB | 15.9x | ~2.7e-08 rel |
> | FALB zlib-6 + stats + diagnostics | 133.1 MB | 3.5x | bit-exact |
>
> Predictions from the default (zlib, f64) are BIT-IDENTICAL to the text model
> across all 3555 features. Load 0.188s text -> 0.160s. zlib-9 buys 0.2 MB for
> 8x the save time, so 6 is the default.
>
> The 31x originally projected here is not reached losslessly: that estimate
> assumed a 31-leaf model whose text ran ~215 bytes per node, and this real
> artifact has ~100 leaves/tree with text already at ~53 bytes/node. 10.3x
> lossless (15.9x with f32 leaves) is the measured figure.
>
> Where the bytes are, and why the format is shaped as it is:
> - **leaf_value f64 was 54.8% of the raw file and is incompressible** (1.15x
>   even byte-shuffled -- f64 mantissas are high entropy). It is the only
>   reason f32 leaves exist as an option, and the only lever that trades
>   exactness for space.
> - everything else compresses well once byte-plane shuffled: threshold
>   indices 3.6x, decision types 4.3x, tree index 11x. The shuffle alone is
>   worth ~4 MB here (49.5 -> 45.6).
> - the **threshold dictionary** did its job: 4.45M thresholds encode as 4.4 MB
>   of indices plus **94.8 KB** of dictionary (11.8k distinct doubles across all
>   45k trees), where raw f64 would have been 35.6 MB.

## 2. Requirements

1. **FALB is the default serialization** (`save_model`, `model_to_*`, pickle).
2. **LightGBM text stays first-class for input AND output** — interop with upstream/stock LightGBM, treelite/nvforest (FIL) conversion, debugging. Loaders auto-detect format.
3. **Importers for foreign GBDT models**: XGBoost and CatBoost models convert into Falcata models, so one engine (incl. FIL GPU predict) serves all of them.

## 3. Format spec (v1)

### Container
```
magic "FALB" | u32 version | u64 flags | header section | section table | sections…
```
- **header**: num_trees, num_features, max_leaves, objective + predict-relevant params only, feature names, average-output/base-score. (No bin-bounds table — see §3.1.)
- **section table**: (id, offset, compressed_len, raw_len, codec) per section. Codec ∈ {raw, zstd}. Raw sections are mmap-able; store raw_len for preallocation.
- **flags**: bitmask of required capabilities; loader errors on unknown *required* flags (forward-compat with a clean failure, never silent misparse).

### Tree data — structure-of-arrays, all trees concatenated, per-tree offsets varint
| array | dtype | notes |
|---|---|---|
| num_leaves per tree | varint | |
| split_feature | u16 | u32 flag if >65k features |
| threshold | **u8/u16/u32 dictionary index** _or_ f32 _or_ f64 | index into the per-feature threshold dictionary (§3.1); bit-exact by construction. Raw-float modes for imported models |
| decision_type | u8 (bit-packed later) | default-left, missing-type |
| left/right_child | i8 when leaves ≤127 else i16 | |
| leaf_value | f64 (default, bit-exact) or f32 (opt-in, ~1e-7 rel) | |
| categoricals | optional section: cat_boundaries u32, cat_threshold bitset u32 | spec'd v1, implementation may land M3 |
| diagnostics | optional section (split_gain, counts, weights…) | written only with `with_diagnostics=True`; absence disables feature_importance(gain) with a clear error |

**M3 DECIDED (2026-08-02, measured on the 45k-tree numerai model): both
optional sections stay OFF by default.** Each roughly DOUBLES the file --
core 45.6 MB (10.3x vs text), +structural stats 87.9 MB (5.4x, +93%),
+diagnostics 90.8 MB (5.2x, +99%), both 133.1 MB (3.5x). Paying +93% so that
`pred_contrib` works without being asked is the wrong default when most models
never call it; better encoding (f32 weights, varint counts) could shave part
of the weight arrays but cannot change that conclusion, so it is not pursued.
What makes the default SAFE is that a stats-less model now REFUSES
`pred_contrib` with a clear message instead of silently returning NaN/Inf --
TreeSHAP divides by each node's data count, so zero-filled counts produce
garbage, not an error.

**Sizing note — structural stats.** counts + weights are ~1.5M internal and
~1.55M leaf entries on the numerai reference; naively (u32 counts, f64 weights)
that is ~+37 MB against a ~21 MB core, i.e. the headline is a core-only number.
They are therefore **behind a flag in M1**, and when they are enabled by
default the encoding must be deliberate: f32 weights (hessian sums used for
SHAP weighting — f32 is ample; leaf VALUES stay f64), varint counts, zstd on
that section. Decide the default from a measurement, not from this estimate.

**Explicit non-goals v1** (refuse with clear error, flag bits reserved): linear trees, CatBoost CTR features. Multi-output-per-leaf is NOT refused — the v1 leaf layout reserves `leaf_dim` (§ROADMAP amendments) so vector-leaf multi-target needs no v2.

**Loader is an untrusted-input parser.** Models are shared as files. Every
section-table offset and length is bounds-checked against the actual file size
before any read, and every count is validated against the space remaining;
the loader must fail cleanly, never UB, on truncated or hostile input. The
fuzzing in §6.3 verifies this property — it is not the mechanism providing it.

**Codec default: raw.** Core sections stay uncompressed so they are mmap-able
and per-tree access via the fixed-width offsets is O(1); zstd is opt-in for
archival. The load-time win is the point, so we do not trade it for the size
headline by default.

### 3.1 Threshold dictionary (replaces the bin-index design)

The original spec had bin-index thresholds reconstructing the exact double
from "the header bin-bounds table (from `feature_infos`)". **That table does
not exist.** `feature_infos` for a numerical feature is only its range —
`bin_info_string()` emits `'[' << min_val_ << ':' << max_val_ << ']'`
(bin.h:236). Bin upper bounds live in the `BinMapper`, i.e. in the **Dataset**,
and never enter the model. So (a) there is nothing to reference, and (b) a
model loaded from text has no Dataset at all, which would make
`convert in.txt out.falb` — most of M2's value — impossible in indexed mode.

v1 instead stores a **per-feature dictionary of the distinct thresholds that
actually occur in the trees**, sorted, as f64; tree nodes store an index into
their feature's dictionary. Properties:

- **Exact by construction.** Dictionary entries are the tree's own doubles,
  copied verbatim. There is no grid to fall off, so the "verify-on-write
  bit-compare, else fall back to raw-f64 for the whole model" rule is dropped:
  refit, text-edited and imported models all encode losslessly with no
  whole-model penalty for one off-grid value.
- **Dataset-free.** Identical path for freshly-trained and text-loaded models.
- **Bounded by the bin table it replaces**: every used threshold is a bin
  bound, deduplicated per feature, so the dictionary is never larger and is
  typically far smaller (numerai: a full 3555×255 f64 table would be ~7.2 MB
  of a ~21 MB budget).
- Index width per feature: u8 (≤256 distinct), u16, else u32 — carried by the
  per-array dtype tag, so it is not a new mechanism.

Raw f32/f64 threshold modes remain for importers (§5); f32 is an import
fidelity mode for f32-native sources, never a compression knob on f64 trees.

### Exactness contract
`text → FALB(f64 leaves, bin-index or f64 thresholds) → text` is byte-idempotent (modulo key ordering, which the test canonicalizes), and predictions are bit-identical. f32 leaf mode is opt-in and documented lossy.

## 4. API surface

- **C++**: `GBDT::SaveModelToBinary/LoadModelFromBinary`, mirroring the existing string pair — the in-memory `Tree` already holds these arrays contiguously, so serialization is mostly per-array writes. C API: `LGBM_BoosterSaveModelBinary`, `LGBM_BoosterCreateFromBinary` (or extend the existing create-from-file with sniffing).
- **Python**:
  - `Booster.save_model(path, format="auto")` — auto by extension: `.falb` → binary (new default suggestion for docs/examples), `.txt` → text, `.gz` wraps either.
  - `Booster.model_to_binary() -> bytes`; `Booster(model_file=…)` sniffs magic; `Booster(model_bin=bytes)`.
  - **`__getstate__` emits FALB** — every existing pickle call site shrinks ~30–45× with zero caller changes. `__setstate__` version-gated: accepts old text-payload pickles forever.
- **CLI**: `falcata convert in.{txt,falb,json} out.{falb,txt}`.

## 5. Foreign-model importers (python-package, no C++ needed beyond raw-float thresholds)

`falcata.importers.from_xgboost(model | json_path)` / `from_catboost(model | json_path)` → Falcata Booster (built via canonical model text or directly via FALB).

- **XGBoost** (M4): parse `save_model` JSON/UBJ, gbtree only. Map: default direction → `decision_type` default-left; `base_score` → init score; thresholds stored raw-f32/f64 (xgb uses `<` vs LightGBM `<=` — adjust threshold by nextafter or emit exact semantics via decision_type bit; settle in impl with parity tests). Refuse: gblinear, multi-target trees.
- **CatBoost** (M5): parse JSON export. Oblivious (symmetric) trees unroll into ordinary binary trees (depth d → 2^d leaves, split repeated per level) — representable directly. Numeric float splits only; **refuse CTR/categorical models** v1. Refuse: multiclass initially.
- Acceptance for both: prediction parity vs the source library on randomized + real data, ≤1e-6 rel (f32) / bit-exact where representable; round-trip through FALB preserves parity.
- Payoff: imported models get Falcata's CUDA/FIL predict and the unified artifact format for free.

## 6. Testing / CI

1. Round-trip: text↔FALB idempotence + bit-identical predict (random configs: missing values, categoricals when landed, max_bin variants, huge feature counts).
2. **Stock-interop guard**: exported *text* must remain parseable by upstream `lightgbm==4.6.0` — CI job in a separate venv with real PyPI lightgbm (the shim shadows it in dev envs, so this MUST be an isolated env).
3. Corrupt/truncated-file fuzzing on the loader: clean errors, never UB.
4. Importer parity suites vs xgboost/catboost as above.
5. Size + load-time benchmarks recorded per release (numerai 50k-tree reference: ≤22 MB f64, ≤15 MB f32+zstd, load <1s).

## 7. Milestones

- **M1** — binary core: save/load predict-only, f64 leaves, dictionary + f64 thresholds, **categoricals included** (our categorical support is a headline feature since the 2026-08-01 lift; a format that cannot store an airline-class model is not shippable), bounds-checked loader, round-trip tests. *(the 90% of value)*
- **M2** — python plumbing: auto-detect, `model_to_binary`, pickle via FALB, zstd sections, CLI convert.
- **M3** — structural-stats + training-diagnostics sections (+ feature_importance support), and the measurement that decides whether structural stats become default-on.
- **M4** — XGBoost importer.
- **M5** — CatBoost importer (numeric, oblivious-unroll).

## 8. Adoption notes (numerai side — do not implement here, context only)

- The prod inference container currently ships **stock lightgbm 4.6.0, which cannot read FALB** — prod artifacts stay `*.txt.gz` (fork-agnostic) until the container ships Falcata (planned with the GPU-inference migration). After that, `.falb` becomes the prod artifact and the container's parse cost disappears.
- Local caches/HPO artifacts can switch to FALB immediately via the pickle path (`__getstate__`), no call-site changes.
