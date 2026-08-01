# Falcata binary model format (FALB) — design plan

**Status:** proposal (handover doc) · **Date:** 2026-08-01 · **Origin:** numerai prod-artifact analysis

## 1. Motivation (measured, real production model)

50k-tree numerai model, 3555 features, 31 leaves/tree:

| representation | size | note |
|---|---|---|
| pickled Booster | 656.8 MB | pickle wraps `model_to_string()` — byte-identical to text |
| model text | 656.8 MB | every double as ~19 ASCII chars |
| gzip -6 text | 206.5 MB | 3.2× |
| **FALB (this plan), f64 bit-exact** | **~21 MB** | ~31× |
| **FALB, f32 leaves + zstd** | **~13–15 MB** | ~45× |

Text field breakdown of those 657 MB: **45% is training diagnostics never read at predict time** (`internal_value` 11.7%, `split_gain` 7.9%, `internal_weight` 7.0%, `internal_count` 6.6%, `leaf_weight` 5.8%, `leaf_count` 5.8%); the remaining 55% is ASCII numbers with small natural dtypes. Load time is the second prize: text load = `atof` on ~40M numbers (seconds per model); binary load = read + memcpy (sub-second, mmap-friendly).

## 2. Requirements

1. **FALB is the default serialization** (`save_model`, `model_to_*`, pickle).
2. **LightGBM text stays first-class for input AND output** — interop with upstream/stock LightGBM, treelite/nvforest (FIL) conversion, debugging. Loaders auto-detect format.
3. **Importers for foreign GBDT models**: XGBoost and CatBoost models convert into Falcata models, so one engine (incl. FIL GPU predict) serves all of them.

## 3. Format spec (v1)

### Container
```
magic "FALB" | u32 version | u64 flags | header section | section table | sections…
```
- **header**: num_trees, num_features, max_leaves, objective + predict-relevant params only, feature names, optional per-feature bin-bounds table (from `feature_infos` — needed for bin-index thresholds), average-output/base-score.
- **section table**: (id, offset, compressed_len, raw_len, codec) per section. Codec ∈ {raw, zstd}. Raw sections are mmap-able; store raw_len for preallocation.
- **flags**: bitmask of required capabilities; loader errors on unknown *required* flags (forward-compat with a clean failure, never silent misparse).

### Tree data — structure-of-arrays, all trees concatenated, per-tree offsets varint
| array | dtype | notes |
|---|---|---|
| num_leaves per tree | varint | |
| split_feature | u16 | u32 flag if >65k features |
| threshold | **u8/u16 bin index** _or_ f32 _or_ f64 | per-model flag. Bin-index reconstructs the exact double from the header bin-bounds table → **bit-exact** and 1–2 bytes. Raw-float mode required for imported models (no bin structure) |
| decision_type | u8 (bit-packed later) | default-left, missing-type |
| left/right_child | i8 when leaves ≤127 else i16 | |
| leaf_value | f64 (default, bit-exact) or f32 (opt-in, ~1e-7 rel) | |
| categoricals | optional section: cat_boundaries u32, cat_threshold bitset u32 | spec'd v1, implementation may land M3 |
| diagnostics | optional section (split_gain, counts, weights…) | written only with `with_diagnostics=True`; absence disables feature_importance(gain) with a clear error |

**Explicit non-goals v1** (refuse with clear error, flag bits reserved): linear trees, CatBoost CTR features, multi-output-per-leaf.

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

- **M1** — binary core: save/load predict-only, f64 leaves, bin-index + f64 thresholds, round-trip tests. *(the 90% of value)*
- **M2** — python plumbing: auto-detect, `model_to_binary`, pickle via FALB, zstd sections, CLI convert.
- **M3** — diagnostics section (+ feature_importance support), categorical section.
- **M4** — XGBoost importer.
- **M5** — CatBoost importer (numeric, oblivious-unroll).

## 8. Adoption notes (numerai side — do not implement here, context only)

- The prod inference container currently ships **stock lightgbm 4.6.0, which cannot read FALB** — prod artifacts stay `*.txt.gz` (fork-agnostic) until the container ships Falcata (planned with the GPU-inference migration). After that, `.falb` becomes the prod artifact and the container's parse cost disappears.
- Local caches/HPO artifacts can switch to FALB immediately via the pickle path (`__getstate__`), no call-site changes.
