# Format compatibility with LightGBM

Falcata was renamed from its LightGBM origins in the C++ interior, the C API,
and every package (see the rename slices in git history). Three things
deliberately keep LightGBM-era spellings, because they are **data formats, not
branding** — changing them breaks interchange for no user-visible gain.

## 1. Model text format (`model_to_string` / `save_model`)

**Status: identical to LightGBM, deliberately.**

The serialized model contains no product name at all — it starts with `tree`,
`version=v4`, `num_class=…`. So Falcata-trained models load in stock LightGBM
and vice versa, with byte-identical predictions.

This is load-bearing in production: models trained on the GPU with Falcata are
served on CPU by stock `lightgbm`, and a deploy gate asserts prediction
equality. **Do not change the model text format** without an explicit decision
and a migration plan for deployed models.

## 2. Binary dataset tokens (`.dataset` files)

**Status: keeps the literal string `______LightGBM_Binary_File_Token______`.**

Defined in `src/io/dataset.cpp` (`Dataset::binary_file_token` and
`Dataset::binary_serialized_reference_token`). These are the magic markers at
the head of a binary dataset file. Renaming them would:

- make every already-written `.dataset` file unreadable by Falcata (including
  large production caches — the numerai v5.3 cache alone is ~12 GB), and
- make Falcata-written datasets unreadable by stock LightGBM.

Nobody sees these strings without hexdumping a file, so the rename buys
nothing and costs interchange.

**When to rename:** only as part of a deliberate binary-format version bump —
at which point existing caches must be regenerated anyway, so the token change
is free. Rename both tokens together and bump
`Dataset::serialized_reference_version`.

## 3. C API compatibility aliases (`LGBM_*`)

**Status: `FLC_*` are the real symbols; `LGBM_*` remain as inline aliases.**

The exported C API is `FLC_*`. A compatibility header keeps the historical
`LGBM_*` names as zero-cost inline forwarders so existing bindings and
externally linked consumers keep building. These aliases are the only place
`LGBM_` appears in the implementation, and they can be dropped whenever
breaking those consumers is acceptable.

## Parameter names

Parameter names (`num_leaves`, `use_quantized_grad`, …) are also unchanged and
should stay that way: they appear in saved models, in user configs, and in
every tuning artifact. Falcata-specific additions use their own names
(`quant_mode`, `cuda_precision`, `cuda_plan`).

## Versioning

Falcata versions independently of LightGBM, starting at **1.0.0** (semver).
The pre-rename `4.6.0.99` advertised the LightGBM release it forked from and is
retired.

Package version and *format* versions are unrelated and must not be conflated:

- the model text format marker stays `version=v4` (LightGBM-compatible, see
  above) regardless of Falcata's package version;
- `Dataset::serialized_reference_version` tracks the binary dataset format;
- neither moves just because the package version does.
