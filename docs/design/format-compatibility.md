# Format compatibility with LightGBM

Falcata was renamed from its LightGBM origins in the C++ interior, the C API,
and every package (see the rename slices in git history). Three things
deliberately keep LightGBM-era spellings, because they are **data formats, not
branding** — changing them breaks interchange for no user-visible gain.

## 1. Model text format (`model_to_string` / `save_model`)

**Status: identical to LightGBM for scalar-leaf models. Vector-leaf models use
a self-describing Falcata extension.**

The serialized model contains no product name at all — it starts with `tree`,
`version=v4`, `num_class=…`. So Falcata-trained models load in stock LightGBM
and vice versa, with byte-identical predictions.

This is load-bearing in production: scalar-leaf models trained on the GPU with
Falcata are served on CPU by stock `lightgbm`, and a deploy gate asserts
prediction equality. **Do not change the scalar model text format** without an
explicit decision and a migration plan for deployed models.

Vector-leaf trees are the explicit exception. Each such tree adds
`leaf_value_dim=T`, and its `leaf_value=` line contains `num_leaves * T`
values in leaf-major, target-minor order. This extension is required to retain
all target outputs in one tree and is written only when `T > 1`; T=1 models
remain byte-identical to the scalar format. The top-level `version=v4` marker
is deliberately unchanged; `leaf_value_dim` is the per-tree capability marker.

Stock LightGBM and Falcata versions from before this extension cannot read a
vector-leaf text model. Their parser expects exactly `num_leaves` values and
fails on the wider array; some old parallel loaders terminate the process
instead of returning a recoverable error. Do not pass these models to an old or
stock reader. Deployments that require stock-LightGBM serving must continue to
use scalar trees (including the round-robin multi-target baseline). There is no
automatic downgrade because splitting a vector tree into scalar trees would
change its structure and semantics. New Falcata versions read both the original
scalar format and this extension.

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
