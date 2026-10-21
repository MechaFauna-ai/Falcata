<p align="center">
  <img src="docs/logo/falcata-mark.svg" alt="Falcata" width="180">
</p>

Falcata
=======

GPU-first gradient boosted decision trees.

*Falcataria moluccana* — the falcata — is one of the fastest-growing trees on
earth. This one grows them faster too.

Falcata is a CUDA-native GBDT library: a leaf-wise learner whose training loop
was rebuilt around batched, level-parallel GPU kernels rather than one split at
a time.

For the story behind the design, see [Falcata: GPU gradient boosting](https://mechafauna.ai/blog/falcata-gpu-gradient-boosting)
on the MechaFauna blog.

What makes it fast
------------------

- **Hybrid level-batched growth.** Whole levels of sibling pairs are scored,
  synchronized, and applied in one launch each instead of per split — turning a
  latency-bound loop into a throughput-bound one, with leaf-wise-identical
  trees.
- **CUDA-graph level loops.** The per-level launch sequence is captured once and
  replayed by a device-side controller, removing host round-trips from the
  inner loop on shallow trees.
- **Per-tree compact column view.** With any `feature_fraction < 1`, only the
  sampled columns are materialized and gathered for histogram construction.
  The win scales with the excluded fraction: ~3.4× end-to-end training at
  `feature_fraction = 0.1` on wide, low-cardinality data, tapering to ~1.1×
  at 0.6 (Numerai deep, measured 30 July 2026 on a pre-launch build).
- **GPU-native dataset construction.** Dense binning, row-data build, and EFB
  pre-checking run on the device; CuPy and `__cuda_array_interface__` inputs are
  ingested without a host round-trip.
- **Quantized training, two ways.** `quant_mode=stochastic` is the speed end:
  gradients packed into 4 bins with seeded stochastic rounding (unbiased in
  expectation). `quant_mode=fixedpoint` is the near-lossless end: deterministic
  rounding, with an internal outlier-robust gradient scale so rare huge
  gradients don't crush the quantization range. Bin counts default to 4 and 64
  respectively; stochastic raises itself automatically to 16 (64 for
  constant-hessian objectives) on large data with small leaves (at least 100k
  rows and fewer than 4,096 expected rows per leaf; not for multiclass or
  ≥50:1-imbalanced binary). Override with `quant_bins` (any value in
  `[2, 65534]` — training refuses counts whose histogram sums could overflow at
  your row count). Both modes are bit-reproducible — not just run to run, but
  across GPU models and host machines: the same seed trains the bit-identical
  model on any CUDA device (verified sm_89 vs sm_120 on a pre-launch build; the
  integer-only Philox stream keeps the guarantee by construction);
  `quant_mode=none` is full precision.
- **An execution planner.** Shape-conditional kernel choices are resolved once
  from the data and parameters (`cuda_plan=auto`) instead of from a pile of
  environment variables — every decision bit-identical except where the
  `cuda_plan` documentation says otherwise (`robust_scale`, and the tie order
  of `batch_kernels`), and individually overridable for experiments.
- **GPU inference via NVIDIA FIL.** `Booster.predict()` transparently runs on
  cuML's Forest Inference Library when available (see below); CuPy arrays stay
  on the device end to end.

Correctness discipline
----------------------

Every optimization above must be *bit-identical* to the reference path, and
that is enforced mechanically rather than trusted. A regression-gate suite runs
locally on a GPU for every merge and nightly (hosted CI has no GPU): a 441-cell
lattice of (config × data-shape) training cells (98 md5-fingerprinted, 341
plan-flip equality cells that prove each planner decision changes nothing,
2 metric cells), validity assertions, metric floors, and a perf gate against a
rolling baseline. The nightly tier adds a config × shape fuzzer with CPU-parity
checks plus full-scale gates on real datasets.

See [tests/gates/README.md](tests/gates/README.md).

Quick start
-----------

```python
import falcata as flc

ds = flc.Dataset(X_train, label=y_train, params={"device_type": "cuda"})
model = flc.train(
    {
        "objective": "regression",
        "device_type": "cuda",
        "num_leaves": 255,
        "quant_mode": "stochastic",   # none | stochastic | fixedpoint
        "cuda_precision": "fp32",     # fp64 (default) | fp32
        "cuda_plan": "auto",          # the planner picks the kernels
    },
    ds,
    num_boost_round=1000,
)
```

GPU inference (NVIDIA FIL)
--------------------------

With cuML installed (`pip install cuml-cu13` alongside the prebuilt CUDA 13
wheel; `cuml-cu12` for a source build against CUDA 12), `Booster.predict()` on a
CUDA-trained model transparently runs on NVIDIA's Forest Inference Library —
no API change. 2-D numpy input returns numpy; CuPy (or any
`__cuda_array_interface__` array) stays on the device end to end. Converted
FIL models are cached per iteration slice and invalidated automatically when
the booster changes.

```python
preds = model.predict(X_test)          # numpy in -> numpy out, FIL under the hood
preds = model.predict(cupy_X)          # device in -> device out, no host round-trip
```

- `FALCATA_FIL=0` disables the FIL path; without cuML installed, predict
  silently falls back to the regular CPU predictor.
- `FALCATA_FIL_PRECISION=single` (default) evaluates thresholds in fp32 —
  rows near a split threshold can route differently than the exact predictor
  (~0.01% of rows in a July 2026 check on a pre-launch build;
  {{NUM:fil_single_routing_fraction}} on 1.1); `double` restores exact routing
  (differences ~1e-13 in July 2026; {{NUM:fil_double_max_abs}} on 1.1),
  `native` uses the model's own precision.
- Training-time validation metrics (and early stopping) do **not** go through
  FIL: they use the booster's internal evaluation on the attached validation
  set, which is already GPU-resident under `device_type=cuda`.

Install
-------

```bash
pip install falcata
```

On Linux x86_64 this installs a **prebuilt wheel** — no compilation, no CUDA
toolkit needed — built with CUDA 13, with native GPU code for sm_75 through
sm_120 (T4, RTX 20xx, A100, RTX 30xx/40xx, H100, B100/B200, RTX 50xx). It
needs an NVIDIA driver that supports CUDA 13 (R580 or newer). Older GPUs
(Pascal and Volta: P100, GTX 10xx, V100) fail fast with a message pointing at
the source build below; build with CUDA 12.x for those.

Everywhere else (and with `pip install --no-binary falcata falcata`), pip
builds from source, which needs the **CUDA toolkit (>= 11.0)**, CMake >= 3.28,
a C++17 compiler and Python >= 3.10. Nothing else:
the source distribution vendors every dependency, so no `git clone` and no
submodule dance.

The build detects the GPU(s) in the machine and **compiles only for those**
(the library compiles in about 50 s on a 24-thread desktop; measured
18 October 2026 with CUDA 13). CUDA 13 cannot
compile for GPUs below sm_75: it skips such a GPU with a warning, and the build
fails if no detected GPU is left, so use CUDA 12.x for those cards. When no GPU is
visible at build time — docker build stages, CPU-only CI — it instead targets
every architecture the toolkit supports (sm_60 upward on CUDA 11/12, sm_75
upward on CUDA 13+, plus PTX for the newest so future cards still run), which
takes about four times as long (about 3.5 minutes on the same machine with
CUDA 13). To build for a card other than the one present, or to pick one
explicitly:

```bash
# RTX 5090 = 120, RTX 4090 = 89, A100 = 80, T4 = 75
pip install falcata --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=89
```

No GPU? There is a CPU build, though it is not what this library is for:

```bash
pip install falcata --config-settings=cmake.define.USE_CUDA=OFF
```

**Multi-GPU training** additionally needs NCCL *and its headers* — on
Debian/Ubuntu `sudo apt-get install libnccl2 libnccl-dev`, or
`conda install -c conda-forge nccl`. `BUILD_WITH_SHARED_NCCL` links
`libnccl.so` instead of the static archive, which avoids nvlink failures
against some static NCCL builds on newer architectures:

```bash
pip install falcata \
  --config-settings=cmake.define.USE_NCCL=ON \
  --config-settings=cmake.define.BUILD_WITH_SHARED_NCCL=ON
```

With `USE_NCCL=ON` but no NCCL headers on the include path, configuration
fails with `Could NOT find NCCL (missing: NCCL_INCLUDE_DIR)`.

Build from source (development)
-------------------------------

```bash
git clone https://github.com/MechaFauna-ai/Falcata.git
cd Falcata
git submodule update --init --recursive
sh build-python.sh install --cuda
```

Development builds, like `pip install`, compile only for the GPU(s) that
`nvidia-smi` reports. Set `CUDAARCHS` for `build-python.sh`, or
`-DCMAKE_CUDA_ARCHITECTURES` for `cmake`, to choose explicitly. Release wheels
are built by `tools/build-release-wheel.sh` for a fixed list of architectures
(override with `FALCATA_WHEEL_ARCHS`), whatever GPU the build machine has.

**Windows.** CUDA builds and runs on Windows too (single-GPU). Use the Ninja
generator from an *x64 Native Tools Command Prompt for VS* (so `nvcc` finds
`cl.exe`), then install against the compiled DLL:

```console
cmake -B build -S . -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sh ./build-python.sh install --precompile
```

See [docs/Installation-Guide.rst](docs/Installation-Guide.rst) for the details —
notably, use Ninja (not the VS generator), and if `nvcc` rejects your MSVC as an
"unsupported Microsoft Visual Studio version" add
`-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler`.

Falcata installs under its own name only, so it coexists with stock LightGBM in
one environment; `import lightgbm` keeps resolving to LightGBM.

Compatibility with LightGBM
---------------------------

Falcata began as a fork of LightGBM and deliberately stays interoperable at the
data boundaries:

- **Models** with scalar leaves written by Falcata load in stock LightGBM (and
  vice versa) with bit-identical predictions — asserted by a production deploy
  gate for scalar-leaf models, not by CI. Vector-leaf models use a format
  extension stock LightGBM cannot read.
- **Binary datasets** (`.dataset`) interchange in both directions.
- **Parameter names** are unchanged; Falcata's additions (`quant_mode`,
  `cuda_precision`, `cuda_plan`) are new names that upstream simply ignores.
- The historical `LGBM_*` C API names remain as aliases for `FLC_*`. The
  Python package is `import falcata` only — code written against
  `import lightgbm` needs its import changed, nothing else.

See [docs/design/format-compatibility.md](docs/design/format-compatibility.md).

Contributing
------------

See [CONTRIBUTING.md](CONTRIBUTING.md). Human and AI contributors are welcome on
the same terms.

License
-------

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Falcata derives from
LightGBM (copyright Microsoft Corporation and the LightGBM developers, MIT);
that copyright is retained. Falcata is not affiliated with, endorsed by, or
supported by Microsoft or the LightGBM maintainers.

Reference papers
----------------

Falcata builds on the algorithms described in:

- Yu Shi, Guolin Ke, Zhuoming Chen, Shuxin Zheng, Tie-Yan Liu. "[Quantized Training of Gradient Boosting Decision Trees](https://papers.nips.cc/paper_files/paper/2022/hash/77911ed9e6e864ca1a3d165b2c3cb258-Abstract.html)". NeurIPS 2022.
- Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". NIPS 2017.
- Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. "[A Communication-Efficient Parallel Algorithm for Decision Tree](https://papers.nips.cc/paper/6380-a-communication-efficient-parallel-algorithm-for-decision-tree)". NIPS 2016.
- Huan Zhang, Si Si, Cho-Jui Hsieh. "[GPU Acceleration for Large-scale Tree Boosting](https://arxiv.org/abs/1706.08359)". SysML 2018.
