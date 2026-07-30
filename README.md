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

What makes it fast
------------------

- **Hybrid level-batched growth.** Whole levels of sibling pairs are scored,
  synchronized, and applied in one launch each instead of per split — turning a
  latency-bound loop into a throughput-bound one, with leaf-wise-identical
  trees.
- **CUDA-graph level loops.** The per-level launch sequence is captured once and
  replayed by a device-side controller, removing host round-trips from the
  inner loop on shallow trees.
- **NVRTC runtime JIT.** Construct kernels are specialized at runtime to the
  actual data shape (bin count, column layout), self-tested against the
  ahead-of-time kernel, and promoted only if bit-identical.
- **Per-tree compact column view.** With any `feature_fraction < 1`, only the
  sampled columns are materialized and gathered for histogram construction.
  The win scales with the excluded fraction: ~3.4× end-to-end training at
  `feature_fraction = 0.1` on wide, low-cardinality data, tapering to ~1.1×
  at 0.6.
- **GPU-native dataset construction.** Dense binning, row-data build, and EFB
  pre-checking run on the device; CuPy and `__cuda_array_interface__` inputs are
  ingested without a host round-trip.
- **Quantized training, two ways.** `quant_mode=stochastic` is the speed end:
  gradients packed into 4 bins with seeded stochastic rounding (unbiased in
  expectation). `quant_mode=fixedpoint` is the near-lossless end: deterministic
  rounding, with an internal outlier-robust gradient scale so rare huge
  gradients don't crush the quantization range. Bin counts default to 4 and 64
  respectively; override with `quant_bins` (any value in `[2, 65534]` —
  training refuses counts whose histogram sums could overflow at your row
  count). Both modes are bit-reproducible run to run; `quant_mode=none` is
  full precision.
- **An execution planner.** Shape-conditional kernel choices are resolved once
  from the data and parameters (`cuda_plan=auto`) instead of from a pile of
  environment variables — every decision guaranteed bit-identical, and
  individually overridable for experiments.
- **GPU inference via NVIDIA FIL.** `Booster.predict()` transparently runs on
  cuML's Forest Inference Library when available (see below); CuPy arrays stay
  on the device end to end.

Correctness discipline
----------------------

Every optimization above must be *bit-identical* to the reference path, and
that is enforced mechanically rather than trusted. A regression-gate suite runs
on every commit against a real GPU: a 38-cell lattice of (config × data-shape)
training cells fingerprinted by model md5, plan-flip equality cells that prove
each planner decision changes nothing, validity assertions, metric floors, and
a perf gate against a rolling baseline. A nightly tier adds a config × shape
fuzzer with CPU-parity checks plus full-scale gates on real datasets.

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

With cuML installed (`pip install cuml-cu12`), `Booster.predict()` on a
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
  ~0.01% of rows near a split threshold can route differently than the exact
  predictor; `double` restores exact routing (~1e-13), `native` uses the
  model's own precision.
- Training-time validation metrics (and early stopping) do **not** go through
  FIL: they use the booster's internal evaluation on the attached validation
  set, which is already GPU-resident under `device_type=cuda`.

Build from source
-----------------

```bash
git clone https://github.com/BelixRogner/Falcata.git
cd Falcata
git submodule update --init --recursive
# Adjust CMAKE_CUDA_ARCHITECTURES for your GPU. RTX 5090 = 120, RTX 4090 = 89.
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120-real;120-virtual -DBUILD_WITH_SHARED_NCCL=ON" \
  sh build-python.sh install --cuda
```

Compatibility with LightGBM
---------------------------

Falcata began as a fork of LightGBM and deliberately stays interoperable at the
data boundaries:

- **Models** written by Falcata load in stock LightGBM (and vice versa) with
  bit-identical predictions — verified in CI.
- **Binary datasets** (`.dataset`) interchange in both directions.
- **Parameter names** are unchanged; Falcata's additions (`quant_mode`,
  `cuda_precision`, `cuda_plan`) are new names that upstream simply ignores.
- `import lightgbm` still works as an alias for `import falcata`, and the
  historical `LGBM_*` C API names remain as aliases for `FLC_*`.

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
- Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. "[A Communication-Efficient Parallel Algorithm for Decision Tree](http://papers.nips.cc/paper/6380-a-communication-efficient-parallel-algorithm-for-decision-tree)". NIPS 2016.
- Huan Zhang, Si Si, Cho-Jui Hsieh. "[GPU Acceleration for Large-scale Tree Boosting](https://arxiv.org/abs/1706.08359)". SysML 2018.
