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

- **Hybrid level-batched growth** — whole levels of sibling pairs are scored,
  synchronized and applied in one launch each instead of per split, turning a
  latency-bound loop into a throughput-bound one, with leaf-wise-identical trees.
- **CUDA-graph level loops** — the per-level launch sequence is captured once and
  replayed by a device-side controller, removing host round-trips.
- **Per-tree compact column view** — with any ``feature_fraction < 1``, only
  the sampled columns are materialized for histogram construction; the win
  scales with the excluded fraction (~3.4× end-to-end at
  ``feature_fraction = 0.1`` on wide, low-cardinality data; Numerai deep,
  measured 30 July 2026 on a pre-launch build).
- **GPU-native dataset construction** — dense binning, row-data build and EFB
  pre-checking run on the device; CuPy and ``__cuda_array_interface__`` inputs
  are ingested without a host round-trip.
- **Quantized training, two ways** — ``quant_mode=stochastic`` is the speed
  end: 4-bin gradients with seeded stochastic rounding; ``quant_mode=fixedpoint``
  is the near-lossless end: deterministic rounding with an internal
  outlier-robust gradient scale. Stochastic raises its 4 bins to 16 (64 for
  constant-hessian objectives) on large data with small leaves; bin counts are
  overridable with ``quant_bins``; both modes are bit-reproducible run to run
  and across GPU models.
- **GPU inference via NVIDIA FIL** — with cuML installed (``cuml-cu13`` for
  the prebuilt CUDA 13 wheel), ``Booster.predict()`` transparently runs on the
  Forest Inference Library; CuPy arrays stay on the device end to end. The
  default fp32 thresholds can route rows near a split threshold differently
  from the exact predictor; ``FALCATA_FIL_PRECISION=double`` restores exact
  routing and ``FALCATA_FIL=0`` disables FIL.
- **An execution planner** — shape-conditional kernel choices are resolved once
  from the data and parameters (``cuda_plan=auto``), every decision
  bit-identical except where the ``cuda_plan`` documentation says otherwise,
  and individually overridable.

Every optimization above is required to be bit-identical to the reference path,
and that is enforced mechanically: a 441-cell regression lattice of
(config × data-shape) training cells (98 md5-fingerprinted, 341 plan-flip
equality, 2 metric cells) runs locally on a GPU for every merge and nightly
(hosted CI has no GPU), alongside validity assertions, metric floors and a perf
gate.

Install
-------

.. code:: sh

    pip install falcata

On Linux x86_64 this installs a **prebuilt wheel** — no compilation, no CUDA
toolkit needed — built with CUDA 13, with native GPU code for sm_75 through
sm_120 (T4, RTX 20xx, A100, RTX 30xx/40xx, H100, B100/B200, RTX 50xx). It needs
an NVIDIA driver that supports CUDA 13 (R580 or newer). Older GPUs (Pascal and
Volta) fail fast with a message pointing at the source build; build with
CUDA 12.x for those.

Everywhere else (and with ``pip install --no-binary falcata falcata``), pip
builds from source, which needs the CUDA toolkit (>= 11.0), CMake >= 3.28, a
C++17 compiler and Python >= 3.10. The source distribution vendors every
dependency. The build compiles only for the GPU(s) in the machine (about 50 s
on a 24-thread desktop, measured 18 October 2026 with CUDA 13); with no GPU
visible it targets every architecture the toolkit supports (about 3.5 minutes
on the same machine). To pick an architecture explicitly:

.. code:: sh

    # RTX 5090 = 120, RTX 4090 = 89, A100 = 80, T4 = 75
    pip install falcata --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=89

There is a CPU build, though it is not what this library is for:

.. code:: sh

    pip install falcata --config-settings=cmake.define.USE_CUDA=OFF

Multi-GPU training additionally needs NCCL *and its headers*
(``libnccl-dev`` on Debian/Ubuntu):

.. code:: sh

    pip install falcata \
      --config-settings=cmake.define.USE_NCCL=ON \
      --config-settings=cmake.define.BUILD_WITH_SHARED_NCCL=ON

Falcata installs under its own name only, so it coexists with stock LightGBM in
one environment; ``import lightgbm`` keeps resolving to LightGBM.

Quick start
-----------

.. code:: python

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

Compatibility with LightGBM
---------------------------

Falcata began as a fork of LightGBM and deliberately stays interoperable at the
data boundaries:

- **Models** with scalar leaves written by Falcata load in stock LightGBM (and
  vice versa) with bit-identical predictions, asserted by a production deploy
  gate; vector-leaf models use a format extension stock LightGBM cannot read.
- **Binary datasets** (``.dataset``) interchange in both directions.
- **Parameter names** are unchanged; Falcata's additions (``quant_mode``,
  ``cuda_precision``, ``cuda_plan``) are new names upstream simply ignores.
- The historical ``LGBM_*`` C API names remain as aliases for ``FLC_*``. The
  Python package is ``import falcata`` only — code written against
  ``import lightgbm`` needs its import changed, nothing else.

License
-------

MIT. Falcata derives from LightGBM (copyright Microsoft Corporation and the
LightGBM developers, MIT); that copyright is retained. Falcata is not affiliated
with, endorsed by, or supported by Microsoft or the LightGBM maintainers.
