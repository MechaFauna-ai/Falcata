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
- **NVRTC runtime JIT** — construct kernels are specialized at runtime to the
  actual data shape, self-tested against the ahead-of-time kernel, and promoted
  only if bit-identical.
- **Per-tree compact column view** — with any ``feature_fraction < 1``, only
  the sampled columns are materialized for histogram construction; the win
  scales with the excluded fraction (~3.4× end-to-end at
  ``feature_fraction = 0.1`` on wide, low-cardinality data).
- **GPU-native dataset construction** — dense binning, row-data build and EFB
  pre-checking run on the device; CuPy and ``__cuda_array_interface__`` inputs
  are ingested without a host round-trip.
- **Quantized training, two ways** — ``quant_mode=stochastic`` is the speed
  end: 4-bin gradients with seeded stochastic rounding; ``quant_mode=fixedpoint``
  is the near-lossless end: deterministic rounding with an internal
  outlier-robust gradient scale. Bin counts are overridable with
  ``quant_bins``; both modes are bit-reproducible run to run.
- **GPU inference via NVIDIA FIL** — with cuML installed,
  ``Booster.predict()`` transparently runs on the Forest Inference Library;
  CuPy arrays stay on the device end to end.
- **An execution planner** — shape-conditional kernel choices are resolved once
  from the data and parameters (``cuda_plan=auto``), every decision guaranteed
  bit-identical and individually overridable.

Every optimization above is required to be bit-identical to the reference path,
and that is enforced mechanically: a 38-cell regression lattice of
(config × data-shape) training cells fingerprinted by model md5 runs on a real
GPU on every commit, alongside plan-flip equality cells, validity assertions,
metric floors and a perf gate.

Install
-------

.. code:: sh

    pip install falcata

That builds the CUDA library from source, so you need the CUDA toolkit
(>= 11.0), CMake >= 3.28, a C++17 compiler and Python >= 3.10. Nothing else:
the source distribution vendors every dependency, so no ``git clone`` and no
submodule dance.

The build targets every GPU architecture your toolkit supports, plus PTX for
the newest, which is why it takes a while. Building for just your own card is
far faster:

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

Installing the wheel also installs a ``lightgbm`` import shim. If the target
environment already has stock LightGBM, uninstall it first or use a fresh
environment — pip will not report the collision.

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

- **Models** written by Falcata load in stock LightGBM (and vice versa) with
  bit-identical predictions — verified in CI.
- **Binary datasets** (``.dataset``) interchange in both directions.
- **Parameter names** are unchanged; Falcata's additions (``quant_mode``,
  ``cuda_precision``, ``cuda_plan``) are new names upstream simply ignores.
- ``import lightgbm`` still works as an alias for ``import falcata``, and the
  historical ``LGBM_*`` C API names remain as aliases for ``FLC_*``.

License
-------

MIT. Falcata derives from LightGBM (copyright Microsoft Corporation and the
LightGBM developers, MIT); that copyright is retained. Falcata is not affiliated
with, endorsed by, or supported by Microsoft or the LightGBM maintainers.
