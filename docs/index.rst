.. image:: ./logo/falcata-mark.svg
   :align: center
   :width: 600
   :alt: Falcata logo.

|

Falcata documentation
=====================

**Falcata** is a CUDA-native gradient boosted decision tree library: a
leaf-wise learner whose training loop was rebuilt around batched,
level-parallel GPU kernels rather than one split at a time.

- Hybrid level-batched growth with leaf-wise-identical trees.
- Quantized training (``quant_mode``), an execution planner (``cuda_plan``),
  and GPU inference via NVIDIA FIL.
- Every optimization gated bit-identical against the reference path in CI.
- Interoperable with LightGBM at the data boundaries: models, binary
  datasets, and parameter names interchange in both directions.

For details, see `Features <./Features.rst>`__ and the
`README <https://github.com/MechaFauna-ai/Falcata#readme>`__.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation Guide <Installation-Guide>
   Quick Start <Quick-Start>
   Python Quick Start <Python-Intro>
   Features <Features>
   Parameters <Parameters>
   Parameters Tuning <Parameters-Tuning>
   C API <C-API>
   Python API <Python-API>
   R API <https://github.com/MechaFauna-ai/Falcata/tree/master/R-package#readme>
   Distributed Learning Guide <Parallel-Learning-Guide>
   Advanced Topics <Advanced-Topics>
   Development Guide <Development-Guide>

.. toctree::
   :hidden:

   README

Indices and Tables
==================

* :ref:`genindex`
