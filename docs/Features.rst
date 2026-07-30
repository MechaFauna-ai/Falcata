Features
========

This is a conceptual overview of how Falcata works. Falcata is a CUDA-native
gradient boosted decision tree library: it inherits LightGBM's
histogram-based, leaf-wise engine\ `[1] <#references>`__ and rebuilds the
training loop around batched, level-parallel GPU kernels. The first half of
this page covers what Falcata adds; the second half covers the shared engine
fundamentals. For measured numbers see the
`README <https://github.com/MechaFauna-ai/Falcata#what-makes-it-fast>`__; for
internals see ``docs/design/``.

The CUDA-Native Training Pipeline
---------------------------------

Everything below runs under ``device_type=cuda`` and is enabled by default;
each decision can be toggled individually through the execution planner (see
`The Execution Planner`_).

Hybrid Level-Batched Growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A classic leaf-wise learner performs one split at a time: construct
histograms, find the best split, synchronize with the host, apply, repeat.
On a GPU that loop is latency-bound -- the device idles between small
kernels. Falcata instead scores, synchronizes, and applies **whole levels of
sibling pairs in one launch each**, turning the loop throughput-bound while
producing **leaf-wise-identical trees**: while the leaf budget cannot bind,
every positive-gain frontier leaf would be split by leaf-wise growth
eventually, so batching a level changes the order of work, not the result.
In depth-limited configurations (``2^max_depth <= num_leaves + 1``) the
batched prefix is exact all the way down. In budget-limited configurations
(``num_leaves`` well below ``2^max_depth``) the *selective* grow-then-prune
mode preserves exact leaf-wise equivalence: levels are grown speculatively
and splits that global greedy ordering would not have taken are collapsed
again before they can influence the model.

A speculative single-sync pipeline (``one_sync``) further removes per-level
host synchronization for the non-quantized path, and a CUDA-graph level loop
(``graph_loop``) captures the per-level launch sequence once and replays it
through a device-side controller -- removing host round-trips entirely on
shallow trees.

NVRTC Runtime JIT
~~~~~~~~~~~~~~~~~

Histogram-construction kernels can be specialized at runtime (via NVRTC) to
the actual data shape -- bin counts, column layout -- instead of relying on
one generic kernel. A JIT kernel is self-tested against the ahead-of-time
kernel on real data and promoted **only if bit-identical**; otherwise the AOT
kernel keeps running. Opt-in via ``cuda_plan=auto,construct_jit:on``.

Per-Tree Compact Column View
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With any ``feature_fraction < 1``, each tree samples a column subset -- but a
row-major bin matrix makes skipping unused columns free of nothing: unread
columns still occupy the same memory sectors. Falcata gathers only the
sampled columns into a dense per-tree compact matrix, so every level pass
reads only what the tree can use. The win scales with the excluded fraction
(measured ~3.4x end-to-end at ``feature_fraction=0.1`` on wide
low-cardinality data, tapering to ~1.1x at 0.6).

GPU-Native Dataset Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dense binning, row-data build, and exclusive-feature-bundling prechecks run
on the device. CuPy arrays and anything exposing
``__cuda_array_interface__`` are ingested without a host round-trip. Row
data for low-cardinality features is packed 4-bit.

Quantized Training, Two Ways
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient quantization\ `[12] <#references>`__ shrinks histogram traffic by
accumulating small integers instead of floating-point pairs. Falcata ships
two modes:

- ``quant_mode=stochastic`` -- the speed end: gradients packed into 4 bins
  with seeded stochastic rounding (unbiased in expectation).
- ``quant_mode=fixedpoint`` -- the near-lossless end: deterministic rounding
  with an internal outlier-robust gradient scale, so rare huge gradients do
  not crush the quantization range.

Bin counts default to 4 and 64 respectively and can be overridden with
``quant_bins``. Both modes are bit-reproducible run to run;
``quant_mode=none`` is full precision. ``cuda_precision=fp32`` additionally
halves global-histogram bandwidth for the non-quantized path (quality-gated,
not bit-identical).

.. _The Execution Planner:

The Execution Planner
~~~~~~~~~~~~~~~~~~~~~

Every shape-conditional kernel choice above is resolved once from the data
and parameters by ``cuda_plan=auto`` -- not from environment variables or
hidden heuristics scattered through the code. Each decision is guaranteed
bit-identical (enforced by per-commit regression gates that flip every key
and require identical models) and individually overridable, e.g.
``cuda_plan=auto,hybrid:off,construct_jit:on``, which also makes every
feature's contribution measurable by leave-one-out ablation.

GPU Inference
~~~~~~~~~~~~~

With cuML installed, ``Booster.predict()`` transparently runs on NVIDIA's
Forest Inference Library; CuPy input stays on the device end to end. See the
README's `GPU inference section
<https://github.com/MechaFauna-ai/Falcata#gpu-inference-nvidia-fil>`__.

A legacy OpenCL backend (``device_type=gpu``, from upstream
LightGBM\ `[11] <#references>`__) still exists but is not developed here.

Shared Engine Fundamentals
--------------------------

The sections below describe the engine Falcata inherits from the LightGBM
lineage; they apply to CPU and CUDA training alike.

Optimization in Speed and Memory Usage
--------------------------------------

Many boosting tools use pre-sort-based algorithms\ `[2, 3] <#references>`__ (e.g. default algorithm in xgboost) for decision tree learning. It is a simple solution, but not easy to optimize.

Falcata uses histogram-based algorithms\ `[4, 5, 6] <#references>`__, which bucket continuous feature (attribute) values into discrete bins. This speeds up training and reduces memory usage. Advantages of histogram-based algorithms include the following:

-  **Reduced cost of calculating the gain for each split**

   -  Pre-sort-based algorithms have time complexity ``O(#data)``

   -  Computing the histogram has time complexity ``O(#data)``, but this involves only a fast sum-up operation. Once the histogram is constructed, a histogram-based algorithm has time complexity ``O(#bins)``, and ``#bins`` is far smaller than ``#data``.

-  **Use histogram subtraction for further speedup**

   -  To get one leaf's histograms in a binary tree, use the histogram subtraction of its parent and its neighbor

   -  So it needs to construct histograms for only one leaf (with smaller ``#data`` than its neighbor). It then can get histograms of its neighbor by histogram subtraction with small cost (``O(#bins)``)

-  **Reduce memory usage**

   -  Replaces continuous values with discrete bins. If ``#bins`` is small, can use small data type, e.g. uint8\_t, to store training data

   -  No need to store additional information for pre-sorting feature values

-  **Reduce communication cost for distributed learning**

Sparse Optimization
-------------------

-  Need only ``O(2 * #non_zero_data)`` to construct histogram for sparse features

Optimization in Accuracy
------------------------

Leaf-wise (Best-first) Tree Growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most decision tree learning algorithms grow trees by level (depth)-wise, like the following image:

.. image:: ./_static/images/level-wise.png
   :align: center
   :alt: A diagram depicting level wise tree growth in which the best possible node is split one level down. The strategy results in a symmetric tree, where every node in a level has child nodes resulting in an additional layer of depth.

Falcata grows trees leaf-wise (best-first)\ `[7] <#references>`__. It will choose the leaf with max delta loss to grow.
Holding ``#leaf`` fixed, leaf-wise algorithms tend to achieve lower loss than level-wise algorithms.

Leaf-wise may cause over-fitting when ``#data`` is small, so Falcata includes the ``max_depth`` parameter to limit tree depth. However, trees still grow leaf-wise even when ``max_depth`` is specified.

.. image:: ./_static/images/leaf-wise.png
   :align: center
   :alt: A diagram depicting leaf wise tree growth in which only the node with the highest loss change is split and not bother with the rest of the nodes in the same level. This results in an asymmetrical tree where subsequent splitting is happening only on one side of the tree.

Optimal Split for Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is common to represent categorical features with one-hot encoding, but this approach is suboptimal for tree learners. Particularly for high-cardinality categorical features, a tree built on one-hot features tends to be unbalanced and needs to grow very deep to achieve good accuracy.

Instead of one-hot encoding, the optimal solution is to split on a categorical feature by partitioning its categories into 2 subsets. If the feature has ``k`` categories, there are ``2^(k-1) - 1`` possible partitions.
But there is an efficient solution for regression trees\ `[8] <#references>`__. It needs about ``O(k * log(k))`` to find the optimal partition.

The basic idea is to sort the categories according to the training objective at each split.
More specifically, Falcata sorts the histogram (for a categorical feature) according to its accumulated values (``sum_gradient / sum_hessian``) and then finds the best split on the sorted histogram.

Optimization in Network Communication
-------------------------------------

It only needs to use some collective communication algorithms, like "All reduce", "All gather" and "Reduce scatter", in distributed learning of Falcata.
Falcata implements state-of-the-art algorithms\ `[9] <#references>`__.
These collective communication algorithms can provide much better performance than point-to-point communication.

.. _Optimization in Parallel Learning:

Optimization in Distributed Learning
------------------------------------

Falcata provides the following distributed learning algorithms.

Feature Parallel
~~~~~~~~~~~~~~~~

Traditional Algorithm
^^^^^^^^^^^^^^^^^^^^^

Feature parallel aims to parallelize the "Find Best Split" in the decision tree. The procedure of traditional feature parallel is:

1. Partition data vertically (different machines have different feature set).

2. Workers find local best split point {feature, threshold} on local feature set.

3. Communicate local best splits with each other and get the best one.

4. Worker with best split to perform split, then send the split result of data to other workers.

5. Other workers split data according to received data.

The shortcomings of traditional feature parallel:

-  Has computation overhead, since it cannot speed up "split", whose time complexity is ``O(#data)``.
   Thus, feature parallel cannot speed up well when ``#data`` is large.

-  Need communication of split result, which costs about ``O(#data / 8)`` (one bit for one data).

Feature Parallel in Falcata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since feature parallel cannot speed up well when ``#data`` is large, we make a little change: instead of partitioning data vertically, every worker holds the full data.
Thus, Falcata doesn't need to communicate for split result of data since every worker knows how to split data.
And ``#data`` won't be larger, so it is reasonable to hold the full data in every machine.

The procedure of feature parallel in Falcata:

1. Workers find local best split point {feature, threshold} on local feature set.

2. Communicate local best splits with each other and get the best one.

3. Perform best split.

However, this feature parallel algorithm still suffers from computation overhead for "split" when ``#data`` is large.
So it will be better to use data parallel when ``#data`` is large.

Data Parallel
~~~~~~~~~~~~~

Traditional Algorithm
^^^^^^^^^^^^^^^^^^^^^

Data parallel aims to parallelize the whole decision learning. The procedure of data parallel is:

1. Partition data horizontally.

2. Workers use local data to construct local histograms.

3. Merge global histograms from all local histograms.

4. Find best split from merged global histograms, then perform splits.

The shortcomings of traditional data parallel:

-  High communication cost.
   If using point-to-point communication algorithm, communication cost for one machine is about ``O(#machine * #feature * #bin)``.
   If using collective communication algorithm (e.g. "All Reduce"), communication cost is about ``O(2 * #feature * #bin)`` (check cost of "All Reduce" in chapter 4.5 at `[9] <#references>`__).

Data Parallel in Falcata
^^^^^^^^^^^^^^^^^^^^^^^^^

We reduce communication cost of data parallel in Falcata:

1. Instead of "Merge global histograms from all local histograms", Falcata uses "Reduce Scatter" to merge histograms of different (non-overlapping) features for different workers.
   Then workers find the local best split on local merged histograms and sync up the global best split.

2. As aforementioned, Falcata uses histogram subtraction to speed up training.
   Based on this, we can communicate histograms only for one leaf, and get its neighbor's histograms by subtraction as well.

All things considered, data parallel in Falcata has time complexity ``O(0.5 * #feature * #bin)``.

Voting Parallel
~~~~~~~~~~~~~~~

Voting parallel further reduces the communication cost in `Data Parallel <#data-parallel>`__ to constant cost.
It uses two-stage voting to reduce the communication cost of feature histograms\ `[10] <#references>`__.

Applications and Metrics
------------------------

Falcata supports the following applications:

-  regression, the objective function is L2 loss

-  binary classification, the objective function is logloss

-  multi classification

-  cross-entropy, the objective function is logloss and supports training on non-binary labels

-  LambdaRank, the objective function is LambdaRank with NDCG

Falcata supports the following metrics:

-  L1 loss

-  L2 loss

-  Log loss

-  Classification error rate

-  AUC

-  NDCG

-  MAP

-  Multi-class log loss

-  Multi-class error rate

-  AUC-mu ``(new in v3.0.0)``

-  Average precision ``(new in v3.1.0)``

-  Fair

-  Huber

-  Poisson

-  Quantile

-  MAPE

-  Kullback-Leibler

-  Gamma

-  Tweedie

For more details, please refer to `Parameters <./Parameters.rst#metric-parameters>`__.

Other Features
--------------

-  Limit ``max_depth`` of tree while grows tree leaf-wise

-  `DART <https://arxiv.org/abs/1505.01866>`__

-  L1/L2 regularization

-  Bagging

-  Column (feature) sub-sample

-  Continued train with input GBDT model

-  Continued train with the input score file

-  Weighted training

-  Validation metric output during training

-  Multiple validation data

-  Multiple metrics

-  Early stopping (both training and prediction)

-  Prediction for leaf index

For more details, please refer to `Parameters <./Parameters.rst>`__.

References
----------

[1] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "`LightGBM: A Highly Efficient Gradient Boosting Decision Tree`_." Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

[2] Mehta, Manish, Rakesh Agrawal, and Jorma Rissanen. "SLIQ: A fast scalable classifier for data mining." International Conference on Extending Database Technology. Springer Berlin Heidelberg, 1996.

[3] Shafer, John, Rakesh Agrawal, and Manish Mehta. "SPRINT: A scalable parallel classifier for data mining." Proc. 1996 Int. Conf. Very Large Data Bases. 1996.

[4] Ranka, Sanjay, and V. Singh. "CLOUDS: A decision tree classifier for large datasets." Proceedings of the 4th Knowledge Discovery and Data Mining Conference. 1998.

[5] Machado, F. P. "Communication and memory efficient parallel decision tree construction." (2003).

[6] Li, Ping, Qiang Wu, and Christopher J. Burges. "Mcrank: Learning to rank using multiple classification and gradient boosting." Advances in Neural Information Processing Systems 20 (NIPS 2007).

[7] Shi, Haijian. "Best-first decision tree learning." Diss. The University of Waikato, 2007.

[8] Walter D. Fisher. "`On Grouping for Maximum Homogeneity`_." Journal of the American Statistical Association. Vol. 53, No. 284 (Dec., 1958), pp. 789-798.

[9] Thakur, Rajeev, Rolf Rabenseifner, and William Gropp. "`Optimization of collective communication operations in MPICH`_." International Journal of High Performance Computing Applications 19.1 (2005), pp. 49-66.

[10] Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. "`A Communication-Efficient Parallel Algorithm for Decision Tree`_." Advances in Neural Information Processing Systems 29 (NIPS 2016), pp. 1279-1287.

[11] Huan Zhang, Si Si and Cho-Jui Hsieh. "`GPU Acceleration for Large-scale Tree Boosting`_." SysML Conference, 2018.

[12] Yu Shi, Guolin Ke, Zhuoming Chen, Shuxin Zheng, Tie-Yan Liu. "`Quantized Training of Gradient Boosting Decision Trees`_." Advances in Neural Information Processing Systems 35 (NeurIPS 2022).

.. _Quantized Training of Gradient Boosting Decision Trees: https://papers.nips.cc/paper_files/paper/2022/hash/77911ed9e6e864ca1a3d165b2c3cb258-Abstract-Conference.html

.. _LightGBM\: A Highly Efficient Gradient Boosting Decision Tree: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

.. _On Grouping for Maximum Homogeneity: https://www.jstor.org/stable/2281952

.. _Optimization of collective communication operations in MPICH: https://www.mpich.org/2012/10/24/optimization-of-collective-communication-operations-in-mpich/

.. _A Communication-Efficient Parallel Algorithm for Decision Tree: https://proceedings.neurips.cc/paper/2016/hash/10a5ab2db37feedfdeaab192ead4ac0e-Abstract.html

.. _GPU Acceleration for Large-scale Tree Boosting: https://arxiv.org/abs/1706.08359
