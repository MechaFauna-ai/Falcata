/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_COMMON_CUH_
#define FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_COMMON_CUH_

#include "cuda_predictor.hpp"

#ifdef USE_CUDA

#include <cstdint>

namespace Falcata {

void LaunchPairwiseCombineKernel(const double* partials, int64_t cells, int num_spans, double* out,
                                 cudaStream_t stream);

namespace cuda_predictor_detail {

constexpr int kRowBlockSize = 256;
constexpr int kMissingZero = 1;   // MissingType::Zero
constexpr int kMissingNaN = 2;    // MissingType::NaN

__device__ __forceinline__ uint32_t ThresholdBits(float t) { return __float_as_uint(t); }
__device__ __forceinline__ uint32_t ThresholdBits(double t) { return static_cast<uint32_t>(__double2loint(t)); }

// RowPairFunctionFromDenseMatrix keeps a value only if |x| > kZeroThreshold (1e-35f) or x is NaN: the CPU predictor
// sees every other value as 0
__device__ __forceinline__ float SnapZero(float x) { return fabsf(x) <= 1e-35f ? 0.0f : x; }
__device__ __forceinline__ double SnapZero(double x) {
  return fabs(x) <= static_cast<double>(1e-35f) ? 0.0 : x;
}

/*! \brief reads features of one row from global memory */
template <typename TIn>
struct GlobalRow {
  const TIn* base;
  int32_t num_col;
  int64_t col_stride;
  __device__ __forceinline__ TIn operator()(uint32_t feature) const {
    // with predict_disable_shape_check a model feature past the input's columns is 0, as in the CPU buffer
    if (feature >= static_cast<uint32_t>(num_col)) {
      return TIn(0);
    }
    const TIn v = col_stride == 1 ? __ldg(base + feature) : __ldg(base + static_cast<int64_t>(feature) * col_stride);
    return SnapZero(v);
  }
};

/*! \brief reads features of one row staged in shared memory over at least the model's features (no bounds check) */
template <typename TIn>
struct StagedRow {
  const TIn* base;
  __device__ __forceinline__ TIn operator()(uint32_t feature) const { return base[feature]; }
};

/*! \brief reads features of one row staged (already zero-snapped, zero past the input's columns) in shared memory */
template <typename TIn>
struct SharedRow {
  const TIn* base;
  int32_t num_col;
  __device__ __forceinline__ TIn operator()(uint32_t feature) const {
    return feature < static_cast<uint32_t>(num_col) ? base[feature] : TIn(0);
  }
};

// Tree::CategoricalDecision on the double value
template <typename TThr>
__device__ __forceinline__ int CategoricalNext(const PackedNode<TThr>& node, double v,
                                               const DeviceForestView<TThr>& forest) {
  if (!(v >= -2147483648.0 && v <= 2147483647.0)) {
    return node.right;
  }
  const int iv = static_cast<int>(v);
  if (iv < 0) {
    return node.right;
  }
  const uint32_t range = ThresholdBits(node.threshold);
  const int64_t word_begin = forest.cat_range[2 * static_cast<int64_t>(range)];
  const int64_t word_count = forest.cat_range[2 * static_cast<int64_t>(range) + 1];
  const int64_t word = iv / 32;
  if (word < word_count && ((forest.cat_words[word_begin + word] >> (iv % 32)) & 1u)) {
    return node.left;
  }
  return node.right;
}

// Tree::NumericalDecision; x is already zero-snapped, so IsZero(x) is x == 0
template <typename TIn, typename TThr>
__device__ __forceinline__ int NumericalNext(const PackedNode<TThr>& node, TIn x) {
  const uint32_t ff = node.feature_flags;
  const int missing = static_cast<int>((ff >> kNodeMissingShift) & 3u);
  if (isnan(x)) {
    if (missing == kMissingNaN || missing == kMissingZero) {
      return (ff & kNodeDefaultLeft) ? node.left : node.right;
    }
    x = TIn(0);
  } else if (missing == kMissingZero && x == TIn(0)) {
    return (ff & kNodeDefaultLeft) ? node.left : node.right;
  }
  return x <= node.threshold ? node.left : node.right;
}

template <typename TIn, typename TThr, bool HAS_CAT, typename Row>
__device__ __forceinline__ int Step(const DeviceForestView<TThr>& forest, const Row& row, int node) {
  const PackedNode<TThr> n = forest.nodes[node];
  const TIn x = row(n.feature_flags & kNodeFeatureMask);
  if (HAS_CAT && (n.feature_flags & kNodeCategorical)) {
    return CategoricalNext(n, static_cast<double>(x), forest);
  }
  return NumericalNext<TIn, TThr>(n, x);
}

/*! \brief the global leaf the row reaches in a tree */
template <typename TIn, typename TThr, bool HAS_CAT, typename Row>
__device__ __forceinline__ int FindLeaf(const DeviceForestView<TThr>& forest, const Row& row, int tree) {
  int node = forest.tree_root[tree];
  while (node >= 0) {
    node = Step<TIn, TThr, HAS_CAT>(forest, row, node);
  }
  return ~node;
}

// Tree::Predict for a linear tree: const + sum coeff_i * x_i in order, or the leaf value when an input is not finite
template <typename TIn, typename TThr, typename Row>
__device__ __forceinline__ double LinearLeafValue(const DeviceForestView<TThr>& forest, const Row& row, int leaf) {
  double out = forest.linear_const[leaf];
  const int64_t end = forest.linear_begin[leaf + 1];
  for (int64_t i = forest.linear_begin[leaf]; i < end; ++i) {
    const double v = static_cast<double>(row(static_cast<uint32_t>(forest.linear_feature[i])));
    if (!isfinite(static_cast<float>(v))) {
      return forest.leaf_values[leaf];
    }
    out = __dadd_rn(out, __dmul_rn(forest.linear_coeff[i], v));
  }
  return out;
}

template <typename TIn, typename TThr, bool HAS_LINEAR, typename Row>
__device__ __forceinline__ double LeafValue(const DeviceForestView<TThr>& forest, const Row& row, int tree, int leaf) {
  if (HAS_LINEAR && forest.tree_is_linear[tree]) {
    return LinearLeafValue<TIn, TThr>(forest, row, leaf);
  }
  return forest.leaf_values[leaf];
}

// adds element `index` (0-based, in order) to a binary-counter stack of aligned subtree sums (entry l at stack[l*stride])
__device__ __forceinline__ void PairwisePush(double* stack, int stride, unsigned index, double x) {
  int level = 0;
  while ((index >> level) & 1u) {
    x = __dadd_rn(stack[level * stride], x);
    ++level;
  }
  stack[level * stride] = x;
}

// the root of the aligned tree over the `count` elements pushed (missing ones count as 0.0, which changes nothing)
__device__ __forceinline__ double PairwiseFinish(const double* stack, int stride, unsigned count) {
  double acc = 0.0;
  bool have = false;
  for (int level = 0; (count >> level) != 0u; ++level) {
    if ((count >> level) & 1u) {
      acc = have ? __dadd_rn(stack[level * stride], acc) : stack[level * stride];
      have = true;
    }
  }
  return acc;
}

inline unsigned int Blocks(int64_t n, int per_block) {
  return static_cast<unsigned int>((n + per_block - 1) / per_block);
}

}  // namespace cuda_predictor_detail

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_COMMON_CUH_
