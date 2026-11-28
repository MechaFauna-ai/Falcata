/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#include "cuda_predictor.hpp"
#include "cuda_predictor_pairwise.cuh"
#include "cuda_predictor_raw.cuh"

#ifdef USE_CUDA

#include <algorithm>

namespace Falcata {

namespace {

using cuda_predictor_detail::Blocks;
using cuda_predictor_detail::FindLeaf;
using cuda_predictor_detail::GlobalRow;
using cuda_predictor_detail::kRowBlockSize;
using cuda_predictor_detail::PairwiseFinish;
using cuda_predictor_detail::PairwisePush;
using cuda_predictor_detail::Step;

// the spans' subtree sums of one (row, output), added by the same aligned tree
__global__ void PairwiseCombineKernel(const double* partials, int64_t cells, int num_spans, double* out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= cells) {
    return;
  }
  double stack[32];
  for (int s = 0; s < num_spans; ++s) {
    PairwisePush(stack, 1, static_cast<unsigned>(s), partials[static_cast<int64_t>(s) * cells + i]);
  }
  out[i] = PairwiseFinish(stack, 1, static_cast<unsigned>(num_spans));
}

template <typename TIn, typename TThr, bool HAS_CAT>
__global__ void PredictLeafKernel(const DeviceForestView<TThr> forest, const DeviceRows<TIn> rows, int tree_begin,
                                  int num_tree, int32_t* out) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows.num_row) {
    return;
  }
  const GlobalRow<TIn> row{rows.data + r * rows.row_stride, rows.num_col, rows.col_stride};
  for (int t = 0; t < num_tree; ++t) {
    const int tree = tree_begin + t;
    out[r * num_tree + t] = FindLeaf<TIn, TThr, HAS_CAT>(forest, row, tree) - forest.tree_leaf_begin[tree];
  }
}

// ------------------------------------------------------------------ feature contributions (TreeSHAP)
// Tree::TreeSHAP with an explicit stack instead of recursion, operation for operation: every product, quotient and
// sum is rounded on its own (the CPU build does not contract), in the CPU's order of evaluation, so the
// contributions are bit-identical to Tree::PredictContrib.

struct ShapPathElement {
  int feature_index;
  double zero_fraction;
  double one_fraction;
  double pweight;
};

struct ShapFrame {
  int node;
  int unique_depth;
  int64_t parent;   // offset of the parent's unique path in the thread's path buffer
  double zero_fraction;
  double one_fraction;
  int feature_index;
};

__device__ __forceinline__ double IntD(int v) { return static_cast<double>(v); }

__device__ void ShapExtendPath(ShapPathElement* p, int d, double zero_fraction, double one_fraction,
                               int feature_index) {
  p[d].feature_index = feature_index;
  p[d].zero_fraction = zero_fraction;
  p[d].one_fraction = one_fraction;
  p[d].pweight = (d == 0 ? 1.0 : 0.0);
  for (int i = d - 1; i >= 0; i--) {
    // pweight[i + 1] += one_fraction * pweight[i] * (i + 1) / (d + 1)
    p[i + 1].pweight = __dadd_rn(p[i + 1].pweight,
                                 __ddiv_rn(__dmul_rn(__dmul_rn(one_fraction, p[i].pweight), IntD(i + 1)), IntD(d + 1)));
    // pweight[i] = zero_fraction * pweight[i] * (d - i) / (d + 1)
    p[i].pweight = __ddiv_rn(__dmul_rn(__dmul_rn(zero_fraction, p[i].pweight), IntD(d - i)), IntD(d + 1));
  }
}

__device__ void ShapUnwindPath(ShapPathElement* p, int d, int path_index) {
  const double one_fraction = p[path_index].one_fraction;
  const double zero_fraction = p[path_index].zero_fraction;
  double next_one_portion = p[d].pweight;
  for (int i = d - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const double tmp = p[i].pweight;
      // next_one_portion * (d + 1) / ((i + 1) * one_fraction)
      p[i].pweight = __ddiv_rn(__dmul_rn(next_one_portion, IntD(d + 1)), __dmul_rn(IntD(i + 1), one_fraction));
      // tmp - pweight[i] * zero_fraction * (d - i) / (d + 1)
      next_one_portion = __dsub_rn(tmp, __ddiv_rn(__dmul_rn(__dmul_rn(p[i].pweight, zero_fraction), IntD(d - i)),
                                                  IntD(d + 1)));
    } else {
      // (pweight[i] * (d + 1)) / (zero_fraction * (d - i))
      p[i].pweight = __ddiv_rn(__dmul_rn(p[i].pweight, IntD(d + 1)), __dmul_rn(zero_fraction, IntD(d - i)));
    }
  }
  for (int i = path_index; i < d; ++i) {
    p[i].feature_index = p[i + 1].feature_index;
    p[i].zero_fraction = p[i + 1].zero_fraction;
    p[i].one_fraction = p[i + 1].one_fraction;
  }
}

__device__ double ShapUnwoundPathSum(const ShapPathElement* p, int d, int path_index) {
  const double one_fraction = p[path_index].one_fraction;
  const double zero_fraction = p[path_index].zero_fraction;
  double next_one_portion = p[d].pweight;
  double total = 0;
  for (int i = d - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      // tmp = next_one_portion * (d + 1) / ((i + 1) * one_fraction)
      const double tmp = __ddiv_rn(__dmul_rn(next_one_portion, IntD(d + 1)), __dmul_rn(IntD(i + 1), one_fraction));
      total = __dadd_rn(total, tmp);
      // pweight[i] - tmp * zero_fraction * ((d - i) / (d + 1))
      next_one_portion = __dsub_rn(p[i].pweight, __dmul_rn(__dmul_rn(tmp, zero_fraction),
                                                           __ddiv_rn(IntD(d - i), IntD(d + 1))));
    } else {
      // (pweight[i] / zero_fraction) / ((d - i) / (d + 1))
      total = __dadd_rn(total, __ddiv_rn(__ddiv_rn(p[i].pweight, zero_fraction), __ddiv_rn(IntD(d - i), IntD(d + 1))));
    }
  }
  return total;
}

template <typename TThr>
__device__ __forceinline__ int DataCount(const DeviceForestView<TThr>& forest, int node) {
  return node >= 0 ? forest.node_count[node] : forest.leaf_count[~node];
}

// one thread per (row, output): every tree of the output in iteration order, as GBDT::PredictContribInRange
template <typename TIn, typename TThr, bool HAS_CAT>
__global__ void PredictContribKernel(const DeviceForestView<TThr> forest, const DeviceRows<TIn> rows, int tree_begin,
                                     int num_iteration, int num_tree_per_iteration, int num_feature, double* out,
                                     ShapPathElement* paths, ShapFrame* frames, int64_t path_len, int stack_len) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows.num_row) {
    return;
  }
  const int k = blockIdx.y;
  const int K = num_tree_per_iteration;
  const int64_t slot = r * K + k;
  ShapPathElement* path = paths + slot * path_len;
  ShapFrame* stack = frames + slot * stack_len;
  double* phi = out + slot * (num_feature + 1);
  for (int f = 0; f <= num_feature; ++f) {
    phi[f] = 0.0;
  }
  const GlobalRow<TIn> row{rows.data + r * rows.row_stride, rows.num_col, rows.col_stride};
  for (int it = 0; it < num_iteration; ++it) {
    const int tree = tree_begin + it * K + k;
    phi[num_feature] = __dadd_rn(phi[num_feature], forest.tree_expected[tree]);
    const int root = forest.tree_root[tree];
    if (root < 0) {
      continue;  // a single leaf contributes its expected value only
    }
    int top = 0;
    stack[top++] = ShapFrame{root, 0, 0, 1.0, 1.0, -1};
    while (top > 0) {
      const ShapFrame fr = stack[--top];
      ShapPathElement* up = path + fr.parent + fr.unique_depth;
      int d = fr.unique_depth;
      if (d > 0) {
        for (int i = 0; i < d; ++i) {
          up[i] = path[fr.parent + i];
        }
      }
      ShapExtendPath(up, d, fr.zero_fraction, fr.one_fraction, fr.feature_index);
      if (fr.node < 0) {
        const double leaf_value = forest.leaf_values[~fr.node];
        for (int i = 1; i <= d; ++i) {
          const double w = ShapUnwoundPathSum(up, d, i);
          const ShapPathElement& el = up[i];
          // phi[feature] += w * (one_fraction - zero_fraction) * leaf_value
          phi[el.feature_index] = __dadd_rn(phi[el.feature_index],
              __dmul_rn(__dmul_rn(w, __dsub_rn(el.one_fraction, el.zero_fraction)), leaf_value));
        }
        continue;
      }
      const PackedNode<TThr> n = forest.nodes[fr.node];
      const int split_feature = static_cast<int>(n.feature_flags & kNodeFeatureMask);
      const int hot = Step<TIn, TThr, HAS_CAT>(forest, row, fr.node);
      const int cold = hot == n.left ? n.right : n.left;
      const double w = IntD(DataCount(forest, fr.node));
      const double hot_zero_fraction = __ddiv_rn(IntD(DataCount(forest, hot)), w);
      const double cold_zero_fraction = __ddiv_rn(IntD(DataCount(forest, cold)), w);
      double incoming_zero_fraction = 1;
      double incoming_one_fraction = 1;
      int path_index = 0;
      for (; path_index <= d; ++path_index) {
        if (up[path_index].feature_index == split_feature) break;
      }
      if (path_index != d + 1) {
        incoming_zero_fraction = up[path_index].zero_fraction;
        incoming_one_fraction = up[path_index].one_fraction;
        ShapUnwindPath(up, d, path_index);
        d -= 1;
      }
      const int64_t own = fr.parent + fr.unique_depth;
      // the hot child is walked first (pushed last), then the cold one, as the CPU recursion does
      stack[top++] = ShapFrame{cold, d + 1, own, __dmul_rn(cold_zero_fraction, incoming_zero_fraction), 0,
                               split_feature};
      stack[top++] = ShapFrame{hot, d + 1, own, __dmul_rn(hot_zero_fraction, incoming_zero_fraction),
                               incoming_one_fraction, split_feature};
    }
  }
}

}  // namespace

void LaunchPairwiseCombineKernel(const double* partials, int64_t cells, int num_spans, double* out,
                                 cudaStream_t stream) {
  PairwiseCombineKernel<<<Blocks(cells, 128), 128, 0, stream>>>(partials, cells, num_spans, out);
}

size_t TileWsSharedBytes(const PredictPlan& plan, int num_col, size_t input_elem, int num_tree_per_iteration) {
  size_t bytes = static_cast<size_t>(plan.tile_rows) * (2 * (plan.tile_trees + 1) + num_tree_per_iteration) *
                 sizeof(double);
  if (plan.stage_rows) {
    bytes += static_cast<size_t>(plan.tile_rows) * (num_col | 1) * input_elem;
  }
  return bytes;
}

size_t MaxSharedBytesPerBlock() {
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  int bytes = 0;
  CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  return static_cast<size_t>(bytes);
}

size_t PairwiseSharedBytes(const PairwisePlan& plan, int num_col, size_t input_elem) {
  const size_t pair_cap = static_cast<size_t>(plan.tile_rows) * plan.out_chunk;
  size_t bytes = (static_cast<size_t>(plan.wave_groups) * (pair_cap | 1) +
                  static_cast<size_t>(plan.wave_groups / plan.seg) * pair_cap +
                  static_cast<size_t>(plan.levels) * pair_cap) * sizeof(double);
  if (plan.stage_rows) {
    bytes += static_cast<size_t>(plan.tile_rows) * (std::max(num_col, plan.stage_cols) | 1) * input_elem;
  }
  return bytes;
}

template <typename TIn, typename TThr>
void LaunchPredictLeafKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                             int num_tree, int32_t* out, cudaStream_t stream) {
  if (rows.num_row == 0 || num_tree == 0) {
    return;
  }
  const dim3 grid(Blocks(rows.num_row, kRowBlockSize));
  if (forest.has_categorical) {
    PredictLeafKernel<TIn, TThr, true><<<grid, kRowBlockSize, 0, stream>>>(forest, rows, tree_begin, num_tree, out);
  } else {
    PredictLeafKernel<TIn, TThr, false><<<grid, kRowBlockSize, 0, stream>>>(forest, rows, tree_begin, num_tree, out);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

size_t ContribScratchBytes(int max_depth) {
  const int64_t path_len = static_cast<int64_t>(max_depth + 1) * (max_depth + 2) / 2 + 1;
  const int stack_len = 2 * (max_depth + 2);
  return static_cast<size_t>(path_len) * sizeof(ShapPathElement) + static_cast<size_t>(stack_len) * sizeof(ShapFrame);
}

template <typename TIn, typename TThr>
void LaunchPredictContribKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                                int num_iteration, int num_tree_per_iteration, int num_feature, double* out,
                                void* scratch, cudaStream_t stream) {
  if (rows.num_row == 0) {
    return;
  }
  const int64_t path_len = static_cast<int64_t>(forest.max_depth + 1) * (forest.max_depth + 2) / 2 + 1;
  const int stack_len = 2 * (forest.max_depth + 2);
  const int64_t slots = rows.num_row * num_tree_per_iteration;
  ShapPathElement* paths = reinterpret_cast<ShapPathElement*>(scratch);
  ShapFrame* frames = reinterpret_cast<ShapFrame*>(paths + slots * path_len);
  const dim3 grid(Blocks(rows.num_row, 128), num_tree_per_iteration);
  if (forest.has_categorical) {
    PredictContribKernel<TIn, TThr, true><<<grid, 128, 0, stream>>>(forest, rows, tree_begin, num_iteration,
        num_tree_per_iteration, num_feature, out, paths, frames, path_len, stack_len);
  } else {
    PredictContribKernel<TIn, TThr, false><<<grid, 128, 0, stream>>>(forest, rows, tree_begin, num_iteration,
        num_tree_per_iteration, num_feature, out, paths, frames, path_len, stack_len);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

template void LaunchPredictContribKernel<float, float>(const DeviceForestView<float>&, const DeviceRows<float>&, int,
                                                       int, int, int, double*, void*, cudaStream_t);
template void LaunchPredictContribKernel<double, double>(const DeviceForestView<double>&, const DeviceRows<double>&,
                                                         int, int, int, int, double*, void*, cudaStream_t);
template void LaunchPredictRawKernel<float, float>(const DeviceForestView<float>&, const DeviceRows<float>&, int, int,
                                                   int, int, bool, const PredictPlan&, double*, cudaStream_t);
template void LaunchPredictRawKernel<double, double>(const DeviceForestView<double>&, const DeviceRows<double>&, int,
                                                     int, int, int, bool, const PredictPlan&, double*, cudaStream_t);
template void LaunchPredictPairwiseJobs<float>(const PairwiseJob<float>*, int, const PairwiseJob<float>*, bool, bool,
                                               bool, size_t, cudaStream_t);
template void LaunchPredictPairwiseJobs<double>(const PairwiseJob<double>*, int, const PairwiseJob<double>*, bool, bool,
                                                bool, size_t, cudaStream_t);
template void LaunchPredictLeafKernel<float, float>(const DeviceForestView<float>&, const DeviceRows<float>&, int, int,
                                                    int32_t*, cudaStream_t);
template void LaunchPredictLeafKernel<double, double>(const DeviceForestView<double>&, const DeviceRows<double>&, int,
                                                      int, int32_t*, cudaStream_t);

}  // namespace Falcata

#endif  // USE_CUDA
