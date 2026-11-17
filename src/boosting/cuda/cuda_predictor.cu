/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#include "cuda_predictor.hpp"

#ifdef USE_CUDA

#include <algorithm>

namespace Falcata {

namespace {

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
    return SnapZero(base[static_cast<int64_t>(feature) * col_stride]);
  }
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

// ------------------------------------------------------------------ kRow: a thread per row, ILP trees in flight

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, int ILP>
__global__ void PredictRowKernel(const DeviceForestView<TThr> forest, const DeviceRows<TIn> rows, int tree_begin,
                                 int num_iteration, int num_tree_per_iteration, double* out) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows.num_row) {
    return;
  }
  const GlobalRow<TIn> row{rows.data + r * rows.row_stride, rows.num_col, rows.col_stride};
  const int k = blockIdx.y;  // output k sums tree k of every iteration, in iteration order
  const int stride = num_tree_per_iteration;
  double acc = 0.0;
  int it = 0;
  for (; it + ILP <= num_iteration; it += ILP) {
    int node[ILP];
    #pragma unroll
    for (int u = 0; u < ILP; ++u) {
      node[u] = forest.tree_root[tree_begin + (it + u) * stride + k];
    }
    // the walks are independent: their loads are in flight together
    bool active = true;
    while (active) {
      active = false;
      #pragma unroll
      for (int u = 0; u < ILP; ++u) {
        if (node[u] >= 0) {
          node[u] = Step<TIn, TThr, HAS_CAT>(forest, row, node[u]);
          active |= node[u] >= 0;
        }
      }
    }
    #pragma unroll
    for (int u = 0; u < ILP; ++u) {
      acc = __dadd_rn(acc, LeafValue<TIn, TThr, HAS_LINEAR>(forest, row, tree_begin + (it + u) * stride + k,
                                                            ~node[u]));
    }
  }
  for (; it < num_iteration; ++it) {
    const int tree = tree_begin + it * stride + k;
    acc = __dadd_rn(acc, LeafValue<TIn, TThr, HAS_LINEAR>(forest, row, tree,
                                                          FindLeaf<TIn, TThr, HAS_CAT>(forest, row, tree)));
  }
  out[r * stride + k] = acc;
}

// ------------------------------------------------------------------ kTileWs: warp-specialized tiles
// Warp 0 keeps each row's running sums (lane r sums row r, tree by tree, so the order is the CPU's) while the other
// warps walk the next tile of trees: the tile's leaf values are double-buffered, one __syncthreads per tile. Rows sit in
// shared memory with an odd stride (lanes reading one feature of different rows hit different banks).

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, bool STAGE, int ILP>
__global__ void PredictTileWsKernel(const DeviceForestView<TThr> forest, const DeviceRows<TIn> rows, int tree_begin,
                                    int num_iteration, int num_tree_per_iteration, int tile_rows, int tile_trees,
                                    double* out) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int K = num_tree_per_iteration;
  const int vals_stride = tile_trees + 1;
  double* vals = reinterpret_cast<double*>(smem);                          // [2][tile_rows][tile_trees + 1]
  double* accs = vals + 2 * static_cast<size_t>(tile_rows) * vals_stride;  // [tile_rows][K]
  TIn* xs = reinterpret_cast<TIn*>(accs + static_cast<size_t>(tile_rows) * K);
  const int num_col = rows.num_col;
  const int x_stride = num_col | 1;
  const int64_t row0 = static_cast<int64_t>(blockIdx.x) * tile_rows;
  if (STAGE) {
    const int64_t total = static_cast<int64_t>(tile_rows) * num_col;
    const bool column_major = rows.row_stride == 1 && rows.col_stride != 1;
    for (int64_t i = threadIdx.x; i < total; i += blockDim.x) {
      const int64_t rr = column_major ? i % tile_rows : i / num_col;
      const int f = static_cast<int>(column_major ? i / tile_rows : i - rr * num_col);
      const int64_t r = row0 + rr;
      xs[rr * x_stride + f] = r < rows.num_row
          ? SnapZero(rows.data[r * rows.row_stride + static_cast<int64_t>(f) * rows.col_stride]) : TIn(0);
    }
  }
  for (int i = threadIdx.x; i < tile_rows * K; i += blockDim.x) {
    accs[i] = 0.0;
  }
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int num_tree = num_iteration * K;
  const int steps = (num_tree + tile_trees - 1) / tile_trees;
  const int workers = blockDim.x - 32;
  const int64_t valid_rows = min(static_cast<int64_t>(tile_rows), rows.num_row - row0);
  for (int step = 0; step <= steps; ++step) {
    if (warp == 0) {
      if (step > 0) {
        const int t0 = (step - 1) * tile_trees;
        const int count = min(tile_trees, num_tree - t0);
        const double* buf = vals + ((step - 1) & 1) * static_cast<size_t>(tile_rows) * vals_stride;
        for (int r = lane; r < valid_rows; r += 32) {
          const double* v = buf + r * vals_stride;
          double* acc = accs + r * K;
          if (K == 1) {
            double a = acc[0];
            for (int c = 0; c < count; ++c) {
              a = __dadd_rn(a, v[c]);
            }
            acc[0] = a;
          } else {
            int k = t0 % K;
            for (int c = 0; c < count; ++c) {
              acc[k] = __dadd_rn(acc[k], v[c]);
              k = k + 1 == K ? 0 : k + 1;
            }
          }
        }
      }
    } else if (step < steps) {
      const int t0 = step * tile_trees;
      const int count = min(tile_trees, num_tree - t0);
      double* buf = vals + (step & 1) * static_cast<size_t>(tile_rows) * vals_stride;
      const int pairs = static_cast<int>(valid_rows) * count;
      // pair p: row p % valid_rows of tree p / valid_rows (a warp's lanes share trees: node reads broadcast)
      for (int p0 = threadIdx.x - 32; p0 < pairs; p0 += workers * ILP) {
        int node[ILP], rr[ILP], cc[ILP];
        #pragma unroll
        for (int u = 0; u < ILP; ++u) {
          const int p = p0 + u * workers;
          if (p < pairs) {
            rr[u] = p % static_cast<int>(valid_rows);
            cc[u] = p / static_cast<int>(valid_rows);
            node[u] = forest.tree_root[tree_begin + t0 + cc[u]];
          } else {
            rr[u] = -1;
            cc[u] = 0;
            node[u] = -1;
          }
        }
        bool active = true;
        while (active) {
          active = false;
          #pragma unroll
          for (int u = 0; u < ILP; ++u) {
            if (node[u] >= 0) {
              if (STAGE) {
                const SharedRow<TIn> row{xs + static_cast<int64_t>(rr[u]) * x_stride, num_col};
                node[u] = Step<TIn, TThr, HAS_CAT>(forest, row, node[u]);
              } else {
                const GlobalRow<TIn> row{rows.data + (row0 + rr[u]) * rows.row_stride, num_col, rows.col_stride};
                node[u] = Step<TIn, TThr, HAS_CAT>(forest, row, node[u]);
              }
              active |= node[u] >= 0;
            }
          }
        }
        #pragma unroll
        for (int u = 0; u < ILP; ++u) {
          if (rr[u] >= 0) {
            const int tree = tree_begin + t0 + cc[u];
            double v;
            if (STAGE) {
              const SharedRow<TIn> row{xs + static_cast<int64_t>(rr[u]) * x_stride, num_col};
              v = LeafValue<TIn, TThr, HAS_LINEAR>(forest, row, tree, ~node[u]);
            } else {
              const GlobalRow<TIn> row{rows.data + (row0 + rr[u]) * rows.row_stride, num_col, rows.col_stride};
              v = LeafValue<TIn, TThr, HAS_LINEAR>(forest, row, tree, ~node[u]);
            }
            buf[rr[u] * vals_stride + cc[u]] = v;
          }
        }
      }
    }
    __syncthreads();
  }
  if (warp == 0) {
    for (int r = lane; r < valid_rows; r += 32) {
      for (int k = 0; k < K; ++k) {
        out[(row0 + r) * K + k] = accs[r * K + k];
      }
    }
  }
}

// ------------------------------------------------------------------ vector leaves and leaf indices

// vector leaves: each tree adds leaf_dim values; the row's outputs are summed in place, tree by tree
template <typename TIn, typename TThr, bool HAS_CAT>
__global__ void PredictRawVectorKernel(const DeviceForestView<TThr> forest, const DeviceRows<TIn> rows,
                                       int tree_begin, int num_tree, int num_output, double* out) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows.num_row) {
    return;
  }
  const GlobalRow<TIn> row{rows.data + r * rows.row_stride, rows.num_col, rows.col_stride};
  double* row_out = out + r * num_output;
  for (int k = 0; k < num_output; ++k) {
    row_out[k] = 0.0;
  }
  for (int t = 0; t < num_tree; ++t) {
    const int leaf = FindLeaf<TIn, TThr, HAS_CAT>(forest, row, tree_begin + t);
    const double* values = forest.leaf_values + static_cast<int64_t>(leaf) * forest.leaf_dim;
    for (int k = 0; k < num_output; ++k) {
      row_out[k] = __dadd_rn(row_out[k], values[k]);
    }
  }
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

inline unsigned int Blocks(int64_t n, int per_block) {
  return static_cast<unsigned int>((n + per_block - 1) / per_block);
}

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR>
void LaunchRow(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin, int num_iteration,
               int num_tree_per_iteration, int ilp, double* out, cudaStream_t stream) {
  const dim3 grid(Blocks(rows.num_row, kRowBlockSize), num_tree_per_iteration);
#define FLC_ROW_KERNEL(U)                                                                                      \
  PredictRowKernel<TIn, TThr, HAS_CAT, HAS_LINEAR, U><<<grid, kRowBlockSize, 0, stream>>>(                     \
      forest, rows, tree_begin, num_iteration, num_tree_per_iteration, out)
  if (ilp >= 8) {
    FLC_ROW_KERNEL(8);
  } else if (ilp >= 4) {
    FLC_ROW_KERNEL(4);
  } else if (ilp >= 2) {
    FLC_ROW_KERNEL(2);
  } else {
    FLC_ROW_KERNEL(1);
  }
#undef FLC_ROW_KERNEL
}

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, bool STAGE>
void LaunchTileWsIlp(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                     int num_iteration, int num_tree_per_iteration, const PredictPlan& plan, double* out,
                     cudaStream_t stream) {
  const dim3 grid(Blocks(rows.num_row, plan.tile_rows));
  const size_t shared = TileWsSharedBytes(plan, rows.num_col, sizeof(TIn), num_tree_per_iteration);
#define FLC_TWS_KERNEL(U)                                                                                      \
  {                                                                                                            \
    auto kernel = PredictTileWsKernel<TIn, TThr, HAS_CAT, HAS_LINEAR, STAGE, U>;                               \
    CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,             \
                                              static_cast<int>(shared)));                                      \
    kernel<<<grid, plan.block_threads, shared, stream>>>(forest, rows, tree_begin, num_iteration,             \
                                                         num_tree_per_iteration, plan.tile_rows,               \
                                                         plan.tile_trees, out);                                \
  }
  if (plan.ilp >= 4) {
    FLC_TWS_KERNEL(4)
  } else if (plan.ilp >= 2) {
    FLC_TWS_KERNEL(2)
  } else {
    FLC_TWS_KERNEL(1)
  }
#undef FLC_TWS_KERNEL
}

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR>
void LaunchScalar(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                  int num_iteration, int num_tree_per_iteration, const PredictPlan& plan, double* out,
                  cudaStream_t stream) {
  if (plan.kernel == PredictPlan::kTileWs) {
    if (plan.stage_rows) {
      LaunchTileWsIlp<TIn, TThr, HAS_CAT, HAS_LINEAR, true>(forest, rows, tree_begin, num_iteration,
                                                            num_tree_per_iteration, plan, out, stream);
    } else {
      LaunchTileWsIlp<TIn, TThr, HAS_CAT, HAS_LINEAR, false>(forest, rows, tree_begin, num_iteration,
                                                             num_tree_per_iteration, plan, out, stream);
    }
  } else {
    LaunchRow<TIn, TThr, HAS_CAT, HAS_LINEAR>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration,
                                              plan.ilp, out, stream);
  }
}

}  // namespace

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

template <typename TIn, typename TThr>
void LaunchPredictRawKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                            int num_iteration, int num_tree_per_iteration, int num_output, bool vector_leaf,
                            const PredictPlan& plan, double* out, cudaStream_t stream) {
  if (rows.num_row == 0) {
    return;
  }
  if (vector_leaf) {
    const dim3 grid(Blocks(rows.num_row, kRowBlockSize));
    if (forest.has_categorical) {
      PredictRawVectorKernel<TIn, TThr, true><<<grid, kRowBlockSize, 0, stream>>>(
          forest, rows, tree_begin, num_iteration, num_output, out);
    } else {
      PredictRawVectorKernel<TIn, TThr, false><<<grid, kRowBlockSize, 0, stream>>>(
          forest, rows, tree_begin, num_iteration, num_output, out);
    }
  } else if (num_iteration == 0) {
    CUDASUCCESS_OR_FATAL(cudaMemsetAsync(out, 0, sizeof(double) * rows.num_row * num_output, stream));
  } else if (forest.has_categorical) {
    if (forest.has_linear) {
      LaunchScalar<TIn, TThr, true, true>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan, out,
                                          stream);
    } else {
      LaunchScalar<TIn, TThr, true, false>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan,
                                           out, stream);
    }
  } else {
    if (forest.has_linear) {
      LaunchScalar<TIn, TThr, false, true>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan,
                                           out, stream);
    } else {
      LaunchScalar<TIn, TThr, false, false>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan,
                                            out, stream);
    }
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
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
template void LaunchPredictLeafKernel<float, float>(const DeviceForestView<float>&, const DeviceRows<float>&, int, int,
                                                    int32_t*, cudaStream_t);
template void LaunchPredictLeafKernel<double, double>(const DeviceForestView<double>&, const DeviceRows<double>&, int,
                                                      int, int32_t*, cudaStream_t);

}  // namespace Falcata

#endif  // USE_CUDA
