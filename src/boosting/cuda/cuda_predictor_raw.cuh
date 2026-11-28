/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_RAW_CUH_
#define FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_RAW_CUH_

#include "cuda_predictor_common.cuh"

#ifdef USE_CUDA

#include <algorithm>

namespace Falcata {

namespace cuda_predictor_detail {

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

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR>
static void LaunchRow(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin, int num_iteration,
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
static void LaunchTileWsIlp(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                     int num_iteration, int num_tree_per_iteration, const PredictPlan& plan, double* out,
                     cudaStream_t stream) {
  const dim3 grid(Blocks(rows.num_row, plan.tile_rows));
  const size_t shared = TileWsSharedBytes(plan, rows.num_col, sizeof(TIn), num_tree_per_iteration);
#define FLC_TWS_KERNEL(U)                                                                                      \
  {                                                                                                            \
    auto kernel = PredictTileWsKernel<TIn, TThr, HAS_CAT, HAS_LINEAR, STAGE, U>;                               \
    CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,             \
                                              static_cast<int>(shared)));                                      \
    kernel<<<grid, plan.block_threads, shared, stream>>>(forest, rows, tree_begin, num_iteration,              \
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
static void LaunchScalar(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
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

}  // namespace cuda_predictor_detail

template <typename TIn, typename TThr>
void LaunchPredictRawKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                            int num_iteration, int num_tree_per_iteration, int num_output, bool vector_leaf,
                            const PredictPlan& plan, double* out, cudaStream_t stream) {
  using cuda_predictor_detail::Blocks;
  using cuda_predictor_detail::kRowBlockSize;
  using cuda_predictor_detail::LaunchScalar;
  using cuda_predictor_detail::PredictRawVectorKernel;
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

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_RAW_CUH_
