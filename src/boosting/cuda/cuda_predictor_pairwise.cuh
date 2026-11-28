/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_PAIRWISE_CUH_
#define FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_PAIRWISE_CUH_

#include "cuda_predictor_common.cuh"

#ifdef USE_CUDA

#include <algorithm>
#include <limits>
#include <utility>

namespace Falcata {

namespace cuda_predictor_detail {

// ------------------------------------------------------------------ kPairwise: the default mode
// The canonical pairwise sum (PairwisePlan): in-order sums of groups of kPairwiseGroupIterations iterations, then the
// complete binary tree over the group index. Blocks compute aligned subtrees of that tree, so the bits do not depend on
// how rows, spans, waves and segments are cut.

// the in-order sum (from 0.0) of the leaf values of `count` trees tree0, tree0 + stride, ...; ILP walks in flight
template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, int ILP, typename Row>
__device__ __forceinline__ double GroupSum(const DeviceForestView<TThr>& forest, const Row& row, int tree0, int stride,
                                           int count) {
  double sum = 0.0;
  for (int j0 = 0; j0 < count; j0 += ILP) {
    int node[ILP];
    #pragma unroll
    for (int u = 0; u < ILP; ++u) {
      node[u] = j0 + u < count ? forest.tree_root[tree0 + (j0 + u) * stride] : -1;
    }
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
      if (j0 + u < count) {
        sum = __dadd_rn(sum, LeafValue<TIn, TThr, HAS_LINEAR>(forest, row, tree0 + (j0 + u) * stride, ~node[u]));
      }
    }
  }
  return sum;
}

// ---- the compact layout (CompactSlot): the walk reads one slot per level, and the slot that reaches a leaf is its value

__device__ __forceinline__ float SlotThreshold(const CompactSlot<float>& s) { return __uint_as_float(s.lo); }
__device__ __forceinline__ double SlotThreshold(const CompactSlot<double>& s) { return s.value; }
__device__ __forceinline__ uint32_t SlotRange(const CompactSlot<float>& s) { return s.lo; }
__device__ __forceinline__ uint32_t SlotRange(const CompactSlot<double>& s) {
  return static_cast<uint32_t>(__double2loint(s.value));
}
__device__ __forceinline__ double SlotValue(const CompactSlot<float>& s) {
  return __hiloint2double(static_cast<int>(s.meta), static_cast<int>(s.lo));
}
__device__ __forceinline__ double SlotValue(const CompactSlot<double>& s) { return s.value; }

__device__ __forceinline__ CompactSlot<float> LoadSlot(const CompactSlot<float>* slots, uint32_t pos) {
  const uint2 u = __ldg(reinterpret_cast<const uint2*>(slots + pos));
  CompactSlot<float> s;
  s.lo = u.x;
  s.meta = u.y;
  return s;
}
__device__ __forceinline__ CompactSlot<double> LoadSlot(const CompactSlot<double>* slots, uint32_t pos) {
  const uint4 u = __ldg(reinterpret_cast<const uint4*>(slots + pos));
  CompactSlot<double> s;
  s.value = __hiloint2double(static_cast<int>(u.y), static_cast<int>(u.x));
  s.meta = u.z;
  s.pad = u.w;
  return s;
}

// Tree::NumericalDecision as a direction (x is zero-snapped)
template <typename TIn, typename TThr>
__device__ __forceinline__ bool GoesLeft(TIn x, TThr threshold, uint32_t meta, bool fast_nan) {
  if (fast_nan) {
    // MissingType::NaN with non-NaN threshold: IEEE unordered/ordered compare handles NaN in one instruction
    return (meta & kSlotDefaultLeft) ? !(x > threshold) : (x <= threshold);
  }
  const int missing = static_cast<int>((meta >> kSlotMissingShift) & 3u);
  if (isnan(x)) {
    if (missing == kMissingNaN || missing == kMissingZero) {
      return (meta & kSlotDefaultLeft) != 0;
    }
    x = TIn(0);
  } else if (missing == kMissingZero && x == TIn(0)) {
    return (meta & kSlotDefaultLeft) != 0;
  }
  return x <= threshold;
}

// Tree::CategoricalDecision as a direction
template <typename TThr>
__device__ __forceinline__ bool CategoryGoesLeft(uint32_t range, double v, const DeviceForestView<TThr>& forest) {
  if (!(v >= -2147483648.0 && v <= 2147483647.0)) {
    return false;
  }
  const int iv = static_cast<int>(v);
  if (iv < 0) {
    return false;
  }
  const int64_t word_begin = forest.cat_range[2 * static_cast<int64_t>(range)];
  const int64_t word_count = forest.cat_range[2 * static_cast<int64_t>(range) + 1];
  const int64_t word = iv / 32;
  return word < word_count && ((forest.cat_words[word_begin + word] >> (iv % 32)) & 1u);
}

// one level from slot `pos` (an internal node): returns the child's slot; *leaf tells whether that child is a leaf
template <typename TIn, typename TThr, bool HAS_CAT, typename Row>
__device__ __forceinline__ uint32_t SlotStep(const DeviceForestView<TThr>& forest, const Row& row, uint32_t pos,
                                             bool* leaf) {
  const CompactSlot<TThr> s = LoadSlot(forest.slots, pos);
  const uint32_t meta = s.meta;
  const uint32_t low = meta & kSlotLowMask;
  const TIn x = row(low & forest.feature_mask);
  bool left;
  if (HAS_CAT && (meta & kSlotCategorical)) {
    left = CategoryGoesLeft(SlotRange(s), static_cast<double>(x), forest);
  } else {
    left = GoesLeft<TIn, TThr>(x, SlotThreshold(s), meta, forest.fast_nan_missing);
  }
  *leaf = (meta & (left ? kSlotLeftLeaf : kSlotRightLeaf)) != 0;
  return pos + (low >> forest.feature_bits) + (left ? 0u : 1u);
}

// GroupSum on the compact layout: a walk in flight is one slot index (the slot is read at each step)
template <typename TIn, typename TThr, bool HAS_CAT, int ILP, typename Row>
__device__ __forceinline__ double SlotGroupSum(const DeviceForestView<TThr>& forest, const Row& row, int tree0,
                                               int stride, int count) {
  double sum = 0.0;
  for (int j0 = 0; j0 < count; j0 += ILP) {
    uint32_t pos[ILP];
    unsigned walking = 0;   // bit u: walk u is at an internal node
    #pragma unroll
    for (int u = 0; u < ILP; ++u) {
      pos[u] = 0;
      if (j0 + u < count) {
        const int32_t start = __ldg(forest.slot_start + tree0 + (j0 + u) * stride);
        pos[u] = static_cast<uint32_t>(start >= 0 ? start : ~start);
        walking |= (start >= 0 ? 1u : 0u) << u;
      }
    }
    while (walking != 0u) {
      #pragma unroll
      for (int u = 0; u < ILP; ++u) {
        if (walking & (1u << u)) {
          bool leaf;
          pos[u] = SlotStep<TIn, TThr, HAS_CAT>(forest, row, pos[u], &leaf);
          if (leaf) {
            walking &= ~(1u << u);
          }
        }
      }
    }
    #pragma unroll
    for (int u = 0; u < ILP; ++u) {
      if (j0 + u < count) {
        sum = __dadd_rn(sum, SlotValue(LoadSlot(forest.slots, pos[u])));
      }
    }
  }
  return sum;
}

// the leaf values (not their sum) of `count` <= 4 trees tree0, tree0 + stride, ..., walked together
template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, bool COMPACT, typename Row>
__device__ __forceinline__ void WalkValues4(const DeviceForestView<TThr>& forest, const Row& row, int tree0,
                                            int stride, int count, double (&v)[4]) {
  if (COMPACT) {
    uint32_t pos[4];
    unsigned walking = 0;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      pos[u] = 0;
      if (u < count) {
        const int32_t start = __ldg(forest.slot_start + tree0 + u * stride);
        pos[u] = static_cast<uint32_t>(start >= 0 ? start : ~start);
        walking |= (start >= 0 ? 1u : 0u) << u;
      }
    }
    while (walking != 0u) {
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        if (walking & (1u << u)) {
          bool leaf;
          pos[u] = SlotStep<TIn, TThr, HAS_CAT>(forest, row, pos[u], &leaf);
          if (leaf) {
            walking &= ~(1u << u);
          }
        }
      }
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      v[u] = u < count ? SlotValue(LoadSlot(forest.slots, pos[u])) : 0.0;
    }
  } else {
    int node[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      node[u] = u < count ? forest.tree_root[tree0 + u * stride] : -1;
    }
    bool active = true;
    while (active) {
      active = false;
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        if (node[u] >= 0) {
          node[u] = Step<TIn, TThr, HAS_CAT>(forest, row, node[u]);
          active |= node[u] >= 0;
        }
      }
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      v[u] = u < count ? LeafValue<TIn, TThr, HAS_LINEAR>(forest, row, tree0 + u * stride, ~node[u]) : 0.0;
    }
  }
}

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, bool COMPACT, int ILP, typename Row>
__device__ __forceinline__ double AnyGroupSum(const DeviceForestView<TThr>& forest, const Row& row, int tree0,
                                              int stride, int count) {
  if (COMPACT) {
    return SlotGroupSum<TIn, TThr, HAS_CAT, ILP>(forest, row, tree0, stride, count);
  }
  return GroupSum<TIn, TThr, HAS_CAT, HAS_LINEAR, ILP>(forest, row, tree0, stride, count);
}

// one block of the default mode: row tile `tile`, span `span`, output chunk `chunk` of one model's prediction;
// TG (1 or 8, compile time so that the one-group kernel keeps its registers) is plan.task_groups
template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, bool COMPACT, bool STAGE, int ILP, int TG>
__device__ __forceinline__ void PairwiseBlock(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows,
                                              int tree_begin, int num_iteration, int num_tree_per_iteration,
                                              const PairwisePlan& plan, double* out, double* partials,
                                              int64_t tile, int span, int chunk, unsigned char* smem) {
  const int K = num_tree_per_iteration;
  const int tile_rows = plan.tile_rows;
  const int k0 = chunk * plan.out_chunk;
  const int kb = min(plan.out_chunk, K - k0);
  const int pairs = tile_rows * kb;            // (row, output) pairs of the block; pair = r + tile_rows * kk
  const int WG = plan.wave_groups;
  const int seg = plan.seg;
  const int segs = WG / seg;
  const int part_stride = pairs | 1;
  const int pair_cap = tile_rows * plan.out_chunk;
  double* part = reinterpret_cast<double*>(smem);                          // [WG][part_stride]
  double* segsum = part + static_cast<size_t>(WG) * (pair_cap | 1);         // [segs][pairs]
  double* stack = segsum + static_cast<size_t>(segs) * pair_cap;            // [levels][pairs]
  TIn* xs = reinterpret_cast<TIn*>(stack + static_cast<size_t>(plan.levels) * pair_cap);
  const int num_col = rows.num_col;
  // staged rows cover the model's features too (zeros past the input's columns), so reads need no bounds check
  const int stage_cols = max(num_col, plan.stage_cols);
  const int x_stride = stage_cols | 1;
  const int64_t row0 = tile * tile_rows;
  const int valid_rows = static_cast<int>(min(static_cast<int64_t>(tile_rows), rows.num_row - row0));
  if (STAGE) {
    if (rows.col_stride == 1 && num_col >= blockDim.x) {
      for (int rr = 0; rr < tile_rows; ++rr) {
        const int64_t r = row0 + rr;
        TIn* dst = xs + static_cast<size_t>(rr) * x_stride;
        if (r < rows.num_row) {
          const TIn* src = rows.data + r * rows.row_stride;
          int f0 = 0;
          if constexpr (sizeof(TIn) == 4) {
            const uintptr_t addr = reinterpret_cast<uintptr_t>(src);
            if ((addr & 3u) == 0u) {
              const int lead = static_cast<int>((4u - ((addr >> 2) & 3u)) & 3u);
              const int pre = min(lead, num_col);
              if (threadIdx.x < pre) {
                dst[threadIdx.x] = SnapZero(__ldg(src + threadIdx.x));
              }
              const int vec_n = (num_col - pre) / 4;
              const float4* src4 = reinterpret_cast<const float4*>(src + pre);
              for (int j = threadIdx.x; j < vec_n; j += blockDim.x) {
                const float4 v = __ldg(src4 + j);
                const int base = pre + j * 4;
                dst[base] = SnapZero(v.x);
                dst[base + 1] = SnapZero(v.y);
                dst[base + 2] = SnapZero(v.z);
                dst[base + 3] = SnapZero(v.w);
              }
              f0 = pre + vec_n * 4;
            }
          }
          for (int f = f0 + threadIdx.x; f < num_col; f += blockDim.x) {
            dst[f] = SnapZero(__ldg(src + f));
          }
          for (int f = num_col + threadIdx.x; f < stage_cols; f += blockDim.x) {
            dst[f] = TIn(0);
          }
        } else {
          for (int f = threadIdx.x; f < stage_cols; f += blockDim.x) {
            dst[f] = TIn(0);
          }
        }
      }
    } else {
      const int total = tile_rows * stage_cols;
      const bool column_major = rows.row_stride == 1 && rows.col_stride != 1;
      for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int rr = column_major ? i % tile_rows : i / stage_cols;
        const int f = column_major ? i / tile_rows : i - rr * stage_cols;
        const int64_t r = row0 + rr;
        xs[static_cast<size_t>(rr) * x_stride + f] = r < rows.num_row && f < num_col
            ? SnapZero(__ldg(rows.data + r * rows.row_stride + static_cast<int64_t>(f) * rows.col_stride)) : TIn(0);
      }
    }
    __syncthreads();
  }
  const int num_groups = (num_iteration + kPairwiseGroupIterations - 1) / kPairwiseGroupIterations;
  const int g_span = span * plan.span_groups;
  const int span_end = min(g_span + plan.span_groups, num_groups);
  // a task's unit is TG consecutive groups (an aligned subtree it adds itself); a wave holds WG units
  const int waves = (span_end - g_span + WG * TG - 1) / (WG * TG);
  const int tasks = pairs * WG;
  const int lane = threadIdx.x % 32;
  for (int w = 0; w < waves; ++w) {
    const int gw0 = g_span + w * WG * TG;
    // 1. one task per (row, output, group); rows vary fastest, so a warp's lanes share trees
    if (TG == 1 && plan.group_lanes > 1) {
      // small batches: a group's trees are shared by group_lanes adjacent lanes, each walking its consecutive part;
      // the in-order sum is chained through the lanes by shuffles, so it is the same sum
      const int GL = plan.group_lanes;
      const int per_lane = kPairwiseGroupIterations / GL;
      const int lane_tasks = tasks * GL;
      const int rounded_lanes = (lane_tasks + 31) / 32 * 32;
      for (int base = threadIdx.x - lane; base < rounded_lanes; base += blockDim.x) {
        const int t = base + lane;
        const int j = t % GL;
        const int task = t / GL;
        const int r = task % tile_rows;
        const int rest = task / tile_rows;
        const int kk = rest % kb;
        const int gw = rest / kb;
        const int g = gw0 + gw;
        double v[4];
        int count = 0;
        if (t < lane_tasks && r < valid_rows && g < num_groups) {
          const int it0 = g * kPairwiseGroupIterations + j * per_lane;
          count = max(0, min(per_lane, num_iteration - it0));
          const int tree0 = tree_begin + it0 * K + k0 + kk;
          if (STAGE) {
            const StagedRow<TIn> row{xs + static_cast<int64_t>(r) * x_stride};
            WalkValues4<TIn, TThr, HAS_CAT, HAS_LINEAR, COMPACT>(forest, row, tree0, K, count, v);
          } else {
            const GlobalRow<TIn> row{rows.data + (row0 + r) * rows.row_stride, num_col, rows.col_stride};
            WalkValues4<TIn, TThr, HAS_CAT, HAS_LINEAR, COMPACT>(forest, row, tree0, K, count, v);
          }
        }
        double acc = 0.0;
        for (int l = 0; l < GL; ++l) {
          if (j == l) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
              if (u < count) {
                acc = __dadd_rn(acc, v[u]);
              }
            }
          }
          acc = __shfl_sync(0xffffffffu, acc, (lane & ~(GL - 1)) + l);
        }
        if (t < lane_tasks && j == 0) {
          part[gw * part_stride + r + tile_rows * kk] = acc;
        }
      }
    } else {
    for (int t = threadIdx.x; t < tasks; t += blockDim.x) {
      const int r = t % tile_rows;
      const int rest = t / tile_rows;
      const int kk = rest % kb;
      const int gw = rest / kb;
      // the unit's groups, added by the aligned tree in registers: l0, l1, l2 hold the pending subtrees of 1, 2 and 4
      // groups; after the last group (TG is a power of two) x is the unit's subtree
      double x = 0.0, l0 = 0.0, l1 = 0.0, l2 = 0.0;
      #pragma unroll
      for (int q = 0; q < TG; ++q) {
        const int g = gw0 + gw * TG + q;
        double p = 0.0;
        if (r < valid_rows && g < num_groups) {
          const int it0 = g * kPairwiseGroupIterations;
          const int count = min(kPairwiseGroupIterations, num_iteration - it0);
          const int tree0 = tree_begin + it0 * K + k0 + kk;
          if (STAGE) {
            const StagedRow<TIn> row{xs + static_cast<int64_t>(r) * x_stride};
            p = AnyGroupSum<TIn, TThr, HAS_CAT, HAS_LINEAR, COMPACT, ILP>(forest, row, tree0, K, count);
          } else {
            const GlobalRow<TIn> row{rows.data + (row0 + r) * rows.row_stride, num_col, rows.col_stride};
            p = AnyGroupSum<TIn, TThr, HAS_CAT, HAS_LINEAR, COMPACT, ILP>(forest, row, tree0, K, count);
          }
        }
        x = p;
        if (q & 1) {
          x = __dadd_rn(l0, x);
          if (q & 2) {
            x = __dadd_rn(l1, x);
            if (q & 4) {
              x = __dadd_rn(l2, x);
            } else {
              l2 = x;
            }
          } else {
            l1 = x;
          }
        } else {
          l0 = x;
        }
      }
      part[gw * part_stride + r + tile_rows * kk] = x;
    }
    }
    __syncthreads();
    // 2. aligned segments of `seg` group sums by warp shuffles; lanes hold consecutive groups of one pair
    const int rounded = (tasks + 31) / 32 * 32;
    for (int base = threadIdx.x - lane; base < rounded; base += blockDim.x) {
      const int u = base + lane;
      const int pair = u / WG;
      const int gw = u % WG;
      double v = u < tasks ? part[gw * part_stride + pair] : 0.0;
      for (int off = 1; off < seg; off <<= 1) {
        v = __dadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
      }
      if (u < tasks && gw % seg == 0) {
        segsum[(gw / seg) * pairs + pair] = v;
      }
    }
    __syncthreads();
    // 3. the wave's segment sums enter each pair's stack, in group order (the next wave rewrites segsum only after
    //    its first barrier, which every pusher reaches after pushing)
    for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
      for (int sg = 0; sg < segs; ++sg) {
        PairwisePush(stack + pair, pairs, static_cast<unsigned>(w * segs + sg), segsum[sg * pairs + pair]);
      }
    }
  }
  __syncthreads();
  for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    const int r = pair % tile_rows;
    const int k = k0 + pair / tile_rows;
    if (r < valid_rows) {
      const double v = PairwiseFinish(stack + pair, pairs, static_cast<unsigned>(waves * segs));
      if (plan.num_spans == 1) {
        out[(row0 + r) * K + k] = v;
      } else {
        partials[(static_cast<int64_t>(span) * rows.num_row + row0 + r) * K + k] = v;
      }
    }
  }
}

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, bool COMPACT, bool STAGE, int ILP, int TG>
__global__ void PredictPairwiseKernel(const DeviceForestView<TThr> forest, const DeviceRows<TIn> rows, int tree_begin,
                                      int num_iteration, int num_tree_per_iteration, const PairwisePlan plan,
                                      double* out, double* partials) {
  extern __shared__ __align__(16) unsigned char smem[];
  PairwiseBlock<TIn, TThr, HAS_CAT, HAS_LINEAR, COMPACT, STAGE, ILP, TG>(
      forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan, out, partials, blockIdx.x, blockIdx.y,
      blockIdx.z, smem);
}

// several models' predictions in one launch: the flattened grid is cut into each job's blocks (PairwiseJob)
template <typename TIn, bool HAS_CAT, bool COMPACT, bool STAGE, int ILP, int TG>
__global__ void PredictPairwiseJobsKernel(const PairwiseJob<TIn>* jobs, int num_jobs) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int64_t block = blockIdx.x;
  int lo = 0, hi = num_jobs - 1;
  while (lo < hi) {   // the last job whose first block is not after this one
    const int mid = (lo + hi + 1) / 2;
    if (jobs[mid].block_begin <= block) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const PairwiseJob<TIn>& job = jobs[lo];
  const int64_t local = block - job.block_begin;
  const int64_t row_tiles = (job.rows.num_row + job.plan.tile_rows - 1) / job.plan.tile_rows;
  const int64_t rest = local / row_tiles;
  PairwiseBlock<TIn, TIn, HAS_CAT, false, COMPACT, STAGE, ILP, TG>(
      job.forest, job.rows, job.tree_begin, job.num_iteration, job.num_tree_per_iteration, job.plan, job.out,
      job.partials, local % row_tiles, static_cast<int>(rest % job.plan.num_spans),
      static_cast<int>(rest / job.plan.num_spans), smem);
}

// the jobs' span sums, added by the aligned tree: one thread per (job, row, output) cell
template <typename TIn>
__global__ void PairwiseJobsCombineKernel(const PairwiseJob<TIn>* jobs, int num_jobs, int64_t total_cells) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total_cells) {
    return;
  }
  int lo = 0, hi = num_jobs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (jobs[mid].cell_begin <= i) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const PairwiseJob<TIn>& job = jobs[lo];
  if (job.plan.num_spans <= 1) {
    return;
  }
  const int64_t cells = job.rows.num_row * job.num_tree_per_iteration;
  const int64_t c = i - job.cell_begin;
  double stack[32];
  for (int s = 0; s < job.plan.num_spans; ++s) {
    PairwisePush(stack, 1, static_cast<unsigned>(s), job.partials[static_cast<int64_t>(s) * cells + c]);
  }
  job.out[c] = PairwiseFinish(stack, 1, static_cast<unsigned>(job.plan.num_spans));
}

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, bool COMPACT, bool STAGE>
static void LaunchPairwiseIlp(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                       int num_iteration, int num_tree_per_iteration, const PairwisePlan& plan, double* out,
                       double* partials, cudaStream_t stream) {
  const dim3 grid(Blocks(rows.num_row, plan.tile_rows), plan.num_spans,
                  (num_tree_per_iteration + plan.out_chunk - 1) / plan.out_chunk);
  const size_t shared = PairwiseSharedBytes(plan, rows.num_col, sizeof(TIn));
#define FLC_PW_KERNEL(U, G)                                                                                    \
  {                                                                                                            \
    auto kernel = PredictPairwiseKernel<TIn, TThr, HAS_CAT, HAS_LINEAR, COMPACT, STAGE, U, G>;                 \
    if (shared > (size_t(48) << 10)) {                                                                         \
      CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,           \
                                                static_cast<int>(shared)));                                    \
    }                                                                                                          \
    kernel<<<grid, plan.block_threads, shared, stream>>>(forest, rows, tree_begin, num_iteration,              \
                                                         num_tree_per_iteration, plan, out, partials);         \
  }
  if (plan.task_groups > 1) {
    if (plan.ilp >= 8) {
      FLC_PW_KERNEL(8, 8)
    } else {
      FLC_PW_KERNEL(4, 8)
    }
  } else {
    if (plan.ilp >= 8) {
      FLC_PW_KERNEL(8, 1)
    } else {
      FLC_PW_KERNEL(4, 1)
    }
  }
#undef FLC_PW_KERNEL
  if (plan.num_spans > 1) {
    const int64_t cells = rows.num_row * num_tree_per_iteration;
    LaunchPairwiseCombineKernel(partials, cells, plan.num_spans, out, stream);
  }
}

template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR, bool COMPACT>
static void LaunchPairwiseStage(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                         int num_iteration, int num_tree_per_iteration, const PairwisePlan& plan, double* out,
                         double* partials, cudaStream_t stream) {
  if (plan.stage_rows) {
    LaunchPairwiseIlp<TIn, TThr, HAS_CAT, HAS_LINEAR, COMPACT, true>(forest, rows, tree_begin, num_iteration,
                                                                     num_tree_per_iteration, plan, out, partials,
                                                                     stream);
  } else {
    LaunchPairwiseIlp<TIn, TThr, HAS_CAT, HAS_LINEAR, COMPACT, false>(forest, rows, tree_begin, num_iteration,
                                                                      num_tree_per_iteration, plan, out, partials,
                                                                      stream);
  }
}

// the compact layout when the model has one (never with linear leaves), else the node layout
template <typename TIn, typename TThr, bool HAS_CAT, bool HAS_LINEAR>
static void LaunchPairwise(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                    int num_iteration, int num_tree_per_iteration, const PairwisePlan& plan, double* out,
                    double* partials, cudaStream_t stream) {
  if (!HAS_LINEAR && plan.compact && forest.slots != nullptr) {
    LaunchPairwiseStage<TIn, TThr, HAS_CAT, false, true>(forest, rows, tree_begin, num_iteration,
                                                         num_tree_per_iteration, plan, out, partials, stream);
  } else {
    LaunchPairwiseStage<TIn, TThr, HAS_CAT, HAS_LINEAR, false>(forest, rows, tree_begin, num_iteration,
                                                               num_tree_per_iteration, plan, out, partials, stream);
  }
}

}  // namespace cuda_predictor_detail

template <typename TIn, typename TThr>
void LaunchPredictPairwiseKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                                 int num_iteration, int num_tree_per_iteration, const PairwisePlan& plan, double* out,
                                 double* partials, cudaStream_t stream) {
  using cuda_predictor_detail::LaunchPairwise;
  if (rows.num_row == 0) {
    return;
  }
  if (num_iteration == 0) {
    CUDASUCCESS_OR_FATAL(cudaMemsetAsync(out, 0, sizeof(double) * rows.num_row * num_tree_per_iteration, stream));
  } else if (forest.has_categorical) {
    if (forest.has_linear) {
      LaunchPairwise<TIn, TThr, true, true>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan, out,
                                            partials, stream);
    } else {
      LaunchPairwise<TIn, TThr, true, false>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan,
                                             out, partials, stream);
    }
  } else {
    if (forest.has_linear) {
      LaunchPairwise<TIn, TThr, false, true>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan,
                                             out, partials, stream);
    } else {
      LaunchPairwise<TIn, TThr, false, false>(forest, rows, tree_begin, num_iteration, num_tree_per_iteration, plan,
                                              out, partials, stream);
    }
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

template <typename T>
void LaunchPredictPairwiseJobs(const PairwiseJob<T>* jobs, int num_jobs, const PairwiseJob<T>* device_jobs,
                               bool has_categorical, bool compact, bool stage, size_t shared_bytes,
                               cudaStream_t stream) {
  using cuda_predictor_detail::Blocks;
  using cuda_predictor_detail::PairwiseJobsCombineKernel;
  using cuda_predictor_detail::PredictPairwiseJobsKernel;
  if (num_jobs <= 0) {
    return;
  }
  const int task_groups = jobs[0].plan.task_groups;   // one value per launch (LaunchJobs groups by it)
  int64_t blocks = 0, cells = 0;
  int threads = 32;
  bool combine = false;
  for (int j = 0; j < num_jobs; ++j) {
    const PairwiseJob<T>& job = jobs[j];
    blocks = job.block_begin + ((job.rows.num_row + job.plan.tile_rows - 1) / job.plan.tile_rows) *
             job.plan.num_spans * ((job.num_tree_per_iteration + job.plan.out_chunk - 1) / job.plan.out_chunk);
    cells = job.cell_begin + job.rows.num_row * job.num_tree_per_iteration;
    threads = std::max(threads, job.plan.block_threads);
    combine = combine || job.plan.num_spans > 1;
  }
  if (blocks > std::numeric_limits<int32_t>::max()) {
    Log::Fatal("CUDA predictor: a batch of %lld blocks is too large for one launch",
               static_cast<long long>(blocks));  // NOLINT(runtime/int)
  }
#define FLC_JOBS_KERNEL(CAT, COMPACT, STAGE)                                                                   \
  {                                                                                                            \
    auto kernel = task_groups > 1 ? PredictPairwiseJobsKernel<T, CAT, COMPACT, STAGE, 8, 8>                    \
                                  : PredictPairwiseJobsKernel<T, CAT, COMPACT, STAGE, 8, 1>;                   \
    if (shared_bytes > (size_t(48) << 10)) {                                                                   \
      CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,           \
                                                static_cast<int>(shared_bytes)));                              \
    }                                                                                                          \
    kernel<<<static_cast<unsigned int>(blocks), threads, shared_bytes, stream>>>(device_jobs, num_jobs);       \
  }
  if (has_categorical) {
    if (compact) {
      if (stage) FLC_JOBS_KERNEL(true, true, true) else FLC_JOBS_KERNEL(true, true, false)
    } else {
      if (stage) FLC_JOBS_KERNEL(true, false, true) else FLC_JOBS_KERNEL(true, false, false)
    }
  } else {
    if (compact) {
      if (stage) FLC_JOBS_KERNEL(false, true, true) else FLC_JOBS_KERNEL(false, true, false)
    } else {
      if (stage) FLC_JOBS_KERNEL(false, false, true) else FLC_JOBS_KERNEL(false, false, false)
    }
  }
#undef FLC_JOBS_KERNEL
  if (combine) {
    PairwiseJobsCombineKernel<T><<<Blocks(cells, 128), 128, 0, stream>>>(device_jobs, num_jobs, cells);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_PAIRWISE_CUH_
