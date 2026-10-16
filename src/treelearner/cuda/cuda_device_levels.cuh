/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief cuda_plan key device_levels: device functions shared by the device level kernels (cuda_device_levels.cu)
 *  and the apply kernel that runs the children's descriptors beside its partition (cuda_data_partition.cu).
 */

#ifndef FALCATA_TREELEARNER_CUDA_CUDA_DEVICE_LEVELS_CUH_
#define FALCATA_TREELEARNER_CUDA_CUDA_DEVICE_LEVELS_CUH_

#if defined(USE_CUDA) && defined(__CUDACC__)

#include <Falcata/cuda/cuda_utils.hu>

#include <algorithm>

#include <Falcata/cuda/cuda_split_info.hpp>

#include "cuda_data_partition.hpp"
#include "cuda_device_levels.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_leaf_splits.hpp"

namespace Falcata {

/*! \brief block-wide exclusive prefix sum (two-level warp-shuffle scan); total in *s_total. s_warp >= 32 ints */
__device__ __forceinline__ int DeviceLevelBlockScan(const int val, int* s_warp, int* s_total) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int num_warps = static_cast<int>(blockDim.x) >> 5;
  __syncthreads();  // s_warp may still be read by a preceding scan's tail
  int incl = val;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, incl, offset);
    if (lane >= offset) {
      incl += other;
    }
  }
  if (lane == 31) {
    s_warp[warp] = incl;
  }
  __syncthreads();
  if (warp == 0) {
    const int w = lane < num_warps ? s_warp[lane] : 0;
    int winc = w;
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, winc, offset);
      if (lane >= offset) {
        winc += other;
      }
    }
    if (lane < num_warps) {
      s_warp[lane] = winc - w;
    }
    if (lane == num_warps - 1) {
      *s_total = winc;
    }
  }
  __syncthreads();
  return s_warp[warp] + incl - val;
}

/*! \brief block-wide max of a non-negative value */
__device__ __forceinline__ int DeviceLevelBlockMax(const int val, int* s_warp, int* s_out) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int num_warps = static_cast<int>(blockDim.x) >> 5;
  __syncthreads();
  int v = val;
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = max(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  if (lane == 0) {
    s_warp[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    int m = lane < num_warps ? s_warp[lane] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
      m = max(m, __shfl_down_sync(0xffffffffu, m, offset));
    }
    if (lane == 0) {
      *s_out = m;
    }
  }
  __syncthreads();
  return *s_out;
}

/*! \brief the quantized histogram bit width of a leaf of n rows: GradientDiscretizer::SetNumBitsInHistogramBin */
__device__ __forceinline__ uint8_t DeviceLevelHistBits(const data_size_t n, const int num_grad_quant_bins) {
  const uint64_t max_stat_per_bin = static_cast<uint64_t>(n) * static_cast<uint64_t>(num_grad_quant_bins);
  return max_stat_per_bin < 256 ? 8 : (max_stat_per_bin < 65536 ? 16 : 32);
}


/*! \brief the children's search descriptors of the level just applied, as EnqueueLevelBestSplitSearch builds them from
 *  the host's bookkeeping (FinishLevelBookkeeping: the partition's exact counts, the split infos' hessian sums, the
 *  bit widths of SetNumBitsInHistogramBin), and the construct's grid layout for them
 *  (LaunchConstructHistogramBatchedKernelInner0's host math). Any block of a multiple of 32 threads: each thread
 *  takes a contiguous run of the level's splits. Reads the aggregate's child counts: after it. */
__device__ __forceinline__ void DeviceLevelPairsBlock(const CUDADeviceLevelConfig& cfg, const int level) {
  __shared__ int s_warp[32];
  __shared__ int s_total;
  __shared__ int s_max;
  CUDADeviceLevelState* st = cfg.state;
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  const int n = st->done ? 0 : st->live_splits;
  if (n == 0) {
    if (tid == 0) {
      st->live_pairs = 0;
      st->construct_total_blocks = 0;
      st->construct_block_start[0] = 0;
    }
    return;
  }
  const int chunk = (n + nt - 1) / nt;
  const int k_begin = min(tid * chunk, n);
  const int k_end = min(k_begin + chunk, n);
  const bool below_max_depth = cfg.max_depth <= 0 || level + 1 < cfg.max_depth;
  const int64_t min_data = static_cast<int64_t>(cfg.min_data_in_leaf);
  data_size_t my_max_smaller = 0;
  int my_has_rows = 0;
  for (int k = k_begin; k < k_end; ++k) {
    const CUDAHybridApplyDescriptor& a = cfg.apply_descs[k];
    const int left = a.left_leaf_index;
    const int right = a.right_leaf_index;
    const data_size_t num_left = cfg.leaf_num_data[left];
    const data_size_t num_right = cfg.leaf_num_data[right];
    const double hess_left = a.best_split_info->left_sum_hessians;
    const double hess_right = a.best_split_info->right_sum_hessians;
    // FinishLevelBookkeeping: the left child is the smaller one only when strictly smaller
    const bool left_smaller = num_left < num_right;
    const data_size_t num_smaller = left_smaller ? num_left : num_right;
    const data_size_t num_larger = left_smaller ? num_right : num_left;
    const double hess_smaller = left_smaller ? hess_left : hess_right;
    const double hess_larger = left_smaller ? hess_right : hess_left;
    const int64_t cs = static_cast<int64_t>(num_smaller);
    const int64_t cl = static_cast<int64_t>(num_larger);
    const bool smaller_unsplittable = cfg.skip_unsplittable != 0 && cs + 2 + (cs >> 20) < 2 * min_data;
    const bool larger_unsplittable = cfg.skip_unsplittable != 0 && cl + 2 + (cl >> 20) < 2 * min_data;
    const uint8_t construct_valid =
      ((num_smaller <= cfg.min_data_in_leaf || smaller_unsplittable ||
        hess_smaller <= cfg.min_sum_hessian_in_leaf) &&
       (num_larger <= cfg.min_data_in_leaf || larger_unsplittable ||
        hess_larger <= cfg.min_sum_hessian_in_leaf)) ? 0 : 1;
    CUDAHybridPairDescriptor& desc = cfg.pair_descs[k];
    // the apply wrote the smaller child's statistics into the split's first slot (HybridSplitTreeStructureSplit)
    desc.smaller_struct = cfg.pair_slots + 2 * k;
    desc.larger_struct = cfg.pair_slots + 2 * k + 1;
    desc.smaller_leaf_index = left_smaller ? left : right;
    desc.larger_leaf_index = left_smaller ? right : left;
    desc.num_data_in_smaller_leaf = num_smaller;
    desc.num_data_in_larger_leaf = num_larger;
    desc.construct_valid = construct_valid;
    desc.smaller_valid = (num_smaller > cfg.min_data_in_leaf && !smaller_unsplittable &&
                          hess_smaller > cfg.min_sum_hessian_in_leaf && below_max_depth) ? 1 : 0;
    desc.larger_valid = (num_larger > cfg.min_data_in_leaf && !larger_unsplittable &&
                         hess_larger > cfg.min_sum_hessian_in_leaf && below_max_depth) ? 1 : 0;
    // the parent's bit width is the one its leaf got at its creation, from its row count
    desc.parent_num_bits = DeviceLevelHistBits(a.num_data_in_leaf, cfg.num_grad_quant_bins);
    desc.smaller_num_bits = DeviceLevelHistBits(num_smaller, cfg.num_grad_quant_bins);
    desc.larger_num_bits = DeviceLevelHistBits(num_larger, cfg.num_grad_quant_bins);
    my_max_smaller = max(my_max_smaller, num_smaller);
    if (construct_valid && num_smaller > 0) {
      my_has_rows = 1;
    }
  }
  // the level's largest smaller leaf over every pair (valid or not, as the host's max)
  const data_size_t max_smaller = DeviceLevelBlockMax(my_max_smaller, s_warp, &s_max);
  // LaunchConstructHistogramBatchedKernelInner0's level geometry: the grid y at the largest smaller leaf (dense:
  // dense_level_row_blocks' floor and dense_block_map; pair-joint: pair_code5_slots' rows per thread or
  // level_row_blocks' floor, per_pair_rows and pair_block_map), as the host computes it from its descriptors
  const CUDADeviceLevelConstructShape& sh = cfg.shape;
  const int block_y = sh.block_y;
  int grid_y = HybridBatchedConstructGridDimYQuant(
    max_smaller, n, block_y, cfg.construct_min_grid_dim_y, cfg.construct_min_rows_per_thread,
    cfg.construct_saturation_floor, cfg.num_grad_quant_bins);
  // a pair's descriptor as this thread wrote it above: whether it has rows to construct, and their count
  const auto pair_rows = [&](const int k) -> data_size_t {
    const CUDAHybridPairDescriptor& desc = cfg.pair_descs[k];
    return desc.construct_valid && desc.num_data_in_smaller_leaf > 0 ? desc.num_data_in_smaller_leaf : 0;
  };
  data_size_t min_rows = 0;
  if (sh.pair_mode == 0) {
    const int64_t dim_y = static_cast<int64_t>(grid_y) * block_y;
    min_rows = cfg.level_row_blocks != 0 && n > 1 && dim_y > 0 ?
      DenseLevelRowsPerThreadFloor(max_smaller, dim_y, cfg.construct_min_rows_per_thread) : 0;
  } else {
    data_size_t slot_rows = 0;
    if (sh.slot != 0 && max_smaller > 0 && DeviceLevelBlockMax(my_has_rows, s_warp, &s_max) != 0) {
      // PairJointSlotRowsPerThread over the valid pairs' smaller counts: the smallest R in [min, max] whose blocks
      // fit the device's slots, max when none does
      const data_size_t max_rows = sh.slot_max_rows;
      data_size_t lo = max(1, min(static_cast<data_size_t>(sh.slot_min_rows), max_rows));
      data_size_t hi = max(lo, max_rows);
      const auto blocks_at = [&](const data_size_t r) {
        const int64_t span = static_cast<int64_t>(block_y) * r;
        int mine = 0;
        for (int k = k_begin; k < k_end; ++k) {
          mine += static_cast<int>((static_cast<int64_t>(pair_rows(k)) + span - 1) / span);
        }
        DeviceLevelBlockScan(mine, s_warp, &s_total);
        return s_total;
      };
      if (blocks_at(hi) > sh.slot_device_slots) {
        slot_rows = hi;
      } else {
        while (lo < hi) {
          const data_size_t mid = lo + (hi - lo) / 2;
          if (blocks_at(mid) <= sh.slot_device_slots) {
            hi = mid;
          } else {
            lo = mid + 1;
          }
        }
        slot_rows = lo;
      }
      grid_y = static_cast<int>((static_cast<int64_t>(max_smaller) + static_cast<int64_t>(block_y) * slot_rows - 1) /
                                (static_cast<int64_t>(block_y) * slot_rows));
    }
    const int64_t pair_dim_y = static_cast<int64_t>(grid_y) * block_y;
    min_rows = slot_rows > 0 ? slot_rows : (cfg.level_row_blocks != 0 && n > 1 && pair_dim_y > 0 ?
      static_cast<data_size_t>((static_cast<int64_t>(max_smaller) + pair_dim_y - 1) / pair_dim_y) : 0);
  }
  // each pair's block rows (dense_block_map / pair_block_map) and their prefix
  const auto pair_blocks = [&](const int k) -> int {
    const data_size_t num_smaller = pair_rows(k);
    if (num_smaller <= 0) {
      return 0;
    }
    int pair_grid_y = grid_y;
    if (sh.pair_mode != 0 && sh.per_pair_min_grid > 0) {
      // per_pair_rows: the pair's own formula, never above the level's
      const int y = HybridBatchedConstructGridDimYQuant(
        num_smaller, n, block_y, sh.per_pair_min_grid, sh.per_pair_min_rows, cfg.construct_saturation_floor,
        cfg.num_grad_quant_bins);
      pair_grid_y = max(1, min(pair_grid_y, y));
    }
    const int64_t dim_y = static_cast<int64_t>(pair_grid_y) * block_y;
    const int64_t rows_by_grid = (static_cast<int64_t>(num_smaller) + dim_y - 1) / dim_y;
    const int64_t rows_per_thread = rows_by_grid > static_cast<int64_t>(min_rows) ? rows_by_grid :
      static_cast<int64_t>(min_rows);
    const int64_t block_rows = static_cast<int64_t>(block_y) * rows_per_thread;
    const int64_t leaf_blocks = (static_cast<int64_t>(num_smaller) + block_rows - 1) / block_rows;
    return static_cast<int>(leaf_blocks < pair_grid_y ? leaf_blocks : static_cast<int64_t>(pair_grid_y));
  };
  int my_blocks = 0;
  for (int k = k_begin; k < k_end; ++k) {
    my_blocks += pair_blocks(k);
  }
  int block_start = DeviceLevelBlockScan(my_blocks, s_warp, &s_total);
  for (int k = k_begin; k < k_end; ++k) {
    st->construct_block_start[k] = block_start;
    block_start += pair_blocks(k);
  }
  if (tid == 0) {
    st->construct_block_start[n] = s_total;
    st->construct_total_blocks = s_total;
    st->construct_grid_y = grid_y;
    st->construct_min_rows = static_cast<int>(min_rows);
    st->max_smaller = max_smaller;
    st->live_pairs = n;
  }
}

}  // namespace Falcata

#endif  // USE_CUDA && __CUDACC__
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_DEVICE_LEVELS_CUH_
