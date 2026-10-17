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
#include <Falcata/cuda/cuda_tree.hpp>

#include "cuda_data_partition.hpp"
#include "cuda_device_levels.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_hybrid_graph.hpp"
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


/*! \brief one level's host decisions between its search and its apply: the leaf-ascending valid scan of
 *  CollectSplittableLeaves, then per split the apply descriptor (SplitLevelBatched's host math), the tree record
 *  (ApplyLevelBatched's CUDATreeBatchSplit) and the journal entry, the terminal-region gaps (the windows of the leaves
 *  this level does not split: they cover what SplitLevelBatched's gaps between the sorted split regions cover; one
 *  descriptor per such leaf, adjacent windows not merged: the same rows, copied once each) and the flat layout. Any
 *  block of a multiple of 32 threads; each thread takes contiguous runs of the leaves and of the splits. Reads every
 *  leaf's best split: after the level's sync. */
__device__ __forceinline__ void DeviceLevelSelectBlock(const CUDADeviceLevelConfig& cfg, const int level) {
  constexpr int kMaxLeaves = 2 * kDeviceLevelMaxPairs;
  __shared__ int s_leaves[kMaxLeaves];
  __shared__ unsigned int s_split_bits[kMaxLeaves / 32];
  __shared__ int s_warp[32];
  __shared__ int s_total;
  CUDADeviceLevelState* st = cfg.state;
  if (st->done) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  // the previous level's values are still in the state: this level's leaves and split-info slab base follow them
  const int cur_leaves = st->num_leaves + st->live_splits;
  const int split_cursor = st->split_info_base + st->live_splits;
  const auto stop = [&](const bool flag) {
    if (tid == 0) {
      if (flag) {
        atomicOr(cfg.journal + 2, 1);
      }
      st->done = 1;
      st->live_splits = 0;
      st->total_flat_blocks = 0;
      st->num_gaps = 0;
      st->gap_flat_blocks = 0;
      st->num_leaves = cur_leaves;
      st->split_info_base = split_cursor;
    }
  };
  if (cur_leaves > kMaxLeaves) {
    stop(true);  // outside the gated regime (the learner gates num_leaves): stop and flag it
    return;
  }
  // ---- CollectSplittableLeaves: leaf-ascending is_valid scan (contiguous per-thread runs keep the order) ----
  const int leaf_chunk = (cur_leaves + nt - 1) / nt;
  const int leaf_begin = min(tid * leaf_chunk, cur_leaves);
  const int leaf_end = min(leaf_begin + leaf_chunk, cur_leaves);
  int my_count = 0;
  for (int leaf = leaf_begin; leaf < leaf_end; ++leaf) {
    if (cfg.leaf_best_split_info[leaf].is_valid) {
      ++my_count;
    }
  }
  int write_pos = DeviceLevelBlockScan(my_count, s_warp, &s_total);
  const int n = s_total;
  if (n > kDeviceLevelMaxPairs) {
    stop(true);
    return;
  }
  for (int leaf = leaf_begin; leaf < leaf_end; ++leaf) {
    if (cfg.leaf_best_split_info[leaf].is_valid) {
      s_leaves[write_pos++] = leaf;
    }
  }
  for (int w = tid; w < kMaxLeaves / 32; w += nt) {
    s_split_bits[w] = 0;
  }
  __syncthreads();
  if (n == 0) {
    stop(false);  // no splittable leaf: the tree is complete; later kernels see done / zero live counts
    return;
  }
  // ---- per split: apply descriptor, tree record, journal ----
  const int split_chunk = (n + nt - 1) / nt;
  const int k_begin = min(tid * split_chunk, n);
  const int k_end = min(k_begin + split_chunk, n);
  int my_slots = 0;
  int my_flat = 0;
  for (int k = k_begin; k < k_end; ++k) {
    const int leaf = s_leaves[k];
    const int right = cur_leaves + k;
    const CUDASplitInfo* info = cfg.leaf_best_split_info + leaf;
    const int inner_feature = info->inner_feature_index;
    const uint32_t threshold = info->threshold;
    const CUDAHybridGraphFeatureMeta m = cfg.feature_meta[inner_feature];
    const CUDAHybridGraphFeatureSource src = cfg.feature_source[inner_feature];
    CUDATreeBatchSplit& ts = cfg.tree_batch_splits[k];
    ts.leaf_index = leaf;
    ts.num_leaves_at_split = right;
    ts.real_feature_index = m.real_feature_index;
    ts.real_threshold = cfg.real_thresholds[m.real_threshold_offset + threshold];
    ts.missing_type = m.missing_type;
    ts.split_info = info;
    uint32_t th = threshold + m.min_bin;
    uint32_t t_zero_bin = m.min_bin + m.default_bin;
    if (m.most_freq_bin == 0) {
      --th;
      --t_zero_bin;
    }
    uint8_t split_default_to_left = 0;
    uint8_t split_missing_default_to_left = 0;
    int default_leaf_index = right;
    int missing_default_leaf_index = right;
    if (m.most_freq_bin <= threshold) {
      split_default_to_left = 1;
      default_leaf_index = leaf;
    }
    if ((m.missing_is_zero || m.missing_is_na) && info->default_left) {
      split_missing_default_to_left = 1;
      missing_default_leaf_index = leaf;
    }
    const data_size_t parent_num_data = cfg.leaf_num_data[leaf];
    CUDAHybridApplyDescriptor& d = cfg.apply_descs[k];
    d.column_data = src.column_data;
    d.best_split_info = info;
    d.smaller_leaf_splits = cfg.pair_slots + 2 * k;
    d.larger_leaf_splits = cfg.pair_slots + 2 * k + 1;
    d.leaf_data_start = cfg.leaf_data_start[leaf];
    d.num_data_in_leaf = parent_num_data;
    d.num_blocks = (parent_num_data + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
      SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    d.left_leaf_index = leaf;
    d.right_leaf_index = right;
    d.th = th;
    d.t_zero_bin = t_zero_bin;
    d.max_bin = m.max_bin;
    d.min_bin = m.min_bin;
    d.default_leaf_index = default_leaf_index;
    d.missing_default_leaf_index = missing_default_leaf_index;
    d.split_default_to_left = split_default_to_left;
    d.split_missing_default_to_left = split_missing_default_to_left;
    d.packed_row_stride = src.packed_row_stride;
    d.bit_type = src.bit_type;
    d.packed_shift = src.packed_shift;
    d.min_is_max = (m.min_bin < m.max_bin) ? 0 : 1;
    d.missing_is_zero = m.missing_is_zero;
    d.missing_is_na = m.missing_is_na;
    d.mfb_is_zero = m.mfb_is_zero;
    d.mfb_is_na = m.mfb_is_na;
    d.max_bin_to_left = (m.max_bin <= th) ? 1 : 0;
    d.use_min_bin = m.use_min_bin;
    d.cat_bitset_len = 0;
    d.cat_bitset = nullptr;
    d.cat_mfb_offset = 0;
    my_slots += d.num_blocks + 1;
    my_flat += d.num_blocks;
    atomicOr(&s_split_bits[leaf >> 5], 1u << (leaf & 31));
    cfg.journal[kDeviceLevelJournalHeader + 2 * (split_cursor + k)] = leaf;
    cfg.journal[kDeviceLevelJournalHeader + 2 * (split_cursor + k) + 1] = inner_feature;
  }
  // block-offset slots (num_blocks + 1 per split) and flat chunks (num_blocks per split), in split order
  int slot_pos = DeviceLevelBlockScan(my_slots, s_warp, &s_total);
  int flat_pos = DeviceLevelBlockScan(my_flat, s_warp, &s_total);
  const int total_flat_blocks = s_total;
  for (int k = k_begin; k < k_end; ++k) {
    CUDAHybridApplyDescriptor& d = cfg.apply_descs[k];
    d.block_offset_start = slot_pos;
    d.flat_block_start = flat_pos;
    slot_pos += d.num_blocks + 1;
    flat_pos += d.num_blocks;
  }
  // ---- terminal-region gaps: the windows of the leaves this level does not split ----
  int my_gaps = 0;
  int my_gap_flat = 0;
  for (int leaf = leaf_begin; leaf < leaf_end; ++leaf) {
    const data_size_t num = cfg.leaf_num_data[leaf];
    if (((s_split_bits[leaf >> 5] >> (leaf & 31)) & 1u) == 0 && num > 0) {
      ++my_gaps;
      my_gap_flat += (num + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    }
  }
  int gap_pos = DeviceLevelBlockScan(my_gaps, s_warp, &s_total);
  const int num_gaps = s_total;
  int gap_flat_pos = DeviceLevelBlockScan(my_gap_flat, s_warp, &s_total);
  const int gap_flat_blocks = s_total;
  for (int leaf = leaf_begin; leaf < leaf_end; ++leaf) {
    const data_size_t num = cfg.leaf_num_data[leaf];
    if (((s_split_bits[leaf >> 5] >> (leaf & 31)) & 1u) == 0 && num > 0) {
      CUDAHybridApplyDescriptor& g = cfg.apply_descs[n + gap_pos];
      memset(&g, 0, sizeof(CUDAHybridApplyDescriptor));
      g.leaf_data_start = cfg.leaf_data_start[leaf];
      g.num_data_in_leaf = num;
      g.num_blocks = (num + cfg.gap_block_dim - 1) / cfg.gap_block_dim;
      g.flat_block_start = total_flat_blocks + gap_flat_pos;
      ++gap_pos;
      gap_flat_pos += (num + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    }
  }
  if (tid == 0) {
    st->live_splits = n;
    st->total_flat_blocks = total_flat_blocks;
    st->num_gaps = num_gaps;
    st->gap_flat_blocks = gap_flat_blocks;
    st->split_info_base = split_cursor;
    st->num_leaves = cur_leaves;
    st->levels_applied = level + 1;
    cfg.journal[kDeviceLevelJournalLevels + level] = n;
    cfg.journal[0] = level + 1;
    cfg.journal[1] = split_cursor + n;
  }
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
  // A complete tree's final apply has no child search. Its partition/child statistics are already written by
  // the apply; leave the live split count intact for the tree record, and omit unused search descriptors/layout.
  const bool final_depth = cfg.whole_tree && level + 1 >= cfg.max_depth;
  const int n = st->done || final_depth ? 0 : st->live_splits;
  if (n == 0) {
    for (int c = tid; c < kDeviceLevelClaimColumns; c += nt) {
      st->construct_next_block[c] = 0;
    }
    if (tid == 0) {
      st->live_pairs = 0;
      st->construct_total_blocks = 0;
      st->construct_block_start[0] = 0;
      st->construct_grid_y = 0;
      st->construct_min_rows = 0;
      st->max_smaller = 0;
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
    for (int c = 0; c < kDeviceLevelClaimColumns; ++c) {
      st->construct_next_block[c] = 0;
    }
    st->construct_grid_y = grid_y;
    st->construct_min_rows = static_cast<int>(min_rows);
    st->max_smaller = max_smaller;
    st->live_pairs = n;
  }
}

}  // namespace Falcata

#endif  // USE_CUDA && __CUDACC__
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_DEVICE_LEVELS_CUH_
