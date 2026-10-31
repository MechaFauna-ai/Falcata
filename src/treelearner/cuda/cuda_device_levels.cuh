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


/*! \brief one compare-exchange of the select's bitonic network at positions i < j = i ^ stride of a stage of width
 *  \p width: candidates order by gain descending, then leaf ascending (the host's stable_sort of leaf-ascending
 *  candidates); padding (leaf -1) sorts after every candidate */
__device__ __forceinline__ void DeviceLevelSortStep(int* leaves, double* gains, const int i, const int j,
                                                    const bool ascending_block) {
  const int a = leaves[i];
  const int b = leaves[j];
  const double ga = gains[i];
  const double gb = gains[j];
  const bool a_first = a >= 0 && (b < 0 || ga > gb || (ga == gb && a < b));
  const bool b_first = b >= 0 && (a < 0 || gb > ga || (gb == ga && b < a));
  if (ascending_block ? b_first : a_first) {
    leaves[i] = b;
    leaves[j] = a;
    gains[i] = gb;
    gains[j] = ga;
  }
}

/*! \brief the select's bitonic network over sort_size (a power of two) entries of global scratch, one block: the same
 *  compare-exchanges as the shared-memory sort of DeviceLevelSelectBlock, stage by stage. Within a stage the pairs are
 *  disjoint, so the order in which they run cannot matter; the stages of stride < kTile touch only pairs inside one
 *  aligned tile of kTile entries and run in shared memory, one tile at a time (each tile all of its consecutive
 *  sub-tile stages before the next tile), the wider strides run over global memory. The result is the shared sort's,
 *  entry for entry. */
__device__ __forceinline__ void DeviceLevelBitonicSortGlobal(int* leaves, double* gains, const int sort_size) {
  constexpr int kTile = kDeviceTreeSharedSplits;
  __shared__ int t_leaves[kTile];
  __shared__ double t_gains[kTile];
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  const int tile = sort_size < kTile ? sort_size : kTile;
  // the tile stages of widths [first_width, last_width] (strides below tile only, from width / 2 or tile / 2)
  const auto tile_pass = [&](const int first_width, const int last_width) {
    for (int base = 0; base < sort_size; base += tile) {
      __syncthreads();
      for (int i = tid; i < tile; i += nt) {
        t_leaves[i] = leaves[base + i];
        t_gains[i] = gains[base + i];
      }
      __syncthreads();
      for (int width = first_width; width <= last_width; width <<= 1) {
        for (int stride = (width < tile ? width : tile) >> 1; stride > 0; stride >>= 1) {
          for (int i = tid; i < tile; i += nt) {
            const int j = i ^ stride;
            if (j > i) {
              DeviceLevelSortStep(t_leaves, t_gains, i, j, ((base + i) & width) == 0);
            }
          }
          __syncthreads();
        }
      }
      for (int i = tid; i < tile; i += nt) {
        leaves[base + i] = t_leaves[i];
        gains[base + i] = t_gains[i];
      }
    }
    __syncthreads();
  };
  tile_pass(2, tile);
  for (int width = tile << 1; width <= sort_size; width <<= 1) {
    for (int stride = width >> 1; stride >= tile; stride >>= 1) {
      for (int i = tid; i < sort_size; i += nt) {
        const int j = i ^ stride;
        if (j > i) {
          DeviceLevelSortStep(leaves, gains, i, j, (i & width) == 0);
        }
      }
      __syncthreads();
    }
    tile_pass(width, width);
  }
}

/*! \brief selective trees (CUDADeviceLevelConfig::r1_binding): whether a level of cur_leaves leaves (cur_leaves - 1
 *  applied splits) with n valid leaves binds, i.e. the host's greedy simulation would not select all of its
 *  candidates (applied + valid > num_leaves - 1). A level with no valid leaf never binds (cur_leaves <= the budget) */
__device__ __forceinline__ bool DeviceLevelR1Binds(const CUDADeviceLevelConfig& cfg, const int cur_leaves,
                                                   const int n) {
  return cfg.r1_binding != 0 && cur_leaves + n > cfg.num_leaves_budget;
}
/*! \brief the first binding level: record it in the journal (the caller stops the tree) */
__device__ __forceinline__ void DeviceLevelStopBinding(const CUDADeviceLevelConfig& cfg, const int tid) {
  if (tid == 0) {
    cfg.journal[3] = 1;
  }
}

/*! \brief one level's host decisions between its search and its apply: the leaf-ascending valid scan of
 *  CollectSplittableLeaves, then per split the apply descriptor (SplitLevelBatched's host math), the tree record
 *  (ApplyLevelBatched's CUDATreeBatchSplit) and the journal entry, the terminal-region gaps (the windows of the leaves
 *  this level does not split: they cover what SplitLevelBatched's gaps between the sorted split regions cover; one
 *  descriptor per such leaf, adjacent windows not merged: the same rows, copied once each) and the flat layout.
 *  A binding final level applies the host's gain-descending, lower-leaf-tie top-K order; nonbinding levels stay
 *  leaf-ascending. Any block of a multiple of 32 threads; each thread takes contiguous runs of leaves and splits.
 *  Reads every leaf's best split after the level's sync. MAX_SPLITS is the block's capacity (static shared memory
 *  12 * MAX_SPLITS + MAX_SPLITS / 4 bytes): kDeviceTreeSharedSplits in the standalone launch of a whole tree. GLOBAL:
 *  a complete tree's level wider than that, with the same arrays in global scratch of cfg.select_capacity splits (the
 *  same steps and sort network, DeviceLevelBitonicSortGlobal). A level past the capacity stops with the error flag. */
template <int MAX_SPLITS, bool GLOBAL = false>
__device__ __forceinline__ void DeviceLevelSelectBlock(const CUDADeviceLevelConfig& cfg, const int level_arg) {
  static_assert(MAX_SPLITS >= 32 && (MAX_SPLITS & (MAX_SPLITS - 1)) == 0, "the bitonic sort needs a power of two");
  constexpr int kSharedSplits = GLOBAL ? 1 : MAX_SPLITS;
  __shared__ int s_leaves_shared[kSharedSplits];
  __shared__ double s_gains_shared[kSharedSplits];
  __shared__ unsigned int s_split_bits_shared[GLOBAL ? 1 : 2 * MAX_SPLITS / 32];
  int* s_leaves = GLOBAL ? cfg.select_leaves : s_leaves_shared;
  double* s_gains = GLOBAL ? cfg.select_gains : s_gains_shared;
  unsigned int* s_split_bits = GLOBAL ? cfg.select_split_bits : s_split_bits_shared;
  const int max_splits = GLOBAL ? cfg.select_capacity : MAX_SPLITS;
  const int kMaxLeaves = 2 * max_splits;
  __shared__ int s_warp[32];
  __shared__ int s_total;
  CUDADeviceLevelState* st = cfg.state;
  if (st->done) {
    return;
  }
  // (a relaunched body of selective levels passes level < 0: the level the state reached)
  const int level = level_arg >= 0 ? level_arg : st->levels_applied;
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  // the previous level's values are still in the state: this level's leaves and split-info slab base follow them
  const int cur_leaves = st->num_leaves + st->live_splits;
  const int split_cursor = st->split_info_base + st->live_splits;
  const int cat_cursor = st->cat_split_base + st->live_cat_splits;
  const auto stop = [&](const bool flag) {
    if (tid == 0) {
      if (flag) {
        atomicOr(cfg.journal + 2, 1);
      }
      st->done = 1;
      st->live_splits = 0;
      st->live_num_splits = 0;
      st->live_cat_splits = 0;
      st->cat_split_base = cat_cursor;
      st->total_flat_blocks = 0;
      st->num_gaps = 0;
      st->gap_flat_blocks = 0;
      st->num_leaves = cur_leaves;
      st->split_info_base = split_cursor;
    }
  };
  if (cur_leaves > kMaxLeaves || (cfg.whole_tree && cur_leaves > cfg.num_leaves_budget)) {
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
  int n = s_total;
  if (DeviceLevelR1Binds(cfg, cur_leaves, n)) {
    DeviceLevelStopBinding(cfg, tid);
    stop(false);
    return;
  }
  if (n > max_splits) {
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
  if (cfg.whole_tree && n > cfg.num_leaves_budget - cur_leaves) {
    // ArbitrateLevelBudget: only the depth-capped final frontier can be truncated without comparing its future
    // children. Any earlier binding budget is outside this controller's admitted regime; do not silently prune it.
    if (level + 1 != cfg.max_depth) {
      stop(true);
      return;
    }
    const int remaining = cfg.num_leaves_budget - cur_leaves;
    if (remaining == 0) {
      stop(false);
      return;
    }
    // Cooperative bounded bitonic sort of valid leaf IDs. Padding sorts after valid candidates; exact gain ties
    // retain the original leaf-ascending order of the host stable_sort. Keep the sorted order for APPLY as well as
    // selection: right-child IDs and the terminal journal depend on it. No sort runs on nonbinding levels.
    int sort_size = 1;
    while (sort_size < n) sort_size <<= 1;
    for (int i = tid; i < sort_size; i += nt) {
      if (i < n) {
        s_gains[i] = cfg.leaf_best_split_info[s_leaves[i]].gain;
      } else {
        s_leaves[i] = -1;
        s_gains[i] = 0.0;
      }
    }
    __syncthreads();
    if constexpr (GLOBAL) {
      DeviceLevelBitonicSortGlobal(s_leaves, s_gains, sort_size);
    } else {
      for (int width = 2; width <= sort_size; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
          for (int i = tid; i < sort_size; i += nt) {
            const int j = i ^ stride;
            if (j > i) {
              DeviceLevelSortStep(s_leaves, s_gains, i, j, (i & width) == 0);
            }
          }
          __syncthreads();
        }
      }
    }
    n = remaining;
  }
  // ---- per split: apply descriptor, tree record, journal ----
  const int split_chunk = (n + nt - 1) / nt;
  const int k_begin = min(tid * split_chunk, n);
  const int k_end = min(k_begin + split_chunk, n);
  // categorical datasets: the level's categorical splits in apply order (ApplyLevelBatched's list for its bitset
  // kernel). The tree record takes CUDATree::SplitLevel's launch order: the numerical splits first (each keeping its
  // rank among them), then the categorical ones in apply order; the splits act on disjoint leaves, so the order of
  // the record's entries does not change what they write. A categorical split is one with categorical thresholds (the
  // host record's test); the partition's test is the feature's type (SplitLevelBatched): they agree, a disagreement
  // stops the tree with error flag 2.
  const bool cat_enabled = cfg.cat.infos != nullptr;
  int cat_seen = 0;
  int num_cat = 0;
  if (cat_enabled) {
    int my_cat = 0;
    for (int k = k_begin; k < k_end; ++k) {
      if (cfg.leaf_best_split_info[s_leaves[k]].num_cat_threshold > 0) {
        ++my_cat;
      }
    }
    cat_seen = DeviceLevelBlockScan(my_cat, s_warp, &s_total);
    num_cat = s_total;
  }
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
    const bool is_cat = cat_enabled && info->num_cat_threshold > 0;
    if (cat_enabled && is_cat != (m.is_categorical != 0)) {
      atomicOr(cfg.journal + 2, 2);
    }
    const int record_index = !cat_enabled ? k : (is_cat ? n - num_cat + cat_seen : k - cat_seen);
    if (cfg.skip_tree_record == 0) {
      CUDATreeBatchSplit& ts = cfg.tree_batch_splits[record_index];
      ts.leaf_index = leaf;
      ts.num_leaves_at_split = right;
      ts.real_feature_index = m.real_feature_index;
      // (a categorical split's threshold is not a bin of the threshold table; its record does not read this field)
      ts.real_threshold = is_cat ? 0.0 : cfg.real_thresholds[m.real_threshold_offset + threshold];
      ts.missing_type = m.missing_type;
      ts.split_info = info;
    }
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
    // built in registers and stored whole (wide stores, as the gap descriptors below); the offsets follow
    CUDAHybridApplyDescriptor d;
    memset(&d, 0, sizeof(CUDAHybridApplyDescriptor));
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
    if (is_cat) {
      // SplitLevelBatched's categorical descriptor: the inner bitset (in the level's slab, where the bitset kernel
      // builds it) read over the feature's bin-range cap; the default directions follow from the bitset (the bitset
      // kernel patches them, as HybridBuildCatBitsetArenaKernel does)
      const uint32_t mfb_offset = m.most_freq_bin == 0 ? 1u : 0u;
      d.cat_bitset_len = static_cast<int>((m.max_bin - m.min_bin + mfb_offset) / 32 + 1);
      d.cat_bitset = cfg.cat.bitset_inner_slab + static_cast<size_t>(cat_seen) * cfg.cat.bitset_inner_stride;
      d.cat_mfb_offset = static_cast<int8_t>(mfb_offset);
      cfg.cat.infos[cat_seen] = info;
      cfg.cat.inner_features[cat_seen] = inner_feature;
      cfg.cat.split_index[cat_seen] = k;
      ++cat_seen;
    }
    cfg.apply_descs[k] = d;
    my_slots += d.num_blocks + 1;
    my_flat += d.num_blocks;
    atomicOr(&s_split_bits[leaf >> 5], 1u << (leaf & 31));
    int* record = cfg.journal + kDeviceLevelJournalHeader + kDeviceLevelJournalRecord * (split_cursor + k);
    record[0] = leaf;
    record[1] = inner_feature;
    record[2] = level;
    if (cfg.split_info_journal != nullptr) {
      // the selective prefix's record of this split (the host rebuilds its applied records from it): the bytes the
      // host's own best-split readback would have copied, before the children's search reuses the leaf's entry
      static_assert(sizeof(CUDASplitInfo) % sizeof(uint64_t) == 0, "CUDASplitInfo is copied as 8-byte words");
      const uint64_t* src_words = reinterpret_cast<const uint64_t*>(info);
      uint64_t* dst_words = reinterpret_cast<uint64_t*>(cfg.split_info_journal + split_cursor + k);
      for (size_t w = 0; w < sizeof(CUDASplitInfo) / sizeof(uint64_t); ++w) {
        dst_words[w] = src_words[w];
      }
    }
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
  if constexpr (GLOBAL) {
    // A deep level has thousands of terminal leaves, one descriptor each, all written by this one block: each thread
    // records its gaps' (leaf, flat chunk start) compactly, then consecutive threads write consecutive 8-byte words
    // of the descriptors (coalesced). The same descriptors as the per-leaf writes below.
    for (int leaf = leaf_begin; leaf < leaf_end; ++leaf) {
      const data_size_t num = cfg.leaf_num_data[leaf];
      if (((s_split_bits[leaf >> 5] >> (leaf & 31)) & 1u) == 0 && num > 0) {
        cfg.select_gaps[gap_pos] = make_int2(leaf, total_flat_blocks + gap_flat_pos);
        ++gap_pos;
        gap_flat_pos += (num + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
      }
    }
    __syncthreads();
    static_assert(sizeof(CUDAHybridApplyDescriptor) % sizeof(uint64_t) == 0, "descriptors are written as words");
    constexpr int kWords = static_cast<int>(sizeof(CUDAHybridApplyDescriptor) / sizeof(uint64_t));
    uint64_t* gap_words = reinterpret_cast<uint64_t*>(cfg.apply_descs + n);
    for (int w = tid; w < num_gaps * kWords; w += nt) {
      const int gap = w / kWords;
      const int2 rec = cfg.select_gaps[gap];
      const data_size_t num = cfg.leaf_num_data[rec.x];
      CUDAHybridApplyDescriptor g;
      memset(&g, 0, sizeof(CUDAHybridApplyDescriptor));
      g.left_leaf_index = rec.x;  // the final whole apply also maps this terminal leaf
      g.leaf_data_start = cfg.leaf_data_start[rec.x];
      g.num_data_in_leaf = num;
      g.num_blocks = (num + cfg.gap_block_dim - 1) / cfg.gap_block_dim;
      g.flat_block_start = rec.y;
      gap_words[w] = reinterpret_cast<const uint64_t*>(&g)[w - gap * kWords];
    }
  } else {
  for (int leaf = leaf_begin; leaf < leaf_end; ++leaf) {
    const data_size_t num = cfg.leaf_num_data[leaf];
    if (((s_split_bits[leaf >> 5] >> (leaf & 31)) & 1u) == 0 && num > 0) {
      // built in registers and stored whole (wide stores): a memset of the global descriptor is byte stores, and a
      // deep level's thousands of terminal leaves made it the select's bottleneck
      CUDAHybridApplyDescriptor g;
      memset(&g, 0, sizeof(CUDAHybridApplyDescriptor));
      g.left_leaf_index = leaf;  // the final whole apply also maps this terminal leaf
      g.leaf_data_start = cfg.leaf_data_start[leaf];
      g.num_data_in_leaf = num;
      g.num_blocks = (num + cfg.gap_block_dim - 1) / cfg.gap_block_dim;
      g.flat_block_start = total_flat_blocks + gap_flat_pos;
      cfg.apply_descs[n + gap_pos] = g;
      ++gap_pos;
      gap_flat_pos += (num + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    }
  }
  }
  if (tid == 0) {
    st->live_splits = n;
    st->live_num_splits = n - num_cat;
    st->live_cat_splits = num_cat;
    st->cat_split_base = cat_cursor;
    st->total_flat_blocks = total_flat_blocks;
    st->num_gaps = num_gaps;
    st->gap_flat_blocks = gap_flat_blocks;
    st->split_info_base = split_cursor;
    st->num_leaves = cur_leaves;
    st->levels_applied = level + 1;
    cfg.journal[0] = level + 1;
    cfg.journal[1] = split_cursor + n;
  }
}

/*! \brief DeviceLevelSelectBlock for a nonbinding prefix level (no whole tree): the copy fused into every prefix's
 *  level sync (cfg staged in shared memory by the caller) and the prefix's standalone select (cfg by value). Each thread holds at most CHUNK leaves and CHUNK splits (2 * MAX_SPLITS <= CHUNK * blockDim.x; past it
 *  the level stops with the error flag) and first loads every input of them into registers, then stores: the plain
 *  block interleaves dependent global loads with stores the compiler must assume alias them, so a thread's leaves,
 *  splits and journal words each cost a memory round trip in sequence. Same results as the plain block on such a
 *  level, element for element:
 *   - split set and order: the valid leaves in leaf-ascending order (contiguous per-thread runs and an exclusive
 *     scan, as there); a nonbinding level splits every one of them, which is also its split mark;
 *   - per split: the same descriptor fields from the same inputs; the slot prefix (num_blocks + 1 per split) equals
 *     the flat prefix plus the split index, so one scan gives both;
 *   - gaps: the leaves not split (not valid) with rows, leaf-ascending, the same descriptors;
 *   - journal: the same ints, and the same 8-byte words of each split's best-split record (read after the sync);
 *   - tree-record descriptors only when cfg.skip_tree_record is 0. */
template <int MAX_SPLITS, int CHUNK = 4>
__device__ __forceinline__ void DeviceLevelSelectBlockFast(const CUDADeviceLevelConfig& cfg, const int level_arg) {
  constexpr int kMaxLeaves = 2 * MAX_SPLITS;
  __shared__ int s_leaves[MAX_SPLITS];
  __shared__ int s_warp[32];
  __shared__ int s_total;
  CUDADeviceLevelState* st = cfg.state;
  if (st->done) {
    return;
  }
  const int level = level_arg >= 0 ? level_arg : st->levels_applied;  // (level < 0: see DeviceLevelSelectBlock)
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  const int cur_leaves = st->num_leaves + st->live_splits;
  const int split_cursor = st->split_info_base + st->live_splits;
  const auto stop = [&](const bool flag) {
    if (tid == 0) {
      if (flag) {
        atomicOr(cfg.journal + 2, 1);
      }
      st->done = 1;
      st->live_splits = 0;
      st->live_num_splits = 0;
      st->total_flat_blocks = 0;
      st->num_gaps = 0;
      st->gap_flat_blocks = 0;
      st->num_leaves = cur_leaves;
      st->split_info_base = split_cursor;
    }
  };
  const int leaf_chunk = (cur_leaves + nt - 1) / nt;
  if (cfg.whole_tree || cur_leaves > kMaxLeaves || leaf_chunk > CHUNK) {
    stop(true);  // outside this variant's regime (the learner launches it only inside it)
    return;
  }
  const int leaf_begin = min(tid * leaf_chunk, cur_leaves);
  const int leaf_end = min(leaf_begin + leaf_chunk, cur_leaves);
  // ---- one load pass over this thread's leaves ----
  bool valid[CHUNK];
  data_size_t num[CHUNK];
  data_size_t start[CHUNK];
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    const int leaf = leaf_begin + c;
    const bool in = leaf < leaf_end;
    valid[c] = in && cfg.leaf_best_split_info[leaf].is_valid;
    num[c] = in ? cfg.leaf_num_data[leaf] : 0;
    start[c] = in ? cfg.leaf_data_start[leaf] : 0;
  }
  int my_count = 0;
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    my_count += valid[c] ? 1 : 0;
  }
  // ---- CollectSplittableLeaves: leaf-ascending is_valid scan ----
  int write_pos = DeviceLevelBlockScan(my_count, s_warp, &s_total);
  const int n = s_total;
  if (DeviceLevelR1Binds(cfg, cur_leaves, n)) {
    DeviceLevelStopBinding(cfg, tid);
    stop(false);
    return;
  }
  if (n > MAX_SPLITS) {
    stop(true);
    return;
  }
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    if (valid[c]) {
      s_leaves[write_pos++] = leaf_begin + c;
    }
  }
  __syncthreads();
  if (n == 0) {
    stop(false);  // no splittable leaf: the tree is complete; later kernels see done / zero live counts
    return;
  }
  const int split_chunk = (n + nt - 1) / nt;
  if (split_chunk > CHUNK) {
    stop(true);
    return;
  }
  const int k_begin = min(tid * split_chunk, n);
  const int k_end = min(k_begin + split_chunk, n);
  const bool record = cfg.skip_tree_record == 0;
  // ---- per split, loads: the leaf's entry and window, then the feature's tables ----
  int s_leaf[CHUNK];
  int s_feature[CHUNK];
  uint32_t s_threshold[CHUNK];
  bool s_default_left[CHUNK];
  data_size_t s_num[CHUNK];
  data_size_t s_start[CHUNK];
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    const int k = k_begin + c;
    if (k < k_end) {
      const int leaf = s_leaves[k];
      const CUDASplitInfo* info = cfg.leaf_best_split_info + leaf;
      s_leaf[c] = leaf;
      s_feature[c] = info->inner_feature_index;
      s_threshold[c] = info->threshold;
      s_default_left[c] = info->default_left;
      s_num[c] = cfg.leaf_num_data[leaf];
      s_start[c] = cfg.leaf_data_start[leaf];
    }
  }
  CUDAHybridGraphFeatureMeta s_meta[CHUNK];
  CUDAHybridGraphFeatureSource s_src[CHUNK];
  double s_real_threshold[CHUNK];
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    if (k_begin + c < k_end) {
      s_meta[c] = cfg.feature_meta[s_feature[c]];
      s_src[c] = cfg.feature_source[s_feature[c]];
    }
  }
  if (record) {
#pragma unroll
    for (int c = 0; c < CHUNK; ++c) {
      if (k_begin + c < k_end) {
        s_real_threshold[c] = cfg.real_thresholds[s_meta[c].real_threshold_offset + s_threshold[c]];
      }
    }
  }
  // ---- per split, stores: apply descriptor, tree record, journal ----
  int my_flat = 0;
  int s_blocks[CHUNK];
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    const int k = k_begin + c;
    if (k >= k_end) {
      continue;
    }
    const int leaf = s_leaf[c];
    const int right = cur_leaves + k;
    const uint32_t threshold = s_threshold[c];
    const CUDAHybridGraphFeatureMeta& m = s_meta[c];
    const CUDAHybridGraphFeatureSource& src = s_src[c];
    const CUDASplitInfo* info = cfg.leaf_best_split_info + leaf;
    if (record) {
      CUDATreeBatchSplit& ts = cfg.tree_batch_splits[k];
      ts.leaf_index = leaf;
      ts.num_leaves_at_split = right;
      ts.real_feature_index = m.real_feature_index;
      ts.real_threshold = s_real_threshold[c];
      ts.missing_type = m.missing_type;
      ts.split_info = info;
    }
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
    if ((m.missing_is_zero || m.missing_is_na) && s_default_left[c]) {
      split_missing_default_to_left = 1;
      missing_default_leaf_index = leaf;
    }
    const data_size_t parent_num_data = s_num[c];
    const int num_blocks = (parent_num_data + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
      SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    s_blocks[c] = num_blocks;
    my_flat += num_blocks;
    CUDAHybridApplyDescriptor& d = cfg.apply_descs[k];
    d.column_data = src.column_data;
    d.best_split_info = info;
    d.smaller_leaf_splits = cfg.pair_slots + 2 * k;
    d.larger_leaf_splits = cfg.pair_slots + 2 * k + 1;
    d.leaf_data_start = s_start[c];
    d.num_data_in_leaf = parent_num_data;
    d.num_blocks = num_blocks;
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
    int* record = cfg.journal + kDeviceLevelJournalHeader + kDeviceLevelJournalRecord * (split_cursor + k);
    record[0] = leaf;
    record[1] = s_feature[c];
    record[2] = level;
  }
  // flat chunks (num_blocks per split) in split order; block-offset slots (num_blocks + 1 per split) = flat + index
  int flat_pos = DeviceLevelBlockScan(my_flat, s_warp, &s_total);
  const int total_flat_blocks = s_total;
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    const int k = k_begin + c;
    if (k < k_end) {
      CUDAHybridApplyDescriptor& d = cfg.apply_descs[k];
      d.block_offset_start = flat_pos + k;
      d.flat_block_start = flat_pos;
      flat_pos += s_blocks[c];
    }
  }
  // ---- terminal-region gaps: this level splits exactly the valid leaves ----
  int my_gaps = 0;
  int my_gap_flat = 0;
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    if (leaf_begin + c < leaf_end && !valid[c] && num[c] > 0) {
      ++my_gaps;
      my_gap_flat += (num[c] + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    }
  }
  int gap_pos = DeviceLevelBlockScan(my_gaps, s_warp, &s_total);
  const int num_gaps = s_total;
  int gap_flat_pos = DeviceLevelBlockScan(my_gap_flat, s_warp, &s_total);
  const int gap_flat_blocks = s_total;
#pragma unroll
  for (int c = 0; c < CHUNK; ++c) {
    if (leaf_begin + c < leaf_end && !valid[c] && num[c] > 0) {
      CUDAHybridApplyDescriptor g;  // registers, stored whole (see DeviceLevelSelectBlock)
      memset(&g, 0, sizeof(CUDAHybridApplyDescriptor));
      g.left_leaf_index = leaf_begin + c;  // the final whole apply also maps this terminal leaf
      g.leaf_data_start = start[c];
      g.num_data_in_leaf = num[c];
      g.num_blocks = (num[c] + cfg.gap_block_dim - 1) / cfg.gap_block_dim;
      g.flat_block_start = total_flat_blocks + gap_flat_pos;
      cfg.apply_descs[n + gap_pos] = g;
      ++gap_pos;
      gap_flat_pos += (num[c] + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    }
  }
  // ---- the journaled best-split records: kUnroll independent word loads per thread before their stores ----
  if (cfg.split_info_journal != nullptr) {
    static_assert(sizeof(CUDASplitInfo) % sizeof(uint64_t) == 0, "CUDASplitInfo is copied as 8-byte words");
    constexpr int kWords = static_cast<int>(sizeof(CUDASplitInfo) / sizeof(uint64_t));
    constexpr int kUnroll = 8;
    const uint64_t* src_words = reinterpret_cast<const uint64_t*>(cfg.leaf_best_split_info);
    uint64_t* dst_words = reinterpret_cast<uint64_t*>(cfg.split_info_journal + split_cursor);
    const int total_words = n * kWords;
    for (int base = 0; base < total_words; base += kUnroll * nt) {
      uint64_t v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int i = base + u * nt + tid;
        if (i < total_words) {
          const int k = i / kWords;
          v[u] = src_words[static_cast<size_t>(s_leaves[k]) * kWords + (i - k * kWords)];
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int i = base + u * nt + tid;
        if (i < total_words) {
          dst_words[i] = v[u];
        }
      }
    }
  }
  if (tid == 0) {
    st->live_splits = n;
    st->live_num_splits = n;  // (no categorical feature reaches the prefix)
    st->total_flat_blocks = total_flat_blocks;
    st->num_gaps = num_gaps;
    st->gap_flat_blocks = gap_flat_blocks;
    st->split_info_base = split_cursor;
    st->num_leaves = cur_leaves;
    st->levels_applied = level + 1;
    cfg.journal[0] = level + 1;
    cfg.journal[1] = split_cursor + n;
  }
}

/*! \brief pair chunking (cfg.num_chunks > 1): chunk c's state, a copy of the level's with live_pairs = its pairs among
 *  [c * chunk_pairs, (c + 1) * chunk_pairs) that are live. A chunk's fix/subtract and find read the copy with the
 *  chunk's descriptors, so they see a level of chunk_pairs pairs. After the level state's last store (block-wide). */
__device__ __forceinline__ void DeviceLevelWriteChunkStates(const CUDADeviceLevelConfig& cfg,
                                                            const CUDADeviceLevelState* st) {
  if (cfg.num_chunks <= 1) {
    return;
  }
  __syncthreads();
  for (int c = static_cast<int>(threadIdx.x); c < cfg.num_chunks; c += static_cast<int>(blockDim.x)) {
    CUDADeviceLevelState* chunk = cfg.chunk_states + c;
    *chunk = *st;
    const int live = st->live_pairs - c * cfg.chunk_pairs;
    chunk->live_pairs = live < 0 ? 0 : (live < cfg.chunk_pairs ? live : cfg.chunk_pairs);
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
    DeviceLevelWriteChunkStates(cfg, st);
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
  DeviceLevelWriteChunkStates(cfg, st);
}

}  // namespace Falcata

#endif  // USE_CUDA && __CUDACC__
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_DEVICE_LEVELS_CUH_
