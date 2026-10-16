/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief cuda_plan key device_levels: the select and pairs kernels of the device-driven level prefix (see
 *  cuda_device_levels.hpp). Each replays, for one level, exactly the host decisions of the two-sync level loop it
 *  stands in for (CollectSplittableLeaves + ApplyLevelBatched + SplitLevelBatched; EnqueueLevelBestSplitSearch +
 *  LaunchConstructHistogramBatchedKernel's grid math), so the kernels after them see the same descriptors.
 */

#ifdef USE_CUDA

#include "cuda_device_levels.hpp"
#include "cuda_device_levels.cuh"

#include <Falcata/cuda/cuda_utils.hu>

#include <algorithm>

#include <Falcata/cuda/cuda_split_info.hpp>
#include <Falcata/cuda/cuda_tree.hpp>

#include "cuda_data_partition.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_hybrid_graph.hpp"
#include "cuda_leaf_splits.hpp"
#include "cuda_pdl_launch.hpp"

namespace Falcata {

namespace {

constexpr int kSelectMaxLeaves = 2 * kDeviceLevelMaxPairs;

__global__ void DeviceLevelResetKernel(const CUDADeviceLevelConfig cfg, const int num_leaves) {
  CUDADeviceLevelState* st = cfg.state;
  if (threadIdx.x == 0) {
    st->live_splits = 0;
    st->total_flat_blocks = 0;
    st->num_gaps = 0;
    st->gap_flat_blocks = 0;
    st->split_info_base = 0;
    st->num_leaves = num_leaves;
    st->root_num_data = cfg.leaf_num_data[0];
    st->levels_applied = 0;
    st->done = 0;
    st->live_pairs = 0;
    st->construct_grid_y = 0;
    st->construct_min_rows = 0;
    st->construct_total_blocks = 0;
    st->max_smaller = 0;
  }
  if (threadIdx.x < kDeviceLevelJournalHeader) {
    cfg.journal[threadIdx.x] = 0;
  }
}

// One level's host decisions between its search and its apply (see the file comment). Block of
// DeviceLevelThreads(2^level) threads: >= 2 x the level's most splits, so one thread per split, one per gap slot
// (splits + 1) and one per sort lane.
__global__ void DeviceLevelSelectKernel(const CUDADeviceLevelConfig cfg, const int level) {
  __shared__ int s_leaves[kSelectMaxLeaves];
  __shared__ int s_rkey[kSelectMaxLeaves];
  __shared__ int s_warp[32];
  __shared__ int s_total;
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the level sync's leaf entries are complete and visible
  CUDADeviceLevelState* st = cfg.state;
  if (st->done) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  // the previous level's values are still in the state: this level's leaves and split-info slab base follow them
  const int cur_leaves = st->num_leaves + st->live_splits;
  const int split_cursor = st->split_info_base + st->live_splits;
  // ---- CollectSplittableLeaves: leaf-ascending is_valid scan (contiguous per-thread chunks keep the order) ----
  const int chunk = (cur_leaves + nt - 1) / nt;
  const int chunk_begin = tid * chunk;
  const int chunk_end = min(chunk_begin + chunk, cur_leaves);
  int my_count = 0;
  for (int leaf = chunk_begin; leaf < chunk_end; ++leaf) {
    if (cfg.leaf_best_split_info[leaf].is_valid) {
      ++my_count;
    }
  }
  const int my_offset = DeviceLevelBlockScan(my_count, s_warp, &s_total);
  const int n = s_total;
  if (n > kDeviceLevelMaxPairs || n >= nt) {
    // outside the gated regime (a level of 2^level leaves splits at most 2^level < nt of them): stop and flag it
    if (tid == 0) {
      atomicOr(cfg.journal + 2, 1);
      st->done = 1;
      st->live_splits = 0;
      st->total_flat_blocks = 0;
      st->num_gaps = 0;
      st->gap_flat_blocks = 0;
    }
    return;
  }
  int write_pos = my_offset;
  for (int leaf = chunk_begin; leaf < chunk_end; ++leaf) {
    if (cfg.leaf_best_split_info[leaf].is_valid) {
      s_leaves[write_pos++] = leaf;
    }
  }
  __syncthreads();
  if (n == 0) {
    // no splittable leaf: the tree is complete; later levels' kernels see done / zero live counts
    if (tid == 0) {
      st->done = 1;
      st->live_splits = 0;
      st->total_flat_blocks = 0;
      st->num_gaps = 0;
      st->gap_flat_blocks = 0;
      st->num_leaves = cur_leaves;
      st->split_info_base = split_cursor;
    }
    return;
  }
  // ---- per split: apply descriptor (SplitLevelBatched), tree record (ApplyLevelBatched), journal ----
  int my_blocks = 0;
  if (tid < n) {
    const int leaf = s_leaves[tid];
    const int right = cur_leaves + tid;
    const CUDASplitInfo* info = cfg.leaf_best_split_info + leaf;
    const int inner_feature = info->inner_feature_index;
    const uint32_t threshold = info->threshold;
    const CUDAHybridGraphFeatureMeta m = cfg.feature_meta[inner_feature];
    const CUDAHybridGraphFeatureSource src = cfg.feature_source[inner_feature];
    CUDATreeBatchSplit& ts = cfg.tree_batch_splits[tid];
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
    CUDAHybridApplyDescriptor& d = cfg.apply_descs[tid];
    d.column_data = src.column_data;
    d.best_split_info = info;
    d.smaller_leaf_splits = cfg.pair_slots + 2 * tid;
    d.larger_leaf_splits = cfg.pair_slots + 2 * tid + 1;
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
    my_blocks = d.num_blocks;
    cfg.journal[kDeviceLevelJournalHeader + 2 * (split_cursor + tid)] = leaf;
    cfg.journal[kDeviceLevelJournalHeader + 2 * (split_cursor + tid) + 1] = inner_feature;
  }
  // block-offset slots (num_blocks + 1 per split) and flat chunks (num_blocks per split), in split order
  const int slot_start = DeviceLevelBlockScan(tid < n ? my_blocks + 1 : 0, s_warp, &s_total);
  const int flat_start = DeviceLevelBlockScan(tid < n ? my_blocks : 0, s_warp, &s_total);
  const int total_flat_blocks = s_total;
  if (tid < n) {
    cfg.apply_descs[tid].block_offset_start = slot_start;
    cfg.apply_descs[tid].flat_block_start = flat_start;
  }
  // ---- terminal-region gaps: the windows of the leaves this level does not split. The leaves' windows partition
  // [0, root rows), so they cover what SplitLevelBatched's gaps between the sorted split regions cover; one
  // descriptor per such leaf (adjacent windows are not merged: the same rows, copied once each) ----
  for (int leaf = tid; leaf < cur_leaves; leaf += nt) {
    s_rkey[leaf] = 0;
  }
  __syncthreads();
  if (tid < n) {
    s_rkey[s_leaves[tid]] = 1;  // split this level
  }
  __syncthreads();
  int my_gaps = 0;
  int my_gap_flat = 0;
  for (int leaf = chunk_begin; leaf < chunk_end; ++leaf) {
    const data_size_t num = cfg.leaf_num_data[leaf];
    if (s_rkey[leaf] == 0 && num > 0) {
      ++my_gaps;
      my_gap_flat += (num + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    }
  }
  int gap_pos = DeviceLevelBlockScan(my_gaps, s_warp, &s_total);
  const int num_gaps = s_total;
  int gap_flat_pos = DeviceLevelBlockScan(my_gap_flat, s_warp, &s_total);
  const int gap_flat_blocks = s_total;
  for (int leaf = chunk_begin; leaf < chunk_end; ++leaf) {
    const data_size_t num = cfg.leaf_num_data[leaf];
    if (s_rkey[leaf] == 0 && num > 0) {
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
    cfg.journal[3 + level] = n;
    cfg.journal[0] = level + 1;
    cfg.journal[1] = split_cursor + n;
  }
}

// The children's search descriptors of the level just applied (DeviceLevelPairsBlock, cuda_device_levels.cuh), as a
// launch of its own where the apply does not run it beside its partition
__global__ void DeviceLevelPairsKernel(const CUDADeviceLevelConfig cfg, const int level) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the apply's counts and child structs are complete and visible
  DeviceLevelPairsBlock(cfg, level);
}

}  // anonymous namespace

void LaunchDeviceLevelReset(const CUDADeviceLevelConfig& config, const int num_leaves, cudaStream_t stream) {
  DeviceLevelResetKernel<<<1, 32, 0, stream>>>(config, num_leaves);
}

void LaunchDeviceLevelSelect(const CUDADeviceLevelConfig& config, const int level, const int max_splits,
                             cudaStream_t stream) {
  // cuda_plan key pdl_level_chain: a programmatic dependent of the level sync launched just before on this stream
  LaunchLevelChainKernel(PdlLevelChainEnabled(), DeviceLevelSelectKernel, dim3(1), dim3(DeviceLevelThreads(max_splits)),
                         0, stream, config, level);
}

void LaunchDeviceLevelPairs(const CUDADeviceLevelConfig& config, const int level, const int max_splits,
                            cudaStream_t stream) {
  // cuda_plan key pdl_level_chain: a programmatic dependent of the apply's partition kernel launched just before
  LaunchLevelChainKernel(PdlLevelChainEnabled(), DeviceLevelPairsKernel, dim3(1), dim3(DeviceLevelThreads(max_splits)),
                         0, stream, config, level);
}

}  // namespace Falcata

#endif  // USE_CUDA
