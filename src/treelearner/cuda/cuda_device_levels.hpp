/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief cuda_plan key device_levels: the first levels of the quantized two-sync level prefix, run device-driven.
 *
 *  The host loop of CUDASingleGPUTreeLearner::TrainLevelWisePrefix reads two things back per level: the level's best
 *  splits (to pick the splittable leaves and build the apply descriptors) and the apply's split info (to build the
 *  children's search descriptors). Where the leaf budget cannot bind (2^(level + 1) <= num_leaves) and the children
 *  stay below max_depth, neither readback feeds a host DECISION: every valid leaf is split. Those levels run here
 *  without a host round trip:
 *   - DeviceLevelSelectKernel (one block, after the level's search): the leaf-ascending valid scan of
 *     CollectSplittableLeaves, then per split the apply descriptor (SplitLevelBatched's host math), the tree record
 *     (ApplyLevelBatched's CUDATreeBatchSplit), the terminal-region gap descriptors and the flat block layout;
 *   - the apply kernels and the tree record, launched for the level's worst case (2^level splits), read the live
 *     counts from CUDADeviceLevelState and exit past them;
 *   - DeviceLevelPairsKernel (one block, after the apply): the children's search descriptors exactly as
 *     EnqueueLevelBestSplitSearch builds them on the host (counts, hessian sums, the skip_unsplittable /
 *     min_data / min_hessian / max_depth gates, the quantized bit widths of SetNumBitsInHistogramBin), the level's
 *     largest smaller leaf and the construct's row grouping and block map (LaunchConstructHistogramBatchedKernel's
 *     host math), so the search kernels see the descriptors and the grid layout the host flow would give them.
 *  After the last device level the host reads the journal, the split infos and the leaf best splits back once,
 *  replays its bookkeeping, and the host loop continues with the next level.
 */

#ifndef FALCATA_TREELEARNER_CUDA_CUDA_DEVICE_LEVELS_HPP_
#define FALCATA_TREELEARNER_CUDA_CUDA_DEVICE_LEVELS_HPP_

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <Falcata/meta.h>

#include <cstdint>

namespace Falcata {

struct CUDASplitInfo;
struct CUDATreeBatchSplit;
struct CUDAHybridApplyDescriptor;
struct CUDAHybridPairDescriptor;
struct CUDALeafSplitsStruct;
struct CUDAHybridGraphFeatureMeta;
struct CUDAHybridGraphFeatureSource;

/*! \brief most pairs of one device level of the prefix (the select / pairs kernels' block covers 2 * this); also the
 *  select capacity of the copy fused into the prefix's level sync (SyncBestSplitForLevelUsedTasksKernel) */
constexpr int kDeviceLevelMaxPairs = 512;
/*! \brief cuda_plan key device_tree: splits of a complete tree's level that the standalone select holds in static
 *  shared memory (its bitonic sort's capacity). Wider levels (max_depth > 12: the final level has 2^(max_depth - 1)
 *  splits) keep the select's arrays in global scratch (CUDADeviceLevelConfig::select_*) and run the same sort network
 *  block-looped over them. */
constexpr int kDeviceTreeSharedSplits = 2048;
/*! \brief launch columns of a device level's construct that claim block rows dynamically */
constexpr int kDeviceLevelClaimColumns = 64;
/*! \brief journal layout (ints): [0] applied levels, [1] total splits, [2] error flags, [3, 5) reserved, then per split
 *  in applied order a record {left leaf, inner feature, level} of kDeviceLevelJournalRecord ints at
 *  kDeviceLevelJournalHeader. The level tags make any level count fit; the records of a level are contiguous and the
 *  tags ascend. */
constexpr int kDeviceLevelJournalHeader = 5;
constexpr int kDeviceLevelJournalRecord = 3;

/*! \brief the applied splits per level from a journal's level tags (host replay): counts[L] for L < num_levels */
inline bool DeviceLevelJournalCounts(const int* journal, const int total_splits, const int num_levels,
                                     int* counts) {
  for (int level = 0; level < num_levels; ++level) {
    counts[level] = 0;
  }
  int previous = 0;
  for (int s = 0; s < total_splits; ++s) {
    const int level = journal[kDeviceLevelJournalHeader + kDeviceLevelJournalRecord * s + 2];
    if (level < previous || level >= num_levels) {
      return false;
    }
    previous = level;
    ++counts[level];
  }
  return true;
}

/*! \brief device-side counts of the level in flight, written by the select and pairs kernels and read by the
 *  worst-case launches (nullptr everywhere on the host-launched path) */
struct CUDADeviceLevelState {
  // ---- written by DeviceLevelSelectKernel (the level's apply) ----
  /*! \brief splits of the level being applied (0 once the tree stopped) */
  int live_splits;
  /*! \brief the level's split chunks (flat_block_start of a split + its num_blocks) */
  int total_flat_blocks;
  /*! \brief terminal-region copy descriptors (after the split descriptors) and their flat chunks */
  int num_gaps;
  int gap_flat_blocks;
  /*! \brief split-info slab slot of the level's first split (splits of the earlier device levels) */
  int split_info_base;
  /*! \brief leaves before the level's apply */
  int num_leaves;
  /*! \brief levels applied so far; set to stop once a level has no splittable leaf */
  int levels_applied;
  int done;
  /*! \brief rows of the tree's root (the in-bag rows), from the partition's leaf 0 at the reset */
  data_size_t root_num_data;
  /*! \brief blocks of the level sync that wrote their leaf entry (the last one runs the next level's select) */
  unsigned int sync_blocks_done;
  // ---- written by DeviceLevelPairsKernel (the children's search) ----
  /*! \brief pairs of the search in flight (== the applied level's splits) */
  int live_pairs;
  /*! \brief the construct's level grid y (HybridBatchedConstructGridDimYQuant of the largest smaller leaf) */
  int construct_grid_y;
  /*! \brief cuda_plan key dense_level_row_blocks: the level's rows-per-thread floor (0 = off) */
  int construct_min_rows;
  /*! \brief block rows with rows over all pairs (the block map's total, cuda_plan key dense_block_map) */
  int construct_total_blocks;
  /*! \brief the construct's next unclaimed block row per launch column (blockIdx.x: a feature partition); its blocks
   *  claim the map's rows one at a time (launches of more columns stride statically) */
  int construct_next_block[kDeviceLevelClaimColumns];
  /*! \brief the level's largest smaller leaf (EnqueueLevelBestSplitSearch's max_num_data_in_smaller_leaf) */
  data_size_t max_smaller;
  int padding_;
  /*! \brief block map: pair p's block rows start at construct_block_start[p] (p < live_pairs; [live_pairs] =
   *  construct_total_blocks), in CUDADeviceLevelConfig::construct_block_start (the reset sets this pointer; a pair
   *  chunk's state copies it) */
  int* construct_block_start;
};

/*! \brief the tree-constant shape of a device level's construct (CUDAHistogramConstructor::DeviceLevelConstructShape):
 *  the dense batched kernel or the pair-joint family, as LaunchConstructHistogramBatchedKernelInner0 picks for a
 *  multi-pair level of this tree */
struct CUDADeviceLevelConstructShape {
  /*! \brief 0: the dense batched kernel; 1: the pair-joint family (cuda_plan key pair_hist) */
  int pair_mode;
  /*! \brief pair-joint kernel: 0 gather, 1 capped gather, 2 code-view gather, 3 capped code-view gather, 4 code words */
  int kernel_kind;
  /*! \brief block rows (block dim y) and the launch's x extent, block dim x and dynamic shared bytes */
  int block_y;
  int grid_x;
  int block_x;
  int smem_bytes;
  /*! \brief cuda_plan key pair_code5_slots: the code-word level's one rows-per-thread over the device's slots */
  int slot;
  int slot_device_slots;
  int slot_min_rows;
  int slot_max_rows;
  /*! \brief kernel parameters: per_pair_rows' grid floor (0 = off) and rows floor, the whole-row partitions and
   *  joint total (0 = per-partition blocks) */
  int per_pair_min_grid;
  int per_pair_min_rows;
  int whole_row_partitions;
  uint32_t whole_row_joint;
  /*! \brief code words: cuda_plan keys pair_code5_flat / pair_code5_epilogue (the instantiation) */
  int code5_flat;
  int code5_epilogue_flat;
};

/*! \brief static inputs of the select and pairs kernels (by value) */
struct CUDADeviceLevelConfig {
  CUDADeviceLevelState* state;
  const CUDASplitInfo* leaf_best_split_info;
  /*! \brief fixed root statistics and pooled tree output; complete trees seed root output before the first split */
  const CUDALeafSplitsStruct* root_struct;
  double* tree_leaf_value;
  const data_size_t* leaf_num_data;
  const data_size_t* leaf_data_start;
  CUDATreeBatchSplit* tree_batch_splits;
  CUDAHybridApplyDescriptor* apply_descs;
  CUDAHybridPairDescriptor* pair_descs;
  CUDALeafSplitsStruct* pair_slots;
  const CUDAHybridGraphFeatureMeta* feature_meta;
  const CUDAHybridGraphFeatureSource* feature_source;
  const double* real_thresholds;
  int* journal;
  int max_depth;
  /*! \brief total leaf budget; whole trees arbitrate a binding final level on the device */
  int num_leaves_budget;
  /*! \brief complete depth-bounded tree: the final apply needs no child histogram/search */
  int whole_tree;
  /*! \brief threads of the host's gap copy blocks (the gap descriptors' num_blocks unit) */
  int gap_block_dim;
  // children's search gates (EnqueueLevelBestSplitSearch)
  data_size_t min_data_in_leaf;
  double min_sum_hessian_in_leaf;
  int skip_unsplittable;
  int num_grad_quant_bins;
  // construct geometry (LaunchConstructHistogramBatchedKernelInner0)
  int construct_min_grid_dim_y;
  int construct_min_rows_per_thread;
  int construct_saturation_floor;
  /*! \brief cuda_plan keys dense_level_row_blocks / level_row_blocks: the rows floor of a multi-pair level */
  int level_row_blocks;
  CUDADeviceLevelConstructShape shape;
  /*! \brief cuda_plan key selective_device_prefix: per applied split (in applied order, at the journal's split
   *  index) a raw copy of the split's best-split record as the select reads it; nullptr: not recorded */
  CUDASplitInfo* split_info_journal;
  /*! \brief the selective prefix (set together with split_info_journal): the select writes no tree-record descriptors
   *  (tree_batch_splits) and the enqueue launches no tree record (the selective flow rebuilds the tree from its own
   *  records) */
  int skip_tree_record;
  /*! \brief host-side launch choice (the kernels do not read it): the selective prefix's partition launches take the
   *  whole tree's joint worker bound */
  int joint_grid_bound;
  /*! \brief the block map (searched pairs + 1 ints; see CUDADeviceLevelState::construct_block_start) */
  int* construct_block_start;
  /*! \brief a complete tree wider than kDeviceTreeSharedSplits splits per level: the select's valid leaves, their gains
   *  (select_capacity each) and the split-leaf bitset (2 * select_capacity bits) in global scratch; null otherwise */
  int* select_leaves;
  double* select_gains;
  unsigned int* select_split_bits;
  /*! \brief the level's terminal gaps (leaf, flat chunk start), 2 * select_capacity of them (the global select writes
   *  their descriptors word-parallel from here) */
  int2* select_gaps;
  int select_capacity;
  /*! \brief pair chunking (a complete tree whose per-pair search scratch for a whole level exceeds the memory budget):
   *  a level's fix/subtract and find run over num_chunks chunks of chunk_pairs pairs. The pairs block writes chunk c's
   *  state (a copy of the level's, live_pairs = the chunk's live pairs) to chunk_states[c]; num_chunks <= 1: none */
  int num_chunks;
  CUDADeviceLevelState* chunk_states;
  int chunk_pairs;
  int padding_;
};

/*! \brief block threads of the select / pairs kernels at a level of at most max_splits splits */
inline int DeviceLevelThreads(const int max_splits) {
  int t = 64;
  while (t < 2 * max_splits && t < 1024) {
    t <<= 1;
  }
  return t;
}

/*! \brief DeviceLevelSelectKernel on \p stream for level \p level (defined in cuda_device_levels.cu) */
void LaunchDeviceLevelSelect(const CUDADeviceLevelConfig& config, const int level, const int max_splits,
                             cudaStream_t stream);
/*! \brief DeviceLevelPairsKernel on \p stream for the children of level \p level */
void LaunchDeviceLevelPairs(const CUDADeviceLevelConfig& config, const int level, const int max_splits,
                            cudaStream_t stream);
/*! \brief zero the state and the journal header before a tree's first device level (on \p stream) */
void LaunchDeviceLevelReset(const CUDADeviceLevelConfig& config, const int num_leaves, cudaStream_t stream);

/*! \brief cuda_plan key device_bynode_masks: the per-node feature masks of a complete device tree, drawn on the device.
 *  A node's mask is ColSampler::GetByNodeKeyed's sample for the node's key (the generator seeded by the high half of
 *  MixKey(MixKey(seed) ^ key), Random::Sample's branch, the candidates' inner features), and its task list is
 *  CUDABestSplitFinder::CommitLevelNodeMasks's: the tasks of the sampled features in ascending task index. A node's
 *  key follows from its parent's as in CUDASingleGPUTreeLearner::BynodeKeySplit. Rows, lists and keys live at fixed
 *  per-leaf offsets, so the level finders read them through a static row table and the captured launches never
 *  change. */
struct CUDADeviceBynodeConfig {
  /*! \brief [0] candidates N, [1] sample size K, [2] Random::Sample's branch (ColSampler::KeyedNodeSample::mode),
   *  [3] / [4] the low / high half of MixKey(seed), then the N candidates' inner features in candidate order */
  const int* sample;
  /*! \brief every split-find task's inner feature (num_tasks) */
  const int* task_inner_feature;
  int num_tasks;
  int num_features;
  /*! \brief leaf slots of the rows, the lists and each key generation */
  int num_slots;
  /*! \brief the most tasks a node sample can list (each list's stride) */
  int max_tasks;
  /*! \brief leaf l's row (num_features bytes, 1 = searched) at rows + l * num_features */
  int8_t* rows;
  /*! \brief (offset, count) per leaf slot, the offset in ints from tasks; leaf l's list at 2 * num_slots + l * max_tasks
   *  (the layout of the finder's level task lists) */
  int* tasks;
  /*! \brief node keys, two generations of num_slots: the children of level L read generation L & 1 (their parents')
   *  and write generation (L + 1) & 1 */
  uint64_t* keys;
  /*! \brief pair chunking (CUDADeviceLevelConfig::num_chunks > 1): rows and lists live in 1 + 2 * chunk_pairs row
   *  slots (slot 0 the root, then both children of each pair of the chunk in flight), and the mask kernels point the
   *  leaf's row table entry (table, num_slots pointers) and list range at its slot. null: row slot = leaf (the static
   *  table) */
  int8_t** table;
};

/*! \brief dynamic shared bytes of the mask kernels (a bitmap over the candidates, at most num_features of them) */
inline size_t DeviceBynodeSharedBytes(const int num_features) {
  return static_cast<size_t>((num_features + 31) / 32) * sizeof(uint32_t);
}
/*! \brief the root's mask and task list (leaf 0) under root_key, which also becomes generation 0's key of leaf 0 */
void LaunchDeviceBynodeRootMask(const CUDADeviceBynodeConfig& config, const uint64_t root_key, cudaStream_t stream);
/*! \brief the masks and task lists of both children of every split level \p level applied (state->live_pairs of them,
 *  after the apply wrote the descriptors), one block per child: splits [chunk_base, chunk_base + max_splits) */
void LaunchDeviceBynodeLevelMasks(const CUDADeviceBynodeConfig& config, const CUDADeviceLevelState* state,
                                  const CUDAHybridApplyDescriptor* apply_descs, const int level, const int max_splits,
                                  const int chunk_base, cudaStream_t stream);

#ifdef __CUDACC__
/*! \brief the apply kernels' guard: this split index lies past the level's live splits */
__device__ __forceinline__ bool DeviceLevelBeyondLiveSplits(const CUDADeviceLevelState* dls, const unsigned int k) {
  return dls != nullptr && k >= static_cast<unsigned int>(dls->live_splits);
}
/*! \brief the search kernels' guard: this pair index lies past the search's live pairs */
__device__ __forceinline__ bool DeviceLevelBeyondLivePairs(const CUDADeviceLevelState* dls, const unsigned int p) {
  return dls != nullptr && p >= static_cast<unsigned int>(dls->live_pairs);
}
/*! \brief block map lookup: the pair whose block rows hold virtual block row vb (start[p] <= vb < start[p + 1]) */
__device__ __forceinline__ int DeviceLevelBlockPair(const CUDADeviceLevelState* dls, const int vb) {
  int lo = 0;
  int hi = dls->live_pairs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (dls->construct_block_start[mid] <= vb) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}
#endif  // __CUDACC__

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_DEVICE_LEVELS_HPP_
