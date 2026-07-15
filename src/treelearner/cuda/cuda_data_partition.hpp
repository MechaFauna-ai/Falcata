/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_DATA_PARTITION_HPP_
#define LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_DATA_PARTITION_HPP_

#ifdef USE_CUDA

#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/tree.h>

#include <vector>

#include <LightGBM/cuda/cuda_column_data.hpp>
#include <LightGBM/cuda/cuda_split_info.hpp>
#include <LightGBM/cuda/cuda_tree.hpp>

#include "cuda_leaf_splits.hpp"
#include "cuda_hybrid_graph.hpp"

#define FILL_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)
#define SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)
#define AGGREGATE_BLOCK_SIZE_DATA_PARTITION (1024)

namespace LightGBM {

/*! \brief Host-side per-split input for the batched (per-level) apply phase of the
 *  hybrid growth: everything the per-split Split() call would receive, restricted
 *  to numerical splits in deferred mode with point_structs_at_main semantics. The
 *  deferred split-info slot of split k is k (18 ints each, see FinishSplitBatch). */
struct CUDAHybridApplySplitInput {
  const CUDASplitInfo* best_split_info;
  int left_leaf_index;
  int right_leaf_index;
  int split_feature;
  uint32_t split_threshold;
  uint8_t split_default_left;
  data_size_t num_data_in_leaf;
  data_size_t leaf_data_start;
  CUDALeafSplitsStruct* smaller_leaf_splits;
  CUDALeafSplitsStruct* larger_leaf_splits;
};

/*! \brief Device-side per-split descriptor for the batched apply kernels. One
 *  entry per split of a level; built host-side by SplitLevelBatched (mirroring the
 *  host metadata math of LaunchGenDataToLeftBitVectorKernel) and uploaded in a
 *  single H2D copy. Data-parallel kernels index splits via a grid dimension and
 *  size themselves by the level's largest split region; blocks beyond a split's
 *  own num_blocks exit early. Per-split scratch lives in per-split regions:
 *  the bit vector and out-indices scratch use the split leaf's own
 *  [leaf_data_start, +num_data_in_leaf) window (mirroring the main index array
 *  layout), the block offset buffers use [block_offset_start, +num_blocks+1). */
struct CUDAHybridApplyDescriptor {
  const void* column_data;
  const CUDASplitInfo* best_split_info;
  CUDALeafSplitsStruct* smaller_leaf_splits;
  CUDALeafSplitsStruct* larger_leaf_splits;
  data_size_t leaf_data_start;
  data_size_t num_data_in_leaf;
  data_size_t block_offset_start;
  /*! \brief per-split grid size: ceil(num_data_in_leaf / 1024) (host-computed) */
  int num_blocks;
  int left_leaf_index;
  int right_leaf_index;
  /*! \brief split decision parameters (see LaunchGenDataToLeftBitVectorKernel) */
  uint32_t th;
  uint32_t t_zero_bin;
  uint32_t max_bin;
  uint32_t min_bin;
  int default_leaf_index;
  int missing_default_leaf_index;
  uint8_t split_default_to_left;
  uint8_t split_missing_default_to_left;
  /*! \brief per-row byte stride of the 4-bit packed compact source (bit_type 4) */
  int packed_row_stride;
  /*! \brief column storage width: 8, 16 or 32 bits, or 4 = 4-bit packed compact
   *  source (column_data points at the column's byte within the packed compact
   *  matrix; the bin is (byte >> packed_shift) & 0xf) */
  uint8_t bit_type;
  /*! \brief nibble shift (0 or 4) of the 4-bit packed compact source */
  uint8_t packed_shift;
  /*! \brief runtime versions of the GenDataToLeftBitVectorKernel template flags */
  uint8_t min_is_max;
  uint8_t missing_is_zero;
  uint8_t missing_is_na;
  uint8_t mfb_is_zero;
  uint8_t mfb_is_na;
  uint8_t max_bin_to_left;
  uint8_t use_min_bin;
};

/*! \brief One collapsed (pruned) subtree of the selective grow-then-prune hybrid
 *  growth: the window [start, start + count) of the main data index array holds
 *  exactly the subtree's rows (child regions nest inside the parent region), and
 *  every row's data-index-to-leaf-index entry is rewritten to target_leaf (the
 *  leaf index the collapsed node reverts to). */
struct CUDACollapseWindow {
  data_size_t start;
  data_size_t count;
  int target_leaf;
};

class CUDADataPartition: public NCCLInfo {
 public:
  CUDADataPartition(
    const Dataset* train_data,
    const int num_total_bin,
    const int num_leaves,
    const int num_threads,
    const bool use_quantized_grad,
    hist_t* cuda_hist);

  ~CUDADataPartition();

  void Init();

  void BeforeTrain();

  void Split(
    // input best split info
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    const int leaf_best_split_feature,
    const uint32_t leaf_best_split_threshold,
    const uint32_t* categorical_bitset,
    const int categorical_bitset_len,
    const uint8_t leaf_best_split_default_left,
    const data_size_t num_data_in_leaf,
    const data_size_t leaf_data_start,
    // for leaf information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians,
    double* left_leaf_sum_of_gradients,
    double* right_leaf_sum_of_gradients,
    data_size_t* global_left_leaf_num_data,
    data_size_t* global_right_leaf_num_data,
    const bool point_structs_at_main = false,
    const int deferred_slot = -1);

  // Deferred split-info readback for the hybrid level-batched growth: one device
  // synchronization + one transfer for all of a level's splits (18 ints per split,
  // laid out by the deferred_slot passed to Split). Layout per slot: [0] left leaf,
  // [1] left count, [2] left start, [4] right count, [5] right start, [6] right leaf,
  // ints [8..15] reinterpreted as 4 doubles: left/right sum_hessians, left/right
  // sum_gradients.
  void FinishSplitBatch(const int num_splits, std::vector<int>* out);

  /*! \brief Batched apply phase for the hybrid level-batched growth: applies ALL
   *  numerical splits of a level with one launch per kernel family (gen bit
   *  vector, update data-index-to-leaf-index, aggregate block offsets, split
   *  inner, split tree structure, copy indices) and ZERO host synchronization.
   *  Equivalent to calling Split(..., point_structs_at_main=true, deferred_slot=k)
   *  for k = 0..n-1: split k writes deferred split-info slot k, read back by the
   *  caller via FinishSplitBatch(n). Requires the level's splits to partition
   *  disjoint index regions (host-known leaf_data_start/num_data from the
   *  previous level's readback). */
  void SplitLevelBatched(const std::vector<CUDAHybridApplySplitInput>& splits);

  /*! \brief Selective grow-then-prune: rewrite the data-index-to-leaf-index
   *  entries of the given collapsed subtree windows to their target leaves.
   *  Enqueued on cuda_streams_[0] (ordered before the next batched apply and,
   *  via the legacy-stream semantics, before subsequent histogram work); the
   *  caller guarantees the device is otherwise idle (post-readback). */
  void CollapseLeafWindows(const std::vector<CUDACollapseWindow>& windows);

  /*! \brief Selective grow-then-prune finalize: remap the data-index-to-leaf-
   *  index entries of all rows of this tree (via the main index array, which
   *  covers exactly root_num_data() rows) through leaf_map (hybrid leaf index ->
   *  final classic leaf index). Async on the default stream; the caller
   *  synchronizes. */
  void RemapDataIndexToLeafIndex(const std::vector<int>& leaf_map);

  /*! \brief Selective grow-then-prune finalize: upload the final per-leaf data
   *  layout (num_data / data_start / data_end) for the first num_leaves leaves,
   *  so post-training consumers (objective renewal, quantized leaf renewal)
   *  see the final classic numbering. */
  void SetLeafDataLayout(const std::vector<data_size_t>& leaf_num_data,
                         const std::vector<data_size_t>& leaf_data_start,
                         int num_leaves);

  void UpdateTrainScore(const Tree* tree, double* cuda_scores);

  void SetUsedDataIndices(const data_size_t* used_indices, const data_size_t num_used_indices);

  void SetBaggingSubset(const Dataset* subset);

  void ResetTrainingData(const Dataset* train_data, const int num_total_bin, hist_t* cuda_hist);

  void ResetConfig(const Config* config, hist_t* cuda_hist);

  void ResetByLeafPred(const std::vector<int>& leaf_pred, int num_leaves);

  void ReduceLeafGradStat(
    const score_t* gradients, const score_t* hessians,
    CUDATree* tree, double* leaf_grad_stat_buffer, double* leaf_hess_state_buffer) const;

  data_size_t root_num_data() const {
    if (use_bagging_) {
      return num_used_indices_;
    } else {
      return num_data_;
    }
  }

  const data_size_t* cuda_data_indices() const { return cuda_data_indices_.RawData(); }

  const data_size_t* cuda_leaf_num_data() const { return cuda_leaf_num_data_.RawData(); }

  const data_size_t* cuda_leaf_data_start() const { return cuda_leaf_data_start_.RawData(); }

  const int* cuda_data_index_to_leaf_index() const { return cuda_data_index_to_leaf_index_.RawData(); }

  bool use_bagging() const { return use_bagging_; }

  /*! \brief recorded on the apply stream after the last batched apply kernel of a
   *  level; the speculative single-sync search phase waits on it before reading
   *  the child leaf-splits structs the apply kernels write */
  cudaEvent_t apply_done_event() const { return indices_copy_done_event_; }

  /*! \brief device array of the current level's actual smaller-child sizes
   *  (min(left, right) count of split k), written by the batched aggregate
   *  kernel; consumed by the speculative batched construct kernel to derive its
   *  exact row grouping before the sizes are host-known */
  const data_size_t* level_smaller_leaf_counts() const { return cuda_level_smaller_counts_.RawDataReadOnly(); }

#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  /*! \brief graphs L1 body capture: launches the five batched apply kernels
   *  with placeholder grids on the capture stream (cuda_streams_[0]) and
   *  collects their graph nodes + roles; the device controller resizes them
   *  per level */
  void CaptureHybridGraphApplyKernels(const CUDAHybridGraphLoopState* gstate,
                                      std::vector<cudaGraphNode_t>* nodes,
                                      std::vector<int>* roles);

  /*! \brief preallocates the apply-descriptor and block-offset buffers for the
   *  graph loop's worst case, so no captured buffer pointer ever reallocates */
  void EnsureHybridGraphCapacity(const data_size_t max_root_num_data);

  /*! \brief graphs L1: static per-feature split metadata of the batched apply
   *  descriptors (everything of SplitLevelBatched's host math that does not
   *  depend on the per-tree column view; real feature/threshold/missing-type
   *  fields are filled by the learner) */
  void BuildHybridGraphFeatureMeta(std::vector<CUDAHybridGraphFeatureMeta>* meta) const;

  /*! \brief graphs L1: per-tree column source of every inner feature (the
   *  packed compact view is per-tree and double-buffered) */
  void BuildHybridGraphFeatureSource(std::vector<CUDAHybridGraphFeatureSource>* source) const;

  /*! \brief device apply-descriptor buffer the graph controller writes */
  CUDAHybridApplyDescriptor* hybrid_graph_apply_descs() { return cuda_apply_descs_.RawData(); }

  /*! \brief current main / out index buffers, written into the loop state
   *  before every graph launch (the controller swaps them per level) */
  data_size_t* hybrid_graph_main_indices() { return cuda_data_indices_.RawData(); }
  data_size_t* hybrid_graph_out_indices() { return cuda_out_data_indices_in_leaf_.RawData(); }

  /*! \brief graphs L1 host bookkeeping after a graph prefix: accounts the
   *  applied splits and realigns the host index-buffer wrappers with the
   *  device loop's per-level buffer swaps */
  void FinishHybridGraphLevels(const int num_levels, const int total_splits);

  /*! \brief graphs L2: stream-ordered async prefetch of the deferred
   *  split-info slab (up to \p max_splits slots) into the pinned staging
   *  buffer, so the learner's single post-graph stream sync covers it;
   *  consume with ReadPrefetchedSplitBatch after synchronizing \p stream */
  void PrefetchSplitBatchAsync(const int max_splits, cudaStream_t stream);

  /*! \brief consume \p num_splits slots of a prefetched slab (host memcpy
   *  only; the caller already synchronized the prefetch stream). Values match
   *  FinishSplitBatch(num_splits) bit-for-bit */
  void ReadPrefetchedSplitBatch(const int num_splits, std::vector<int>* out) const;
#endif  // EXABOOST_HYBRID_GRAPH_SUPPORTED

  /*! \brief the stream the batched apply kernels run on (graphs L1 captures
   *  the body on it) */
  cudaStream_t apply_stream() const { return cuda_streams_[0]; }

 private:
  void CalcBlockDim(const data_size_t num_data_in_leaf);

  /*! \brief launch the batched apply kernels of one level, in dependency order
   *  on cuda_streams_[0] (fused gen-bit-vector+update-leaf-index, aggregate
   *  block offsets, split inner into the out buffer, split tree structure, and
   *  a sparse gap copy carrying terminal leaves' regions into the out buffer);
   *  descriptors (splits first, then gaps) already uploaded. The caller swaps
   *  the out buffer in as the new main index array afterwards. */
  void LaunchSplitLevelBatchedKernels(const int num_splits, const int max_num_blocks,
                                      const int num_gaps, const int max_gap_blocks);

  void GenDataToLeftBitVector(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint32_t* categorical_bitset,
    const int categorical_bitset_len,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index);

  void SplitInner(
    // input best split info
    const data_size_t num_data_in_leaf,
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    // for leaf splits information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians,
    double* left_leaf_sum_of_gradients,
    double* right_leaf_sum_of_gradients,
    data_size_t* global_left_leaf_num_data,
    data_size_t* global_right_leaf_num_data,
    const bool point_structs_at_main,
    const int deferred_slot,
    const data_size_t leaf_data_start_for_copy);

  // kernel launch functions
  void LaunchFillDataIndicesBeforeTrain();

  void LaunchSplitInnerKernel(
    // input best split info
    const data_size_t num_data_in_leaf,
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    // for leaf splits information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians,
    double* left_leaf_sum_of_gradients,
    double* right_leaf_sum_of_gradients,
    data_size_t* global_left_leaf_num_data,
    data_size_t* global_right_leaf_num_data,
    const bool point_structs_at_main,
    const int deferred_slot,
    const data_size_t leaf_data_start_for_copy);

  void LaunchGenDataToLeftBitVectorKernel(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index);

  void LaunchGenDataToLeftBitVectorCategoricalKernel(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t* bitset,
    const int bitset_len,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index);

#define GenDataToLeftBitVectorKernel_PARAMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin, \
  const uint32_t min_bin, \
  const uint8_t split_default_to_left, \
  const uint8_t split_missing_default_to_left

  template <typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool missing_is_zero,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner0(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner1(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner2(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner3(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner4(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool is_single_feature_in_column);

#undef GenDataToLeftBitVectorKernel_PARAMS

#define UpdateDataIndexToLeafIndexKernel_PARAMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin_ref, \
  const uint32_t min_bin_ref, \
  const int left_leaf_index, \
  const int right_leaf_index, \
  const int default_leaf_index, \
  const int missing_default_leaf_index

  template <typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool missing_is_zero,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner0(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner1(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner2(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool mfb_is_na,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner3(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner4(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool is_single_feature_in_column);

#undef UpdateDataIndexToLeafIndexKernel_PARAMS

  void LaunchAddPredictionToScoreKernel(const double* leaf_value, double* cuda_scores);

  void LaunchFillDataIndexToLeafIndex();

  void LaunchReduceLeafGradStat(
    const score_t* gradients, const score_t* hessians,
    CUDATree* tree, double* leaf_grad_stat_buffer, double* leaf_hess_state_buffer) const;

  // Host memory

  // dataset information
  /*! \brief number of training data */
  data_size_t num_data_;
  /*! \brief number of features in training data */
  int num_features_;
  /*! \brief number of total bins in training data */
  int num_total_bin_;
  /*! \brief bin data stored by column */
  const CUDAColumnData* cuda_column_data_;
  /*! \brief grid dimension when splitting one leaf */
  int grid_dim_;
  /*! \brief block dimension when splitting one leaf */
  int block_dim_;
  /*! \brief data indices used in this iteration */
  const data_size_t* used_indices_;
  /*! \brief marks whether a feature is a categorical feature */
  std::vector<bool> is_categorical_feature_;
  /*! \brief marks whether a feature is the only feature in its group */
  std::vector<bool> is_single_feature_in_column_;

  // config information
  /*! \brief maximum number of leaves in a tree */
  int num_leaves_;
  /*! \brief number of threads */
  int num_threads_;
  /*! \brief whether to use quantized gradients */
  bool use_quantized_grad_;

  // per iteration information
  /*! \brief whether bagging is used in this iteration */
  bool use_bagging_;
  /*! \brief number of used data indices in this iteration */
  data_size_t num_used_indices_;

  // tree structure information
  /*! \brief current number of leaves in tree */
  int cur_num_leaves_;

  // split algorithm related
  /*! \brief maximum number of blocks to aggregate after finding bit vector by blocks */
  int max_num_split_indices_blocks_;

  // CUDA streams
  /*! \brief cuda streams used for asynchronizing kernel computing and memory copy */
  std::vector<cudaStream_t> cuda_streams_;
  // recorded on cuda_streams_[2] after CopyDataIndicesKernel; consumed at the next
  // Split() so back-to-back splits (hybrid level growth) don't race on the shared
  // cuda_out_data_indices_in_leaf_ scratch while the copy is still in flight
  cudaEvent_t indices_copy_done_event_ = nullptr;
  // host staging for the batched apply descriptors (one entry per split of a level)
  std::vector<CUDAHybridApplyDescriptor> host_apply_descs_;


  // CUDA memory, held by this object

  // tree structure information
  /*! \brief data indices by leaf */
  CUDAVector<data_size_t> cuda_data_indices_;
  /*! \brief start position of each leaf in cuda_data_indices_ */
  CUDAVector<data_size_t> cuda_leaf_data_start_;
  /*! \brief end position of each leaf in cuda_data_indices_ */
  CUDAVector<data_size_t> cuda_leaf_data_end_;
  /*! \brief number of data in each leaf */
  CUDAVector<data_size_t> cuda_leaf_num_data_;
  /*! \brief records the histogram of each leaf */
  CUDAVector<hist_t*> cuda_hist_pool_;
  /*! \brief records the value of each leaf */
  CUDAVector<double> cuda_leaf_output_;

  // split data algorithm related
  CUDAVector<uint16_t> cuda_block_to_left_offset_;
  /*! \brief maps data index to leaf index, for adding scores to training data set */
  CUDAVector<int> cuda_data_index_to_leaf_index_;
  /*! \brief prefix sum of number of data going to left in all blocks */
  CUDAVector<data_size_t> cuda_block_data_to_left_offset_;
  /*! \brief prefix sum of number of data going to right in all blocks */
  CUDAVector<data_size_t> cuda_block_data_to_right_offset_;
  /*! \brief buffer for splitting data indices, will be copied back to cuda_data_indices_ after split */
  CUDAVector<data_size_t> cuda_out_data_indices_in_leaf_;
  /*! \brief device copy of one level's batched apply descriptors */
  CUDAVector<CUDAHybridApplyDescriptor> cuda_apply_descs_;
  /*! \brief selective grow-then-prune: device copy of one level's collapsed
   *  subtree windows and host staging (see CollapseLeafWindows) */
  CUDAVector<CUDACollapseWindow> cuda_collapse_windows_;
  std::vector<CUDACollapseWindow> host_collapse_windows_;
  /*! \brief selective grow-then-prune: device copy of the hybrid-to-final leaf
   *  index map (see RemapDataIndexToLeafIndex) */
  CUDAVector<int> cuda_leaf_index_remap_;

  // split tree structure algorithm related
  /*! \brief buffer to store split information, prepared to be copied to cpu */
  CUDAVector<int> cuda_split_info_buffer_;
  /*! \brief pinned staging buffer of FinishSplitBatch /
   *  PrefetchSplitBatchAsync */
  int* pinned_split_info_ = nullptr;
  size_t pinned_split_info_size_ = 0;
  /*! \brief grow the pinned split-info staging buffer to >= num_ints ints */
  void EnsurePinnedSplitInfoCapacity(const size_t num_ints);
  /*! \brief per-split smaller-child size of the current level's batched apply
   *  (see level_smaller_leaf_counts()) */
  CUDAVector<data_size_t> cuda_level_smaller_counts_;

  // dataset information
  /*! \brief number of data in training set, for initialization of cuda_leaf_num_data_ and cuda_leaf_data_end_ */
  CUDAVector<data_size_t> cuda_num_data_;


  // CUDA memory, held by other object

  // dataset information
  /*! \brief beginning of histograms, for initialization of cuda_hist_pool_ */
  hist_t* cuda_hist_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_DATA_PARTITION_HPP_
