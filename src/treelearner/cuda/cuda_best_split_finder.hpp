/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_HPP_

#ifdef USE_CUDA

#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>

#include <cstdio>
#include <string>
#include <vector>

#include <LightGBM/cuda/cuda_random.hpp>
#include <LightGBM/cuda/cuda_split_info.hpp>

#include "cuda_leaf_splits.hpp"
#include "cuda_hybrid_graph.hpp"

#define NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER (256)
#define NUM_THREADS_FIND_BEST_LEAF (256)
#define NUM_TASKS_PER_SYNC_BLOCK (1024)

namespace LightGBM {

/*! \brief fp32 per-bin gain arithmetic in the best-split find kernels
 *  (EXABOOST_FP32_GAIN=1 enables; default off = fp64, the historical behavior).
 *  Leaf-level sums stay double and are converted once per task; results are
 *  quality-gated, not bit-identical. */
inline bool ExaboostFP32GainEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("EXABOOST_FP32_GAIN");
    return env != nullptr && std::string(env) == "1";
  }();
  return enabled;
}

struct SplitFindTask {
  int inner_feature_index;
  bool reverse;
  bool skip_default_bin;
  bool na_as_missing;
  bool assume_out_default_left;
  bool is_categorical;
  bool is_one_hot;
  uint32_t hist_offset;
  uint8_t mfb_offset;
  uint32_t num_bin;
  uint32_t default_bin;
  int rand_threshold;
};

class CUDABestSplitFinder {
 public:
  CUDABestSplitFinder(
    const hist_t* cuda_hist,
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets,
    const bool select_features_by_node,
    const Config* config);

  ~CUDABestSplitFinder();

  void InitFeatureMetaInfo(const Dataset* train_data);

  void Init();

  void InitCUDAFeatureMetaInfo();

  /*! \brief Provide the histogram constructor's completion events so the per-leaf
   *  FindBestSplits kernels can be ordered after histogram construction/subtraction
   *  via cudaStreamWaitEvent instead of a per-split device sync. Called once after
   *  both objects are initialized. */
  void SetHistogramEvents(const cudaEvent_t* construct_done_events, const cudaEvent_t* subtract_done_events) {
    hist_construct_done_events_ = construct_done_events;
    hist_subtract_done_events_ = subtract_done_events;
  }

  /*! \brief select which histogram pipeline's completion events the next
   *  FindBestSplitsForLeaf call waits on (hybrid level-batched growth) */
  void SetActiveHistPipeline(const int pipeline) { active_hist_pipeline_ = pipeline; }

  /*! \brief non-quantized histogram storage is float pairs (EXABOOST_FP32_HIST);
   *  decided by the histogram constructor, wired in by the tree learner */
  void SetHistFP32(const bool hist_fp32) { hist_fp32_ = hist_fp32; }

  /*! \brief large-bin fallback stays fp64: the tree learner disables the fp32
   *  histogram mode when this is set */
  bool use_global_memory() const { return use_global_memory_; }

  void BeforeTrain(const std::vector<int8_t>& is_feature_used_bytree);

  void FindBestSplitsForLeaf(
    const CUDALeafSplitsStruct* smaller_leaf_splits,
    const CUDALeafSplitsStruct* larger_leaf_splits,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    const data_size_t num_data_in_smaller_leaf,
    const data_size_t num_data_in_larger_leaf,
    const double sum_hessians_in_smaller_leaf,
    const double sum_hessians_in_larger_leaf,
    const score_t* grad_scale,
    const score_t* hess_scale,
    const uint8_t smaller_num_bits_in_histogram_bins,
    const uint8_t larger_num_bits_in_histogram_bins,
    const bool smaller_leaf_below_max_depth,
    const bool larger_leaf_below_max_depth,
    const bool synchronize = true);

  /*! \brief whether the batched per-level find+sync path supports the current
   *  configuration (only the template combination the benchmarks exercise:
   *  no extra_trees / L1 / path smoothing, shared-memory histograms, no
   *  per-node feature selection, no categorical features). Wide shapes
   *  (num_tasks > one sync block) use the multi-block batched sync mirroring
   *  the per-pair two-stage reduction; EXABOOST_BATCH_WIDE=0 disables that. */
  bool SupportsBatchedLevel() const {
    return !extra_trees_ && lambda_l1_ <= 0.0f && !use_smoothing_ &&
           !use_global_memory_ && !select_features_by_node_ &&
           !has_categorical_feature_ &&
           (num_tasks_ <= NUM_TASKS_PER_SYNC_BLOCK || ExaboostBatchWideEnabled());
  }

  /*! \brief one-line gate dump for EXABOOST_HYBRID_DIAG */
  std::string BatchedLevelGateDiag() const {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "finder: extra_trees=%d l1=%f smoothing=%d global_mem=%d "
             "by_node=%d categorical=%d num_tasks=%d (cap %d, batch_wide=%d)",
             static_cast<int>(extra_trees_), lambda_l1_,
             static_cast<int>(use_smoothing_), static_cast<int>(use_global_memory_),
             static_cast<int>(select_features_by_node_),
             static_cast<int>(has_categorical_feature_), num_tasks_,
             NUM_TASKS_PER_SYNC_BLOCK, static_cast<int>(ExaboostBatchWideEnabled()));
    return std::string(buf);
  }

  /*! \brief Batched per-level best-split search for hybrid growth: one find launch
   *  and one sync launch cover every sibling pair of a level. Each pair writes its
   *  own region of the task output buffer and its own (disjoint) per-leaf slots of
   *  the best-split cache. Waits on the histogram constructor's batched
   *  subtract-done event before reading any histogram; grad_scale/hess_scale are
   *  null for non-quantized training. The caller is expected to synchronize via
   *  SyncAllLeafBestSplitsToHost afterwards. */
  void FindBestSplitsForLevel(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs,
    const score_t* grad_scale,
    const score_t* hess_scale);

  const CUDASplitInfo* FindBestFromAllSplits(
    const int cur_num_leaves,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left,
    int* best_leaf_index,
    int* num_cat_threshold);

  void ResetTrainingData(
    const hist_t* cuda_hist,
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets);

  void ResetConfig(const Config* config, const hist_t* cuda_hist);

  void SetUsedFeatureByNode(const std::vector<int8_t>& is_feature_used_by_smaller_node,
                            const std::vector<int8_t>& is_feature_used_by_larger_node);

  // ---- forced-split support (forcedsplits_filename) ----
  // Compute split information for a FORCED split of the given leaf at the given
  // (numerical) feature and bin threshold, mirroring the CPU
  // FeatureHistogram::GatherInfoForThresholdNumerical. The result is written to the
  // given leaf's slot of an internal device buffer (so it survives later searches);
  // *out_host receives a host copy for gain/validity checks.
  const CUDASplitInfo* ComputeForcedSplit(
    const CUDALeafSplitsStruct* leaf_splits,
    const int inner_feature_index,
    const uint32_t threshold,
    const data_size_t num_data_in_leaf,
    const int leaf_slot,
    CUDASplitInfo* out_host);

  // Invalidate the cached per-leaf best-split entries of the given leaves (after a
  // forced split is applied, the parent's cached entry describes the pre-split leaf).
  void InvalidateLeafBestSplit(const int left_leaf_index, const int right_leaf_index);

  // Copy the cached best-split (feature, threshold, default_left) of the given leaves
  // from the device cache to the host arrays. The main loop's FindBestFromAllSplits
  // does this implicitly; the forced-split pre-pass must do it explicitly after each
  // search so the host arrays stay consistent with the device cache for every leaf.
  void SyncLeafBestSplitToHost(
    const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left);

  // Copy the whole device per-leaf best-split cache for leaves [0, num_leaves) to the
  // host in a single transfer (synchronizes the device). Used by the hybrid
  // level-batched growth phase, which needs every frontier leaf's candidate at once.
  // The categorical-threshold pointers inside the copied structs are device pointers
  // and must not be dereferenced on the host.
  void SyncAllLeafBestSplitsToHost(const int num_leaves, std::vector<CUDASplitInfo>* out) const;

#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  /*! \brief graphs L1 body capture: find + sync kernels with placeholder pair
   *  counts on cuda_streams_[0]; node handles + roles collected for the device
   *  controller (which sets the find grid x per tree from num_used_tasks) */
  void CaptureHybridGraphFindKernels(const CUDAHybridPairDescriptor* pair_descs,
                                     std::vector<cudaGraphNode_t>* nodes,
                                     std::vector<int>* roles,
                                     std::vector<int>* role_static_x);

  /*! \brief preallocates the per-pair best-split scratch for the graph loop's
   *  worst case, so the captured buffer pointer never reallocates */
  void EnsureHybridGraphCapacity(const int max_pairs) { EnsureHybridLevelCapacity(max_pairs); }

  /*! \brief the batched find kernel's x-grid extent of the current tree */
  int hybrid_graph_find_grid_x() const {
    const bool compact_tasks = num_used_tasks_ > 0 && num_used_tasks_ < num_tasks_;
    return compact_tasks ? num_used_tasks_ : num_tasks_;
  }

  int num_used_tasks() const { return num_used_tasks_; }

  /*! \brief per-tree pointer/flag choices baked into captured find/sync kernel
   *  params: a graph instance is only valid for trees whose key matches */
  void HybridGraphKeyPointers(std::vector<const void*>* key) const {
    const bool compact_tasks = num_used_tasks_ > 0 && num_used_tasks_ < num_tasks_;
    key->push_back(compact_tasks ?
      static_cast<const void*>(cuda_used_task_indices_.RawDataReadOnly()) : nullptr);
    key->push_back(static_cast<const void*>(cuda_is_feature_used_bytree_.RawDataReadOnly()));
  }

  cudaStream_t find_stream() const { return cuda_streams_[0]; }
#endif  // EXABOOST_HYBRID_GRAPH_SUPPORTED

  // Device pointer to a leaf's cached best split, for passing to CUDATree::Split.
  const CUDASplitInfo* leaf_best_split_info_ptr(const int leaf_index) const {
    return cuda_leaf_best_split_info_.RawDataReadOnly() + leaf_index;
  }

 private:
  void LaunchComputeForcedSplitKernel(
    const CUDALeafSplitsStruct* leaf_splits,
    const SplitFindTask* device_task,
    const uint32_t threshold,
    const data_size_t num_data_in_leaf,
    CUDASplitInfo* out);

  void LaunchInvalidateLeafBestSplitKernel(const int left_leaf_index, const int right_leaf_index);

  #define LaunchFindBestSplitsForLeafKernel_PARAMS \
    const CUDALeafSplitsStruct* smaller_leaf_splits, \
    const CUDALeafSplitsStruct* larger_leaf_splits, \
    const int smaller_leaf_index, \
    const int larger_leaf_index, \
    const bool is_smaller_leaf_valid, \
    const bool is_larger_leaf_valid, \
    const data_size_t global_num_data_in_smaller_leaf, \
    const data_size_t global_num_data_in_larger_leaf

  void LaunchFindBestSplitsForLeafKernel(LaunchFindBestSplitsForLeafKernel_PARAMS);

  template <bool USE_RAND>
  void LaunchFindBestSplitsForLeafKernelInner0(LaunchFindBestSplitsForLeafKernel_PARAMS);

  template <bool USE_RAND, bool USE_L1>
  void LaunchFindBestSplitsForLeafKernelInner1(LaunchFindBestSplitsForLeafKernel_PARAMS);

  template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING>
  void LaunchFindBestSplitsForLeafKernelInner2(LaunchFindBestSplitsForLeafKernel_PARAMS);

  template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, typename GAIN_T>
  void LaunchFindBestSplitsForLeafKernelInner3(LaunchFindBestSplitsForLeafKernel_PARAMS);

  #undef LaunchFindBestSplitsForLeafKernel_PARAMS

  #define LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS \
  const CUDALeafSplitsStruct* smaller_leaf_splits, \
  const CUDALeafSplitsStruct* larger_leaf_splits, \
  const int smaller_leaf_index, \
  const int larger_leaf_index, \
  const bool is_smaller_leaf_valid, \
  const bool is_larger_leaf_valid, \
  const score_t* grad_scale, \
  const score_t* hess_scale, \
  const uint8_t smaller_num_bits_in_histogram_bins, \
  const uint8_t larger_num_bits_in_histogram_bins, \
  const data_size_t global_num_data_in_smaller_leaf, \
  const data_size_t global_num_data_in_larger_leaf

  void LaunchFindBestSplitsDiscretizedForLeafKernel(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS);

  template <bool USE_RAND>
  void LaunchFindBestSplitsDiscretizedForLeafKernelInner0(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS);

  template <bool USE_RAND, bool USE_L1>
  void LaunchFindBestSplitsDiscretizedForLeafKernelInner1(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS);

  template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING>
  void LaunchFindBestSplitsDiscretizedForLeafKernelInner2(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS);

  template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, typename GAIN_T>
  void LaunchFindBestSplitsDiscretizedForLeafKernelInner3(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS);

  #undef LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS

  void LaunchSyncBestSplitForLeafKernel(
    const int host_smaller_leaf_index,
    const int host_larger_leaf_index,
    const bool is_smaller_leaf_valid,
    const bool is_larger_leaf_valid);

  // ---- batched per-level launchers (hybrid growth) ----
  // Grow the task output buffer so every pair of a level has its own region of
  // 2 * num_tasks_ CUDASplitInfo slots. No-op once capacity is reached.
  void EnsureHybridLevelCapacity(const int num_pairs);

  void LaunchFindBestSplitsForLevelKernel(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs);

  void LaunchFindBestSplitsDiscretizedForLevelKernel(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs,
    const score_t* grad_scale,
    const score_t* hess_scale);

  void LaunchSyncBestSplitForLevelKernel(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs);

  void LaunchFindBestFromAllSplitsKernel(
    const int cur_num_leaves,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left,
    int* best_leaf_index,
    data_size_t* num_cat_threshold);

  void AllocateCatVectors(CUDASplitInfo* cuda_split_infos, uint32_t* cat_threshold_vec, int* cat_threshold_real_vec, size_t len);

  void LaunchAllocateCatVectorsKernel(CUDASplitInfo* cuda_split_infos, uint32_t* cat_threshold_vec, int* cat_threshold_real_vec, size_t len);

  void LaunchInitCUDARandomKernel();

  // Host memory
  int num_features_;
  int num_leaves_;
  /*! \brief pinned staging buffer of SyncAllLeafBestSplitsToHost (raw bytes;
   *  CUDASplitInfo is never constructed/destructed in it) */
  mutable CUDASplitInfo* pinned_leaf_best_split_info_ = nullptr;
  mutable size_t pinned_leaf_best_split_info_size_ = 0;
  int max_num_bin_in_feature_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_default_bins_;
  std::vector<uint32_t> feature_num_bins_;
  std::vector<MissingType> feature_missing_type_;
  double lambda_l1_;
  double lambda_l2_;
  data_size_t min_data_in_leaf_;
  double min_sum_hessian_in_leaf_;
  double min_gain_to_split_;
  double cat_smooth_;
  double cat_l2_;
  int max_cat_threshold_;
  int min_data_per_group_;
  int max_cat_to_onehot_;
  bool extra_trees_;
  int extra_seed_;
  bool use_smoothing_;
  double path_smooth_;
  std::vector<cudaStream_t> cuda_streams_;
  /*! \brief histogram constructor completion events (not owned); used to order the
   *  per-leaf FindBestSplits kernels after histogram construction/subtraction. */
  const cudaEvent_t* hist_construct_done_events_ = nullptr;
  const cudaEvent_t* hist_subtract_done_events_ = nullptr;
  int active_hist_pipeline_ = 0;
  // for best split find tasks
  std::vector<SplitFindTask> split_find_tasks_;
  int num_tasks_;
  /*! \brief hybrid batched find: per-tree list of the task indices whose feature
   *  is in this tree's feature_fraction sample. The batched find launches one
   *  block per USED task only (a 10x saving for wide sampled datasets); output
   *  slots keep the ORIGINAL task index so the batched sync reduces the exact
   *  same lanes as the per-pair kernels (unused lanes masked not-found, exactly
   *  as if the find had written is_valid=false there). Empty/full -> nullptr
   *  passed to the kernels (zero overhead without sampling). */
  std::vector<int> host_used_task_indices_;
  CUDAVector<int> cuda_used_task_indices_;
  int num_used_tasks_ = 0;
  // use global memory
  bool use_global_memory_;
  // non-quantized histograms stored as float pairs (EXABOOST_FP32_HIST)
  bool hist_fp32_ = false;
  // number of total bins in the dataset
  const int num_total_bin_;
  // has categorical feature
  bool has_categorical_feature_;
  // maximum number of bins of categorical features
  int max_num_categorical_bin_;
  // marks whether a feature is categorical
  std::vector<int8_t> is_categorical_;
  // whether need to select features by node
  bool select_features_by_node_;

  // CUDA memory, held by this object
  // for per leaf best split information
  CUDAVector<CUDASplitInfo> cuda_leaf_best_split_info_;
  // for best split information when finding best split
  CUDAVector<CUDASplitInfo> cuda_best_split_info_;
  // best split information buffer, to be copied to host
  CUDAVector<int> cuda_best_split_info_buffer_;
  // find best split task information
  CUDAVector<SplitFindTask> cuda_split_find_tasks_;
  // per-leaf device buffer for forced-split info
  CUDAVector<CUDASplitInfo> cuda_forced_split_info_;
  // maps inner feature index -> index of its first task in split_find_tasks_
  std::vector<int> feature_to_task_index_;
  CUDAVector<int8_t> cuda_is_feature_used_bytree_;
  // used when finding best split with global memory
  CUDAVector<hist_t> cuda_feature_hist_grad_buffer_;
  CUDAVector<hist_t> cuda_feature_hist_hess_buffer_;
  CUDAVector<hist_t> cuda_feature_hist_stat_buffer_;
  CUDAVector<data_size_t> cuda_feature_hist_index_buffer_;
  CUDAVector<uint32_t> cuda_cat_threshold_leaf_;
  CUDAVector<int> cuda_cat_threshold_real_leaf_;
  CUDAVector<uint32_t> cuda_cat_threshold_feature_;
  CUDAVector<int> cuda_cat_threshold_real_feature_;
  int max_num_categories_in_split_;
  // used for extremely randomized trees
  CUDAVector<CUDARandom> cuda_randoms_;
  // features used by node
  CUDAVector<int8_t> is_feature_used_by_smaller_node_;
  CUDAVector<int8_t> is_feature_used_by_larger_node_;

  // CUDA memory, held by other object
  const hist_t* cuda_hist_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_HPP_
