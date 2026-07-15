/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
#define LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_row_data.hpp>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/feature_group.h>
#include <LightGBM/tree.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "cuda_leaf_splits.hpp"
#include "cuda_hybrid_graph.hpp"

#define NUM_DATA_PER_THREAD (400)
#define NUM_THREADS_PER_BLOCK (504)
#define NUM_FEATURE_PER_THREAD_GROUP (28)
#define SUBTRACT_BLOCK_SIZE (1024)
#define FIX_HISTOGRAM_SHARED_MEM_SIZE (1024)
#define FIX_HISTOGRAM_BLOCK_SIZE (512)
#define USED_HISTOGRAM_BUFFER_NUM (8)

namespace LightGBM {

/*! \brief fp32-pair global histogram storage for the non-quantized CUDA path
 *  (EXABOOST_FP32_HIST=1 requests; default off = hist_t/double pairs, the
 *  historical behavior). The actual engagement additionally requires the dense
 *  shared-memory layout (see CUDAHistogramConstructor::Init) and is exposed via
 *  hist_fp32(). Halves global histogram bandwidth (construct merge, fix,
 *  subtract, find reads); results are quality-gated, not bit-identical. */
inline bool ExaboostFP32HistRequested() {
  static const bool requested = []() {
    const char* env = std::getenv("EXABOOST_FP32_HIST");
    return env != nullptr && std::string(env) == "1";
  }();
  return requested;
}

/*! \brief y-grid sizing formula of the batched per-level construct kernel,
 *  shared verbatim by the host launch sizing (CalcConstructHistogramBatchedKernelDim)
 *  and the device row-grouping replica used by the speculative single-sync flow
 *  (which launches an upper-bound grid but must group rows exactly as the classic
 *  host sizing would, so histograms stay bit-identical). See the .cpp comment for
 *  the rationale of the cap/floor structure. */
__host__ __device__ inline int HybridBatchedConstructGridDimY(
    const data_size_t max_num_data_in_smaller_leaf,
    const int num_pairs,
    const int block_dim_y,
    const int min_grid_dim_y,
    const int min_rows_per_thread,
    const int saturation_floor_total) {
  // rows-per-thread target of the per-leaf sizing
  const int y_full_rate = ((max_num_data_in_smaller_leaf + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD +
    block_dim_y - 1) / block_dim_y;
  if (min_rows_per_thread <= 0) {
    // disabled: fall back to the per-leaf sizing
    return min_grid_dim_y > y_full_rate ? min_grid_dim_y : y_full_rate;
  }
  const int y_min_rate = ((max_num_data_in_smaller_leaf + min_rows_per_thread - 1) / min_rows_per_thread +
    block_dim_y - 1) / block_dim_y;
  const int saturation_floor = num_pairs > 0 ?
    (saturation_floor_total + num_pairs - 1) / num_pairs : saturation_floor_total;
  int inner = y_min_rate > 1 ? y_min_rate : 1;
  inner = inner > saturation_floor ? inner : saturation_floor;
  inner = inner < min_grid_dim_y ? inner : min_grid_dim_y;
  return y_full_rate > inner ? y_full_rate : inner;
}

class CUDAHistogramConstructor {
 public:
  CUDAHistogramConstructor(
    const Dataset* train_data,
    const int num_leaves,
    const int num_threads,
    const std::vector<uint32_t>& feature_hist_offsets,
    const int min_data_in_leaf,
    const double min_sum_hessian_in_leaf,
    const int gpu_device_id,
    const bool gpu_use_dp,
    const bool use_discretized_grad,
    const int grad_discretized_bins);

  ~CUDAHistogramConstructor();

  void Init(const Dataset* train_data, TrainingShareStates* share_state);

  /*! \brief Event recorded on cuda_stream_ after the smaller-leaf histogram is
   *  constructed. The best split finder waits on it (cudaStreamWaitEvent) before
   *  reading the smaller-leaf histogram, replacing a per-split device sync. */
  cudaEvent_t construct_done_event() const { return construct_done_events_[0]; }

  /*! \brief number of independent (stream, event-pair) pipelines for hybrid
   *  level-batched growth; pairs of one level round-robin across them so their
   *  small histogram kernels can execute concurrently */
  static constexpr int kNumHistPipelines = 4;
  void SetActivePipeline(const int pipeline) { active_pipeline_ = pipeline; }
  cudaStream_t current_stream() const { return pipeline_streams_[active_pipeline_]; }
  const cudaEvent_t* construct_done_events() const { return construct_done_events_; }
  const cudaEvent_t* subtract_done_events() const { return subtract_done_events_; }
  /*! \brief Event recorded on cuda_stream_ after the larger-leaf histogram is
   *  produced by subtraction. The best split finder waits on it before reading
   *  the larger-leaf histogram. */
  cudaEvent_t subtract_done_event() const { return subtract_done_events_[0]; }

  void ZeroHistForLeaf(int leaf_index);

  /*! \brief Selective grow-then-prune: zero the histogram slots freed by
   *  collapsed subtrees so they can be reallocated to later right children
   *  (the construct kernels accumulate into assumed-zero slots). Async memsets
   *  on the legacy default stream: ordered before any subsequently enqueued
   *  construct kernel on the (blocking) histogram streams. */
  void ZeroHistSlots(const std::vector<int>& slots);

  void ConstructHistogramForLeaf(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
    const data_size_t global_num_data_in_smaller_leaf,
    const data_size_t global_num_data_in_larger_leaf,
    const data_size_t num_data_in_smaller_leaf,
    const data_size_t num_data_in_larger_leaf,
    const double sum_hessians_in_smaller_leaf,
    const double sum_hessians_in_larger_leaf,
    const uint8_t num_bits_in_histogram_bins);

  void SubtractHistogramForLeaf(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
    const bool use_discretized_grad,
    const uint8_t parent_num_bits_in_histogram_bins,
    const uint8_t smaller_num_bits_in_histogram_bins,
    const uint8_t larger_num_bits_in_histogram_bins);

  /*! \brief stream the batched per-level histogram kernels run on; the hybrid
   *  learner uploads the per-level pair descriptors asynchronously on this
   *  stream so the copy is ordered before the construct kernel that reads them */
  cudaStream_t hist_stream() const { return cuda_stream_; }

  /*! \brief whether the batched per-level construct/fix/subtract path supports the
   *  current data layout (dense, shared-memory histograms). The compact column
   *  view (feature_fraction sampling gathered into a dense per-tree copy) is
   *  supported through the same batched dense kernel fed with the compact
   *  pointers/dims; EXABOOST_BATCH_WIDE=0 restores the per-pair fallback for it. */
  bool SupportsBatchedLevel() const {
    return cuda_row_data_ != nullptr &&
           !cuda_row_data_->is_sparse() &&
           cuda_row_data_->NumLargeBinPartition() == 0 &&
           (!use_compact_view_ ||
            (ExaboostBatchWideEnabled() && !compact_is_col_major_ &&
             cuda_row_data_->bit_type() == 8));
  }

  /*! \brief whether the tree learner's per-tree column-view build can source its
   *  gather from this constructor's compact matrix (row-major-in-partition over
   *  the tree's sampled columns) instead of the full bin matrix */
  bool CompactViewSourceAvailable() const {
    return use_compact_view_ && !compact_is_col_major_;
  }
  const std::vector<int>& compact_src_cols() const { return compact_src_cols_host_; }
  const std::vector<size_t>& compact_slot_bytes() const { return compact_slot_byte_host_; }
  const std::vector<int>& compact_slot_strides() const { return compact_slot_stride_host_; }
  const std::vector<int>& compact_slot_cols() const { return compact_slot_col_host_; }
  /*! \brief whether the compact matrix is 4-bit packed (two columns per byte,
   *  per-partition even-column padding; mirrors CUDARowData::is_4bit_packed) */
  bool compact_src_is_4bit() const { return compact_is_4bit_; }
  const uint8_t* compact_data_device() const { return compact_data_uint8_t_.RawDataReadOnly(); }
  /*! \brief whether this tree's compact build also produced the column-major
   *  view (compact_col_major_device()[slot * num_data + row]); the tree learner
   *  then uses it directly instead of gathering its own copy */
  bool CompactColMajorFilled() const { return compact_col_major_filled_; }
  const uint8_t* compact_col_major_device() const { return compact_staging_col_major_.RawDataReadOnly(); }

  /*! \brief one-line gate dump for EXABOOST_HYBRID_DIAG */
  std::string BatchedLevelGateDiag() const {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "hist: row_data=%d sparse=%d large_bin_parts=%d compact_view=%d "
             "col_major=%d bit_type=%d batch_wide=%d",
             static_cast<int>(cuda_row_data_ != nullptr),
             cuda_row_data_ != nullptr ? static_cast<int>(cuda_row_data_->is_sparse()) : -1,
             cuda_row_data_ != nullptr ? cuda_row_data_->NumLargeBinPartition() : -1,
             static_cast<int>(use_compact_view_),
             static_cast<int>(compact_is_col_major_),
             cuda_row_data_ != nullptr ? cuda_row_data_->bit_type() : -1,
             static_cast<int>(ExaboostBatchWideEnabled()));
    return std::string(buf);
  }

  /*! \brief Batched per-level histogram phase for hybrid growth: one construct, one
   *  fix and one subtract launch cover every sibling pair of a level (pairs indexed
   *  by a grid dimension). All kernels run on pipeline_streams_[0] (cuda_stream_) in
   *  order; subtract_done_events_[0] is recorded at the end so the best split finder
   *  can order its batched find kernel after the whole histogram phase.
   *  level_smaller_num_data is null in the classic (two-sync) flow, where
   *  max_num_data_in_smaller_leaf is the level's exact maximum smaller-leaf size.
   *  In the speculative (single-sync) flow it points at the device array of this
   *  level's actual smaller-child sizes (written by the batched apply kernels);
   *  max_num_data_in_smaller_leaf is then only an upper BOUND used to size the
   *  launch grid, and the construct kernel derives the exact per-thread row
   *  grouping on-device from the array so histograms are bit-identical to the
   *  classic sizing (non-quantized dense path only). */
  void ConstructHistogramsForLevel(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs,
    const data_size_t max_num_data_in_smaller_leaf,
    const bool any_pair_needs_bit_change_copy,
    const data_size_t* level_smaller_num_data = nullptr);

  /*! \brief minimum rows-per-thread cap of the batched construct grid sizing
   *  (EXABOOST_BATCH_CONSTRUCT_MINROWS; 0 = per-leaf sizing). Shared by the host
   *  grid sizing and the device row-grouping replica. */
  static int BatchConstructMinRowsPerThread() {
    static const int min_rows_per_thread = []() {
      const char* env = std::getenv("EXABOOST_BATCH_CONSTRUCT_MINROWS");
      return env != nullptr ? std::atoi(env) : 64;
    }();
    return min_rows_per_thread;
  }

  /*! \brief device-saturation floor (total y-blocks shared across the level's
   *  pairs) of the batched construct grid sizing
   *  (EXABOOST_BATCH_CONSTRUCT_FLOOR; default 160 = historical behavior) */
  static int BatchConstructSaturationFloor() {
    static const int floor_total = []() {
      const char* env = std::getenv("EXABOOST_BATCH_CONSTRUCT_FLOOR");
      return env != nullptr ? std::atoi(env) : 160;
    }();
    return floor_total;
  }

  /*! \brief interleaved float2 gradient/hessian copy for the non-quantized
   *  dense construct kernels: their scattered per-row reads then touch ONE
   *  32B sector per row instead of two (gradients and hessians live in two
   *  separate arrays). Values are bit-identical to the separate reads.
   *  EXABOOST_GH_INTERLEAVE=0 disables; float score_t only. */
  static bool GHInterleaveEnabled() {
    static const bool enabled = []() {
      const char* env = std::getenv("EXABOOST_GH_INTERLEAVE");
      return env == nullptr || std::string(env) != "0";
    }();
    return enabled && sizeof(score_t) == sizeof(float);
  }

  /*! \brief kill-switch of the small-leaf construct path
   *  (EXABOOST_SMALL_LEAF_CONSTRUCT=0 restores the shared-memory batched
   *  kernels for every level) */
  static bool SmallLeafConstructEnabled() {
    static const bool enabled = []() {
      const char* env = std::getenv("EXABOOST_SMALL_LEAF_CONSTRUCT");
      return env == nullptr || std::string(env) != "0";
    }();
    return enabled;
  }

  /*! \brief leaf-size threshold of the DIRECT small-leaf construct body: a pair
   *  whose actual smaller-leaf size (device-side check) is at most this many
   *  rows skips the shared-memory histogram accumulation and adds straight to
   *  the global histogram (non-quantized training only). Default 0 = disabled:
   *  the direct body changes the float accumulation (per-row double adds vs
   *  per-block float partial sums), which trades the fraud 63/6 exact-quality
   *  reproduction for a measured ~1-2% fraud-deep gain -- not worth it by
   *  default. Set EXABOOST_SMALL_LEAF_ROWS=8192 to enable. */
  static data_size_t SmallLeafRowThreshold() {
    static const data_size_t threshold = []() {
      const char* env = std::getenv("EXABOOST_SMALL_LEAF_ROWS");
      return env != nullptr ? static_cast<data_size_t>(std::atoi(env)) : 0;
    }();
    return threshold;
  }

#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  /*! \brief graphs L1 body capture: construct + fix/subtract kernels with
   *  placeholder grids on the hist stream, node handles + roles collected for
   *  the device controller; records subtract_done_events_[0] (the find phase
   *  waits on it inside the graph exactly like the host flow) */
  void CaptureHybridGraphSearchKernels(const CUDAHybridPairDescriptor* pair_descs,
                                       const data_size_t* level_smaller_num_data,
                                       std::vector<cudaGraphNode_t>* nodes,
                                       std::vector<int>* roles,
                                       std::vector<int>* role_static_x);

  /*! \brief static launch extents of the batched construct kernel (grid x and
   *  block dim y), for the controller's per-level grid-y formula */
  void HybridGraphConstructDims(int* grid_x, int* block_dim_y) const;

  int hybrid_graph_min_grid_dim_y() const { return min_grid_dim_y_; }

  /*! \brief per-tree pointers/flags baked into captured construct kernel
   *  params: a graph instance is only valid for trees whose key matches (the
   *  compact metadata buffers grow to the running-max sampled column count,
   *  so their pointers stabilize after the first few trees) */
  void HybridGraphKeyPointers(std::vector<const void*>* key) const {
    key->push_back(use_compact_view_ ?
      static_cast<const void*>(compact_data_uint8_t_.RawDataReadOnly()) : nullptr);
    key->push_back(use_compact_view_ ?
      static_cast<const void*>(compact_column_hist_offsets_.RawDataReadOnly()) : nullptr);
    key->push_back(use_compact_view_ ?
      static_cast<const void*>(compact_feature_partition_column_index_offsets_.RawDataReadOnly()) : nullptr);
    key->push_back(use_compact_view_ ?
      static_cast<const void*>(compact_packed_partition_byte_offsets_.RawDataReadOnly()) : nullptr);
    key->push_back(static_cast<const void*>(cuda_gradients_));
    key->push_back(static_cast<const void*>(cuda_hessians_));
    key->push_back(static_cast<const void*>(cuda_gradients_hessians_.RawDataReadOnly()));
    key->push_back(static_cast<const void*>(cuda_is_feature_used_bytree_.RawDataReadOnly()));
    key->push_back(any_feature_unused_bytree_ ? this : nullptr);
  }

  /*! \brief graphs L1 + compact view: the batched construct BLOCK dims follow
   *  the per-tree sampled column count and cannot be updated inside a graph
   *  (only grids are device-updatable), so the construct shape is part of the
   *  graph key: one instance per distinct shape, each captured with the exact
   *  host block dims (keeps results bit-identical to the host loop) */
  int HybridGraphCompactShapeKey() const {
    return use_compact_view_ ? std::max(1, max_num_compact_cols_per_partition_) : 0;
  }
#endif  // EXABOOST_HYBRID_GRAPH_SUPPORTED

  void ResetTrainingData(const Dataset* train_data, TrainingShareStates* share_states);

  void ResetConfig(const Config* config);

  void BeforeTrain(const score_t* gradients, const score_t* hessians);

  /*! \brief number of leaf histogram slots the finished tree dirtied (== its
   *  final leaf count; leaf k's histogram is slot k). BeforeTrain only zeroes
   *  that prefix of cuda_hist_ instead of all num_leaves slots. -1 (initial /
   *  after resets, and kept on the NCCL path) zeroes everything. */
  void SetNumDirtyLeaves(const int num_dirty_leaves) { num_dirty_leaves_ = num_dirty_leaves; }

  // Per-tree feature sampling mask (host-side vector, length == num_features_).
  // Copied to cuda_is_feature_used_bytree_ so the histogram kernel can skip
  // features not selected by feature_fraction sampling.
  void SetFeatureUsedBytree(const std::vector<int8_t>& is_feature_used_bytree);

  // Build a compact view of the bin matrix that contains ONLY the columns
  // selected by this tree's feature_fraction sample. When this returns true,
  // subsequent ConstructHistogramForLeaf launches will use the compact
  // buffers (smaller block_dim_x, fewer warps per block, ~10x speedup at
  // feature_fraction=0.1). Falls back automatically to the full-data path
  // when sampling is disabled or the data layout is not supported.
  bool BuildCompactView(const std::vector<int8_t>& is_feature_used_bytree);

  // Expose internal CUDARowData so the tree learner can build a per-tree compact
  // column view from the on-GPU row-major bin matrix.
  const CUDARowData* cuda_row_data_internal() const { return cuda_row_data_.get(); }

  const hist_t* cuda_hist() const { return cuda_hist_.RawData(); }

  hist_t* cuda_hist_pointer() { return cuda_hist_.RawData(); }

  /*! \brief non-quantized global histograms are stored as float pairs in the
   *  leading half of each leaf's hist_t slot (EXABOOST_FP32_HIST; slot strides
   *  and the leaf histogram pool arithmetic are unchanged) */
  bool hist_fp32() const { return hist_fp32_; }

  /*! \brief consumers the fp32 layout does not cover (large-bin global-memory
   *  find, forced splits, NCCL) make the tree learner fall back to fp64 */
  void DisableHistFP32() { hist_fp32_ = false; }

 private:
  void InitFeatureMetaInfo(const Dataset* train_data, const std::vector<uint32_t>& feature_hist_offsets);

  /*! \brief (re)build cuda_fix_mfb_mask_ from the need-fix feature metadata */
  void InitFixMFBMask();

  void CalcConstructHistogramKernelDim(
    int* grid_dim_x,
    int* grid_dim_y,
    int* block_dim_x,
    int* block_dim_y,
    const data_size_t num_data_in_smaller_leaf);

  /*! \brief Grid sizing for the batched per-level construct kernel. Shares the
   *  min_grid_dim_y_ device-saturation floor across the level's pairs and caps
   *  the y-grid at a minimum rows-per-thread: every active block pays a fixed
   *  shared-histogram zero + global-merge cost, which dominates small leaves.
   *  Identical to the per-leaf sizing for single-pair levels and for leaves
   *  large enough that the cap is inactive. */
  void CalcConstructHistogramBatchedKernelDim(
    int* grid_dim_x,
    int* grid_dim_y,
    int* block_dim_x,
    int* block_dim_y,
    const data_size_t max_num_data_in_smaller_leaf,
    const int num_pairs);

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE>
  void LaunchConstructHistogramKernelInner(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE>
  void LaunchConstructHistogramKernelInner0(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE>
  void LaunchConstructHistogramKernelInner1(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE, bool USE_GLOBAL_MEM_BUFFER>
  void LaunchConstructHistogramKernelInner2(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  void LaunchConstructHistogramKernel(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  void LaunchSubtractHistogramKernel(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
    const bool use_discretized_grad,
    const uint8_t parent_num_bits_in_histogram_bins,
    const uint8_t smaller_num_bits_in_histogram_bins,
    const uint8_t larger_num_bits_in_histogram_bins);

  // ---- batched per-level launchers (hybrid growth) ----

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE>
  void LaunchConstructHistogramBatchedKernelInner(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs,
    const data_size_t max_num_data_in_smaller_leaf,
    const data_size_t* level_smaller_num_data);

  /*! \brief USE_GH2: feed the construct kernel the interleaved float2
   *  gradient/hessian copy (see GHInterleaveEnabled); compile-time so the
   *  non-interleaved instantiation stays identical to the historical kernel */
  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, bool USE_GH2>
  void LaunchConstructHistogramBatchedKernelInner0(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs,
    const data_size_t max_num_data_in_smaller_leaf,
    const data_size_t* level_smaller_num_data);

  void LaunchConstructHistogramBatchedKernel(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs,
    const data_size_t max_num_data_in_smaller_leaf,
    const data_size_t* level_smaller_num_data);

  void LaunchSubtractHistogramBatchedKernel(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs,
    const bool any_pair_needs_bit_change_copy);

  // ---- small-leaf level launchers (hybrid growth, non-quantized only) ----

  /*! \brief fused fix + subtract of the small-leaf path: one launch does both
   *  the most-frequent-bin fix of the smaller leaf and the parent-minus-smaller
   *  subtraction of the larger leaf. Subtract blocks skip the fix features'
   *  most-frequent-bin entries (cuda_fix_mfb_mask_); the fix blocks write those
   *  for both leaves with the identical arithmetic, so the result is
   *  bit-identical to the sequential fix -> subtract launches. */
  void LaunchFixSubtractHistogramSmallLeafBatchedKernel(
    const CUDAHybridPairDescriptor* pair_descs,
    const int num_pairs);

  /*! \brief one tiny launch per speculative level: reduce the level's actual
   *  smaller-child sizes and apply the exact host grid-sizing formula, writing
   *  the effective row-grouping extent to cuda_hybrid_construct_dim_y_ */
  void LaunchComputeBatchedConstructDimYKernel(
    const data_size_t* level_smaller_num_data,
    const int num_pairs,
    const int block_dim_y);

  // Host memory

  /*! \brief size of training data */
  data_size_t num_data_;
  /*! \brief number of features in training data */
  int num_features_;
  /*! \brief maximum number of leaves */
  int num_leaves_;
  /*! \brief number of threads */
  int num_threads_;
  /*! \brief total number of bins in histogram */
  int num_total_bin_;
  /*! \brief number of bins per feature */
  std::vector<uint32_t> feature_num_bins_;
  /*! \brief offsets in histogram of all features */
  std::vector<uint32_t> feature_hist_offsets_;
  /*! \brief most frequent bins in each feature */
  std::vector<uint32_t> feature_most_freq_bins_;
  /*! \brief minimum number of data allowed per leaf */
  int min_data_in_leaf_;
  /*! \brief minimum sum value of hessians allowed per leaf */
  double min_sum_hessian_in_leaf_;
  /*! \brief cuda stream for histogram construction */
  cudaStream_t cuda_stream_;
  cudaStream_t pipeline_streams_[kNumHistPipelines];
  cudaEvent_t construct_done_events_[kNumHistPipelines];
  cudaEvent_t subtract_done_events_[kNumHistPipelines];
  int active_pipeline_ = 0;
  /*! \brief recorded on cuda_stream_ after smaller-leaf histogram construction */
  cudaEvent_t construct_done_event_ = nullptr;
  /*! \brief recorded on cuda_stream_ after larger-leaf histogram subtraction */
  cudaEvent_t subtract_done_event_ = nullptr;
  /*! \brief indices of feature whose histograms need to be fixed */
  std::vector<int> need_fix_histogram_features_;
  /*! \brief aligned number of bins of the features whose histograms need to be fixed */
  std::vector<uint32_t> need_fix_histogram_features_num_bin_aligend_;
  /*! \brief minimum number of blocks allowed in the y dimension */
  const int min_grid_dim_y_ = 160;
  /*! \brief leaf histogram slots dirtied by the previous tree (-1 = all) */
  int num_dirty_leaves_ = -1;
  /*! \brief dataset has categorical features (their find kernel reads hist_t;
   *  disables the fp32 histogram mode) */
  bool has_categorical_feature_ = false;
  /*! \brief non-quantized global histograms stored as float pairs (see hist_fp32()) */
  bool hist_fp32_ = false;


  // CUDA memory, held by this object

  /*! \brief CUDA row wise data */
  std::unique_ptr<CUDARowData> cuda_row_data_;
  /*! \brief number of bins per feature */
  CUDAVector<uint32_t> cuda_feature_num_bins_;
  /*! \brief offsets in histogram of all features */
  CUDAVector<uint32_t> cuda_feature_hist_offsets_;
  /*! \brief most frequent bins in each feature */
  CUDAVector<uint32_t> cuda_feature_most_freq_bins_;
  /*! \brief CUDA histograms */
  CUDAVector<hist_t> cuda_hist_;
  /*! \brief CUDA histograms buffer for each block */
  CUDAVector<float> cuda_hist_buffer_;
  /*! \brief Per-tree feature mask (1 = feature in this tree's sample, 0 = skip).
   *  Indexed by column_index (== inner_feature_index for dense single-feature groups). */
  CUDAVector<int8_t> cuda_is_feature_used_bytree_;
  /*! \brief effective row-grouping extent of the speculative batched construct
   *  launch (see LaunchComputeBatchedConstructDimYKernel) */
  CUDAVector<int> cuda_hybrid_construct_dim_y_;
  /*! \brief mask over the 2 * num_total_bin_ histogram entries marking the
   *  most-frequent-bin gradient/hessian slots of the need-fix features; the
   *  fused small-leaf fix+subtract kernel's subtract blocks skip these (the fix
   *  blocks own them) */
  CUDAVector<uint8_t> cuda_fix_mfb_mask_;
  /*! \brief per-tree bin-level used mask (1 = bin belongs to a feature in this
   *  tree's feature_fraction sample). The batched fix/subtract kernels and the
   *  batched construct's shared-histogram zero/merge skip unused bins: nothing
   *  of this tree ever reads them (the find kernels are feature-masked), so
   *  their content is dead storage. Only active when sampling is on. */
  std::vector<uint8_t> host_bin_used_bytree_;
  CUDAVector<uint8_t> cuda_bin_used_bytree_;
  bool any_feature_unused_bytree_ = false;
  /*! \brief every feature fits the register-accumulation bin cap (<= 8 bins);
   *  selects the contention-free construct body on the batched compact path
   *  (non-quantized only; EXABOOST_BATCH_REGHIST=0 disables) */
  bool construct_reg_bins_ = false;

  // ========================================================================
  // Compact-view buffers: when feature_fraction < 1.0, build a contiguous
  // bin matrix containing only the sampled columns. Used by the histogram
  // kernel as a drop-in replacement when use_compact_view_ == true.
  // Single-partition layout: compact[row * num_compact_cols + i].
  // ========================================================================
  /*! \brief uint8 bin matrix containing only used columns (per-partition compact layout).
   *  Double-buffered: while tree N trains using current, we async-fill next for tree N+1.
   *  cuda_stream_ waits on prefetch_event_[current] before starting histograms. */
  CUDAVector<uint8_t> compact_data_uint8_t_;
  CUDAVector<uint8_t> compact_data_uint8_t_alt_;
  bool active_buffer_is_alt_ = false;
  cudaEvent_t fill_done_event_ = nullptr;
  cudaEvent_t fill_done_event_alt_ = nullptr;
  cudaStream_t prefetch_stream_ = nullptr;
  /*! \brief number of columns in the compact view (sum across partitions) */
  int num_compact_columns_;
  /*! \brief max compact cols per partition (sets block_dim_x for compact launches) */
  int max_num_compact_cols_per_partition_;

  /*! \brief whether compact view is active for current tree */
  bool use_compact_view_;
  /*! \brief if true, compact_data_uint8_t_ is column-major-in-partition (used when source is host) */
  bool compact_is_col_major_ = false;
  /*! \brief host copies of the compact layout (source column per slot and the
   *  row-major-in-partition placement), kept so the tree learner's per-tree
   *  column-view build can gather from the ~10x smaller compact matrix instead
   *  of re-reading the full bin matrix (see CompactViewSource*) */
  std::vector<int> compact_src_cols_host_;
  std::vector<size_t> compact_slot_byte_host_;
  std::vector<int> compact_slot_stride_host_;
  std::vector<int> compact_slot_col_host_;
  /*! \brief compact matrix is 4-bit packed (set iff the row data is packed) */
  bool compact_is_4bit_ = false;
  /*! \brief prefix over partitions of packed compact per-row byte widths (4-bit mode) */
  CUDAVector<int> compact_packed_partition_byte_offsets_;
  /*! \brief per destination-byte-slot metadata of the 4-bit compact fill kernel
   *  (source nibble bases of the byte's two columns, per-row nibble stride,
   *  destination byte base/stride) */
  CUDAVector<size_t> cuda_bs_src_nib0_;
  CUDAVector<size_t> cuda_bs_src_nib1_;
  CUDAVector<int> cuda_bs_src_stride_nib_;
  CUDAVector<size_t> cuda_bs_dst_byte_;
  CUDAVector<int> cuda_bs_dst_stride_;
  /*! \brief whether compact_staging_col_major_ holds this tree's column-major
   *  compact view (fused second output of the compact fill) */
  bool compact_col_major_filled_ = false;
  /*! \brief column_hist_offsets for compact view (length num_compact_columns+1) */
  CUDAVector<uint32_t> compact_column_hist_offsets_;
  /*! \brief partition_hist_offsets for compact view: [0, total_compact_bins] */
  CUDAVector<uint32_t> compact_partition_hist_offsets_;
  /*! \brief partition column offsets for compact view: [0, num_compact_columns] */
  CUDAVector<int> compact_feature_partition_column_index_offsets_;
  /*! \brief per-slot precomputed src/dst metadata for the new fill kernel. */
  CUDAVector<size_t> cuda_slot_src_byte_;
  CUDAVector<int> cuda_slot_src_stride_;
  CUDAVector<size_t> cuda_slot_dst_byte_;
  CUDAVector<int> cuda_slot_dst_stride_;
  /*! \brief Maps compact column index -> source byte offset in cuda_data_uint8_t_
   *  for row 0. To get source for row r at compact col c:
   *    src = source_offsets_[c] + r * source_strides_[c]
   *  (stride differs per partition; same for all cols in same partition). */
  CUDAVector<size_t> compact_source_offsets_;
  CUDAVector<int> compact_source_strides_;
  /*! \brief Pinned host staging buffer for compact data (used when source bin matrix
   *  is host-mapped — host-side gather + bulk cudaMemcpy is ~10× faster than the
   *  GPU fill kernel reading strided bytes via PCIe). Allocated via cudaMallocHost. */
  uint8_t* compact_data_host_ = nullptr;
  size_t compact_data_host_size_ = 0;
  /*! \brief GPU staging buffer for column-major transfer from host. Per-tree: cudaMemcpy
   *  each used column contiguously into this buffer, then transpose-kernel into
   *  compact_data_uint8_t_ which is row-major-in-partition (matches histogram kernel).  */
  CUDAVector<uint8_t> compact_staging_col_major_;
  /*! \brief indices of feature whose histograms need to be fixed */
  CUDAVector<int> cuda_need_fix_histogram_features_;
  /*! \brief aligned number of bins of the features whose histograms need to be fixed */
  CUDAVector<uint32_t> cuda_need_fix_histogram_features_num_bin_aligned_;
  /*! \brief histogram buffer used in histogram subtraction with different number of bits for histogram bins */
  CUDAVector<hist_t> hist_buffer_for_num_bit_change_;
  /*! \brief per-tree interleaved (gradient, hessian) pairs (see GHInterleaveEnabled);
   *  filled by BeforeTrain, consumed by the non-quantized dense construct kernels */
  CUDAVector<float2> cuda_gradients_hessians_;
  /*! \brief whether cuda_gradients_hessians_ holds this tree's gradients */
  bool gh_interleave_valid_ = false;

  // CUDA memory, held by other object

  /*! \brief gradients on CUDA */
  const score_t* cuda_gradients_;
  /*! \brief hessians on CUDA */
  const score_t* cuda_hessians_;

  /*! \brief GPU device index */
  const int gpu_device_id_;
  /*! \brief use double precision histogram per block */
  const bool gpu_use_dp_;
  /*! \brief whether to use quantized gradients */
  const bool use_quantized_grad_;
  /*! \brief the number of bins to quantized gradients */
  const int num_grad_quant_bins_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
