/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_TREE_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_TREE_HPP_

#ifdef USE_CUDA

#include <Falcata/bin.h>
#include <Falcata/tree.h>

#include <vector>

#include <Falcata/cuda/cuda_column_data.hpp>
#include <Falcata/cuda/cuda_split_info.hpp>

namespace Falcata {

__device__ void SetDecisionTypeCUDA(int8_t* decision_type, bool input, int8_t mask);

__device__ void SetMissingTypeCUDA(int8_t* decision_type, int8_t input);

__device__ bool GetDecisionTypeCUDA(int8_t decision_type, int8_t mask);

__device__ int8_t GetMissingTypeCUDA(int8_t decision_type);


/*! \brief Per-split descriptor for the batched tree-structure update of the hybrid
 *  level-batched growth phase: one SplitBatchKernel launch applies all numerical
 *  splits of a level (splits indexed by blockIdx.x). All fields are host-known
 *  before the level is applied; split VALUES (sums, gains, leaf outputs) come from
 *  the per-leaf device CUDASplitInfo pointer. */
struct CUDATreeBatchSplit {
  /*! \brief the leaf being split (left child keeps this index) */
  int leaf_index;
  /*! \brief tree num_leaves at the time of this split (== the right child's leaf
   *  index; the new internal node index is num_leaves_at_split - 1) */
  int num_leaves_at_split;
  int real_feature_index;
  double real_threshold;
  /*! \brief MissingType of the split feature, stored as int for POD-ness */
  int missing_type;
  const CUDASplitInfo* split_info;
};

/*! \brief One split of a host-side tree rebuild (RebuildFromHostSplits): all the
 *  values SplitKernel would read from the device CUDASplitInfo, captured on the
 *  host. Used by the selective (grow-then-prune) hybrid growth, which replays the
 *  final greedy split sequence host-side after pruning displaced splits. */
struct CUDATreeHostSplit {
  /*! \brief the leaf being split, in FINAL (classic leaf-wise) numbering */
  int leaf_index;
  int real_feature_index;
  double real_threshold;
  /*! \brief MissingType of the split feature, stored as int for POD-ness */
  int missing_type;
  int inner_feature_index;
  uint32_t threshold_in_bin;
  double gain;
  uint8_t default_left;
  double left_sum_hessians;
  double right_sum_hessians;
  data_size_t left_count;
  data_size_t right_count;
  double left_value;
  double right_value;
  /*! \brief categorical split payload: >0 selects the categorical replay.
   *  The bitsets are prebuilt on host in the model-format layout (uint32 words,
   *  bit v set for member value v): cat_bitset over REAL category values,
   *  cat_bitset_inner over inner bin values. */
  int num_cat_threshold = 0;
  std::vector<uint32_t> cat_bitset;
  std::vector<uint32_t> cat_bitset_inner;
};

class CUDATree : public Tree {
 public:
  /*!
  * \brief Constructor
  * \param max_leaves The number of max leaves
  * \param track_branch_features Whether to keep track of ancestors of leaf nodes
  * \param is_linear Whether the tree has linear models at each leaf
  * \param pooled_device_buffer Optional externally owned device slab of at least
  *        PooledDeviceBufferSize(max_leaves) bytes. When given, all per-tree
  *        device arrays are carved from it as non-owning views instead of ~17
  *        individual cudaMallocs, and ToHost() reads them back with ONE fused
  *        D2H copy. The caller must guarantee exclusive use of the slab until
  *        ToHost() has been called (the tree learner trains one tree at a time
  *        and calls ToHost() before returning it). Only cuda_leaf_value_ is
  *        needed after ToHost(); it is materialized into owned memory there.
  */
  explicit CUDATree(int max_leaves, bool track_branch_features, bool is_linear,
    const int gpu_device_id, const bool has_categorical_feature,
    uint8_t* pooled_device_buffer = nullptr, int leaf_value_dim = 1);

  /*! \brief bytes needed by the pooled per-tree device arrays (16 arrays of
   *  max_leaves entries + the batched split descriptors), 8-byte aligned */
  static size_t PooledDeviceBufferSize(const int max_leaves) {
    const size_t L = static_cast<size_t>(max_leaves);
    const size_t num_batch_splits = L / 2 + 2;
    return 5 * sizeof(double) * L +                       // threshold, internal_weight/value, leaf_value/weight
           sizeof(CUDATreeBatchSplit) * num_batch_splits +  // batched split descriptors (8-byte aligned)
           10 * 4 * L +                                    // int / uint32_t / data_size_t / float arrays
           ((sizeof(int8_t) * L + 7) / 8) * 8;             // decision_type, padded
  }

  explicit CUDATree(const Tree* host_tree);

  ~CUDATree() noexcept;

  int Split(const int leaf_index,
            const int real_feature_index,
            const double real_threshold,
            const MissingType missing_type,
            const CUDASplitInfo* cuda_split_info);

  /*! \brief Apply all numerical splits of one level in a single kernel launch
   *  (device tree-structure updates) plus the usual host-side mirror updates
   *  (num_leaves_, leaf_depth_, branch features). Equivalent to calling Split()
   *  once per entry in order: splits[k].num_leaves_at_split must equal the tree's
   *  num_leaves before split k is applied (right children take consecutive
   *  indices). The level's splits touch disjoint leaves, so the device updates
   *  are safe to run concurrently. */
  void SplitBatch(const std::vector<CUDATreeBatchSplit>& splits);

  /*! \brief vector-leaf counterpart of SetVectorLeafValuesFromSplit for a
   *  batch: writes both children's T leaf values of every split of the batch
   *  most recently passed to SplitBatch (the descriptors still live in the
   *  device batch buffer), one launch instead of one per split. */
  void SetVectorLeafValuesFromSplitBatch(const int num_splits);

  /*! \brief graphs L1 body capture: launch SplitBatchKernel with a placeholder
   *  grid on \p stream (the graph controller resizes it per level); the batch
   *  split descriptors are read from the pooled device buffer the controller
   *  writes (hybrid_graph_batch_splits()). \p graph_live_split_count is the
   *  device address of the loop state's live split count: the controller
   *  freezes the grid at a pow2 bucket and idle blocks guard on it (A2) */
  void CaptureHybridGraphSplitBatchKernel(cudaStream_t stream,
                                          const int* graph_live_split_count);

  /*! \brief pooled device buffer of the batched split descriptors (written by
   *  the graphs L1 device controller; stable across pooled trees) */
  CUDATreeBatchSplit* hybrid_graph_batch_splits() { return cuda_batch_splits_.RawData(); }

  /*! \brief Replace the recorded node counts with the data partition's exact
   *  per-leaf row counts and rebuild the internal counts from them. The split
   *  finder can only estimate counts (count = round(hessian * num_data /
   *  sum_hessians), since a CUDA histogram bin carries gradient and hessian but
   *  no count), which is exact only for a constant-hessian objective. */
  void SyncNodeCountsFromPartition(const std::vector<data_size_t>& leaf_num_data);

  /*! \brief graphs L1 post-prefix replay: the host-mirror half of SplitBatch
   *  (branch features, leaf depth, num_leaves_) for ONE split whose device
   *  updates already ran inside the graph */
  void ReplaySplitHostMirrors(const int leaf_index, const int real_feature_index) {
    RecordBranchFeatures(leaf_index, num_leaves_, real_feature_index);
    leaf_depth_[num_leaves_] = leaf_depth_[leaf_index] + 1;
    leaf_depth_[leaf_index]++;
    ++num_leaves_;
  }

  /*! \brief Build the final tree structure host-side from a replayed split
   *  sequence (selective grow-then-prune hybrid growth). The device tree arrays
   *  were never written during growth; this replays SplitKernel's per-field
   *  arithmetic on the host mirrors (bit-identical: same IEEE double operations
   *  on the same captured CUDASplitInfo values), uploads the final leaf values
   *  to the device (AddPredictionToScore / Shrinkage / renewal kernels read
   *  them), and then performs ToHost()'s cleanup (host vector shrink, leaf
   *  value materialization, device array release, stream destroy). The caller
   *  must NOT call ToHost() afterwards. leaf_value_[0] must hold the root
   *  output before the call (it seeds the internal_value_ chain). */
  void RebuildFromHostSplits(const std::vector<CUDATreeHostSplit>& splits);

  int SplitCategorical(
    const int leaf_index,
    const int real_feature_index,
    const MissingType missing_type,
    const CUDASplitInfo* cuda_split_info,
    uint32_t* cuda_bitset,
    size_t cuda_bitset_len,
    uint32_t* cuda_bitset_inner,
    size_t cuda_bitset_inner_len);

  /*!
  * \brief Adding prediction value of this tree model to scores
  * \param data The dataset
  * \param num_data Number of total data
  * \param score Will add prediction to score
  */
  void AddPredictionToScore(const Dataset* data,
                            data_size_t num_data,
                            double* score) const override;

  /*!
  * \brief Adding prediction value of this tree model to scores
  * \param data The dataset
  * \param used_data_indices Indices of used data
  * \param num_data Number of total data
  * \param score Will add prediction to score
  */
  void AddPredictionToScore(const Dataset* data,
                            const data_size_t* used_data_indices,
                            data_size_t num_data, double* score) const override;

  inline void AsConstantTree(double val, int count) override;

  const int* cuda_leaf_parent() const { return cuda_leaf_parent_.RawData(); }

  const int* cuda_left_child() const { return cuda_left_child_.RawData(); }

  const int* cuda_right_child() const { return cuda_right_child_.RawData(); }

  const int* cuda_split_feature_inner() const { return cuda_split_feature_inner_.RawData(); }

  const int* cuda_split_feature() const { return cuda_split_feature_.RawData(); }

  const uint32_t* cuda_threshold_in_bin() const { return cuda_threshold_in_bin_.RawData(); }

  const double* cuda_threshold() const { return cuda_threshold_.RawData(); }

  const int8_t* cuda_decision_type() const { return cuda_decision_type_.RawData(); }

  const double* cuda_leaf_value() const { return cuda_leaf_value_.RawData(); }

  double* cuda_leaf_value_ref() { return cuda_leaf_value_.RawData(); }

  /*! \brief vector-leaf mode: flat [leaf * leaf_value_dim + target] device
   *  leaf outputs (empty when leaf_value_dim == 1). Retained after ToHost()
   *  like cuda_leaf_value_ so score updates and shrinkage can read it. */
  const double* cuda_leaf_values_vec() const { return cuda_leaf_values_vec_.RawData(); }

  /*! \brief vector-leaf mode: write both children's per-target leaf outputs
   *  from the winning split's vector payload (called right after Split()) */
  void SetVectorLeafValuesFromSplit(const int left_leaf_index, const int right_leaf_index,
                                    const CUDASplitInfo* cuda_split_info);

  inline void Shrinkage(double rate) override;

  inline void AddBias(double val) override;

  void ToHost();

  void SyncLeafOutputFromHostToCUDA();

  void SyncLeafOutputFromCUDAToHost();

 private:
  /*! \brief shared tail of ToHost() and RebuildFromHostSplits(): shrink the host
   *  vectors to num_leaves_, materialize the retained device leaf values and
   *  release every other per-tree device array + the per-tree stream */
  void ShrinkHostVectorsAndReleaseDevice();

  void InitCUDAMemory(uint8_t* pooled_device_buffer);

  void InitCUDA();

  void LaunchSplitKernel(const int leaf_index,
                         const int real_feature_index,
                         const double real_threshold,
                         const MissingType missing_type,
                         const CUDASplitInfo* cuda_split_info);

  void LaunchSplitBatchKernel(const int num_splits);

  void LaunchSplitCategoricalKernel(
    const int leaf_index,
    const int real_feature_index,
    const MissingType missing_type,
    const CUDASplitInfo* cuda_split_info,
    size_t cuda_bitset_len,
    size_t cuda_bitset_inner_len);

  void LaunchAddPredictionToScoreKernel(const Dataset* data,
                                        const data_size_t* used_data_indices,
                                        data_size_t num_data, double* score) const;

  void LaunchShrinkageKernel(const double rate);

  void LaunchAddBiasKernel(const double val);

  void LaunchSetVectorLeafValuesFromSplitKernel(const int left_leaf_index,
    const int right_leaf_index, const CUDASplitInfo* cuda_split_info);

  void LaunchSetVectorLeafValuesFromSplitBatchKernel(const int num_splits);

  /*! \brief internal_count_ of the subtree at node, bottom-up from leaf_count_ */
  data_size_t RebuildInternalCounts(const int node);

  void RecordBranchFeatures(const int left_leaf_index,
                            const int right_leaf_index,
                            const int real_feature_index);

  CUDAVector<int> cuda_left_child_;
  CUDAVector<int> cuda_right_child_;
  CUDAVector<int> cuda_split_feature_inner_;
  CUDAVector<int> cuda_split_feature_;
  CUDAVector<int> cuda_leaf_depth_;
  CUDAVector<int> cuda_leaf_parent_;
  CUDAVector<uint32_t> cuda_threshold_in_bin_;
  CUDAVector<double> cuda_threshold_;
  CUDAVector<double> cuda_internal_weight_;
  CUDAVector<double> cuda_internal_value_;
  CUDAVector<int8_t> cuda_decision_type_;
  CUDAVector<double> cuda_leaf_value_;
  /*! \brief vector-leaf outputs [leaf * leaf_value_dim + target]; own device
   *  memory (never pooled), empty in scalar mode */
  CUDAVector<double> cuda_leaf_values_vec_;
  CUDAVector<data_size_t> cuda_leaf_count_;
  CUDAVector<double> cuda_leaf_weight_;
  CUDAVector<data_size_t> cuda_internal_count_;
  CUDAVector<float> cuda_split_gain_;
  CUDAVector<uint32_t> cuda_bitset_;
  CUDAVector<uint32_t> cuda_bitset_inner_;
  CUDAVector<int> cuda_cat_boundaries_;
  CUDAVector<int> cuda_cat_boundaries_inner_;
  /*! \brief device copy of one level's batched split descriptors (SplitBatch) */
  CUDAVector<CUDATreeBatchSplit> cuda_batch_splits_;
  /*! \brief whether the per-tree arrays are views into a pooled slab */
  bool pooled_ = false;

  cudaStream_t cuda_stream_;

 public:
  /*! \brief stream the tree-structure kernels (SplitBatch/SplitCategorical)
   *  run on; the hybrid apply's categorical-slab consumers are ordered
   *  against the next level's best-split sync through it */
  cudaStream_t tree_stream() const { return cuda_stream_; }

 private:
  const int num_threads_per_block_add_prediction_to_score_;
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_TREE_HPP_
