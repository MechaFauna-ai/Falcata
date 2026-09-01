/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */


#ifdef USE_CUDA

#include <Falcata/cuda/cuda_column_data.hpp>
#include <Falcata/cuda/cuda_tree.hpp>

namespace Falcata {

// Decision-type bit helpers, used only by the kernels below. They stay in this
// translation unit: a __device__ function called across TUs would need
// relocatable device code, which the build deliberately does not use.
__device__ void SetDecisionTypeCUDA(int8_t* decision_type, bool input, int8_t mask) {
  if (input) {
    (*decision_type) |= mask;
  } else {
    (*decision_type) &= (127 - mask);
  }
}

__device__ void SetMissingTypeCUDA(int8_t* decision_type, int8_t input) {
  (*decision_type) &= 3;
  (*decision_type) |= (input << 2);
}

__device__ bool GetDecisionTypeCUDA(int8_t decision_type, int8_t mask) {
  return (decision_type & mask) > 0;
}

__device__ int8_t GetMissingTypeCUDA(int8_t decision_type) {
  return (decision_type >> 2) & 3;
}

template<typename T>
__device__ bool FindInBitsetCUDA(const uint32_t* bits, int n, T pos) {
  int i1 = pos / 32;
  if (i1 >= n) {
    return false;
  }
  int i2 = pos % 32;
  return (bits[i1] >> i2) & 1;
}

__global__ void SplitKernel(  // split information
                            const int leaf_index,
                            const int real_feature_index,
                            const double real_threshold,
                            const MissingType missing_type,
                            const CUDASplitInfo* cuda_split_info,
                            // tree structure
                            const int num_leaves,
                            int* leaf_parent,
                            int* leaf_depth,
                            int* left_child,
                            int* right_child,
                            int* split_feature_inner,
                            int* split_feature,
                            float* split_gain,
                            double* internal_weight,
                            double* internal_value,
                            data_size_t* internal_count,
                            double* leaf_weight,
                            double* leaf_value,
                            data_size_t* leaf_count,
                            int8_t* decision_type,
                            uint32_t* threshold_in_bin,
                            double* threshold) {
  const int new_node_index = num_leaves - 1;
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int parent_index = leaf_parent[leaf_index];
  if (thread_index == 0) {
    if (parent_index >= 0) {
      // if cur node is left child
      if (left_child[parent_index] == ~leaf_index) {
        left_child[parent_index] = new_node_index;
      } else {
        right_child[parent_index] = new_node_index;
      }
    }
    left_child[new_node_index] = ~leaf_index;
    right_child[new_node_index] = ~num_leaves;
    leaf_parent[leaf_index] = new_node_index;
    leaf_parent[num_leaves] = new_node_index;
  } else if (thread_index == 1) {
    // add new node
    split_feature_inner[new_node_index] = cuda_split_info->inner_feature_index;
  } else if (thread_index == 2) {
    split_feature[new_node_index] = real_feature_index;
  } else if (thread_index == 3) {
    split_gain[new_node_index] = static_cast<float>(cuda_split_info->gain);
  } else if (thread_index == 4) {
    // save current leaf value to internal node before change
    internal_weight[new_node_index] = cuda_split_info->left_sum_hessians + cuda_split_info->right_sum_hessians;
    leaf_weight[leaf_index] = cuda_split_info->left_sum_hessians;
  } else if (thread_index == 5) {
    internal_value[new_node_index] = leaf_value[leaf_index];
    leaf_value[leaf_index] = isnan(cuda_split_info->left_value) ? 0.0f : cuda_split_info->left_value;
  } else if (thread_index == 6) {
    internal_count[new_node_index] = cuda_split_info->left_count + cuda_split_info->right_count;
  } else if (thread_index == 7) {
    leaf_count[leaf_index] = cuda_split_info->left_count;
  } else if (thread_index == 8) {
    leaf_value[num_leaves] = isnan(cuda_split_info->right_value) ? 0.0f : cuda_split_info->right_value;
  } else if (thread_index == 9) {
    leaf_weight[num_leaves] = cuda_split_info->right_sum_hessians;
  } else if (thread_index == 10) {
    leaf_count[num_leaves] = cuda_split_info->right_count;
  } else if (thread_index == 11) {
    // update leaf depth
    leaf_depth[num_leaves] = leaf_depth[leaf_index] + 1;
    leaf_depth[leaf_index]++;
  } else if (thread_index == 12) {
    decision_type[new_node_index] = 0;
    SetDecisionTypeCUDA(&decision_type[new_node_index], false, kCategoricalMask);
    SetDecisionTypeCUDA(&decision_type[new_node_index], cuda_split_info->default_left, kDefaultLeftMask);
    SetMissingTypeCUDA(&decision_type[new_node_index], static_cast<int8_t>(missing_type));
  } else if (thread_index == 13) {
    threshold_in_bin[new_node_index] = cuda_split_info->threshold;
  } else if (thread_index == 14) {
    threshold[new_node_index] = real_threshold;
  }
}

void CUDATree::LaunchSplitKernel(const int leaf_index,
                                 const int real_feature_index,
                                 const double real_threshold,
                                 const MissingType missing_type,
                                 const CUDASplitInfo* cuda_split_info) {
  SplitKernel<<<3, 5, 0, cuda_stream_>>>(
    // split information
    leaf_index,
    real_feature_index,
    real_threshold,
    missing_type,
    cuda_split_info,
    // tree structure
    num_leaves_,
    cuda_leaf_parent_.RawData(),
    cuda_leaf_depth_.RawData(),
    cuda_left_child_.RawData(),
    cuda_right_child_.RawData(),
    cuda_split_feature_inner_.RawData(),
    cuda_split_feature_.RawData(),
    cuda_split_gain_.RawData(),
    cuda_internal_weight_.RawData(),
    cuda_internal_value_.RawData(),
    cuda_internal_count_.RawData(),
    cuda_leaf_weight_.RawData(),
    cuda_leaf_value_.RawData(),
    cuda_leaf_count_.RawData(),
    cuda_decision_type_.RawData(),
    cuda_threshold_in_bin_.RawData(),
    cuda_threshold_.RawData());
}

// Batched variant of SplitKernel for the hybrid level-batched growth phase: one
// block per split (blockIdx.x indexes CUDATreeBatchSplit descriptors), threadIdx.x
// plays the role of SplitKernel's global thread index. The level's splits act on
// disjoint leaves; the only shared cells are a sibling pair's parent child-pointers,
// where the compare-then-write is race-free because node indices (>= 0) can never
// equal leaf codes (~leaf < 0).
__global__ void SplitBatchKernel(  // split information
                                 const CUDATreeBatchSplit* batch_splits,
                                 // tree structure
                                 int* leaf_parent,
                                 int* leaf_depth,
                                 int* left_child,
                                 int* right_child,
                                 int* split_feature_inner,
                                 int* split_feature,
                                 float* split_gain,
                                 double* internal_weight,
                                 double* internal_value,
                                 data_size_t* internal_count,
                                 double* leaf_weight,
                                 double* leaf_value,
                                 data_size_t* leaf_count,
                                 int8_t* decision_type,
                                 uint32_t* threshold_in_bin,
                                 double* threshold,
                                 // graphs A2: live split count of the current graph level body
                                 // (nullptr on the exact-grid host launches); the graph-frozen
                                 // grid is a pow2 bucket of it, so excess blocks exit here
                                 const int* graph_live_split_count) {
  if (graph_live_split_count != nullptr &&
      blockIdx.x >= static_cast<unsigned int>(*graph_live_split_count)) {
    return;
  }
  const CUDATreeBatchSplit batch_split = batch_splits[blockIdx.x];
  const int leaf_index = batch_split.leaf_index;
  const int num_leaves = batch_split.num_leaves_at_split;
  const CUDASplitInfo* cuda_split_info = batch_split.split_info;
  const int new_node_index = num_leaves - 1;
  const int thread_index = static_cast<int>(threadIdx.x);
  const int parent_index = leaf_parent[leaf_index];
  if (thread_index == 0) {
    if (parent_index >= 0) {
      // if cur node is left child
      if (left_child[parent_index] == ~leaf_index) {
        left_child[parent_index] = new_node_index;
      } else {
        right_child[parent_index] = new_node_index;
      }
    }
    left_child[new_node_index] = ~leaf_index;
    right_child[new_node_index] = ~num_leaves;
    leaf_parent[leaf_index] = new_node_index;
    leaf_parent[num_leaves] = new_node_index;
  } else if (thread_index == 1) {
    // add new node
    split_feature_inner[new_node_index] = cuda_split_info->inner_feature_index;
  } else if (thread_index == 2) {
    split_feature[new_node_index] = batch_split.real_feature_index;
  } else if (thread_index == 3) {
    split_gain[new_node_index] = static_cast<float>(cuda_split_info->gain);
  } else if (thread_index == 4) {
    // save current leaf value to internal node before change
    internal_weight[new_node_index] = cuda_split_info->left_sum_hessians + cuda_split_info->right_sum_hessians;
    leaf_weight[leaf_index] = cuda_split_info->left_sum_hessians;
  } else if (thread_index == 5) {
    internal_value[new_node_index] = leaf_value[leaf_index];
    leaf_value[leaf_index] = isnan(cuda_split_info->left_value) ? 0.0f : cuda_split_info->left_value;
  } else if (thread_index == 6) {
    internal_count[new_node_index] = cuda_split_info->left_count + cuda_split_info->right_count;
  } else if (thread_index == 7) {
    leaf_count[leaf_index] = cuda_split_info->left_count;
  } else if (thread_index == 8) {
    leaf_value[num_leaves] = isnan(cuda_split_info->right_value) ? 0.0f : cuda_split_info->right_value;
  } else if (thread_index == 9) {
    leaf_weight[num_leaves] = cuda_split_info->right_sum_hessians;
  } else if (thread_index == 10) {
    leaf_count[num_leaves] = cuda_split_info->right_count;
  } else if (thread_index == 11) {
    // update leaf depth
    leaf_depth[num_leaves] = leaf_depth[leaf_index] + 1;
    leaf_depth[leaf_index]++;
  } else if (thread_index == 12) {
    decision_type[new_node_index] = 0;
    SetDecisionTypeCUDA(&decision_type[new_node_index], false, kCategoricalMask);
    SetDecisionTypeCUDA(&decision_type[new_node_index], cuda_split_info->default_left, kDefaultLeftMask);
    SetMissingTypeCUDA(&decision_type[new_node_index], static_cast<int8_t>(batch_split.missing_type));
  } else if (thread_index == 13) {
    threshold_in_bin[new_node_index] = cuda_split_info->threshold;
  } else if (thread_index == 14) {
    threshold[new_node_index] = batch_split.real_threshold;
  }
}

void CUDATree::CaptureHybridGraphSplitBatchKernel(cudaStream_t stream,
                                                  const int* graph_live_split_count) {
  // graphs L1 body capture: placeholder grid (the device controller resizes it
  // per level), descriptors from the pooled batch-split buffer the controller
  // writes; parameter set identical to LaunchSplitBatchKernel plus the live
  // split count the pow2-frozen grid's idle blocks guard on (graphs A2)
  SplitBatchKernel<<<1, 32, 0, stream>>>(
    cuda_batch_splits_.RawDataReadOnly(),
    cuda_leaf_parent_.RawData(),
    cuda_leaf_depth_.RawData(),
    cuda_left_child_.RawData(),
    cuda_right_child_.RawData(),
    cuda_split_feature_inner_.RawData(),
    cuda_split_feature_.RawData(),
    cuda_split_gain_.RawData(),
    cuda_internal_weight_.RawData(),
    cuda_internal_value_.RawData(),
    cuda_internal_count_.RawData(),
    cuda_leaf_weight_.RawData(),
    cuda_leaf_value_.RawData(),
    cuda_leaf_count_.RawData(),
    cuda_decision_type_.RawData(),
    cuda_threshold_in_bin_.RawData(),
    cuda_threshold_.RawData(),
    graph_live_split_count);
}

void CUDATree::LaunchSplitBatchKernel(const int num_splits) {
  SplitBatchKernel<<<num_splits, 32, 0, cuda_stream_>>>(
    cuda_batch_splits_.RawDataReadOnly(),
    cuda_leaf_parent_.RawData(),
    cuda_leaf_depth_.RawData(),
    cuda_left_child_.RawData(),
    cuda_right_child_.RawData(),
    cuda_split_feature_inner_.RawData(),
    cuda_split_feature_.RawData(),
    cuda_split_gain_.RawData(),
    cuda_internal_weight_.RawData(),
    cuda_internal_value_.RawData(),
    cuda_internal_count_.RawData(),
    cuda_leaf_weight_.RawData(),
    cuda_leaf_value_.RawData(),
    cuda_leaf_count_.RawData(),
    cuda_decision_type_.RawData(),
    cuda_threshold_in_bin_.RawData(),
    cuda_threshold_.RawData(),
    nullptr);
}

__global__ void SplitCategoricalKernel(  // split information
  const int leaf_index,
  const int real_feature_index,
  const MissingType missing_type,
  const CUDASplitInfo* cuda_split_info,
  // tree structure
  const int num_leaves,
  int* leaf_parent,
  int* leaf_depth,
  int* left_child,
  int* right_child,
  int* split_feature_inner,
  int* split_feature,
  float* split_gain,
  double* internal_weight,
  double* internal_value,
  data_size_t* internal_count,
  double* leaf_weight,
  double* leaf_value,
  data_size_t* leaf_count,
  int8_t* decision_type,
  uint32_t* threshold_in_bin,
  double* threshold,
  size_t cuda_bitset_len,
  size_t cuda_bitset_inner_len,
  int num_cat,
  int* cuda_cat_boundaries,
  int* cuda_cat_boundaries_inner) {
  const int new_node_index = num_leaves - 1;
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int parent_index = leaf_parent[leaf_index];
  if (thread_index == 0) {
    if (parent_index >= 0) {
      // if cur node is left child
      if (left_child[parent_index] == ~leaf_index) {
        left_child[parent_index] = new_node_index;
      } else {
        right_child[parent_index] = new_node_index;
      }
    }
    left_child[new_node_index] = ~leaf_index;
    right_child[new_node_index] = ~num_leaves;
    leaf_parent[leaf_index] = new_node_index;
    leaf_parent[num_leaves] = new_node_index;
  } else if (thread_index == 1) {
    // add new node
    split_feature_inner[new_node_index] = cuda_split_info->inner_feature_index;
  } else if (thread_index == 2) {
    split_feature[new_node_index] = real_feature_index;
  } else if (thread_index == 3) {
    split_gain[new_node_index] = static_cast<float>(cuda_split_info->gain);
  } else if (thread_index == 4) {
    // save current leaf value to internal node before change
    internal_weight[new_node_index] = cuda_split_info->left_sum_hessians + cuda_split_info->right_sum_hessians;
    leaf_weight[leaf_index] = cuda_split_info->left_sum_hessians;
  } else if (thread_index == 5) {
    internal_value[new_node_index] = leaf_value[leaf_index];
    leaf_value[leaf_index] = isnan(cuda_split_info->left_value) ? 0.0f : cuda_split_info->left_value;
  } else if (thread_index == 6) {
    internal_count[new_node_index] = cuda_split_info->left_count + cuda_split_info->right_count;
  } else if (thread_index == 7) {
    leaf_count[leaf_index] = cuda_split_info->left_count;
  } else if (thread_index == 8) {
    leaf_value[num_leaves] = isnan(cuda_split_info->right_value) ? 0.0f : cuda_split_info->right_value;
  } else if (thread_index == 9) {
    leaf_weight[num_leaves] = cuda_split_info->right_sum_hessians;
  } else if (thread_index == 10) {
    leaf_count[num_leaves] = cuda_split_info->right_count;
  } else if (thread_index == 11) {
    // update leaf depth
    leaf_depth[num_leaves] = leaf_depth[leaf_index] + 1;
    leaf_depth[leaf_index]++;
  } else if (thread_index == 12) {
    decision_type[new_node_index] = 0;
    SetDecisionTypeCUDA(&decision_type[new_node_index], true, kCategoricalMask);
    SetMissingTypeCUDA(&decision_type[new_node_index], static_cast<int8_t>(missing_type));
  } else if (thread_index == 13) {
    threshold_in_bin[new_node_index] = num_cat;
  } else if (thread_index == 14) {
    threshold[new_node_index] = num_cat;
  } else if (thread_index == 15) {
    if (num_cat == 0) {
      cuda_cat_boundaries[num_cat] = 0;
    }
    cuda_cat_boundaries[num_cat + 1] = cuda_cat_boundaries[num_cat] + cuda_bitset_len;
  } else if (thread_index == 16) {
    if (num_cat == 0) {
      cuda_cat_boundaries_inner[num_cat] = 0;
    }
    cuda_cat_boundaries_inner[num_cat + 1] = cuda_cat_boundaries_inner[num_cat] + cuda_bitset_inner_len;
  }
}

void CUDATree::LaunchSplitCategoricalKernel(const int leaf_index,
  const int real_feature_index,
  const MissingType missing_type,
  const CUDASplitInfo* cuda_split_info,
  size_t cuda_bitset_len,
  size_t cuda_bitset_inner_len) {
  SplitCategoricalKernel<<<3, 6, 0, cuda_stream_>>>(
    // split information
    leaf_index,
    real_feature_index,
    missing_type,
    cuda_split_info,
    // tree structure
    num_leaves_,
    cuda_leaf_parent_.RawData(),
    cuda_leaf_depth_.RawData(),
    cuda_left_child_.RawData(),
    cuda_right_child_.RawData(),
    cuda_split_feature_inner_.RawData(),
    cuda_split_feature_.RawData(),
    cuda_split_gain_.RawData(),
    cuda_internal_weight_.RawData(),
    cuda_internal_value_.RawData(),
    cuda_internal_count_.RawData(),
    cuda_leaf_weight_.RawData(),
    cuda_leaf_value_.RawData(),
    cuda_leaf_count_.RawData(),
    cuda_decision_type_.RawData(),
    cuda_threshold_in_bin_.RawData(),
    cuda_threshold_.RawData(),
    cuda_bitset_len,
    cuda_bitset_inner_len,
    num_cat_,
    cuda_cat_boundaries_.RawData(),
    cuda_cat_boundaries_inner_.RawData());
}

__global__ void ShrinkageKernel(const double rate, double* cuda_leaf_value, const int num_leaves) {
  const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (leaf_index < num_leaves) {
    cuda_leaf_value[leaf_index] *= rate;
  }
}

void CUDATree::LaunchShrinkageKernel(const double rate) {
  const int num_threads_per_block = 1024;
  const int num_blocks = (num_leaves_ + num_threads_per_block - 1) / num_threads_per_block;
  ShrinkageKernel<<<num_blocks, num_threads_per_block>>>(rate, cuda_leaf_value_.RawData(), num_leaves_);
  if (leaf_value_dim_ > 1 && cuda_leaf_values_vec_.Size() > 0) {
    const int num_values = num_leaves_ * leaf_value_dim_;
    const int num_vec_blocks = (num_values + num_threads_per_block - 1) / num_threads_per_block;
    ShrinkageKernel<<<num_vec_blocks, num_threads_per_block>>>(rate, cuda_leaf_values_vec_.RawData(), num_values);
  }
}

__global__ void AddBiasKernel(const double val, double* cuda_leaf_value, const int num_leaves) {
  const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (leaf_index < num_leaves) {
    cuda_leaf_value[leaf_index] += val;
  }
}

void CUDATree::LaunchAddBiasKernel(const double val) {
  const int num_threads_per_block = 1024;
  const int num_blocks = (num_leaves_ + num_threads_per_block - 1) / num_threads_per_block;
  AddBiasKernel<<<num_blocks, num_threads_per_block>>>(val, cuda_leaf_value_.RawData(), num_leaves_);
  if (leaf_value_dim_ > 1 && cuda_leaf_values_vec_.Size() > 0) {
    const int num_values = num_leaves_ * leaf_value_dim_;
    const int num_vec_blocks = (num_values + num_threads_per_block - 1) / num_threads_per_block;
    AddBiasKernel<<<num_vec_blocks, num_threads_per_block>>>(val, cuda_leaf_values_vec_.RawData(), num_values);
  }
}

__global__ void SetVectorLeafValuesFromSplitKernel(
  const int left_leaf_index, const int right_leaf_index, const int leaf_value_dim,
  const CUDASplitInfo* cuda_split_info, double* cuda_leaf_values_vec) {
  const int target = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (target < leaf_value_dim) {
    const double* payload = cuda_split_info->vec_payload;
    const double left_value = payload[kVecLeftValue * leaf_value_dim + target];
    const double right_value = payload[kVecRightValue * leaf_value_dim + target];
    cuda_leaf_values_vec[static_cast<size_t>(left_leaf_index) * leaf_value_dim + target] =
      isnan(left_value) ? 0.0 : left_value;
    cuda_leaf_values_vec[static_cast<size_t>(right_leaf_index) * leaf_value_dim + target] =
      isnan(right_value) ? 0.0 : right_value;
  }
}

// Batched form: blockIdx.y selects the split of the level's batch (its left and
// right leaf indices are the descriptor's leaf_index / num_leaves_at_split,
// exactly what SplitBatchKernel uses).
__global__ void SetVectorLeafValuesFromSplitBatchKernel(
  const CUDATreeBatchSplit* splits, const int num_splits, const int leaf_value_dim,
  double* cuda_leaf_values_vec) {
  const int split_index = static_cast<int>(blockIdx.y);
  if (split_index >= num_splits) {
    return;
  }
  const int target = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (target < leaf_value_dim) {
    const CUDATreeBatchSplit& split = splits[split_index];
    const double* payload = split.split_info->vec_payload;
    const double left_value = payload[kVecLeftValue * leaf_value_dim + target];
    const double right_value = payload[kVecRightValue * leaf_value_dim + target];
    cuda_leaf_values_vec[static_cast<size_t>(split.leaf_index) * leaf_value_dim + target] =
      isnan(left_value) ? 0.0 : left_value;
    cuda_leaf_values_vec[static_cast<size_t>(split.num_leaves_at_split) * leaf_value_dim + target] =
      isnan(right_value) ? 0.0 : right_value;
  }
}

void CUDATree::LaunchSetVectorLeafValuesFromSplitBatchKernel(const int num_splits) {
  const int num_threads_per_block = 32;
  const dim3 grid_dim((leaf_value_dim_ + num_threads_per_block - 1) / num_threads_per_block,
                      num_splits);
  SetVectorLeafValuesFromSplitBatchKernel<<<grid_dim, num_threads_per_block, 0, cuda_stream_>>>(
    cuda_batch_splits_.RawDataReadOnly(), num_splits, leaf_value_dim_,
    cuda_leaf_values_vec_.RawData());
}

void CUDATree::LaunchSetVectorLeafValuesFromSplitKernel(const int left_leaf_index,
  const int right_leaf_index, const CUDASplitInfo* cuda_split_info) {
  const int num_threads_per_block = 32;
  const int num_blocks = (leaf_value_dim_ + num_threads_per_block - 1) / num_threads_per_block;
  SetVectorLeafValuesFromSplitKernel<<<num_blocks, num_threads_per_block, 0, cuda_stream_>>>(
    left_leaf_index, right_leaf_index, leaf_value_dim_, cuda_split_info,
    cuda_leaf_values_vec_.RawData());
}

template <bool USE_INDICES, bool USE_PACKED>
__global__ void AddPredictionToScoreKernel(
  // dataset information
  const data_size_t num_data,
  uint8_t* const* cuda_data_by_column,
  const uint8_t* cuda_column_bit_type,
  const uint32_t* cuda_feature_min_bin,
  const uint32_t* cuda_feature_max_bin,
  const uint32_t* cuda_feature_offset,
  const uint32_t* cuda_feature_default_bin,
  const uint32_t* cuda_feature_most_freq_bin,
  const int* cuda_feature_to_column,
  // per-tree packed view (USE_PACKED only): per-column base pointer, per-row
  // byte stride and nibble shift into the row matrix; sparse-encoded columns
  // carry their own buffer at their real width (bit type 8/16/32)
  const uint8_t* const* cuda_packed_column_ptr,
  const int* cuda_packed_column_stride,
  const uint8_t* cuda_packed_column_shift,
  const uint8_t* cuda_packed_column_bit_type,
  const data_size_t* cuda_used_indices,
  // tree information
  const uint32_t* cuda_threshold_in_bin,
  const int8_t* cuda_decision_type,
  const int* cuda_split_feature_inner,
  const int* cuda_left_child,
  const int* cuda_right_child,
  const double* cuda_leaf_value,
  // vector-leaf outputs (nullptr in scalar mode): one traversal feeds every
  // target's score plane (plane stride == the dataset's total row count)
  const double* cuda_leaf_values_vec,
  const int leaf_value_dim,
  const data_size_t score_plane_stride,
  const uint32_t* cuda_bitset_inner,
  const int* cuda_cat_boundaries_inner,
  // output
  double* score) {
  const data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (inner_data_index < num_data) {
    const data_size_t data_index = USE_INDICES ? cuda_used_indices[inner_data_index] : inner_data_index;
    int node = 0;
    while (node >= 0) {
      const int split_feature_inner = cuda_split_feature_inner[node];
      const int column = cuda_feature_to_column[split_feature_inner];
      const uint32_t default_bin = cuda_feature_default_bin[split_feature_inner];
      const uint32_t most_freq_bin = cuda_feature_most_freq_bin[split_feature_inner];
      const uint32_t max_bin = cuda_feature_max_bin[split_feature_inner];
      const uint32_t min_bin = cuda_feature_min_bin[split_feature_inner];
      const uint32_t offset = cuda_feature_offset[split_feature_inner];
      const uint8_t column_bit_type = cuda_column_bit_type[column];
      // min_bin/max_bin index the RAW column, which may bundle several features;
      // default_bin, most_freq_bin and threshold_in_bin are all feature-local.
      // The row's bin is converted to feature-local below, so the NaN test needs
      // the feature-local max bin too -- comparing a converted bin against the
      // raw max_bin never matches, which routed every NaN row by value instead
      // of down the split's default direction.
      const uint32_t max_bin_local = max_bin - min_bin + offset;
      uint32_t bin = 0;
      if (USE_PACKED) {
        // the values match the classic gathered view byte for byte: the
        // gather kernel only copies these nibbles/bytes, so the bin logic
        // below is untouched.
        // Widths here are 4 (row-matrix nibble at a fixed shift, stride apart)
        // or a sparse column's own 8/16/32 -- never kNibbleColumnBitType, which
        // describes a dense column's own flat two-rows-per-byte buffer and is
        // read by the non-packed branch below.
        const uint8_t* base = cuda_packed_column_ptr[column];
        const uint8_t packed_bit_type = cuda_packed_column_bit_type[column];
        if (packed_bit_type == 4) {
          const size_t byte_index = static_cast<size_t>(data_index) *
            static_cast<size_t>(cuda_packed_column_stride[column]);
          bin = static_cast<uint32_t>((base[byte_index] >> cuda_packed_column_shift[column]) & 0xf);
        } else if (packed_bit_type == 8) {
          bin = static_cast<uint32_t>(base[data_index]);
        } else if (packed_bit_type == 16) {
          bin = static_cast<uint32_t>((reinterpret_cast<const uint16_t*>(base))[data_index]);
        } else {
          bin = static_cast<uint32_t>((reinterpret_cast<const uint32_t*>(base))[data_index]);
        }
      } else if (column_bit_type == 8) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint8_t*>(cuda_data_by_column[column]))[data_index]);
      } else if (column_bit_type == kNibbleColumnBitType) {
        // two rows per byte; see kNibbleColumnBitType
        const uint8_t packed =
          (reinterpret_cast<const uint8_t*>(cuda_data_by_column[column]))[data_index >> 1];
        bin = static_cast<uint32_t>((packed >> ((data_index & 1) << 2)) & 0xf);
      } else if (column_bit_type == 16) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint16_t*>(cuda_data_by_column[column]))[data_index]);
      } else if (column_bit_type == 32) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint32_t*>(cuda_data_by_column[column]))[data_index]);
      }
      if (bin >= min_bin && bin <= max_bin) {
        bin = bin - min_bin + offset;
      } else {
        bin = most_freq_bin;
      }
      const int8_t decision_type = cuda_decision_type[node];
      if (GetDecisionTypeCUDA(decision_type, kCategoricalMask)) {
        int cat_idx = static_cast<int>(cuda_threshold_in_bin[node]);
        if (FindInBitsetCUDA(cuda_bitset_inner + cuda_cat_boundaries_inner[cat_idx],
                             cuda_cat_boundaries_inner[cat_idx + 1] - cuda_cat_boundaries_inner[cat_idx], bin)) {
          node = cuda_left_child[node];
        } else {
          node = cuda_right_child[node];
        }
      } else {
        const uint32_t threshold_in_bin = cuda_threshold_in_bin[node];
        const int8_t missing_type = GetMissingTypeCUDA(decision_type);
        const bool default_left = ((decision_type & kDefaultLeftMask) > 0);
        if ((missing_type == 1 && bin == default_bin) || (missing_type == 2 && bin == max_bin_local)) {
          if (default_left) {
            node = cuda_left_child[node];
          } else {
            node = cuda_right_child[node];
          }
        } else {
          if (bin <= threshold_in_bin) {
            node = cuda_left_child[node];
          } else {
            node = cuda_right_child[node];
          }
        }
      }
    }
    if (cuda_leaf_values_vec == nullptr) {
      score[data_index] += cuda_leaf_value[~node];
    } else {
      const int leaf = ~node;
      for (int target = 0; target < leaf_value_dim; ++target) {
        score[static_cast<size_t>(target) * score_plane_stride + data_index] +=
          cuda_leaf_values_vec[static_cast<size_t>(leaf) * leaf_value_dim + target];
      }
    }
  }
}

void CUDATree::LaunchAddPredictionToScoreKernel(
  const Dataset* data,
  const data_size_t* used_data_indices,
  data_size_t num_data,
  double* score) const {
  const CUDAColumnData* cuda_column_data = data->cuda_column_data();
  // A launch with used_data_indices on the TRAIN data is the out-of-bag score
  // update of the tree trained this very iteration (nothing else scores a row
  // subset), so a live packed per-tree view is that tree's own view and the
  // packed kernel variant traverses it directly — no classic per-column
  // gather. Every other caller scores against a plain per-column view.
  const bool packed_serves_oob =
    cuda_column_data->packed_column_view_active() && used_data_indices != nullptr;
  if (packed_serves_oob) {
    const_cast<CUDAColumnData*>(cuda_column_data)->EnsurePackedViewOnDevice();
  } else {
  // This kernel may traverse ANY tree, including one grown many iterations ago
  // (DART re-scores dropped trees). Under feature_fraction the tree learner
  // publishes a per-tree compact column view: columns the current tree did not
  // sample are null, and the sampled ones point into a scratch buffer the next
  // tree overwrites. Traversing an older tree against that view dereferences a
  // null column ("an illegal memory access was encountered") or, when the
  // pointer happens to be live, silently reads another tree's bins. Put the
  // original per-column table back; the learner reinstalls its view per tree.
  // A tree still holding live device arrays is the one being grown right now,
  // so the view in force is the one it was grown under and serves it as-is.
  const bool host_structure_authoritative = cuda_split_feature_inner_.Size() == 0;
  bool view_serves_this_tree = cuda_column_data->original_column_view_active() ||
                               (!cuda_column_data->packed_column_view_active() &&
                                !host_structure_authoritative);
  if (!view_serves_this_tree && host_structure_authoritative &&
      !cuda_column_data->packed_column_view_active() &&
      static_cast<int>(split_feature_inner_.size()) >= num_leaves_ - 1) {
    view_serves_this_tree = true;
    for (int node = 0; node < num_leaves_ - 1; ++node) {
      if (!cuda_column_data->ColumnAvailableInCurrentView(
              cuda_column_data->feature_to_column(split_feature_inner_[node]))) {
        view_serves_this_tree = false;
        break;
      }
    }
  }
  if (!view_serves_this_tree) {
    if (!cuda_column_data->has_original_column_view()) {
      // Reaching here means an OLDER tree — DART re-scoring a dropped tree
      // whose sampled columns are no longer materialized (the freshly trained
      // tree's out-of-bag pass is served by the packed branch above).
      Log::Fatal("Scoring a previously trained tree on CUDA needs the full "
                 "per-column data, which was skipped because it would not fit "
                 "in GPU memory. This configuration (e.g. DART on a very wide "
                 "dataset) needs fewer features or device_type=cpu.");
    }
    const_cast<CUDAColumnData*>(cuda_column_data)->RestoreOriginalColumnView();
  }
  }
  const int num_blocks = (num_data + num_threads_per_block_add_prediction_to_score_ - 1) / num_threads_per_block_add_prediction_to_score_;
  // ToHost() frees the per-tree GPU tree-structure arrays to bound device memory
  // across many boosting rounds, keeping only cuda_leaf_value_. This kernel,
  // however, traverses the *whole* tree (split_feature_inner / children /
  // thresholds / decision_type), so when it runs post-ToHost -- which happens in
  // GBDT::UpdateScore's out-of-bag pass under bagging -- those arrays are empty
  // and the kernel dereferences freed/null device pointers, crashing with
  // "an illegal memory access was encountered". Re-upload the structure from the
  // (still-populated, shrunk-to-n_internal) host arrays for the duration of this
  // launch, then free it again so the memory optimization is preserved. Stumps
  // (num_leaves_ <= 1) have no internal nodes and never enter the traversal.
  const bool restore_tree_structure =
    num_leaves_ > 1 && cuda_split_feature_inner_.Size() == 0;
  // Same treatment for the categorical arrays, on their own trigger: the
  // selective flow builds trees on host and leaves the device copies empty,
  // and a categorical decision would dereference the null bases. The classic
  // flow keeps live device copies, so only restore what is actually missing.
  const bool restore_cat_arrays =
    num_cat_ > 0 && cuda_bitset_inner_.Size() == 0;
  if (restore_tree_structure) {
    CUDATree* self = const_cast<CUDATree*>(this);
    self->cuda_left_child_.InitFromHostVector(left_child_);
    self->cuda_right_child_.InitFromHostVector(right_child_);
    self->cuda_split_feature_inner_.InitFromHostVector(split_feature_inner_);
    self->cuda_threshold_in_bin_.InitFromHostVector(threshold_in_bin_);
    self->cuda_decision_type_.InitFromHostVector(decision_type_);
  }
  if (restore_cat_arrays) {
    CUDATree* self = const_cast<CUDATree*>(this);
    self->cuda_bitset_inner_.InitFromHostVector(cat_threshold_inner_);
    self->cuda_cat_boundaries_inner_.InitFromHostVector(cat_boundaries_inner_);
  }
  // vector-leaf outputs (retained through ToHost) and the score plane stride:
  // score planes are laid out [target][row] over the DATASET's full row count
  const double* score_leaf_values_vec =
    leaf_value_dim_ > 1 ? cuda_leaf_values_vec_.RawData() : nullptr;
  const data_size_t score_plane_stride = data->num_data();
  if (packed_serves_oob) {
    AddPredictionToScoreKernel<true, true><<<num_blocks, num_threads_per_block_add_prediction_to_score_>>>(
      // dataset information
      num_data,
      cuda_column_data->cuda_data_by_column(),
      cuda_column_data->cuda_column_bit_type(),
      cuda_column_data->cuda_feature_min_bin(),
      cuda_column_data->cuda_feature_max_bin(),
      cuda_column_data->cuda_feature_offset(),
      cuda_column_data->cuda_feature_default_bin(),
      cuda_column_data->cuda_feature_most_freq_bin(),
      cuda_column_data->cuda_feature_to_column(),
      cuda_column_data->cuda_packed_column_ptr(),
      cuda_column_data->cuda_packed_column_stride(),
      cuda_column_data->cuda_packed_column_shift(),
      cuda_column_data->cuda_packed_column_bit_type(),
      used_data_indices,
      // tree information
      cuda_threshold_in_bin_.RawData(),
      cuda_decision_type_.RawData(),
      cuda_split_feature_inner_.RawData(),
      cuda_left_child_.RawData(),
      cuda_right_child_.RawData(),
      cuda_leaf_value_.RawData(),
      score_leaf_values_vec,
      leaf_value_dim_,
      score_plane_stride,
      cuda_bitset_inner_.RawDataReadOnly(),
      cuda_cat_boundaries_inner_.RawDataReadOnly(),
      // output
      score);
  } else if (used_data_indices == nullptr) {
    AddPredictionToScoreKernel<false, false><<<num_blocks, num_threads_per_block_add_prediction_to_score_>>>(
      // dataset information
      num_data,
      cuda_column_data->cuda_data_by_column(),
      cuda_column_data->cuda_column_bit_type(),
      cuda_column_data->cuda_feature_min_bin(),
      cuda_column_data->cuda_feature_max_bin(),
      cuda_column_data->cuda_feature_offset(),
      cuda_column_data->cuda_feature_default_bin(),
      cuda_column_data->cuda_feature_most_freq_bin(),
      cuda_column_data->cuda_feature_to_column(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      // tree information
      cuda_threshold_in_bin_.RawData(),
      cuda_decision_type_.RawData(),
      cuda_split_feature_inner_.RawData(),
      cuda_left_child_.RawData(),
      cuda_right_child_.RawData(),
      cuda_leaf_value_.RawData(),
      score_leaf_values_vec,
      leaf_value_dim_,
      score_plane_stride,
      cuda_bitset_inner_.RawDataReadOnly(),
      cuda_cat_boundaries_inner_.RawDataReadOnly(),
      // output
      score);
  } else {
    AddPredictionToScoreKernel<true, false><<<num_blocks, num_threads_per_block_add_prediction_to_score_>>>(
      // dataset information
      num_data,
      cuda_column_data->cuda_data_by_column(),
      cuda_column_data->cuda_column_bit_type(),
      cuda_column_data->cuda_feature_min_bin(),
      cuda_column_data->cuda_feature_max_bin(),
      cuda_column_data->cuda_feature_offset(),
      cuda_column_data->cuda_feature_default_bin(),
      cuda_column_data->cuda_feature_most_freq_bin(),
      cuda_column_data->cuda_feature_to_column(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      used_data_indices,
      // tree information
      cuda_threshold_in_bin_.RawData(),
      cuda_decision_type_.RawData(),
      cuda_split_feature_inner_.RawData(),
      cuda_left_child_.RawData(),
      cuda_right_child_.RawData(),
      cuda_leaf_value_.RawData(),
      score_leaf_values_vec,
      leaf_value_dim_,
      score_plane_stride,
      cuda_bitset_inner_.RawDataReadOnly(),
      cuda_cat_boundaries_inner_.RawDataReadOnly(),
      // output
      score);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (restore_tree_structure) {
    CUDATree* self = const_cast<CUDATree*>(this);
    self->cuda_left_child_.Clear();
    self->cuda_right_child_.Clear();
    self->cuda_split_feature_inner_.Clear();
    self->cuda_threshold_in_bin_.Clear();
    self->cuda_decision_type_.Clear();
  }
  if (restore_cat_arrays) {
    // free only what THIS launch uploaded
    CUDATree* self = const_cast<CUDATree*>(this);
    self->cuda_bitset_inner_.Clear();
    self->cuda_cat_boundaries_inner_.Clear();
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
