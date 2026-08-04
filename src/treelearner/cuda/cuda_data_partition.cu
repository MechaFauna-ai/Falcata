/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_data_partition.hpp"

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_rocm_interop.h>
#include <Falcata/tree.h>

#include <algorithm>
#include <vector>

namespace Falcata {

// Number of thread blocks assigned to each leaf in RenewDiscretizedTreeLeavesKernel.
// Shallow trees have few leaves; with 1 block/leaf the reduction launches too few
// blocks to fill the GPU. 16 blocks/leaf saturates a large SM count while keeping
// per-leaf atomic contention bounded.
#define RENEW_BLOCKS_PER_LEAF (16)

__global__ void FillDataIndicesBeforeTrainKernel(const data_size_t num_data,
  data_size_t* data_indices, int* cuda_data_index_to_leaf_index) {
  const unsigned int data_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (data_index < num_data) {
    data_indices[data_index] = data_index;
    cuda_data_index_to_leaf_index[data_index] = 0;
  }
}

__global__ void FillDataIndexToLeafIndexKernel(
  const data_size_t num_data,
  const data_size_t* data_indices,
  int* data_index_to_leaf_index) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    data_index_to_leaf_index[data_indices[data_index]] = 0;
  }
}

void CUDADataPartition::LaunchFillDataIndicesBeforeTrain() {
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  FillDataIndicesBeforeTrainKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(num_data_in_root, cuda_data_indices_.RawData(), cuda_data_index_to_leaf_index_.RawData());
}

void CUDADataPartition::LaunchFillDataIndexToLeafIndex() {
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  FillDataIndexToLeafIndexKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(num_data_in_root, cuda_data_indices_.RawData(), cuda_data_index_to_leaf_index_.RawData());
}

__device__ __forceinline__ void PrepareOffsetAt(const unsigned int block_x,
  const data_size_t num_data_in_leaf, uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const uint16_t thread_to_left_offset_cnt, uint16_t* shared_mem_buffer) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const uint16_t thread_to_left_offset = ShufflePrefixSum<uint16_t>(thread_to_left_offset_cnt, shared_mem_buffer);
  const data_size_t num_data_in_block = (block_x + 1) * blockDim_x <= num_data_in_leaf ? static_cast<data_size_t>(blockDim_x) :
    num_data_in_leaf - static_cast<data_size_t>(block_x * blockDim_x);
  if (static_cast<data_size_t>(threadIdx_x) < num_data_in_block) {
    block_to_left_offset[threadIdx_x] = thread_to_left_offset;
  }
  if (threadIdx_x == blockDim_x - 1) {
    if (num_data_in_block > 0) {
      const data_size_t data_to_left = static_cast<data_size_t>(thread_to_left_offset);
      block_to_left_offset_buffer[block_x + 1] = data_to_left;
      block_to_right_offset_buffer[block_x + 1] = num_data_in_block - data_to_left;
    } else {
      block_to_left_offset_buffer[block_x + 1] = 0;
      block_to_right_offset_buffer[block_x + 1] = 0;
    }
  }
}

__device__ __forceinline__ void PrepareOffset(const data_size_t num_data_in_leaf, uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const uint16_t thread_to_left_offset_cnt, uint16_t* shared_mem_buffer) {
  PrepareOffsetAt(blockIdx.x, num_data_in_leaf, block_to_left_offset,
    block_to_left_offset_buffer, block_to_right_offset_buffer,
    thread_to_left_offset_cnt, shared_mem_buffer);
}

template <typename T>
__device__ bool CUDAFindInBitset(const uint32_t* bits, int n, T pos) {
  int i1 = pos / 32;
  if (i1 >= n) {
    return false;
  }
  int i2 = pos % 32;
  return (bits[i1] >> i2) & 1;
}



#define UpdateDataIndexToLeafIndexKernel_PARAMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin, \
  const uint32_t min_bin, \
  const int left_leaf_index, \
  const int right_leaf_index, \
  const int default_leaf_index, \
  const int missing_default_leaf_index

#define UpdateDataIndexToLeafIndex_ARGS \
  column_data, \
  num_data_in_leaf, \
  data_indices_in_leaf, th, \
  t_zero_bin, \
  max_bin, \
  min_bin, \
  left_leaf_index, \
  right_leaf_index, \
  default_leaf_index, \
  missing_default_leaf_index

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, bool USE_MIN_BIN, typename BIN_TYPE>
__global__ void UpdateDataIndexToLeafIndexKernel(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  int* cuda_data_index_to_leaf_index) {
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (!MIN_IS_MAX) {
      if ((MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
        (MISSING_IS_NA && !MFB_IS_NA && bin == max_bin)) {
        cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
      } else if ((USE_MIN_BIN && (bin < min_bin || bin > max_bin)) ||
                 (!USE_MIN_BIN && bin == 0)) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
        }
      } else if (bin > th) {
        cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
      } else {
        cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
      }
    } else {
      if (MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
      } else if (bin != max_bin) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
        }
      } else {
        if (MISSING_IS_NA && !MFB_IS_NA) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          if (!MAX_TO_LEFT) {
            cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
          } else {
            cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
          }
        }
      }
    }
  }
}

template <typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool missing_is_zero,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (min_bin < max_bin) {
    if (!missing_is_zero) {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<false, false, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
    } else {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<false, true, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
    }
  } else {
    if (!missing_is_zero) {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<true, false, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
    } else {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<true, true, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
    }
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner0(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (!missing_is_na) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner1<MIN_IS_MAX, MISSING_IS_ZERO, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner1<MIN_IS_MAX, MISSING_IS_ZERO, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner1(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (!mfb_is_zero) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_na, max_to_left, is_single_feature_in_column);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_na, max_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner2(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool mfb_is_na,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (!mfb_is_na) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner3<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, max_to_left, is_single_feature_in_column);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner3<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, max_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner3(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (!max_to_left) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner4<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, is_single_feature_in_column);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner4<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner4(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool is_single_feature_in_column) {
  if (!is_single_feature_in_column) {
    UpdateDataIndexToLeafIndexKernel<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, true, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index_.RawData());
  } else {
    UpdateDataIndexToLeafIndexKernel<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, false, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index_.RawData());
  }
}

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

#define GenBitVector_ARGS \
  column_data, \
  num_data_in_leaf, \
  data_indices_in_leaf, \
  th, \
  t_zero_bin, \
  max_bin, \
  min_bin, \
  split_default_to_left,  \
  split_missing_default_to_left

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, bool USE_MIN_BIN, typename BIN_TYPE>
__global__ void GenDataToLeftBitVectorKernel(
  GenDataToLeftBitVectorKernel_PARAMS,
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer) {
  __shared__ uint16_t shared_mem_buffer[WARPSIZE];
  uint16_t thread_to_left_offset_cnt = 0;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (!MIN_IS_MAX) {
      if ((MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
        (MISSING_IS_NA && !MFB_IS_NA && bin == max_bin)) {
        thread_to_left_offset_cnt = split_missing_default_to_left;
      } else if ((USE_MIN_BIN && (bin < min_bin || bin > max_bin)) ||
                 (!USE_MIN_BIN && bin == 0)) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO || MFB_IS_ZERO)) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else {
          thread_to_left_offset_cnt = split_default_to_left;
        }
      } else if (bin <= th) {
        thread_to_left_offset_cnt = 1;
      }
    } else {
      if (MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        thread_to_left_offset_cnt = split_missing_default_to_left;
      } else if (bin != max_bin) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else {
          thread_to_left_offset_cnt = split_default_to_left;
        }
      } else {
        if (MISSING_IS_NA && !MFB_IS_NA) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else if (MAX_TO_LEFT) {
          thread_to_left_offset_cnt = 1;
        }
      }
    }
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, block_to_left_offset + blockIdx.x * blockDim.x, block_to_left_offset_buffer, block_to_right_offset_buffer,
    thread_to_left_offset_cnt, shared_mem_buffer);
}

template <typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool missing_is_zero,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (min_bin < max_bin) {
    if (!missing_is_zero) {
      LaunchGenDataToLeftBitVectorKernelInner0<false, false, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
    } else {
      LaunchGenDataToLeftBitVectorKernelInner0<false, true, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
    }
  } else {
    if (!missing_is_zero) {
      LaunchGenDataToLeftBitVectorKernelInner0<true, false, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
    } else {
      LaunchGenDataToLeftBitVectorKernelInner0<true, true, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
    }
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner0(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (!missing_is_na) {
    LaunchGenDataToLeftBitVectorKernelInner1<MIN_IS_MAX, MISSING_IS_ZERO, false, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner1<MIN_IS_MAX, MISSING_IS_ZERO, true, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner1(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (!mfb_is_zero) {
    LaunchGenDataToLeftBitVectorKernelInner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, false, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, true, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner2(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (!mfb_is_na) {
    LaunchGenDataToLeftBitVectorKernelInner3
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, false, BIN_TYPE>
      (GenBitVector_ARGS, max_bin_to_left, is_single_feature_in_column);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner3
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, true, BIN_TYPE>
      (GenBitVector_ARGS, max_bin_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner3(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (!max_bin_to_left) {
    LaunchGenDataToLeftBitVectorKernelInner4
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, false, BIN_TYPE>
      (GenBitVector_ARGS, is_single_feature_in_column);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner4
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, true, BIN_TYPE>
      (GenBitVector_ARGS, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner4(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool is_single_feature_in_column) {
  if (!is_single_feature_in_column) {
    GenDataToLeftBitVectorKernel
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, true, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_ARGS,
        cuda_block_to_left_offset_.RawData(), cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData());
  } else {
    GenDataToLeftBitVectorKernel
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, false, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_ARGS,
        cuda_block_to_left_offset_.RawData(), cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData());
  }
}

void CUDADataPartition::LaunchGenDataToLeftBitVectorKernel(
  const data_size_t num_data_in_leaf,
  const int split_feature_index,
  const uint32_t split_threshold,
  const uint8_t split_default_left,
  const data_size_t leaf_data_start,
  const int left_leaf_index,
  const int right_leaf_index) {
  const bool missing_is_zero = static_cast<bool>(cuda_column_data_->feature_missing_is_zero(split_feature_index));
  const bool missing_is_na = static_cast<bool>(cuda_column_data_->feature_missing_is_na(split_feature_index));
  const bool mfb_is_zero = static_cast<bool>(cuda_column_data_->feature_mfb_is_zero(split_feature_index));
  const bool mfb_is_na = static_cast<bool>(cuda_column_data_->feature_mfb_is_na(split_feature_index));
  const bool is_single_feature_in_column = is_single_feature_in_column_[split_feature_index];
  const uint32_t default_bin = cuda_column_data_->feature_default_bin(split_feature_index);
  const uint32_t most_freq_bin = cuda_column_data_->feature_most_freq_bin(split_feature_index);
  const uint32_t min_bin = is_single_feature_in_column ? 1 : cuda_column_data_->feature_min_bin(split_feature_index);
  const uint32_t max_bin = cuda_column_data_->feature_max_bin(split_feature_index);
  uint32_t th = split_threshold + min_bin;
  uint32_t t_zero_bin = min_bin + default_bin;
  if (most_freq_bin == 0) {
    --th;
    --t_zero_bin;
  }
  uint8_t split_default_to_left = 0;
  uint8_t split_missing_default_to_left = 0;
  int default_leaf_index = right_leaf_index;
  int missing_default_leaf_index = right_leaf_index;
  if (most_freq_bin <= split_threshold) {
    split_default_to_left = 1;
    default_leaf_index = left_leaf_index;
  }
  if (missing_is_zero || missing_is_na) {
    if (split_default_left) {
      split_missing_default_to_left = 1;
      missing_default_leaf_index = left_leaf_index;
    }
  }
  const int column_index = cuda_column_data_->feature_to_column(split_feature_index);
  const uint8_t bit_type = cuda_column_data_->column_bit_type(column_index);

  const bool max_bin_to_left = (max_bin <= th);

  const data_size_t* data_indices_in_leaf = cuda_data_indices_.RawData() + leaf_data_start;
  const void* column_data_pointer = cuda_column_data_->GetColumnData(column_index);

  if (bit_type == 8) {
    const uint8_t* column_data = reinterpret_cast<const uint8_t*>(column_data_pointer);
    LaunchGenDataToLeftBitVectorKernelInner<uint8_t>(
      GenBitVector_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
    LaunchUpdateDataIndexToLeafIndexKernel<uint8_t>(
      UpdateDataIndexToLeafIndex_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
  } else if (bit_type == 16) {
    const uint16_t* column_data = reinterpret_cast<const uint16_t*>(column_data_pointer);
    LaunchGenDataToLeftBitVectorKernelInner<uint16_t>(
      GenBitVector_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
    LaunchUpdateDataIndexToLeafIndexKernel<uint16_t>(
      UpdateDataIndexToLeafIndex_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
  } else if (bit_type == 32) {
    const uint32_t* column_data = reinterpret_cast<const uint32_t*>(column_data_pointer);
    LaunchGenDataToLeftBitVectorKernelInner<uint32_t>(
      GenBitVector_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
    LaunchUpdateDataIndexToLeafIndexKernel<uint32_t>(
      UpdateDataIndexToLeafIndex_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
  }
}

#undef UpdateDataIndexToLeafIndexKernel_PARAMS
#undef UpdateDataIndexToLeafIndex_ARGS
#undef GenDataToLeftBitVectorKernel_PARAMS
#undef GenBitVector_ARGS

template <typename BIN_TYPE, bool USE_MIN_BIN>
__global__ void UpdateDataIndexToLeafIndexKernel_Categorical(
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t* bitset, const int bitset_len, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t max_bin, const uint32_t min_bin, const int8_t mfb_offset,
  int* cuda_data_index_to_leaf_index, const int left_leaf_index, const int right_leaf_index,
  const int default_leaf_index) {
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (USE_MIN_BIN && (bin < min_bin || bin > max_bin)) {
      cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
    } else if (!USE_MIN_BIN && bin == 0) {
      cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
    } else if (CUDAFindInBitset(bitset, bitset_len, bin - min_bin + mfb_offset)) {
      cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
    } else {
      cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
    }
  }
}

// for categorical features
template <typename BIN_TYPE, bool USE_MIN_BIN>
__global__ void GenDataToLeftBitVectorKernel_Categorical(
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t* bitset, int bitset_len, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t max_bin, const uint32_t min_bin, const int8_t mfb_offset,
  const uint8_t split_default_to_left,
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer) {
  __shared__ uint16_t shared_mem_buffer[WARPSIZE];
  uint16_t thread_to_left_offset_cnt = 0;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (USE_MIN_BIN && (bin < min_bin || bin > max_bin)) {
      thread_to_left_offset_cnt = split_default_to_left;
    } else if (!USE_MIN_BIN && bin == 0) {
      thread_to_left_offset_cnt = split_default_to_left;
    } else if (CUDAFindInBitset(bitset, bitset_len, bin - min_bin + mfb_offset)) {
      thread_to_left_offset_cnt = 1;
    }
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, block_to_left_offset + blockIdx.x * blockDim.x, block_to_left_offset_buffer, block_to_right_offset_buffer,
    thread_to_left_offset_cnt, shared_mem_buffer);
}

#define GenBitVector_Categorical_ARGS \
  num_data_in_leaf, data_indices_in_leaf, \
  bitset, bitset_len, \
  column_data, max_bin, min_bin, mfb_offset, split_default_to_left, \
  cuda_block_to_left_offset_.RawData(), cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData()

#define UpdateDataIndexToLeafIndex_Categorical_ARGS \
  num_data_in_leaf, data_indices_in_leaf, \
  bitset, bitset_len, \
  column_data, max_bin, min_bin, mfb_offset,  \
  cuda_data_index_to_leaf_index_.RawData(), left_leaf_index, right_leaf_index, default_leaf_index

void CUDADataPartition::LaunchGenDataToLeftBitVectorCategoricalKernel(
  const data_size_t num_data_in_leaf,
  const int split_feature_index,
  const uint32_t* bitset,
  const int bitset_len,
  const uint8_t split_default_left,
  const data_size_t leaf_data_start,
  const int left_leaf_index,
  const int right_leaf_index) {
  const data_size_t* data_indices_in_leaf = cuda_data_indices_.RawData() + leaf_data_start;
  const int column_index = cuda_column_data_->feature_to_column(split_feature_index);
  const uint8_t bit_type = cuda_column_data_->column_bit_type(column_index);
  const bool is_single_feature_in_column = is_single_feature_in_column_[split_feature_index];
  const uint32_t min_bin = is_single_feature_in_column ? 1 : cuda_column_data_->feature_min_bin(split_feature_index);
  const uint32_t max_bin = cuda_column_data_->feature_max_bin(split_feature_index);
  const uint32_t most_freq_bin = cuda_column_data_->feature_most_freq_bin(split_feature_index);
  const uint32_t default_bin = cuda_column_data_->feature_default_bin(split_feature_index);
  const void* column_data_pointer = cuda_column_data_->GetColumnData(column_index);
  const int8_t mfb_offset = static_cast<int8_t>(most_freq_bin == 0);
  std::vector<uint32_t> host_bitset(bitset_len, 0);
  CopyFromCUDADeviceToHost<uint32_t>(host_bitset.data(), bitset, bitset_len, __FILE__, __LINE__);
  uint8_t split_default_to_left = 0;
  int default_leaf_index = right_leaf_index;
  if (most_freq_bin > 0 && Common::FindInBitset(host_bitset.data(), bitset_len, most_freq_bin)) {
    split_default_to_left = 1;
    default_leaf_index = left_leaf_index;
  }
  if (bit_type == 8) {
    const uint8_t* column_data = reinterpret_cast<const uint8_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint8_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint8_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint8_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint8_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  } else if (bit_type == 16) {
    const uint16_t* column_data = reinterpret_cast<const uint16_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint16_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint16_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint16_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint16_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  } else if (bit_type == 32) {
    const uint32_t* column_data = reinterpret_cast<const uint32_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint32_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint32_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint32_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint32_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  }
}

#undef GenBitVector_Categorical_ARGS
#undef UpdateDataIndexToLeafIndex_Categorical_ARGS

__global__ void AggregateBlockOffsetKernel0(
  const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t num_blocks) {
  __shared__ uint32_t shared_mem_buffer[WARPSIZE];
  __shared__ uint32_t to_left_total_count;
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index];
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int threadIdx_x = threadIdx.x;
  const data_size_t num_blocks_plus_1 = num_blocks + 1;
  const uint32_t num_blocks_per_thread = (num_blocks_plus_1 + blockDim_x - 1) / blockDim_x;
  const uint32_t remain = num_blocks_plus_1 - ((num_blocks_per_thread - 1) * blockDim_x);
  const uint32_t remain_offset = remain * num_blocks_per_thread;
  uint32_t thread_start_block_index = 0;
  uint32_t thread_end_block_index = 0;
  if (threadIdx_x < remain) {
    thread_start_block_index = threadIdx_x * num_blocks_per_thread;
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread, num_blocks_plus_1);
  } else {
    thread_start_block_index = remain_offset + (num_blocks_per_thread - 1) * (threadIdx_x - remain);
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread - 1, num_blocks_plus_1);
  }
  if (threadIdx.x == 0) {
    block_to_right_offset_buffer[0] = 0;
  }
  __syncthreads();
  for (uint32_t block_index = thread_start_block_index + 1; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += block_to_left_offset_buffer[block_index - 1];
    block_to_right_offset_buffer[block_index] += block_to_right_offset_buffer[block_index - 1];
  }
  __syncthreads();
  uint32_t block_to_left_offset = 0;
  uint32_t block_to_right_offset = 0;
  if (thread_start_block_index < thread_end_block_index && thread_start_block_index > 1) {
    block_to_left_offset = block_to_left_offset_buffer[thread_start_block_index - 1];
    block_to_right_offset = block_to_right_offset_buffer[thread_start_block_index - 1];
  }
  block_to_left_offset = ShufflePrefixSum<uint32_t>(block_to_left_offset, shared_mem_buffer);
  __syncthreads();
  block_to_right_offset = ShufflePrefixSum<uint32_t>(block_to_right_offset, shared_mem_buffer);
  if (threadIdx_x == blockDim_x - 1) {
    to_left_total_count = block_to_left_offset + block_to_left_offset_buffer[num_blocks];
  }
  __syncthreads();
  const uint32_t to_left_thread_block_offset = block_to_left_offset;
  const uint32_t to_right_thread_block_offset = block_to_right_offset + to_left_total_count;
  for (uint32_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += to_left_thread_block_offset;
    block_to_right_offset_buffer[block_index] += to_right_thread_block_offset;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_num_data[left_leaf_index] = static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = num_data_in_leaf - static_cast<data_size_t>(to_left_total_count);
  }
}

__global__ void AggregateBlockOffsetKernel1(
  const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t num_blocks) {
  __shared__ uint32_t shared_mem_buffer[WARPSIZE];
  __shared__ uint32_t to_left_total_count;
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index];
  const unsigned int threadIdx_x = threadIdx.x;
  uint32_t block_to_left_offset = 0;
  uint32_t block_to_right_offset = 0;
  if (threadIdx_x < static_cast<unsigned int>(num_blocks)) {
    block_to_left_offset = block_to_left_offset_buffer[threadIdx_x + 1];
    block_to_right_offset = block_to_right_offset_buffer[threadIdx_x + 1];
  }
  block_to_left_offset = ShufflePrefixSum<uint32_t>(block_to_left_offset, shared_mem_buffer);
  __syncthreads();
  block_to_right_offset = ShufflePrefixSum<uint32_t>(block_to_right_offset, shared_mem_buffer);
  if (threadIdx.x == blockDim.x - 1) {
    to_left_total_count = block_to_left_offset;
  }
  __syncthreads();
  if (threadIdx_x < static_cast<unsigned int>(num_blocks)) {
    block_to_left_offset_buffer[threadIdx_x + 1] = block_to_left_offset;
    block_to_right_offset_buffer[threadIdx_x + 1] = block_to_right_offset + to_left_total_count;
  }
  if (threadIdx_x == 0) {
    block_to_right_offset_buffer[0] = to_left_total_count;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_num_data[left_leaf_index] = static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = num_data_in_leaf - static_cast<data_size_t>(to_left_total_count);
  }
}

// NOTE: this template parameter selects the multi-GPU (NCCL) partition path at
// compile time. It is deliberately NOT named `USE_NCCL`: the build macro of that
// name (added by the USE_NCCL cmake option) would be preprocessor-substituted
// into the parameter, breaking compilation on NCCL-enabled builds.
template <bool USE_NCCL_REDUCE, bool USE_GRAD_DISCRETIZED>
__global__ void SplitTreeStructureKernel(const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t* cuda_data_indices_main,
  const bool point_structs_at_main,
  const CUDASplitInfo* best_split_info,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  const int num_total_bin,
  hist_t* cuda_hist, hist_t** cuda_hist_pool,
  double* cuda_leaf_output,
  int* cuda_split_info_buffer) {
  const unsigned int to_left_total_cnt = cuda_leaf_num_data[left_leaf_index];
  double* cuda_split_info_buffer_for_hessians = reinterpret_cast<double*>(cuda_split_info_buffer + 8);
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_index == 0) {
    cuda_leaf_output[left_leaf_index] = best_split_info->left_value;
  } else if (global_thread_index == 1) {
    cuda_leaf_output[right_leaf_index] = best_split_info->right_value;
  } else if (global_thread_index == 2) {
    cuda_split_info_buffer[0] = left_leaf_index;
  } else if (global_thread_index == 3) {
    cuda_split_info_buffer[1] = cuda_leaf_num_data[left_leaf_index];
  } else if (global_thread_index == 4) {
    cuda_split_info_buffer[2] = cuda_leaf_data_start[left_leaf_index];
  } else if (global_thread_index == 5) {
    cuda_split_info_buffer[3] = right_leaf_index;
  } else if (global_thread_index == 6) {
    cuda_split_info_buffer[4] = cuda_leaf_num_data[right_leaf_index];
  } else if (global_thread_index == 7) {
    cuda_split_info_buffer[5] = cuda_leaf_data_start[right_leaf_index];
  } else if (global_thread_index == 8) {
    cuda_split_info_buffer_for_hessians[0] = best_split_info->left_sum_hessians;
    cuda_split_info_buffer_for_hessians[2] = best_split_info->left_sum_gradients;
  } else if (global_thread_index == 9) {
    cuda_split_info_buffer_for_hessians[1] = best_split_info->right_sum_hessians;
    cuda_split_info_buffer_for_hessians[3] = best_split_info->right_sum_gradients;
  } else if (global_thread_index == 10 && USE_NCCL_REDUCE) {
    // Slots 16/17 are the GLOBAL child counts the host reads back into
    // global_num_data_in_leaf_. Nothing used to write them: the buffer is
    // reused across splits and never cleared, so every rank read stale
    // garbage -- independently. That desynchronised both the smaller/larger
    // choice below (ranks then all-reduce DIFFERENT leaves) and the
    // min_data_in_leaf validity gates (one rank stops growing while the other
    // continues), which is exactly how 2-GPU training deadlocked.
    // best_split_info's counts come from the ALL-REDUCED histogram, so they
    // are already global and identical on every rank.
    cuda_split_info_buffer[16] = best_split_info->left_count;
    cuda_split_info_buffer[17] = best_split_info->right_count;
  }

  // Read the counts from the split info directly rather than through the
  // buffer: the writes above land in another thread, and there is no barrier
  // between them and this read.
  bool left_is_smaller = USE_NCCL_REDUCE ?
    best_split_info->left_count < best_split_info->right_count :
    cuda_leaf_num_data[left_leaf_index] < cuda_leaf_num_data[right_leaf_index];

  if (left_is_smaller) {
    if (global_thread_index == 0) {
      hist_t* parent_hist_ptr = cuda_hist_pool[left_leaf_index];
      cuda_hist_pool[right_leaf_index] = parent_hist_ptr;
      // Histogram slots are allocated at a 2 * num_total_bin stride per leaf (see the
      // cuda_hist_ size num_total_bin * 2 * num_leaves and the right-is-smaller branch
      // below, which always uses 2 * right_leaf_index * num_total_bin). The discretized
      // path here previously used a 1x stride (right_leaf_index * num_total_bin), so a
      // left-is-smaller child could be handed a slot that collides with an existing
      // 2x-strided leaf -- its histogram then accumulated on top of that leaf's data,
      // producing phantom splits and exploded leaf outputs under use_quantized_grad.
      // Use the same 2x stride here for consistency.
      cuda_hist_pool[left_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 1) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
      if (USE_GRAD_DISCRETIZED) {
        // The discretized best-split finder uses the int64 packed gradient/hessian
        // sum as the leaf total; it must be refreshed on split or the child inherits
        // the parent's total and the finder scores phantom (parent-remainder) splits.
        smaller_leaf_splits->sum_of_gradients_hessians = best_split_info->left_sum_of_gradients_hessians;
      }
    } else if (global_thread_index == 2) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      smaller_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      smaller_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      smaller_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      smaller_leaf_splits->data_indices_in_leaf = point_structs_at_main ?
        cuda_data_indices_main + cuda_leaf_data_start[left_leaf_index] : cuda_data_indices;
    } else if (global_thread_index == 7) {
      larger_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      larger_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
      if (USE_GRAD_DISCRETIZED) {
        larger_leaf_splits->sum_of_gradients_hessians = best_split_info->right_sum_of_gradients_hessians;
      }
    } else if (global_thread_index == 9) {
      larger_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      larger_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      larger_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      larger_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      larger_leaf_splits->data_indices_in_leaf = point_structs_at_main ?
        cuda_data_indices_main + cuda_leaf_data_start[right_leaf_index] : cuda_data_indices + cuda_leaf_num_data[left_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_split_info_buffer[6] = left_leaf_index;
    } else if (global_thread_index == 15) {
      cuda_split_info_buffer[7] = right_leaf_index;
    } else if (global_thread_index == 16) {
      smaller_leaf_splits->leaf_index = left_leaf_index;
    }
  } else {
    if (global_thread_index == 0) {
      larger_leaf_splits->leaf_index = left_leaf_index;
    } else if (global_thread_index == 1) {
      larger_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
      if (USE_GRAD_DISCRETIZED) {
        larger_leaf_splits->sum_of_gradients_hessians = best_split_info->left_sum_of_gradients_hessians;
      }
    } else if (global_thread_index == 2) {
      larger_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      larger_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      larger_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      larger_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      larger_leaf_splits->data_indices_in_leaf = point_structs_at_main ?
        cuda_data_indices_main + cuda_leaf_data_start[left_leaf_index] : cuda_data_indices;
    } else if (global_thread_index == 7) {
      smaller_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
      if (USE_GRAD_DISCRETIZED) {
        smaller_leaf_splits->sum_of_gradients_hessians = best_split_info->right_sum_of_gradients_hessians;
      }
    } else if (global_thread_index == 9) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      smaller_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      smaller_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      smaller_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      smaller_leaf_splits->data_indices_in_leaf = point_structs_at_main ?
        cuda_data_indices_main + cuda_leaf_data_start[right_leaf_index] : cuda_data_indices + cuda_leaf_num_data[left_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_hist_pool[right_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 15) {
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
    } else if (global_thread_index == 16) {
      cuda_split_info_buffer[6] = right_leaf_index;
    } else if (global_thread_index == 17) {
      cuda_split_info_buffer[7] = left_leaf_index;
    }
  }
}

__global__ void SplitInnerKernel(const int left_leaf_index, const int right_leaf_index,
  const data_size_t* cuda_leaf_data_start, const data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices,
  const data_size_t* block_to_left_offset_buffer, const data_size_t* block_to_right_offset_buffer,
  const uint16_t* block_to_left_offset, data_size_t* out_data_indices_in_leaf) {
  const data_size_t leaf_num_data_offset = cuda_leaf_data_start[left_leaf_index];
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index] + cuda_leaf_num_data[right_leaf_index];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim_x + threadIdx_x;
  const data_size_t* cuda_data_indices_in_leaf = cuda_data_indices + leaf_num_data_offset;
  const uint16_t* block_to_left_offset_ptr = block_to_left_offset + blockIdx.x * blockDim_x;
  const uint32_t to_right_block_offset = block_to_right_offset_buffer[blockIdx.x];
  const uint32_t to_left_block_offset = block_to_left_offset_buffer[blockIdx.x];
  data_size_t* left_out_data_indices_in_leaf = out_data_indices_in_leaf + to_left_block_offset;
  data_size_t* right_out_data_indices_in_leaf = out_data_indices_in_leaf + to_right_block_offset;
  if (static_cast<data_size_t>(global_thread_index) < num_data_in_leaf) {
    const uint32_t thread_to_left_offset = (threadIdx_x == 0 ? 0 : block_to_left_offset_ptr[threadIdx_x - 1]);
    const bool to_left = block_to_left_offset_ptr[threadIdx_x] > thread_to_left_offset;
    if (to_left) {
      left_out_data_indices_in_leaf[thread_to_left_offset] = cuda_data_indices_in_leaf[global_thread_index];
    } else {
      const uint32_t thread_to_right_offset = threadIdx.x - thread_to_left_offset;
      right_out_data_indices_in_leaf[thread_to_right_offset] = cuda_data_indices_in_leaf[global_thread_index];
    }
  }
}

__global__ void CopyDataIndicesKernel(
  const data_size_t num_data_in_leaf,
  const data_size_t* out_data_indices_in_leaf,
  data_size_t* cuda_data_indices) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx_x;
  if (global_thread_index < num_data_in_leaf) {
    cuda_data_indices[global_thread_index] = out_data_indices_in_leaf[global_thread_index];
  }
}

void CUDADataPartition::LaunchSplitInnerKernel(
  const data_size_t num_data_in_leaf,
  const CUDASplitInfo* best_split_info,
  const int left_leaf_index,
  const int right_leaf_index,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  data_size_t* left_leaf_num_data_ref,
  data_size_t* right_leaf_num_data_ref,
  data_size_t* left_leaf_start_ref,
  data_size_t* right_leaf_start_ref,
  double* left_leaf_sum_of_hessians_ref,
  double* right_leaf_sum_of_hessians_ref,
  double* left_leaf_sum_of_gradients_ref,
  double* right_leaf_sum_of_gradients_ref,
  data_size_t* global_left_leaf_num_data,
  data_size_t* global_right_leaf_num_data,
  const bool point_structs_at_main,
  const int deferred_slot,
  const data_size_t leaf_data_start_for_copy) {
  int num_blocks_final_ref = grid_dim_ - 1;
  int num_blocks_final_aligned = 1;
  while (num_blocks_final_ref > 0) {
    num_blocks_final_aligned <<= 1;
    num_blocks_final_ref >>= 1;
  }
  global_timer.Start("CUDADataPartition::AggregateBlockOffsetKernel");

  if (grid_dim_ > AGGREGATE_BLOCK_SIZE_DATA_PARTITION) {
    AggregateBlockOffsetKernel0<<<1, AGGREGATE_BLOCK_SIZE_DATA_PARTITION, 0, cuda_streams_[0]>>>(
      left_leaf_index,
      right_leaf_index,
      cuda_block_data_to_left_offset_.RawData(),
      cuda_block_data_to_right_offset_.RawData(), cuda_leaf_data_start_.RawData(), cuda_leaf_data_end_.RawData(),
      cuda_leaf_num_data_.RawData(), cuda_data_indices_.RawData(),
      grid_dim_);
  } else {
    AggregateBlockOffsetKernel1<<<1, num_blocks_final_aligned, 0, cuda_streams_[0]>>>(
      left_leaf_index,
      right_leaf_index,
      cuda_block_data_to_left_offset_.RawData(),
      cuda_block_data_to_right_offset_.RawData(), cuda_leaf_data_start_.RawData(), cuda_leaf_data_end_.RawData(),
      cuda_leaf_num_data_.RawData(), cuda_data_indices_.RawData(),
      grid_dim_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::AggregateBlockOffsetKernel");

  if (nccl_communicator_ != nullptr) {
    NCCLGroupStart();
    NCCLAllReduce<data_size_t>(
      cuda_leaf_num_data_.RawData() + left_leaf_index,
      cuda_split_info_buffer_.RawData() + 16,
      1, ncclInt32, ncclSum, nccl_communicator_, cuda_streams_[0]);
      NCCLAllReduce<data_size_t>(
        cuda_leaf_num_data_.RawData() + right_leaf_index,
        cuda_split_info_buffer_.RawData() + 17,
        1, ncclInt32, ncclSum, nccl_communicator_, cuda_streams_[0]);
    NCCLGroupEnd();
  }

  global_timer.Start("CUDADataPartition::SplitInnerKernel");
  SplitInnerKernel<<<grid_dim_, block_dim_, 0, cuda_streams_[1]>>>(
    left_leaf_index, right_leaf_index, cuda_leaf_data_start_.RawData(), cuda_leaf_num_data_.RawData(), cuda_data_indices_.RawData(),
    cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData(), cuda_block_to_left_offset_.RawData(),
    cuda_out_data_indices_in_leaf_.RawData());
  global_timer.Stop("CUDADataPartition::SplitInnerKernel");
  SynchronizeCUDADevice(__FILE__, __LINE__);

  global_timer.Start("CUDADataPartition::SplitTreeStructureKernel");

  int* split_info_ptr = cuda_split_info_buffer_.RawData() +
    (deferred_slot >= 0 ? 18 * static_cast<size_t>(deferred_slot) : 0);
#define SPLIT_TREE_ARGS \
  left_leaf_index, right_leaf_index, \
  cuda_block_data_to_left_offset_.RawData(), \
  cuda_block_data_to_right_offset_.RawData(), cuda_leaf_data_start_.RawData(), cuda_leaf_data_end_.RawData(), \
  cuda_leaf_num_data_.RawData(), cuda_out_data_indices_in_leaf_.RawData(), \
  cuda_data_indices_.RawData(), \
  point_structs_at_main, \
  best_split_info, \
  smaller_leaf_splits, \
  larger_leaf_splits, \
  num_total_bin_, \
  cuda_hist_, \
  cuda_hist_pool_.RawData(), \
  cuda_leaf_output_.RawData(), split_info_ptr

  if (nccl_communicator_ != nullptr) {
    if (use_quantized_grad_) {
      SplitTreeStructureKernel<true, true><<<4, 5, 0, cuda_streams_[0]>>>(SPLIT_TREE_ARGS);
    } else {
      SplitTreeStructureKernel<true, false><<<4, 5, 0, cuda_streams_[0]>>>(SPLIT_TREE_ARGS);
    }
  } else {
    if (use_quantized_grad_) {
      SplitTreeStructureKernel<false, true><<<4, 5, 0, cuda_streams_[0]>>>(SPLIT_TREE_ARGS);
    } else {
      SplitTreeStructureKernel<false, false><<<4, 5, 0, cuda_streams_[0]>>>(SPLIT_TREE_ARGS);
    }
  }

#undef SPLIT_TREE_ARGS
  global_timer.Stop("CUDADataPartition::SplitTreeStructureKernel");
  if (deferred_slot >= 0) {
    // deferred (level-batched) mode: no per-split synchronization. The copy's
    // destination and size are host-known (the split leaf's region); split info
    // is read back once per level via FinishSplitBatch.
    CopyDataIndicesKernel<<<grid_dim_, block_dim_, 0, cuda_streams_[2]>>>(
      num_data_in_leaf, cuda_out_data_indices_in_leaf_.RawData(),
      cuda_data_indices_.RawData() + leaf_data_start_for_copy);
    CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_copy_done_event_, cuda_streams_[2]));
    return;
  }
  std::vector<int> cpu_split_info_buffer(18);
  const double* cpu_sum_hessians_info = reinterpret_cast<const double*>(cpu_split_info_buffer.data() + 8);
  global_timer.Start("CUDADataPartition::CopyFromCUDADeviceToHostAsync");
  CopyFromCUDADeviceToHostAsync<int>(cpu_split_info_buffer.data(), cuda_split_info_buffer_.RawData(), 18, cuda_streams_[0], __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::CopyFromCUDADeviceToHostAsync");
  const data_size_t left_leaf_num_data = cpu_split_info_buffer[1];
  const data_size_t left_leaf_data_start = cpu_split_info_buffer[2];
  const data_size_t right_leaf_num_data = cpu_split_info_buffer[4];
  global_timer.Start("CUDADataPartition::CopyDataIndicesKernel");
  CopyDataIndicesKernel<<<grid_dim_, block_dim_, 0, cuda_streams_[2]>>>(
    left_leaf_num_data + right_leaf_num_data, cuda_out_data_indices_in_leaf_.RawData(), cuda_data_indices_.RawData() + left_leaf_data_start);
  CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_copy_done_event_, cuda_streams_[2]));
  global_timer.Stop("CUDADataPartition::CopyDataIndicesKernel");
  const data_size_t right_leaf_data_start = cpu_split_info_buffer[5];
  *left_leaf_num_data_ref = left_leaf_num_data;
  *left_leaf_start_ref = left_leaf_data_start;
  *right_leaf_num_data_ref = right_leaf_num_data;
  *right_leaf_start_ref = right_leaf_data_start;
  *left_leaf_sum_of_hessians_ref = cpu_sum_hessians_info[0];
  *right_leaf_sum_of_hessians_ref = cpu_sum_hessians_info[1];
  *left_leaf_sum_of_gradients_ref = cpu_sum_hessians_info[2];
  *right_leaf_sum_of_gradients_ref = cpu_sum_hessians_info[3];
  if (nccl_communicator_ != nullptr) {
    *global_left_leaf_num_data = cpu_split_info_buffer[16];
    *global_right_leaf_num_data = cpu_split_info_buffer[17];
  }
}

// ---- batched apply kernels for the hybrid level-batched growth ----
// One launch per kernel family covers ALL splits of a level. Data-parallel
// kernels use grid (max_num_blocks, num_splits) with a fixed 1024-thread block;
// blocks beyond a split's own num_blocks exit early. The results are identical
// to the per-split kernels for any block size because the partition is stable
// (per-block stable scatter + exact per-block counts), so the batched path may
// use a fixed block size where CalcBlockDim would have chosen a smaller one.
// Per-split scratch regions: the uint16 bit vector and the out-indices buffer
// use the split leaf's own [leaf_data_start, +num_data_in_leaf) window; the
// block offset buffers use [block_offset_start, +num_blocks+1).

__device__ __forceinline__ uint32_t HybridLoadBin(
  const CUDAHybridApplyDescriptor& d, const data_size_t global_data_index) {
  if (d.bit_type == 8) {
    return static_cast<uint32_t>(static_cast<const uint8_t*>(d.column_data)[global_data_index]);
  } else if (d.bit_type == 4) {
    // 4-bit packed compact source: column_data already points at this column's
    // byte of row 0; rows are packed_row_stride bytes apart, the bin is the
    // packed_shift nibble (see SetCompactPackedColumnView)
    const uint8_t packed = static_cast<const uint8_t*>(d.column_data)[
      static_cast<size_t>(global_data_index) * static_cast<size_t>(d.packed_row_stride)];
    return static_cast<uint32_t>(packed >> d.packed_shift) & 0xfu;
  } else if (d.bit_type == 16) {
    return static_cast<uint32_t>(static_cast<const uint16_t*>(d.column_data)[global_data_index]);
  } else {
    return static_cast<const uint32_t*>(d.column_data)[global_data_index];
  }
}

// runtime-branch replica of GenDataToLeftBitVectorKernel's templated decision
// (branches are block-uniform descriptor fields, so divergence-free); the
// (MISSING_IS_ZERO || MFB_IS_ZERO) condition mirrors the template verbatim
__device__ __forceinline__ uint16_t HybridGenBitVectorDecision(
  const CUDAHybridApplyDescriptor& d, const uint32_t bin) {
  if (d.cat_bitset_len > 0) {
    // categorical: mirror GenDataToLeftBitVectorKernel_Categorical exactly
    if ((d.use_min_bin && (bin < d.min_bin || bin > d.max_bin)) ||
        (!d.use_min_bin && bin == 0)) {
      return d.split_default_to_left;
    }
    return CUDAFindInBitset(d.cat_bitset, d.cat_bitset_len,
                            bin - d.min_bin + static_cast<uint32_t>(d.cat_mfb_offset)) ? 1 : 0;
  }
  if (!d.min_is_max) {
    if ((d.missing_is_zero && !d.mfb_is_zero && bin == d.t_zero_bin) ||
      (d.missing_is_na && !d.mfb_is_na && bin == d.max_bin)) {
      return d.split_missing_default_to_left;
    } else if ((d.use_min_bin && (bin < d.min_bin || bin > d.max_bin)) ||
               (!d.use_min_bin && bin == 0)) {
      if ((d.missing_is_na && d.mfb_is_na) || (d.missing_is_zero || d.mfb_is_zero)) {
        return d.split_missing_default_to_left;
      } else {
        return d.split_default_to_left;
      }
    } else if (bin <= d.th) {
      return 1;
    } else {
      return 0;
    }
  } else {
    if (d.missing_is_zero && !d.mfb_is_zero && bin == d.t_zero_bin) {
      return d.split_missing_default_to_left;
    } else if (bin != d.max_bin) {
      if ((d.missing_is_na && d.mfb_is_na) || (d.missing_is_zero && d.mfb_is_zero)) {
        return d.split_missing_default_to_left;
      } else {
        return d.split_default_to_left;
      }
    } else {
      if (d.missing_is_na && !d.mfb_is_na) {
        return d.split_missing_default_to_left;
      } else if (d.max_bin_to_left) {
        return 1;
      } else {
        return 0;
      }
    }
  }
}

// runtime-branch replica of UpdateDataIndexToLeafIndexKernel's templated decision
__device__ __forceinline__ int HybridUpdateLeafIndexDecision(
  const CUDAHybridApplyDescriptor& d, const uint32_t bin) {
  if (d.cat_bitset_len > 0) {
    // categorical: mirror UpdateDataIndexToLeafIndexKernel_Categorical exactly
    if ((d.use_min_bin && (bin < d.min_bin || bin > d.max_bin)) ||
        (!d.use_min_bin && bin == 0)) {
      return d.default_leaf_index;
    }
    return CUDAFindInBitset(d.cat_bitset, d.cat_bitset_len,
                            bin - d.min_bin + static_cast<uint32_t>(d.cat_mfb_offset)) ?
        d.left_leaf_index : d.right_leaf_index;
  }
  if (!d.min_is_max) {
    if ((d.missing_is_zero && !d.mfb_is_zero && bin == d.t_zero_bin) ||
      (d.missing_is_na && !d.mfb_is_na && bin == d.max_bin)) {
      return d.missing_default_leaf_index;
    } else if ((d.use_min_bin && (bin < d.min_bin || bin > d.max_bin)) ||
               (!d.use_min_bin && bin == 0)) {
      if ((d.missing_is_na && d.mfb_is_na) || (d.missing_is_zero && d.mfb_is_zero)) {
        return d.missing_default_leaf_index;
      } else {
        return d.default_leaf_index;
      }
    } else if (bin > d.th) {
      return d.right_leaf_index;
    } else {
      return d.left_leaf_index;
    }
  } else {
    if (d.missing_is_zero && !d.mfb_is_zero && bin == d.t_zero_bin) {
      return d.missing_default_leaf_index;
    } else if (bin != d.max_bin) {
      if ((d.missing_is_na && d.mfb_is_na) || (d.missing_is_zero && d.mfb_is_zero)) {
        return d.missing_default_leaf_index;
      } else {
        return d.default_leaf_index;
      }
    } else {
      if (d.missing_is_na && !d.mfb_is_na) {
        return d.missing_default_leaf_index;
      } else if (d.max_bin_to_left) {
        return d.left_leaf_index;
      } else {
        return d.right_leaf_index;
      }
    }
  }
}


// One block per categorical split of the level: build the split's INNER
// bitset in its arena region from the finder's cat thresholds, then patch the
// (device-resident) descriptor's default-direction fields from on-device MFB
// membership. Mirrors LaunchGenDataToLeftBitVectorCategoricalKernel's host
// logic (including its raw-most_freq_bin membership test) without its D2H.
__global__ void HybridBuildCatBitsetArenaKernel(
  CUDAHybridApplyDescriptor* descs,
  const int* cat_desc_indices,
  const uint32_t* cat_mfb_bins) {
  CUDAHybridApplyDescriptor* d = descs + cat_desc_indices[blockIdx.x];
  uint32_t* bits = const_cast<uint32_t*>(d->cat_bitset);
  const int len = d->cat_bitset_len;
  for (int i = static_cast<int>(threadIdx.x); i < len; i += static_cast<int>(blockDim.x)) {
    bits[i] = 0;
  }
  __syncthreads();
  const CUDASplitInfo* info = d->best_split_info;
  const int n = info->num_cat_threshold;
  const uint32_t* th = info->cat_threshold;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    const uint32_t pos = th[i];
    if (static_cast<int>(pos >> 5) < len) {
      atomicOr(&bits[pos >> 5], 1u << (pos & 31));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t mfb = cat_mfb_bins[blockIdx.x];
    uint8_t to_left = 0;
    if (mfb > 0 && CUDAFindInBitset(bits, len, mfb)) {
      to_left = 1;
    }
    d->split_default_to_left = to_left;
    d->split_missing_default_to_left = to_left;
    d->default_leaf_index = to_left ? d->left_leaf_index : d->right_leaf_index;
    d->missing_default_leaf_index = d->default_leaf_index;
  }
}

void CUDADataPartition::LaunchBuildCatBitsetArenaKernel(
    const int num_cat_splits, const std::vector<int>& cat_desc_indices,
    const std::vector<uint32_t>& cat_mfb_bins) {
  if (cuda_cat_desc_indices_.Size() < cat_desc_indices.size()) {
    cuda_cat_desc_indices_.Resize(cat_desc_indices.size() * 2);
    cuda_cat_mfb_bins_.Resize(cat_desc_indices.size() * 2);
  }
  CopyFromHostToCUDADeviceAsync<int>(cuda_cat_desc_indices_.RawData(), cat_desc_indices.data(),
                                     cat_desc_indices.size(), cuda_streams_[0], __FILE__, __LINE__);
  CopyFromHostToCUDADeviceAsync<uint32_t>(cuda_cat_mfb_bins_.RawData(), cat_mfb_bins.data(),
                                          cat_mfb_bins.size(), cuda_streams_[0], __FILE__, __LINE__);
  HybridBuildCatBitsetArenaKernel<<<num_cat_splits, 128, 0, cuda_streams_[0]>>>(
    cuda_apply_descs_.RawData(), cuda_cat_desc_indices_.RawDataReadOnly(),
    cuda_cat_mfb_bins_.RawDataReadOnly());
}

// fused GenDataToLeftBitVector + UpdateDataIndexToLeafIndex: both derive from
// the SAME (data index, bin) load, so fusing halves the column/index reads that
// dominate the apply phase's memory traffic. The written values are identical
// to the two per-split kernels'.
// flat block id -> descriptor index: largest desc with flat_block_start <= flat
__device__ __forceinline__ int HybridFlatBlockDesc(
  const CUDAHybridApplyDescriptor* descs, const int num_split_descs, const int flat) {
  int lo = 0;
  int hi = num_split_descs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (descs[mid].flat_block_start <= flat) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// shared per-chunk body of the gen-bit-vector kernel: one 1024-row chunk of
// one split descriptor. data_index_to_leaf_index is NOT updated here: every
// consumer of the map is a tree-end operation (AddPredictionToScore, refit
// stats, linear trees), so the per-level 4-byte random scatter over every row
// -- one full DRAM sector per row at deep levels -- is deferred to ONE
// MaterializeLeafMapKernel pass per tree reading the final windows.
// Packed-bit chunk body: the per-row direction bits travel between the
// gen-bit and split-inner kernels as warp ballot words (1 bit/row) instead of
// a per-row uint16 running prefix -- 32x less inter-kernel traffic. The
// per-block left/right totals still land in the offset buffers for the
// aggregate kernel, unchanged.
__device__ __forceinline__ void HybridGenBitChunk(
  const CUDAHybridApplyDescriptor& d,
  const unsigned int block_x,
  const data_size_t* cuda_data_indices,
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer,
  uint16_t* shared_mem_buffer) {
  uint16_t to_left = 0;
  const data_size_t local_data_index = static_cast<data_size_t>(block_x * blockDim.x + threadIdx.x);
  if (local_data_index < d.num_data_in_leaf) {
    const data_size_t global_data_index = cuda_data_indices[d.leaf_data_start + local_data_index];
    const uint32_t bin = HybridLoadBin(d, global_data_index);
    to_left = HybridGenBitVectorDecision(d, bin);
  }
  const uint32_t ballot = __ballot_sync(0xffffffffu, to_left != 0);
  const unsigned int lane = threadIdx.x & (WARPSIZE - 1);
  const unsigned int warp = threadIdx.x / WARPSIZE;
  // bit words live in the (reinterpreted) block_to_left_offset buffer: one
  // uint32 per warp-of-rows, at word index (row position / 32)
  // chunk-slot word indexing: leaf windows are not 32-aligned, so bits are
  // stored per (descriptor, block) slot -- d.block_offset_start is a unique
  // monotone per-desc slot base in both the host flat path and the graph path
  uint32_t* bit_words = reinterpret_cast<uint32_t*>(block_to_left_offset);
  if (lane == 0) {
    bit_words[(static_cast<size_t>(d.block_offset_start) + block_x) * (blockDim.x / WARPSIZE) + warp] = ballot;
  }
  // block totals for the aggregate kernel (unchanged layout)
  uint16_t warp_left = static_cast<uint16_t>(__popc(ballot));
  if (lane == 0) {
    shared_mem_buffer[warp] = warp_left;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t block_left = 0;
    const unsigned int num_warps = blockDim.x / WARPSIZE;
    for (unsigned int w = 0; w < num_warps; ++w) {
      block_left += shared_mem_buffer[w];
    }
    const data_size_t num_data_in_block =
      (static_cast<data_size_t>(block_x) + 1) * blockDim.x <= d.num_data_in_leaf ?
        static_cast<data_size_t>(blockDim.x) :
        d.num_data_in_leaf - static_cast<data_size_t>(block_x) * blockDim.x;
    data_size_t* left_buf = block_to_left_offset_buffer + d.block_offset_start;
    data_size_t* right_buf = block_to_right_offset_buffer + d.block_offset_start;
    if (num_data_in_block > 0) {
      left_buf[block_x + 1] = static_cast<data_size_t>(block_left);
      right_buf[block_x + 1] = num_data_in_block - static_cast<data_size_t>(block_left);
    } else {
      left_buf[block_x + 1] = 0;
      right_buf[block_x + 1] = 0;
    }
  }
}

__global__ void HybridGenBitVectorUpdateLeafIndexBatchKernel(
  const CUDAHybridApplyDescriptor* descs,
  const data_size_t* cuda_data_indices_param,
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer,
  int* cuda_data_index_to_leaf_index,
  const CUDAHybridGraphLoopStateOpt gstate,
  const int num_split_descs,
  const int total_flat_blocks) {
  __shared__ uint16_t shared_mem_buffer[WARPSIZE];
  (void)cuda_data_index_to_leaf_index;
  if (total_flat_blocks > 0) {
    // host-launched flat grid: a grid-stride loop over the level's REAL chunk
    // count with binary-searched (descriptor, local block) mapping. Skewed
    // levels previously launched (largest leaf's blocks x num_splits), which
    // is mostly-empty at deep levels; the flat form launches exactly the work.
    for (int flat = static_cast<int>(blockIdx.x); flat < total_flat_blocks;
         flat += static_cast<int>(gridDim.x)) {
      const int desc_index = HybridFlatBlockDesc(descs, num_split_descs, flat);
      const CUDAHybridApplyDescriptor d = descs[desc_index];
      HybridGenBitChunk(d, static_cast<unsigned int>(flat - d.flat_block_start),
        cuda_data_indices_param, block_to_left_offset,
        block_to_left_offset_buffer, block_to_right_offset_buffer, shared_mem_buffer);
      __syncthreads();
    }
    return;
  }
  // graph-captured 2D path (grids resized per level by the device controller)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const data_size_t* cuda_data_indices =
    HybridGraphMainIndices(gstate, cuda_data_indices_param);
  const CUDAHybridApplyDescriptor d = descs[blockIdx.y];
  if (static_cast<int>(blockIdx.x) >= d.num_blocks) {
    return;
  }
  HybridGenBitChunk(d, blockIdx.x, cuda_data_indices, block_to_left_offset,
    block_to_left_offset_buffer, block_to_right_offset_buffer, shared_mem_buffer);
}

// per-split block-offset prefix sum + children leaf metadata update; one block
// per split, generalizing AggregateBlockOffsetKernel0 to arbitrary num_blocks.
// The split's region slot 0 is zeroed explicitly (a slot that held a non-zero
// value in a previous level's layout can become a region start in this one).
__global__ void HybridAggregateBlockOffsetBatchKernel(
  const CUDAHybridApplyDescriptor* descs,
  data_size_t* block_to_left_offset_buffer_base,
  data_size_t* block_to_right_offset_buffer_base,
  data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end,
  data_size_t* cuda_leaf_num_data,
  data_size_t* level_smaller_counts,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the gen-bit-vector kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.x)) {
    return;
  }
  __shared__ uint32_t shared_mem_buffer[WARPSIZE];
  __shared__ uint32_t to_left_total_count;
  const CUDAHybridApplyDescriptor d = descs[blockIdx.x];
  data_size_t* block_to_left_offset_buffer = block_to_left_offset_buffer_base + d.block_offset_start;
  data_size_t* block_to_right_offset_buffer = block_to_right_offset_buffer_base + d.block_offset_start;
  const data_size_t num_blocks = static_cast<data_size_t>(d.num_blocks);
  const data_size_t num_data_in_leaf = d.num_data_in_leaf;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int threadIdx_x = threadIdx.x;
  const data_size_t num_blocks_plus_1 = num_blocks + 1;
  const uint32_t num_blocks_per_thread = (num_blocks_plus_1 + blockDim_x - 1) / blockDim_x;
  const uint32_t remain = num_blocks_plus_1 - ((num_blocks_per_thread - 1) * blockDim_x);
  const uint32_t remain_offset = remain * num_blocks_per_thread;
  uint32_t thread_start_block_index = 0;
  uint32_t thread_end_block_index = 0;
  if (threadIdx_x < remain) {
    thread_start_block_index = threadIdx_x * num_blocks_per_thread;
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread, num_blocks_plus_1);
  } else {
    thread_start_block_index = remain_offset + (num_blocks_per_thread - 1) * (threadIdx_x - remain);
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread - 1, num_blocks_plus_1);
  }
  if (threadIdx.x == 0) {
    block_to_left_offset_buffer[0] = 0;
    block_to_right_offset_buffer[0] = 0;
  }
  __syncthreads();
  for (uint32_t block_index = thread_start_block_index + 1; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += block_to_left_offset_buffer[block_index - 1];
    block_to_right_offset_buffer[block_index] += block_to_right_offset_buffer[block_index - 1];
  }
  __syncthreads();
  uint32_t block_to_left_offset = 0;
  uint32_t block_to_right_offset = 0;
  if (thread_start_block_index < thread_end_block_index && thread_start_block_index > 1) {
    block_to_left_offset = block_to_left_offset_buffer[thread_start_block_index - 1];
    block_to_right_offset = block_to_right_offset_buffer[thread_start_block_index - 1];
  }
  block_to_left_offset = ShufflePrefixSum<uint32_t>(block_to_left_offset, shared_mem_buffer);
  __syncthreads();
  block_to_right_offset = ShufflePrefixSum<uint32_t>(block_to_right_offset, shared_mem_buffer);
  if (threadIdx_x == blockDim_x - 1) {
    to_left_total_count = block_to_left_offset + block_to_left_offset_buffer[num_blocks];
  }
  __syncthreads();
  const uint32_t to_left_thread_block_offset = block_to_left_offset;
  const uint32_t to_right_thread_block_offset = block_to_right_offset + to_left_total_count;
  for (uint32_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += to_left_thread_block_offset;
    block_to_right_offset_buffer[block_index] += to_right_thread_block_offset;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int left_leaf_index = d.left_leaf_index;
    const int right_leaf_index = d.right_leaf_index;
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    const data_size_t left_count = static_cast<data_size_t>(to_left_total_count);
    const data_size_t right_count = num_data_in_leaf - left_count;
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + left_count;
    cuda_leaf_num_data[left_leaf_index] = left_count;
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = right_count;
    // actual smaller-child size of this split, consumed by the speculative
    // batched construct kernel's device row-grouping (single-sync flow)
    level_smaller_counts[blockIdx.x] = left_count < right_count ? left_count : right_count;
  }
}

template <bool WRITE_LEAF_MAP>
__device__ __forceinline__ void HybridSplitInnerChunk(
  const CUDAHybridApplyDescriptor& d,
  const unsigned int block_x,
  const data_size_t* cuda_data_indices,
  const data_size_t* block_to_left_offset_buffer_base,
  const data_size_t* block_to_right_offset_buffer_base,
  const uint16_t* block_to_left_offset,
  data_size_t* out_data_indices_in_leaf,
  int* cuda_data_index_to_leaf_index,
  uint16_t* shared_warp_scan) {
  const data_size_t num_data_in_leaf = d.num_data_in_leaf;
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int global_thread_index = block_x * blockDim_x + threadIdx_x;
  const data_size_t* cuda_data_indices_in_leaf = cuda_data_indices + d.leaf_data_start;
  const uint32_t to_right_block_offset = block_to_right_offset_buffer_base[d.block_offset_start + block_x];
  const uint32_t to_left_block_offset = block_to_left_offset_buffer_base[d.block_offset_start + block_x];
  data_size_t* left_out_data_indices_in_leaf = out_data_indices_in_leaf + d.leaf_data_start + to_left_block_offset;
  data_size_t* right_out_data_indices_in_leaf = out_data_indices_in_leaf + d.leaf_data_start + to_right_block_offset;
  // reconstruct per-thread positions from the packed ballot words
  const uint32_t* bit_words = reinterpret_cast<const uint32_t*>(block_to_left_offset);
  const unsigned int lane = threadIdx_x & (WARPSIZE - 1);
  const unsigned int warp = threadIdx_x / WARPSIZE;
  const uint32_t word = bit_words[(static_cast<size_t>(d.block_offset_start) + block_x) * (blockDim_x / WARPSIZE) + warp];
  const uint16_t warp_left_total = static_cast<uint16_t>(__popc(word));
  if (lane == 0) {
    shared_warp_scan[warp] = warp_left_total;
  }
  __syncthreads();
  // exclusive scan of per-warp left counts (few warps; thread 0 serial)
  uint32_t warp_left_base = 0;
  for (unsigned int w = 0; w < warp; ++w) {
    warp_left_base += shared_warp_scan[w];
  }
  if (static_cast<data_size_t>(global_thread_index) < num_data_in_leaf) {
    const bool to_left = (word >> lane) & 1u;
    const uint32_t lefts_before_in_warp = __popc(word & ((1u << lane) - 1u));
    const uint32_t thread_to_left_offset = warp_left_base + lefts_before_in_warp;
    const data_size_t row_index = cuda_data_indices_in_leaf[global_thread_index];
    if (to_left) {
      left_out_data_indices_in_leaf[thread_to_left_offset] = row_index;
    } else {
      const uint32_t thread_to_right_offset = threadIdx_x - thread_to_left_offset;
      right_out_data_indices_in_leaf[thread_to_right_offset] = row_index;
    }
    if (WRITE_LEAF_MAP) {
      // final level of the tree: the row index is already in a register, so
      // the map write costs one store here instead of a separate full-data
      // materialize pass re-reading every window
      cuda_data_index_to_leaf_index[row_index] = to_left ? d.left_leaf_index : d.right_leaf_index;
    }
  }
  __syncthreads();
}

__global__ void HybridSplitInnerBatchKernel(
  const CUDAHybridApplyDescriptor* descs,
  const data_size_t* cuda_data_indices_param,
  const data_size_t* block_to_left_offset_buffer_base,
  const data_size_t* block_to_right_offset_buffer_base,
  const uint16_t* block_to_left_offset,
  data_size_t* out_data_indices_param,
  const CUDAHybridGraphLoopStateOpt gstate,
  const int num_split_descs,
  const int total_flat_blocks,
  int* cuda_data_index_to_leaf_index,
  const int write_leaf_map) {
  __shared__ uint16_t shared_warp_scan[WARPSIZE];
  if (total_flat_blocks > 0) {
    // host-launched flat grid (see the gen-bit-vector kernel)
    if (write_leaf_map != 0) {
      for (int flat = static_cast<int>(blockIdx.x); flat < total_flat_blocks;
           flat += static_cast<int>(gridDim.x)) {
        const int desc_index = HybridFlatBlockDesc(descs, num_split_descs, flat);
        const CUDAHybridApplyDescriptor d = descs[desc_index];
        HybridSplitInnerChunk<true>(d, static_cast<unsigned int>(flat - d.flat_block_start),
          cuda_data_indices_param, block_to_left_offset_buffer_base,
          block_to_right_offset_buffer_base, block_to_left_offset,
          out_data_indices_param, cuda_data_index_to_leaf_index, shared_warp_scan);
      }
    } else {
      for (int flat = static_cast<int>(blockIdx.x); flat < total_flat_blocks;
           flat += static_cast<int>(gridDim.x)) {
        const int desc_index = HybridFlatBlockDesc(descs, num_split_descs, flat);
        const CUDAHybridApplyDescriptor d = descs[desc_index];
        HybridSplitInnerChunk<false>(d, static_cast<unsigned int>(flat - d.flat_block_start),
          cuda_data_indices_param, block_to_left_offset_buffer_base,
          block_to_right_offset_buffer_base, block_to_left_offset,
          out_data_indices_param, cuda_data_index_to_leaf_index, shared_warp_scan);
      }
    }
    return;
  }
  // graphs A2 idle-block guard (pow2-frozen grid; see the gen-bit-vector kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const data_size_t* cuda_data_indices =
    HybridGraphMainIndices(gstate, cuda_data_indices_param);
  data_size_t* out_data_indices_in_leaf =
    HybridGraphOutIndices(gstate, out_data_indices_param);
  const CUDAHybridApplyDescriptor d = descs[blockIdx.y];
  if (static_cast<int>(blockIdx.x) >= d.num_blocks) {
    return;
  }
  HybridSplitInnerChunk<false>(d, blockIdx.x, cuda_data_indices,
    block_to_left_offset_buffer_base, block_to_right_offset_buffer_base,
    block_to_left_offset, out_data_indices_in_leaf, cuda_data_index_to_leaf_index,
    shared_warp_scan);
}

// replica of SplitTreeStructureKernel<false, USE_GRAD_DISCRETIZED> with
// point_structs_at_main fixed to true (batched apply always points the child
// structs' data_indices_in_leaf at the main index array); one 32-thread block
// per split, threadIdx.x playing the original global thread index.
// USE_NCCL_REDUCE mirrors SplitTreeStructureKernel: slots 16/17 carry the
// GLOBAL child counts and the smaller/larger role assignment reads the
// global counts from the split info, so every rank makes the same choice.
template <bool USE_NCCL_REDUCE, bool USE_GRAD_DISCRETIZED>
__global__ void HybridSplitTreeStructureBatchKernel(
  const CUDAHybridApplyDescriptor* descs,
  data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices_main_param,
  const int num_total_bin,
  hist_t* cuda_hist, hist_t** cuda_hist_pool,
  double* cuda_leaf_output,
  int* cuda_split_info_buffer_base,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the gen-bit-vector kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.x)) {
    return;
  }
  const CUDAHybridApplyDescriptor d = descs[blockIdx.x];
  const int left_leaf_index = d.left_leaf_index;
  const int right_leaf_index = d.right_leaf_index;
  const CUDASplitInfo* best_split_info = d.best_split_info;
  CUDALeafSplitsStruct* smaller_leaf_splits = d.smaller_leaf_splits;
  CUDALeafSplitsStruct* larger_leaf_splits = d.larger_leaf_splits;
  // graphs L1: the level's new main array is the loop's out buffer, and the
  // deferred split info lands in the level's own slab region
  const data_size_t* cuda_data_indices_main =
    HybridGraphOutIndices(gstate, const_cast<data_size_t*>(cuda_data_indices_main_param));
  int* cuda_split_info_buffer = cuda_split_info_buffer_base +
    18 * (HybridGraphSplitInfoBase(gstate) + blockIdx.x);
  const unsigned int to_left_total_cnt = cuda_leaf_num_data[left_leaf_index];
  double* cuda_split_info_buffer_for_hessians = reinterpret_cast<double*>(cuda_split_info_buffer + 8);
  const unsigned int global_thread_index = threadIdx.x;
  if (global_thread_index == 0) {
    cuda_leaf_output[left_leaf_index] = best_split_info->left_value;
  } else if (global_thread_index == 1) {
    cuda_leaf_output[right_leaf_index] = best_split_info->right_value;
  } else if (global_thread_index == 2) {
    cuda_split_info_buffer[0] = left_leaf_index;
  } else if (global_thread_index == 3) {
    cuda_split_info_buffer[1] = cuda_leaf_num_data[left_leaf_index];
  } else if (global_thread_index == 4) {
    cuda_split_info_buffer[2] = cuda_leaf_data_start[left_leaf_index];
  } else if (global_thread_index == 5) {
    cuda_split_info_buffer[3] = right_leaf_index;
  } else if (global_thread_index == 6) {
    cuda_split_info_buffer[4] = cuda_leaf_num_data[right_leaf_index];
  } else if (global_thread_index == 7) {
    cuda_split_info_buffer[5] = cuda_leaf_data_start[right_leaf_index];
  } else if (global_thread_index == 8) {
    cuda_split_info_buffer_for_hessians[0] = best_split_info->left_sum_hessians;
    cuda_split_info_buffer_for_hessians[2] = best_split_info->left_sum_gradients;
  } else if (global_thread_index == 9) {
    cuda_split_info_buffer_for_hessians[1] = best_split_info->right_sum_hessians;
    cuda_split_info_buffer_for_hessians[3] = best_split_info->right_sum_gradients;
  } else if (global_thread_index == 10 && USE_NCCL_REDUCE) {
    // GLOBAL child counts for the host's global_num_data_in_leaf_ (identical
    // on every rank: derived from the all-reduced histograms). Same contract
    // as SplitTreeStructureKernel slots 16/17.
    cuda_split_info_buffer[16] = best_split_info->left_count;
    cuda_split_info_buffer[17] = best_split_info->right_count;
  }

  // Under NCCL the smaller/larger role must be the same on every rank -- the
  // level all-reduce pairs up smaller-leaf histograms BY ROLE across ranks --
  // so the choice reads the global counts from the split info; local
  // partition counts can order differently per rank on skewed leaves.
  const bool left_is_smaller = USE_NCCL_REDUCE ?
    best_split_info->left_count < best_split_info->right_count :
    cuda_leaf_num_data[left_leaf_index] < cuda_leaf_num_data[right_leaf_index];

  if (left_is_smaller) {
    if (global_thread_index == 0) {
      hist_t* parent_hist_ptr = cuda_hist_pool[left_leaf_index];
      cuda_hist_pool[right_leaf_index] = parent_hist_ptr;
      cuda_hist_pool[left_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 1) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
      if (USE_GRAD_DISCRETIZED) {
        smaller_leaf_splits->sum_of_gradients_hessians = best_split_info->left_sum_of_gradients_hessians;
      }
    } else if (global_thread_index == 2) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      smaller_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      smaller_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      smaller_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      smaller_leaf_splits->data_indices_in_leaf = cuda_data_indices_main + cuda_leaf_data_start[left_leaf_index];
    } else if (global_thread_index == 7) {
      larger_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      larger_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
      if (USE_GRAD_DISCRETIZED) {
        larger_leaf_splits->sum_of_gradients_hessians = best_split_info->right_sum_of_gradients_hessians;
      }
    } else if (global_thread_index == 9) {
      larger_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      larger_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      larger_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      larger_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      larger_leaf_splits->data_indices_in_leaf = cuda_data_indices_main + cuda_leaf_data_start[right_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_split_info_buffer[6] = left_leaf_index;
    } else if (global_thread_index == 15) {
      cuda_split_info_buffer[7] = right_leaf_index;
    } else if (global_thread_index == 16) {
      smaller_leaf_splits->leaf_index = left_leaf_index;
    }
  } else {
    if (global_thread_index == 0) {
      larger_leaf_splits->leaf_index = left_leaf_index;
    } else if (global_thread_index == 1) {
      larger_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
      if (USE_GRAD_DISCRETIZED) {
        larger_leaf_splits->sum_of_gradients_hessians = best_split_info->left_sum_of_gradients_hessians;
      }
    } else if (global_thread_index == 2) {
      larger_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      larger_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      larger_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      larger_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      larger_leaf_splits->data_indices_in_leaf = cuda_data_indices_main + cuda_leaf_data_start[left_leaf_index];
    } else if (global_thread_index == 7) {
      smaller_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
      if (USE_GRAD_DISCRETIZED) {
        smaller_leaf_splits->sum_of_gradients_hessians = best_split_info->right_sum_of_gradients_hessians;
      }
    } else if (global_thread_index == 9) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      smaller_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      smaller_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      smaller_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      smaller_leaf_splits->data_indices_in_leaf = cuda_data_indices_main + cuda_leaf_data_start[right_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_hist_pool[right_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 15) {
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
    } else if (global_thread_index == 16) {
      cuda_split_info_buffer[6] = right_leaf_index;
    } else if (global_thread_index == 17) {
      cuda_split_info_buffer[7] = left_leaf_index;
    }
  }
}

// selective grow-then-prune: rewrite the data-index-to-leaf-index entries of the
// collapsed subtree windows (blockIdx.y indexes windows; blocks beyond a window's
// own row count exit early). The window rows in the main index array are exactly
// the subtree's rows (child regions nest inside the parent region).
__global__ void CollapseLeafWindowsKernel(
  const CUDACollapseWindow* windows,
  const data_size_t* cuda_data_indices,
  int* cuda_data_index_to_leaf_index) {
  const CUDACollapseWindow w = windows[blockIdx.y];
  const data_size_t local_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (local_index < w.count) {
    cuda_data_index_to_leaf_index[cuda_data_indices[w.start + local_index]] = w.target_leaf;
  }
}

void CUDADataPartition::CollapseLeafWindows(const std::vector<CUDACollapseWindow>& windows) {
  const int num_windows = static_cast<int>(windows.size());
  if (num_windows <= 0) {
    return;
  }
  host_collapse_windows_ = windows;
  if (cuda_collapse_windows_.Size() < windows.size()) {
    cuda_collapse_windows_.Resize(std::max(windows.size(),
      static_cast<size_t>(num_leaves_ / 2 + 2)));
  }
  // async H2D on the apply stream: ordered before the kernel below and any later
  // batched apply of the same level (same stream). The host staging buffer is
  // only rewritten after the next full sync, and a pageable async H2D returns
  // only once the data is staged, so reuse is safe.
  CopyFromHostToCUDADeviceAsync<CUDACollapseWindow>(
    cuda_collapse_windows_.RawData(), host_collapse_windows_.data(),
    windows.size(), cuda_streams_[0], __FILE__, __LINE__);
  data_size_t max_count = 0;
  for (const CUDACollapseWindow& w : windows) {
    max_count = std::max(max_count, w.count);
  }
  const int max_blocks = (max_count + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
    SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
  const dim3 grid(static_cast<unsigned int>(max_blocks), static_cast<unsigned int>(num_windows));
  CollapseLeafWindowsKernel<<<grid, SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION, 0, cuda_streams_[0]>>>(
    cuda_collapse_windows_.RawDataReadOnly(), cuda_data_indices_.RawData(),
    cuda_data_index_to_leaf_index_.RawData());
}

// selective grow-then-prune finalize: map every used row's leaf index from the
// hybrid numbering to the final classic numbering (pruned leaves map to their
// surviving ancestor's final leaf)
__global__ void RemapDataIndexToLeafIndexKernel(
  const data_size_t num_data,
  const data_size_t* cuda_data_indices,
  const int* leaf_map,
  int* cuda_data_index_to_leaf_index) {
  const data_size_t i = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < num_data) {
    const data_size_t data_index = cuda_data_indices[i];
    cuda_data_index_to_leaf_index[data_index] = leaf_map[cuda_data_index_to_leaf_index[data_index]];
  }
}

void CUDADataPartition::RemapDataIndexToLeafIndex(const std::vector<int>& leaf_map) {
  const data_size_t num_data_in_root = root_num_data();
  if (num_data_in_root <= 0 || leaf_map.empty()) {
    return;
  }
  if (cuda_leaf_index_remap_.Size() < leaf_map.size()) {
    cuda_leaf_index_remap_.Resize(std::max(leaf_map.size(), static_cast<size_t>(num_leaves_)));
  }
  CopyFromHostToCUDADevice<int>(cuda_leaf_index_remap_.RawData(), leaf_map.data(),
    leaf_map.size(), __FILE__, __LINE__);
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
    FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  RemapDataIndexToLeafIndexKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
    num_data_in_root, cuda_data_indices_.RawData(), cuda_leaf_index_remap_.RawDataReadOnly(),
    cuda_data_index_to_leaf_index_.RawData());
}

// region copy at identical offsets (src and dst share the main array layout);
// used to carry terminal (non-split) leaves' regions from the old main index
// array into the out buffer before it is swapped in as the new main array
__global__ void HybridCopyDataIndicesBatchKernel(
  const CUDAHybridApplyDescriptor* descs,
  const data_size_t* src_data_indices_param,
  data_size_t* dst_data_indices_param,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid, live GAP count)
  if (HybridGraphBeyondLiveGaps(gstate, blockIdx.y)) {
    return;
  }
  const data_size_t* src_data_indices =
    HybridGraphMainIndices(gstate, src_data_indices_param);
  data_size_t* dst_data_indices =
    HybridGraphOutIndices(gstate, dst_data_indices_param);
  const CUDAHybridApplyDescriptor d = descs[blockIdx.y];
  if (static_cast<int>(blockIdx.x) >= d.num_blocks) {
    return;
  }
  const data_size_t local_data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (local_data_index < d.num_data_in_leaf) {
    dst_data_indices[d.leaf_data_start + local_data_index] =
      src_data_indices[d.leaf_data_start + local_data_index];
  }
}

void CUDADataPartition::LaunchSplitLevelBatchedKernels(const int num_splits, const int max_num_blocks,
                                                       const int num_gaps, const int max_gap_blocks,
                                                       const int total_flat_blocks,
                                                       const bool write_leaf_map) {
  const CUDAHybridApplyDescriptor* descs = cuda_apply_descs_.RawDataReadOnly();
  (void)max_num_blocks;
  // flat 1D grid over the level's REAL chunk count (grid-stride in-kernel):
  // the old (largest leaf's blocks x num_splits) grid is mostly empty blocks
  // at skewed deep levels
  const unsigned int flat_grid = static_cast<unsigned int>(
    total_flat_blocks < 8192 ? total_flat_blocks : 8192);
  constexpr int block_dim = SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
  // out buffer of this level == new main array after the caller's swap
  data_size_t* new_main_indices = cuda_out_data_indices_in_leaf_.RawData();
  // the level is bracketed by device synchronizations (best-split readback before,
  // FinishSplitBatch after), but wait on the last recorded copy event anyway so a
  // still-in-flight per-split CopyDataIndicesKernel can never race the scratch
  if (indices_copy_done_event_ != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[0], indices_copy_done_event_, 0));
  }
  HybridGenBitVectorUpdateLeafIndexBatchKernel<<<flat_grid, block_dim, 0, cuda_streams_[0]>>>(
    descs, cuda_data_indices_.RawData(), cuda_block_to_left_offset_.RawData(),
    cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData(),
    cuda_data_index_to_leaf_index_.RawData(), nullptr, num_splits, total_flat_blocks);
  HybridAggregateBlockOffsetBatchKernel<<<num_splits, AGGREGATE_BLOCK_SIZE_DATA_PARTITION, 0, cuda_streams_[0]>>>(
    descs, cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData(),
    cuda_leaf_data_start_.RawData(), cuda_leaf_data_end_.RawData(), cuda_leaf_num_data_.RawData(),
    cuda_level_smaller_counts_.RawData(), nullptr);
  HybridSplitInnerBatchKernel<<<flat_grid, block_dim, 0, cuda_streams_[0]>>>(
    descs, cuda_data_indices_.RawData(), cuda_block_data_to_left_offset_.RawData(),
    cuda_block_data_to_right_offset_.RawData(), cuda_block_to_left_offset_.RawData(),
    new_main_indices, nullptr, num_splits, total_flat_blocks,
    cuda_data_index_to_leaf_index_.RawData(), write_leaf_map ? 1 : 0);
  if (nccl_communicator_ != nullptr) {
    // quantized multi-GPU is fenced at the NCCLGBDT layer, so only the
    // non-quantized variant needs the NCCL role/count contract here
    HybridSplitTreeStructureBatchKernel<true, false><<<num_splits, 32, 0, cuda_streams_[0]>>>(
      descs, cuda_leaf_data_start_.RawData(), cuda_leaf_num_data_.RawData(),
      new_main_indices, num_total_bin_, cuda_hist_, cuda_hist_pool_.RawData(),
      cuda_leaf_output_.RawData(), cuda_split_info_buffer_.RawData(), nullptr);
  } else if (use_quantized_grad_) {
    HybridSplitTreeStructureBatchKernel<false, true><<<num_splits, 32, 0, cuda_streams_[0]>>>(
      descs, cuda_leaf_data_start_.RawData(), cuda_leaf_num_data_.RawData(),
      new_main_indices, num_total_bin_, cuda_hist_, cuda_hist_pool_.RawData(),
      cuda_leaf_output_.RawData(), cuda_split_info_buffer_.RawData(), nullptr);
  } else {
    HybridSplitTreeStructureBatchKernel<false, false><<<num_splits, 32, 0, cuda_streams_[0]>>>(
      descs, cuda_leaf_data_start_.RawData(), cuda_leaf_num_data_.RawData(),
      new_main_indices, num_total_bin_, cuda_hist_, cuda_hist_pool_.RawData(),
      cuda_leaf_output_.RawData(), cuda_split_info_buffer_.RawData(), nullptr);
  }
  if (num_gaps > 0) {
    // gap descriptors follow the split descriptors in cuda_apply_descs_
    const dim3 gap_grid(static_cast<unsigned int>(max_gap_blocks), static_cast<unsigned int>(num_gaps));
    HybridCopyDataIndicesBatchKernel<<<gap_grid, block_dim, 0, cuda_streams_[0]>>>(
      descs + num_splits, cuda_data_indices_.RawData(), new_main_indices, nullptr);
  }
  CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_copy_done_event_, cuda_streams_[0]));
}

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
void CUDADataPartition::CaptureHybridGraphApplyKernels(
    const CUDAHybridGraphLoopState* gstate,
    std::vector<cudaGraphNode_t>* nodes,
    std::vector<int>* roles) {
  // graphs L1 body capture: the five batched apply kernels with PLACEHOLDER
  // grids (the controller resizes them per level through the device-updatable
  // node handles collected here). Parameters are the same persistent buffers
  // the host launcher uses; the level-swapping index pointers and the split
  // info slab base come from the loop state instead.
  const CUDAHybridApplyDescriptor* descs = cuda_apply_descs_.RawDataReadOnly();
  constexpr int block_dim = SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
  cudaStream_t stream = cuda_streams_[0];
  HybridGenBitVectorUpdateLeafIndexBatchKernel<<<dim3(1, 1), block_dim, 0, stream>>>(
    descs, cuda_data_indices_.RawData(), cuda_block_to_left_offset_.RawData(),
    cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData(),
    cuda_data_index_to_leaf_index_.RawData(), gstate, /*num_split_descs=*/0,
    /*total_flat_blocks=*/0);
  if (!AppendCapturedNode(stream, nodes)) return;
  roles->push_back(kHybridGraphNodeGenBitVector);
  HybridAggregateBlockOffsetBatchKernel<<<1, AGGREGATE_BLOCK_SIZE_DATA_PARTITION, 0, stream>>>(
    descs, cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData(),
    cuda_leaf_data_start_.RawData(), cuda_leaf_data_end_.RawData(), cuda_leaf_num_data_.RawData(),
    cuda_level_smaller_counts_.RawData(), gstate);
  if (!AppendCapturedNode(stream, nodes)) return;
  roles->push_back(kHybridGraphNodeAggregate);
  HybridSplitInnerBatchKernel<<<dim3(1, 1), block_dim, 0, stream>>>(
    descs, cuda_data_indices_.RawData(), cuda_block_data_to_left_offset_.RawData(),
    cuda_block_data_to_right_offset_.RawData(), cuda_block_to_left_offset_.RawData(),
    cuda_out_data_indices_in_leaf_.RawData(), gstate, /*num_split_descs=*/0,
    /*total_flat_blocks=*/0, cuda_data_index_to_leaf_index_.RawData(),
    /*write_leaf_map=*/0);
  if (!AppendCapturedNode(stream, nodes)) return;
  roles->push_back(kHybridGraphNodeSplitInner);
  // template selection mirrors LaunchSplitLevelBatchedKernels (the quantized
  // variant also writes the child structs' packed int64 gradient/hessian sums)
  if (use_quantized_grad_) {
    // graph flow is single-GPU only (HybridGraphPrefixUsable), no NCCL variant
    HybridSplitTreeStructureBatchKernel<false, true><<<1, 32, 0, stream>>>(
      descs, cuda_leaf_data_start_.RawData(), cuda_leaf_num_data_.RawData(),
      cuda_out_data_indices_in_leaf_.RawData(), num_total_bin_, cuda_hist_,
      cuda_hist_pool_.RawData(), cuda_leaf_output_.RawData(),
      cuda_split_info_buffer_.RawData(), gstate);
  } else {
    HybridSplitTreeStructureBatchKernel<false, false><<<1, 32, 0, stream>>>(
      descs, cuda_leaf_data_start_.RawData(), cuda_leaf_num_data_.RawData(),
      cuda_out_data_indices_in_leaf_.RawData(), num_total_bin_, cuda_hist_,
      cuda_hist_pool_.RawData(), cuda_leaf_output_.RawData(),
      cuda_split_info_buffer_.RawData(), gstate);
  }
  if (!AppendCapturedNode(stream, nodes)) return;
  roles->push_back(kHybridGraphNodeTreeStructure);
  HybridCopyDataIndicesBatchKernel<<<dim3(1, 1), block_dim, 0, stream>>>(
    descs + kHybridGraphGapDescBase, cuda_data_indices_.RawData(),
    cuda_out_data_indices_in_leaf_.RawData(), gstate);
  if (!AppendCapturedNode(stream, nodes)) return;
  roles->push_back(kHybridGraphNodeCopyGaps);
}

void CUDADataPartition::EnsureHybridGraphCapacity(const data_size_t max_root_num_data) {
  // apply descriptors: split slots [0, kGapDescBase) + gap slots after
  const size_t desc_capacity = static_cast<size_t>(kHybridGraphGapDescBase) +
    static_cast<size_t>(kHybridGraphMaxSplitsPerLevel) + 2;
  if (cuda_apply_descs_.Size() < desc_capacity) {
    cuda_apply_descs_.Resize(desc_capacity);
  }
  // worst-case per-level block-offset slots: the split regions partition at
  // most the whole root window, plus one sentinel slot per split
  const data_size_t worst_blocks =
    (max_root_num_data + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
      SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION +
    static_cast<data_size_t>(num_leaves_) + 4;
  if (cuda_block_data_to_left_offset_.Size() < static_cast<size_t>(worst_blocks)) {
    cuda_block_data_to_left_offset_.Resize(static_cast<size_t>(worst_blocks));
    cuda_block_data_to_right_offset_.Resize(static_cast<size_t>(worst_blocks));
    SetCUDAMemory<data_size_t>(cuda_block_data_to_left_offset_.RawData(), 0, cuda_block_data_to_left_offset_.Size(), __FILE__, __LINE__);
    SetCUDAMemory<data_size_t>(cuda_block_data_to_right_offset_.RawData(), 0, cuda_block_data_to_right_offset_.Size(), __FILE__, __LINE__);
  }
}

void CUDADataPartition::FinishHybridGraphLevels(const int num_levels, const int total_splits) {
  cur_num_leaves_ += total_splits;
  // the graph loop swapped the device roles of the two index buffers once per
  // applied level; realign the host wrappers
  if ((num_levels & 1) != 0) {
    cuda_data_indices_.Swap(&cuda_out_data_indices_in_leaf_);
  }
}
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

// One pass per tree: write the row -> leaf map from the final leaf windows
// (sequential window reads + one 4-byte scatter per row, instead of the same
// scatter at EVERY level of the batched apply). blockIdx.y = leaf; grid-stride
// over the leaf's window.
__global__ void MaterializeLeafMapKernel(
  const data_size_t* cuda_data_indices,
  const data_size_t* cuda_leaf_data_start,
  const data_size_t* cuda_leaf_num_data,
  int* cuda_data_index_to_leaf_index) {
  const int leaf = static_cast<int>(blockIdx.y);
  const data_size_t start = cuda_leaf_data_start[leaf];
  const data_size_t num = cuda_leaf_num_data[leaf];
  const data_size_t stride = static_cast<data_size_t>(gridDim.x) * blockDim.x;
  for (data_size_t j = static_cast<data_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < num; j += stride) {
    cuda_data_index_to_leaf_index[cuda_data_indices[start + j]] = leaf;
  }
}

__global__ void MaterializeLeafMapSubsetKernel(
  const data_size_t* cuda_data_indices,
  const data_size_t* cuda_leaf_data_start,
  const data_size_t* cuda_leaf_num_data,
  const int* leaf_list,
  int* cuda_data_index_to_leaf_index) {
  const int leaf = leaf_list[blockIdx.y];
  const data_size_t start = cuda_leaf_data_start[leaf];
  const data_size_t num = cuda_leaf_num_data[leaf];
  const data_size_t stride = static_cast<data_size_t>(gridDim.x) * blockDim.x;
  for (data_size_t j = static_cast<data_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < num; j += stride) {
    cuda_data_index_to_leaf_index[cuda_data_indices[start + j]] = leaf;
  }
}

void CUDADataPartition::MaterializeHybridLeafMapSubset(const std::vector<int>& leaves) {
  if (leaves.empty()) {
    return;
  }
  global_timer.Start("CUDADataPartition::MaterializeHybridLeafMap");
  if (cuda_materialize_leaf_list_.Size() < leaves.size()) {
    cuda_materialize_leaf_list_.Resize(leaves.size());
  }
  CopyFromHostToCUDADevice<int>(cuda_materialize_leaf_list_.RawData(), leaves.data(),
                                leaves.size(), __FILE__, __LINE__);
  dim3 grid(64, static_cast<unsigned int>(leaves.size()));
  MaterializeLeafMapSubsetKernel<<<grid, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
    cuda_data_indices_.RawData(), cuda_leaf_data_start_.RawData(),
    cuda_leaf_num_data_.RawData(), cuda_materialize_leaf_list_.RawData(),
    cuda_data_index_to_leaf_index_.RawData());
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::MaterializeHybridLeafMap");
}

void CUDADataPartition::MaterializeHybridLeafMap(const int num_leaves) {
  if (num_leaves <= 0) {
    return;
  }
  global_timer.Start("CUDADataPartition::MaterializeHybridLeafMap");
  const int blocks_x = 64;
  dim3 grid(blocks_x, num_leaves);
  MaterializeLeafMapKernel<<<grid, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
    cuda_data_indices_.RawData(), cuda_leaf_data_start_.RawData(),
    cuda_leaf_num_data_.RawData(), cuda_data_index_to_leaf_index_.RawData());
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::MaterializeHybridLeafMap");
}

template <bool USE_BAGGING>
__global__ void AddPredictionToScoreKernel(
  const data_size_t* data_indices_in_leaf,
  const double* leaf_value, double* cuda_scores,
  const int* cuda_data_index_to_leaf_index, const data_size_t num_data) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockIdx_x = blockIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const data_size_t local_data_index = static_cast<data_size_t>(blockIdx_x * blockDim_x + threadIdx_x);
  if (local_data_index < num_data) {
    if (USE_BAGGING) {
      const data_size_t global_data_index = data_indices_in_leaf[local_data_index];
      const int leaf_index = cuda_data_index_to_leaf_index[global_data_index];
      const double leaf_prediction_value = leaf_value[leaf_index];
      cuda_scores[global_data_index] += leaf_prediction_value;
    } else {
      const int leaf_index = cuda_data_index_to_leaf_index[local_data_index];
      const double leaf_prediction_value = leaf_value[leaf_index];
      cuda_scores[local_data_index] += leaf_prediction_value;
    }
  }
}

void CUDADataPartition::LaunchAddPredictionToScoreKernel(const double* leaf_value, double* cuda_scores) {
  global_timer.Start("CUDADataPartition::AddPredictionToScoreKernel");
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  if (use_bagging_) {
    AddPredictionToScoreKernel<true><<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
      cuda_data_indices_.RawData(), leaf_value, cuda_scores, cuda_data_index_to_leaf_index_.RawData(), num_data_in_root);
  } else {
    AddPredictionToScoreKernel<false><<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
      cuda_data_indices_.RawData(), leaf_value, cuda_scores, cuda_data_index_to_leaf_index_.RawData(), num_data_in_root);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::AddPredictionToScoreKernel");
}

// Multi-block-per-leaf reduction: each leaf is covered by RENEW_BLOCKS_PER_LEAF
// blocks so that shallow trees (few leaves) still saturate the GPU. Blocks
// grid-stride over their leaf's data slice and atomicAdd their partial sum
// into the (pre-zeroed) leaf buffers. Double summation is not associative, so
// the reduced value can differ in the last ULPs from the single-block version;
// this only affects quant_train_renew_leaf runs, which carry no md5 lock.
__global__ void RenewDiscretizedTreeLeavesKernel(
  const score_t* gradients,
  const score_t* hessians,
  const data_size_t* data_indices,
  const data_size_t* leaf_data_start,
  const data_size_t* leaf_num_data,
  double* leaf_grad_stat_buffer,
  double* leaf_hess_stat_buffer,
  double* /*leaf_values*/) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const int leaf_index = static_cast<int>(blockIdx.x) / RENEW_BLOCKS_PER_LEAF;
  const int block_in_leaf = static_cast<int>(blockIdx.x) % RENEW_BLOCKS_PER_LEAF;
  const data_size_t* data_indices_in_leaf = data_indices + leaf_data_start[leaf_index];
  const data_size_t num_data_in_leaf = leaf_num_data[leaf_index];
  double sum_gradients = 0.0;
  double sum_hessians = 0.0;
  const data_size_t stride = static_cast<data_size_t>(blockDim.x) * RENEW_BLOCKS_PER_LEAF;
  for (data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.x) +
         static_cast<data_size_t>(block_in_leaf) * static_cast<data_size_t>(blockDim.x);
       inner_data_index < num_data_in_leaf; inner_data_index += stride) {
    const data_size_t data_index = data_indices_in_leaf[inner_data_index];
    sum_gradients += static_cast<double>(gradients[data_index]);
    sum_hessians += static_cast<double>(hessians[data_index]);
  }
  sum_gradients = ShuffleReduceSum<double>(sum_gradients, shared_mem_buffer, blockDim.x);
  __syncthreads();
  sum_hessians = ShuffleReduceSum<double>(sum_hessians, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    atomicAdd(leaf_grad_stat_buffer + leaf_index, sum_gradients);
    atomicAdd(leaf_hess_stat_buffer + leaf_index, sum_hessians);
  }
}

void CUDADataPartition::LaunchReduceLeafGradStat(
  const score_t* gradients, const score_t* hessians,
  CUDATree* tree, double* leaf_grad_stat_buffer, double* leaf_hess_state_buffer) const {
  const int num_leaves = tree->num_leaves();
  // atomicAdd accumulation requires zeroed buffers
  SetCUDAMemory<double>(leaf_grad_stat_buffer, 0, static_cast<size_t>(num_leaves), __FILE__, __LINE__);
  SetCUDAMemory<double>(leaf_hess_state_buffer, 0, static_cast<size_t>(num_leaves), __FILE__, __LINE__);
  const int num_blocks = num_leaves * RENEW_BLOCKS_PER_LEAF;
  RenewDiscretizedTreeLeavesKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
    gradients,
    hessians,
    cuda_data_indices_.RawData(),
    cuda_leaf_data_start_.RawData(),
    cuda_leaf_num_data_.RawData(),
    leaf_grad_stat_buffer,
    leaf_hess_state_buffer,
    tree->cuda_leaf_value_ref());
}

}  // namespace Falcata

#endif  // USE_CUDA
