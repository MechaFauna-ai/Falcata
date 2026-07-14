/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_rocm_interop.h>

#include <algorithm>
#include <vector>

namespace LightGBM {

// =====================================================================
// Compaction kernel: copies sampled (used) columns from the partitioned
// row-major bin matrix into a compact partitioned buffer.
//
// Source layout (per partition p):
//   src_data[partition_byte_offset[p] + row * src_stride[p] + col_in_partition]
// where partition_byte_offset[p] = src_partition_column_start[p] * num_data
//       src_stride[p]            = src_num_columns_in_partition[p]
//
// Compact layout (per partition p, packed contiguous over USED columns):
//   compact_data[compact_byte_offset[p] + row * compact_stride[p] + i_in_partition]
// where compact_byte_offset[p] = compact_partition_column_start[p] * num_data
//       compact_stride[p]      = num_used_in_partition[p]
//
// One thread copies one (row, compact_col) pair. Grid is sized as
// (ceil(total_compact_cols / TX), ceil(num_data / TY)).
// =====================================================================
// Per-slot precomputed metadata fill kernel. Writes row-major-in-partition
// output. Block is 32 slots × 32 rows. Each thread copies bytes down a column
// to keep grid_y under CUDA's 65535 limit on large datasets.
__global__ void CUDAFillCompactDataKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ compact_data,
  const size_t* __restrict__ slot_src_byte,
  const int* __restrict__ slot_src_stride,
  const size_t* __restrict__ slot_dst_byte,
  const int* __restrict__ slot_dst_stride,
  const int total_compact_cols,
  const data_size_t num_data,
  uint8_t* __restrict__ colmajor_out) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= total_compact_cols) return;
  const size_t src_byte = slot_src_byte[slot];
  const size_t src_stride = static_cast<size_t>(slot_src_stride[slot]);
  const size_t dst_byte = slot_dst_byte[slot];
  const size_t dst_stride = static_cast<size_t>(slot_dst_stride[slot]);
  const size_t colmajor_base = static_cast<size_t>(slot) * static_cast<size_t>(num_data);
  const data_size_t row_stride = static_cast<data_size_t>(gridDim.y) * static_cast<data_size_t>(blockDim.y);
  for (data_size_t row = blockIdx.y * blockDim.y + threadIdx.y; row < num_data; row += row_stride) {
    const uint8_t val = src_data[src_byte + static_cast<size_t>(row) * src_stride];
    compact_data[dst_byte + static_cast<size_t>(row) * dst_stride] = val;
    if (colmajor_out != nullptr) {
      // fused second output: the tree learner's column-major compact view
      // (compact_col_buf[slot * num_data + row]), produced from the same source
      // read so the full bin matrix streams through L2 only once per tree
      colmajor_out[colmajor_base + static_cast<size_t>(row)] = val;
    }
  }
}

// Transpose row-major-in-partition source data into column-major compact buffer.
// Source: cuda_data_uint8_t_ (row-major-in-partition).
// Dest: compact_col_buf, layout = compact_col_buf[slot * num_data + row].
//
// Args:
//   src_data:                       cuda_data_uint8_t_
//   compact_col_buf:                destination
//   src_partition_column_offsets:   [P+1] cumulative source col counts (partition byte offset = col_offset * num_data)
//   src_partition_stride:           [P]   columns per source partition (byte stride per row)
//   slot_for_col:                   [num_total_cols] -> compact slot, or -1 if not in sample
//   num_data, num_partitions
// Per-slot precomputed source-frame metadata: for compact slot s,
//   slot_p_byte[s] = partition_byte_offset for that slot's source column
//   slot_p_stride[s] = partition row stride
//   slot_col_in_p[s] = column index within partition
template <bool IS_4BIT>
__global__ void CUDARowToColCompactKernel(
    const uint8_t* __restrict__ src_data,
    uint8_t* __restrict__ compact_col_buf,
    const size_t* __restrict__ slot_p_byte,
    const int* __restrict__ slot_p_stride,
    const int* __restrict__ slot_col_in_p,
    const int num_compact_cols,
    const data_size_t num_data) {
  // Block is (32 rows, 32 slots). Each block tile-transposes a 32×32 chunk via
  // shared memory: coalesced reads of 32 contiguous slots × 1 row, coalesced
  // writes of 1 slot × 32 contiguous rows. To stay under CUDA's 65535 grid_y
  // limit on large datasets (>~2M rows), grid_y is capped and the block strides
  // down its column.
  __shared__ uint8_t tile[32][33];

  const int slot_block = blockIdx.x * 32;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const data_size_t y_stride = static_cast<data_size_t>(gridDim.y) * 32;

  for (data_size_t row_block = static_cast<data_size_t>(blockIdx.y) * 32;
       row_block < num_data; row_block += y_stride) {
    // Phase 1: load src[row_block+ty, slot_block+tx] -> tile[ty][tx].
    {
      const int slot = slot_block + tx;
      const data_size_t row = row_block + ty;
      uint8_t val = 0;
      if (slot < num_compact_cols && row < num_data) {
        const int col_in_p = slot_col_in_p[slot];
        if (IS_4BIT) {
          // packed source: slot_p_byte/slot_p_stride are the partition's packed
          // byte base/row width; column j sits in byte (j >> 1), nibble (j & 1)
          const size_t base = slot_p_byte[slot]
                            + static_cast<size_t>(row) * static_cast<size_t>(slot_p_stride[slot])
                            + static_cast<size_t>(col_in_p >> 1);
          val = (src_data[base] >> ((col_in_p & 1) << 2)) & 0xf;
        } else {
          const size_t base = slot_p_byte[slot]
                            + static_cast<size_t>(row) * static_cast<size_t>(slot_p_stride[slot])
                            + static_cast<size_t>(col_in_p);
          val = src_data[base];
        }
      }
      tile[ty][tx] = val;
    }
    __syncthreads();

    // Phase 2: write compact[slot_block+ty, row_block+tx] = tile[tx][ty].
    {
      const int slot = slot_block + ty;
      const data_size_t row = row_block + tx;
      if (slot < num_compact_cols && row < num_data) {
        compact_col_buf[static_cast<size_t>(slot) * static_cast<size_t>(num_data) + static_cast<size_t>(row)] = tile[tx][ty];
      }
    }
    __syncthreads();
  }
}

void LaunchRowToColCompactKernel(
    cudaStream_t stream,
    const uint8_t* src_data,
    uint8_t* compact_col_buf,
    const size_t* slot_p_byte,
    const int* slot_p_stride,
    const int* slot_col_in_p,
    int num_compact_cols,
    data_size_t num_data,
    bool src_is_4bit) {
  const int TX = 32;
  const int TY = 32;
  // Cap grid_y at 32k to stay under CUDA's 65535 limit; kernel strides over rows.
  int grid_y = (num_data + TY - 1) / TY;
  if (grid_y > 32768) grid_y = 32768;
  dim3 block_dim(TX, TY);
  dim3 grid_dim((num_compact_cols + TX - 1) / TX, grid_y);
  if (src_is_4bit) {
    CUDARowToColCompactKernel<true><<<grid_dim, block_dim, 0, stream>>>(
        src_data, compact_col_buf, slot_p_byte, slot_p_stride, slot_col_in_p,
        num_compact_cols, num_data);
  } else {
    CUDARowToColCompactKernel<false><<<grid_dim, block_dim, 0, stream>>>(
        src_data, compact_col_buf, slot_p_byte, slot_p_stride, slot_col_in_p,
        num_compact_cols, num_data);
  }
}

// Transpose column-major staging into row-major-in-partition compact_data.
// staging layout: staging[c * num_data + r] for compact col c (in partition order).
// dst layout:     compact[part_offset[p] + r * stride[p] + c_in_p].
//
// One thread = one (compact_col, row) cell. Threads in same warp have consecutive
// compact_col → consecutive byte writes in dst (within partition; coalesced).
// Reads from staging are stride num_data per col → 32 different cache lines per warp.
// But staging is GPU-resident (HBM, 1.79 TB/s bandwidth) so this is fine.
__global__ void CUDATransposeColMajorToRowMajorKernel(
    const uint8_t* __restrict__ staging,        // col-major: [c * num_data + r]
    uint8_t* __restrict__ compact_data,         // row-major-in-partition
    const int* __restrict__ partition_for_compact,
    const int* __restrict__ compact_partition_column_offsets,
    const data_size_t num_data,
    const int total_compact_cols) {
  const int compact_col = blockIdx.x * blockDim.x + threadIdx.x;
  const data_size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (compact_col >= total_compact_cols) return;
  if (row >= num_data) return;

  const int p = partition_for_compact[compact_col];
  const int p_compact_start = compact_partition_column_offsets[p];
  const int compact_col_in_p = compact_col - p_compact_start;
  const int compact_stride_p = compact_partition_column_offsets[p + 1] - p_compact_start;
  const size_t compact_part_byte_offset = static_cast<size_t>(p_compact_start) * static_cast<size_t>(num_data);

  const uint8_t val = staging[static_cast<size_t>(compact_col) * static_cast<size_t>(num_data) + row];
  compact_data[compact_part_byte_offset + static_cast<size_t>(row) * static_cast<size_t>(compact_stride_p) + compact_col_in_p] = val;
}

void LaunchTransposeColMajorToRowMajor(
    cudaStream_t stream,
    const uint8_t* staging,
    uint8_t* compact_data,
    const int* partition_for_compact,
    const int* compact_partition_column_offsets,
    data_size_t num_data,
    int total_compact_cols) {
  const int TX = 8;
  const int TY = 128;  // grid_y under 65535 for 8M rows
  dim3 block_dim(TX, TY);
  dim3 grid_dim((total_compact_cols + TX - 1) / TX, (num_data + TY - 1) / TY);
  CUDATransposeColMajorToRowMajorKernel<<<grid_dim, block_dim, 0, stream>>>(
      staging, compact_data, partition_for_compact, compact_partition_column_offsets,
      num_data, total_compact_cols);
}

// Interleave the per-row gradient and hessian arrays into float2 pairs so the
// scattered per-row reads of the dense construct kernels touch one 32B sector
// per row instead of two. Bit-identical values, purely a layout change.
__global__ void InterleaveGradHessKernel(
  const score_t* __restrict__ gradients,
  const score_t* __restrict__ hessians,
  float2* __restrict__ gradients_hessians,
  const data_size_t num_data) {
  const data_size_t i = static_cast<data_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < num_data) {
    gradients_hessians[i] = make_float2(static_cast<float>(gradients[i]),
                                        static_cast<float>(hessians[i]));
  }
}

// Host wrapper called from cuda_histogram_constructor.cpp. Legacy default
// stream: ordered before subsequently enqueued work on the blocking streams.
void LaunchInterleaveGradHessKernel(
  const score_t* gradients,
  const score_t* hessians,
  float2* gradients_hessians,
  data_size_t num_data) {
  const int block_size = 1024;
  const int num_blocks = static_cast<int>((num_data + block_size - 1) / block_size);
  InterleaveGradHessKernel<<<num_blocks, block_size>>>(
    gradients, hessians, gradients_hessians, num_data);
}

// Diagnostic kernel: read N bytes from a (possibly host-mapped) source pointer.
__global__ void DiagReadKernel(const uint8_t* __restrict__ src, uint8_t* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
void LaunchDiagRead(cudaStream_t stream, const uint8_t* src, uint8_t* dst, int n) {
  DiagReadKernel<<<(n + 31) / 32, 32, 0, stream>>>(src, dst, n);
}

// Host wrapper called from cuda_histogram_constructor.cpp.
void LaunchFillCompactDataKernel(
  cudaStream_t stream,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* slot_src_byte,
  const int* slot_src_stride,
  const size_t* slot_dst_byte,
  const int* slot_dst_stride,
  int total_compact_cols,
  data_size_t num_data,
  uint8_t* colmajor_out) {
  const int TX = 32;
  const int TY = 32;
  // Cap grid_y at 32k so we stay well under CUDA's 65535 limit; the kernel
  // strides each thread down the column to cover all rows.
  int grid_y = (num_data + TY - 1) / TY;
  if (grid_y > 32768) grid_y = 32768;
  dim3 block_dim(TX, TY);
  dim3 grid_dim((total_compact_cols + TX - 1) / TX, grid_y);
  CUDAFillCompactDataKernel<<<grid_dim, block_dim, 0, stream>>>(
    src_data,
    compact_data,
    slot_src_byte,
    slot_src_stride,
    slot_dst_byte,
    slot_dst_stride,
    total_compact_cols,
    num_data,
    colmajor_out);
}

// 4-bit variant of the compaction: both the SOURCE (full packed bin matrix) and
// the DESTINATION (packed compact matrix) hold two columns per byte. One thread
// owns one (destination byte, row) cell, i.e. a PAIR of adjacent compact slots,
// so no two threads read-modify-write the same output byte. Source positions are
// precomputed per byte-slot as NIBBLE indices (byte * 2 + nibble): the base
// nibble of each of the two source columns plus a per-row nibble stride
// (2 * packed source row width). bs_src_nib1 == SIZE_MAX marks the padding
// nibble of an odd-width partition (writes 0).
__global__ void CUDAFillCompactData4BitKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ compact_data,
  const size_t* __restrict__ bs_src_nib0,
  const size_t* __restrict__ bs_src_nib1,
  const int* __restrict__ bs_src_stride_nib,
  const size_t* __restrict__ bs_dst_byte,
  const int* __restrict__ bs_dst_stride,
  const int total_byte_slots,
  const data_size_t num_data) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= total_byte_slots) return;
  const size_t src_nib0 = bs_src_nib0[slot];
  const size_t src_nib1 = bs_src_nib1[slot];
  const size_t stride_nib = static_cast<size_t>(bs_src_stride_nib[slot]);
  const size_t dst_byte = bs_dst_byte[slot];
  const size_t dst_stride = static_cast<size_t>(bs_dst_stride[slot]);
  const bool has_hi = src_nib1 != ~static_cast<size_t>(0);
  const data_size_t row_stride = static_cast<data_size_t>(gridDim.y) * static_cast<data_size_t>(blockDim.y);
  for (data_size_t row = blockIdx.y * blockDim.y + threadIdx.y; row < num_data; row += row_stride) {
    const size_t nib0 = src_nib0 + static_cast<size_t>(row) * stride_nib;
    const uint8_t lo = (src_data[nib0 >> 1] >> ((nib0 & 1) << 2)) & 0xf;
    uint8_t hi = 0;
    if (has_hi) {
      const size_t nib1 = src_nib1 + static_cast<size_t>(row) * stride_nib;
      hi = (src_data[nib1 >> 1] >> ((nib1 & 1) << 2)) & 0xf;
    }
    compact_data[dst_byte + static_cast<size_t>(row) * dst_stride] = static_cast<uint8_t>(lo | (hi << 4));
  }
}

// Host wrapper called from cuda_histogram_constructor.cpp.
void LaunchFillCompactData4BitKernel(
  cudaStream_t stream,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* bs_src_nib0,
  const size_t* bs_src_nib1,
  const int* bs_src_stride_nib,
  const size_t* bs_dst_byte,
  const int* bs_dst_stride,
  int total_byte_slots,
  data_size_t num_data) {
  const int TX = 32;
  const int TY = 32;
  int grid_y = (num_data + TY - 1) / TY;
  if (grid_y > 32768) grid_y = 32768;
  dim3 block_dim(TX, TY);
  dim3 grid_dim((total_byte_slots + TX - 1) / TX, grid_y);
  CUDAFillCompactData4BitKernel<<<grid_dim, block_dim, 0, stream>>>(
    src_data, compact_data, bs_src_nib0, bs_src_nib1, bs_src_stride_nib,
    bs_dst_byte, bs_dst_stride, total_byte_slots, num_data);
}

// Column-major-in-partition variant of the dense histogram kernel.
// Used by the CompactView host-mapped path on large datasets, where compact_data
// is laid out column-major-per-partition for cheap per-column cudaMemcpy fill.
template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramDenseColMajorKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  // data is column-major-in-partition: column c at offset partition_column_start + threadIdx.x
  // starts at byte (partition_column_start + threadIdx.x) * num_data.
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(partition_column_start) * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    HIST_TYPE* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    // Column-major: this thread's column starts at data_ptr + threadIdx.x * num_data.
    const BIN_TYPE* col_ptr = data_ptr + static_cast<size_t>(threadIdx.x) * static_cast<size_t>(num_data);
    for (data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y); inner_data_index < block_num_data; inner_data_index += blockDim.y) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(col_ptr[data_index]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

// Shared body of the dense histogram kernel; the shared-memory histogram is
// declared by the calling __global__ kernel and passed in so a kernel that
// instantiates the helper more than once does not duplicate the allocation.
// Blocks whose row range lies beyond this leaf's data exit before touching
// shared/global memory (they would only add zeros); this keeps over-provisioned
// batched grids (sized for the level's largest pair) cheap. dim_y is the row
// grouping extent: gridDim.y * blockDim.y in the classic flow, or the
// device-computed effective value in the speculative single-sync flow (whose
// launch grid is only an upper bound).
/*! \brief bin cap of the register-accumulation construct body (USE_REG_BINS):
 *  active only when EVERY feature has at most this many bins (host-gated) */
#define kRegHistMaxBins (8)

// 4-bit packed dense bin matrix (IS_4BIT construct variants): the partition's
// packed row width is ceil(num_columns_in_partition / 2) bytes (columns padded
// to an even count PER PARTITION, so each row segment is byte-aligned); column
// j lives in byte (j >> 1), nibble (j & 1), low nibble = even column. The
// packed per-partition byte-width prefix comes in via
// packed_partition_byte_offsets (nullptr and unused in the 8-bit variants).
template <bool IS_4BIT, typename BIN_TYPE>
__device__ __forceinline__ uint32_t ReadDenseBin(
  const BIN_TYPE* row_ptr, const unsigned int column_in_partition) {
  if (IS_4BIT) {
    const uint32_t packed = static_cast<uint32_t>(row_ptr[column_in_partition >> 1]);
    return (packed >> ((column_in_partition & 1) << 2)) & 0xfu;
  } else {
    return static_cast<uint32_t>(row_ptr[column_in_partition]);
  }
}

// USE_GH2: read the per-row (gradient, hessian) pair from cuda_gh, the
// interleaved float2 copy of the two score_t arrays (see GHInterleaveEnabled):
// same bits, one scattered 32B sector per row instead of two. Compile-time so
// the non-interleaved instantiation stays byte-identical to the historical
// kernel (a runtime branch measurably raised the kernel's latency).
template <typename BIN_TYPE, typename HIST_TYPE, bool USE_REG_BINS = false, bool IS_4BIT = false, bool USE_GH2 = false>
__device__ __forceinline__ void ConstructHistogramDenseInner(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  HIST_TYPE* shared_hist,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const float2* cuda_gh,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used_bytree,
  const data_size_t num_data,
  const int dim_y,
  const uint8_t* bin_used = nullptr) {
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  if (block_start >= num_data_in_smaller_leaf) {
    return;
  }
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const int row_stride = IS_4BIT ?
    packed_partition_byte_offsets[blockIdx.x + 1] - packed_partition_byte_offsets[blockIdx.x] :
    num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(
    IS_4BIT ? packed_partition_byte_offsets[blockIdx.x] : partition_column_start) * num_data;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  // bin_used (per-tree feature_fraction bin mask, may be null) skips the zeroing
  // and global merge of histogram entries belonging to features outside this
  // tree's sample: only used columns accumulate into shared memory, and unused
  // global entries are dead storage this tree, so both loops may skip them.
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    if (bin_used == nullptr || bin_used[partition_hist_start + (i >> 1)]) {
      shared_hist[i] = 0.0f;
    }
  }
  __syncthreads();
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  // Skip features that are not in the per-tree feature_fraction sample.
  // is_feature_used_bytree may be nullptr (no sampling); treat as all-used.
  const bool feat_used = (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) &&
      (is_feature_used_bytree == nullptr || is_feature_used_bytree[column_index]);
  if (feat_used) {
    HIST_TYPE* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    if (USE_REG_BINS) {
      // Few-bin datasets (every feature <= kRegHistMaxBins bins): accumulate the
      // thread's rows into registers and flush once, instead of two same-address
      // shared atomics per row. With ~7 bins and blockDim.y threads per column
      // the per-row atomics serialize heavily; the register accumulation is
      // contention-free. Float accumulation ORDER differs from the atomic
      // per-row order, so this path is non-quantized-only (quality-parity, not
      // bit-parity, is the contract for non-quantized training).
      HIST_TYPE reg_grad[kRegHistMaxBins];
      HIST_TYPE reg_hess[kRegHistMaxBins];
#pragma unroll
      for (int b = 0; b < kRegHistMaxBins; ++b) {
        reg_grad[b] = 0.0f;
        reg_hess[b] = 0.0f;
      }
      for (data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y); inner_data_index < block_num_data; inner_data_index += blockDim.y) {
        const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
        score_t grad, hess;
        if (USE_GH2) {
          const float2 gh = cuda_gh[data_index];
          grad = gh.x;
          hess = gh.y;
        } else {
          grad = cuda_gradients[data_index];
          hess = cuda_hessians[data_index];
        }
        const uint32_t bin = ReadDenseBin<IS_4BIT>(data_ptr + static_cast<size_t>(data_index) * row_stride, threadIdx.x);
#pragma unroll
        for (int b = 0; b < kRegHistMaxBins; ++b) {
          if (bin == static_cast<uint32_t>(b)) {
            reg_grad[b] += grad;
            reg_hess[b] += hess;
          }
        }
      }
#pragma unroll
      for (int b = 0; b < kRegHistMaxBins; ++b) {
        // (0, 0) sums are no-ops; skipping them saves most of the flush atomics
        if (reg_grad[b] != 0.0f || reg_hess[b] != 0.0f) {
          atomicAdd_block(shared_hist_ptr + (b << 1), reg_grad[b]);
          atomicAdd_block(shared_hist_ptr + (b << 1) + 1, reg_hess[b]);
        }
      }
    } else {
      for (data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y); inner_data_index < block_num_data; inner_data_index += blockDim.y) {
        const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
        score_t grad, hess;
        if (USE_GH2) {
          const float2 gh = cuda_gh[data_index];
          grad = gh.x;
          hess = gh.y;
        } else {
          grad = cuda_gradients[data_index];
          hess = cuda_hessians[data_index];
        }
        const uint32_t bin = ReadDenseBin<IS_4BIT>(data_ptr + static_cast<size_t>(data_index) * row_stride, threadIdx.x);
        const uint32_t pos = bin << 1;
        HIST_TYPE* pos_ptr = shared_hist_ptr + pos;
        atomicAdd_block(pos_ptr, grad);
        atomicAdd_block(pos_ptr + 1, hess);
      }
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    if (bin_used == nullptr || bin_used[partition_hist_start + (i >> 1)]) {
      atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
    }
  }
}

template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE, bool IS_4BIT = false>
__global__ void CUDAConstructHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used_bytree,
  const data_size_t num_data) {
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  ConstructHistogramDenseInner<BIN_TYPE, HIST_TYPE, false, IS_4BIT>(
    smaller_leaf_splits, shared_hist, cuda_gradients, cuda_hessians, nullptr, data,
    column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
    packed_partition_byte_offsets,
    is_feature_used_bytree, num_data, static_cast<int>(gridDim.y * blockDim.y));
}

// Computes the effective row-grouping extent (dim_y) of the speculative batched
// construct launch: the exact host sizing formula applied to the level's ACTUAL
// smaller-child sizes (written by the batched apply's aggregate kernel), so the
// row grouping -- and hence the float histograms -- are bit-identical to the
// classic host sizing. One tiny single-block launch per level; the construct
// blocks then read a single int.
__global__ void ComputeBatchedConstructDimYKernel(
  const data_size_t* level_smaller_num_data,
  const int num_pairs,
  const int block_dim_y,
  const int min_grid_dim_y,
  const int min_rows_per_thread,
  const int saturation_floor_total,
  int* out_dim_y) {
  __shared__ data_size_t shared_max[32];
  data_size_t thread_max = 0;
  for (int i = static_cast<int>(threadIdx.x); i < num_pairs; i += static_cast<int>(blockDim.x)) {
    const data_size_t n = level_smaller_num_data[i];
    if (n > thread_max) {
      thread_max = n;
    }
  }
  const uint32_t warp_id = threadIdx.x / warpSize;
  const uint32_t lane = threadIdx.x % warpSize;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    const data_size_t other = __shfl_down_sync(0xffffffffu, thread_max, offset);
    if (other > thread_max) {
      thread_max = other;
    }
  }
  if (lane == 0) {
    shared_max[warp_id] = thread_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    data_size_t max_num_data_in_smaller_leaf = 0;
    const uint32_t num_warps = (blockDim.x + warpSize - 1) / warpSize;
    for (uint32_t w = 0; w < num_warps; ++w) {
      if (shared_max[w] > max_num_data_in_smaller_leaf) {
        max_num_data_in_smaller_leaf = shared_max[w];
      }
    }
    out_dim_y[0] = HybridBatchedConstructGridDimY(
      max_num_data_in_smaller_leaf, num_pairs, block_dim_y,
      min_grid_dim_y, min_rows_per_thread, saturation_floor_total) * block_dim_y;
  }
}

void CUDAHistogramConstructor::LaunchComputeBatchedConstructDimYKernel(
  const data_size_t* level_smaller_num_data,
  const int num_pairs,
  const int block_dim_y) {
  ComputeBatchedConstructDimYKernel<<<1, 128, 0, cuda_stream_>>>(
    level_smaller_num_data, num_pairs, block_dim_y, min_grid_dim_y_,
    BatchConstructMinRowsPerThread(), BatchConstructSaturationFloor(),
    cuda_hybrid_construct_dim_y_.RawData());
}

// Small-leaf direct body (hybrid growth, non-quantized only): adds each row's
// gradient/hessian pair straight to the leaf's global histogram with plain
// device-scope atomicAdd, skipping the shared-memory accumulation entirely.
// The shared-memory body pays a fixed per-block cost (zero + merge of up to
// 2 * num_bins_per_partition shared entries) that dwarfs the row work when a
// pair's leaves are tiny; at <= SmallLeafRowThreshold() rows per leaf global
// atomic contention is negligible. Rows are covered by a grid-stride loop, so
// any launch grid works: blocks whose row range lies beyond the leaf exit
// after one comparison (no shared zero, no merge), which is what makes the
// over-provisioned batched grids (sized for the level's LARGEST pair) cheap
// for the tiny pairs of the same level.
template <typename BIN_TYPE, bool IS_4BIT = false, bool USE_GH2 = false>
__device__ __forceinline__ void ConstructHistogramDenseDirectInner(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const float2* cuda_gh,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used_bytree,
  const data_size_t num_data) {
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  const bool feat_used = (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) &&
      (is_feature_used_bytree == nullptr || is_feature_used_bytree[column_index]);
  if (!feat_used) {
    return;
  }
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const int data_row_stride = IS_4BIT ?
    packed_partition_byte_offsets[blockIdx.x + 1] - packed_partition_byte_offsets[blockIdx.x] :
    num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(
    IS_4BIT ? packed_partition_byte_offsets[blockIdx.x] : partition_column_start) * num_data;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  // column_hist_offsets is PARTITION-RELATIVE (it indexes the per-partition
  // shared histogram in the shared-memory body); the global histogram position
  // additionally needs the partition's own start offset
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  hist_t* hist_ptr = smaller_leaf_splits->hist_in_leaf +
    ((partition_hist_start + column_hist_offsets[column_index]) << 1);
  const data_size_t row_stride = static_cast<data_size_t>(gridDim.y) * static_cast<data_size_t>(blockDim.y);
  for (data_size_t row = static_cast<data_size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
       row < num_data_in_smaller_leaf; row += row_stride) {
    const data_size_t data_index = data_indices_ref[row];
    score_t grad, hess;
    if (USE_GH2) {
      const float2 gh = cuda_gh[data_index];
      grad = gh.x;
      hess = gh.y;
    } else {
      grad = cuda_gradients[data_index];
      hess = cuda_hessians[data_index];
    }
    const uint32_t pos = ReadDenseBin<IS_4BIT>(
      data_ptr + static_cast<size_t>(data_index) * data_row_stride, threadIdx.x) << 1;
    atomicAdd(hist_ptr + pos, static_cast<hist_t>(grad));
    atomicAdd(hist_ptr + pos + 1, static_cast<hist_t>(hess));
  }
}

// Batched per-level variant (hybrid growth): one launch covers all sibling pairs
// of a level; blockIdx.z selects the pair. The x/y grid is sized for the pair with
// the most data; blocks beyond a pair's own data exit early inside the helper.
// The construct gating (host-mirrored min_data/min_hessian early return) is
// evaluated on-device from the pair structs, so the speculative single-sync flow
// can enqueue the level before the child statistics are read back; in the classic
// flow desc->construct_valid carries the identical host decision and the device
// check is a no-op. When level_dim_y is non-null (speculative flow), the launch
// grid is only an upper bound and the row-grouping extent comes from the scalar
// precomputed by ComputeBatchedConstructDimYKernel (bit-identical to the classic
// host sizing).
template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE, bool USE_REG_BINS = false, bool IS_4BIT = false, bool USE_GH2 = false>
__global__ void CUDAConstructHistogramDenseBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const float2* cuda_gh,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used_bytree,
  const data_size_t num_data,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const int* level_dim_y,
  const data_size_t* level_smaller_num_data,
  const int min_grid_dim_y,
  const int min_rows_per_thread,
  const int saturation_floor_total,
  const data_size_t small_leaf_threshold,
  const uint8_t* bin_used) {
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (!desc->construct_valid) {
    return;
  }
  // device mirror of ConstructHistogramForLeaf's min_data/min_hessian early
  // return (block-uniform, so the early return is divergence-free)
  const CUDALeafSplitsStruct* smaller_struct = desc->smaller_struct;
  const CUDALeafSplitsStruct* larger_struct = desc->larger_struct;
  const data_size_t num_data_smaller = smaller_struct->num_data_in_leaf;
  const double sum_hessians_smaller = smaller_struct->sum_of_hessians;
  const bool has_larger = larger_struct->leaf_index >= 0;
  const data_size_t num_data_larger = has_larger ? larger_struct->num_data_in_leaf : 0;
  const double sum_hessians_larger = has_larger ? larger_struct->sum_of_hessians : 0.0;
  if ((num_data_smaller <= min_data_in_leaf || sum_hessians_smaller <= min_sum_hessian_in_leaf) &&
      (num_data_larger <= min_data_in_leaf || sum_hessians_larger <= min_sum_hessian_in_leaf)) {
    return;
  }
  // small-leaf pairs (decided on-device from the ACTUAL smaller-leaf size, so
  // the speculative flow's host-side upper bounds never mask a tiny pair) skip
  // the shared-memory accumulation entirely; see ConstructHistogramDenseDirectInner
  if (num_data_smaller <= small_leaf_threshold) {
    ConstructHistogramDenseDirectInner<BIN_TYPE, IS_4BIT, USE_GH2>(
      smaller_struct, cuda_gradients, cuda_hessians, cuda_gh, data,
      column_hist_offsets, column_hist_offsets_full,
      feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      is_feature_used_bytree, num_data);
    return;
  }
  // effective row-grouping extent: the launch grid in the classic flow; the
  // precomputed scalar (ComputeBatchedConstructDimYKernel) for many-pair
  // speculative levels; or -- for few-pair speculative levels -- the identical
  // formula evaluated right here from the level's actual smaller-child sizes,
  // saving that kernel launch (block-uniform, <= 32 loads)
  int dim_y;
  if (level_dim_y != nullptr) {
    dim_y = level_dim_y[0];
  } else if (level_smaller_num_data == nullptr) {
    dim_y = static_cast<int>(gridDim.y * blockDim.y);
  } else {
    data_size_t max_num_data = 0;
    const int num_pairs = static_cast<int>(gridDim.z);
    for (int i = 0; i < num_pairs; ++i) {
      const data_size_t n = level_smaller_num_data[i];
      if (n > max_num_data) {
        max_num_data = n;
      }
    }
    dim_y = HybridBatchedConstructGridDimY(
      max_num_data, num_pairs, static_cast<int>(blockDim.y),
      min_grid_dim_y, min_rows_per_thread, saturation_floor_total) * static_cast<int>(blockDim.y);
  }
  if (USE_REG_BINS) {
    ConstructHistogramDenseInner<BIN_TYPE, HIST_TYPE, true, IS_4BIT, USE_GH2>(
      smaller_struct, shared_hist, cuda_gradients, cuda_hessians, cuda_gh, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      is_feature_used_bytree, num_data, dim_y, bin_used);
  } else {
    ConstructHistogramDenseInner<BIN_TYPE, HIST_TYPE, false, IS_4BIT, USE_GH2>(
      smaller_struct, shared_hist, cuda_gradients, cuda_hessians, cuda_gh, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      is_feature_used_bytree, num_data, dim_y, bin_used);
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramSparseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + static_cast<size_t>(blockIdx.x) * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename HIST_TYPE>
__global__ void CUDAConstructHistogramDenseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data,
  HIST_TYPE* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(partition_column_start) * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  HIST_TYPE* shared_hist = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start) * 2;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    HIST_TYPE* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[static_cast<size_t>(data_index) * num_columns_in_partition + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename HIST_TYPE, typename DATA_PTR_TYPE>
__global__ void CUDAConstructHistogramSparseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data,
  HIST_TYPE* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + static_cast<size_t>(blockIdx.x) * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  HIST_TYPE* shared_hist = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start) * 2;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

// Shared body of the discretized dense histogram kernel (see
// ConstructHistogramDenseInner for the shared-memory-passing and early-exit
// rationale).
template <typename BIN_TYPE, bool USE_16BIT_HIST, bool IS_4BIT = false>
__device__ __forceinline__ void ConstructDiscretizedHistogramDenseInner(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  int16_t* shared_hist,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const data_size_t num_data,
  const int8_t* is_feature_used_bytree = nullptr,
  const uint8_t* bin_used = nullptr) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  if (block_start >= num_data_in_smaller_leaf) {
    return;
  }
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  int32_t* shared_hist_packed = reinterpret_cast<int32_t*>(shared_hist);
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const int row_stride = IS_4BIT ?
    packed_partition_byte_offsets[blockIdx.x + 1] - packed_partition_byte_offsets[blockIdx.x] :
    num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(
    IS_4BIT ? packed_partition_byte_offsets[blockIdx.x] : partition_column_start) * num_data;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  // bin_used / is_feature_used_bytree (per-tree feature_fraction masks, null in
  // the per-pair path and without sampling): unused features' bins are dead
  // storage this tree, so their zero/accumulate/merge work is skipped. Used
  // bins see the identical arithmetic (integer atomics are order-invariant).
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    if (bin_used == nullptr || bin_used[partition_hist_start + i]) {
      shared_hist_packed[i] = 0;
    }
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition) &&
      (is_feature_used_bytree == nullptr || is_feature_used_bytree[column_index])) {
    int32_t* shared_hist_ptr = shared_hist_packed + (column_hist_offsets[column_index]);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = ReadDenseBin<IS_4BIT>(data_ptr + static_cast<size_t>(data_index) * row_stride, threadIdx.x);
      int32_t* pos_ptr = shared_hist_ptr + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      if (bin_used != nullptr && !bin_used[partition_hist_start + i]) {
        continue;
      }
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      if (bin_used != nullptr && !bin_used[partition_hist_start + i]) {
        continue;
      }
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

template <typename BIN_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST, bool IS_4BIT = false>
__global__ void CUDAConstructDiscretizedHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const data_size_t num_data) {
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  ConstructDiscretizedHistogramDenseInner<BIN_TYPE, USE_16BIT_HIST, IS_4BIT>(
    smaller_leaf_splits, shared_hist, cuda_gradients_and_hessians, data,
    column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
    packed_partition_byte_offsets,
    num_data);
}

// Batched per-level variant (hybrid growth): blockIdx.z selects the pair. The
// per-pair histogram bit width is a runtime (block-uniform) branch so pairs with
// 16-bit and 32-bit histograms share a single launch.
template <typename BIN_TYPE, int SHARED_HIST_SIZE, bool IS_4BIT = false>
__global__ void CUDAConstructDiscretizedHistogramDenseBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const data_size_t num_data,
  const int8_t* is_feature_used_bytree,
  const uint8_t* bin_used) {
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (!desc->construct_valid) {
    return;
  }
  if (desc->smaller_num_bits <= 16) {
    ConstructDiscretizedHistogramDenseInner<BIN_TYPE, true, IS_4BIT>(
      desc->smaller_struct, shared_hist, cuda_gradients_and_hessians, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      num_data, is_feature_used_bytree, bin_used);
  } else {
    ConstructDiscretizedHistogramDenseInner<BIN_TYPE, false, IS_4BIT>(
      desc->smaller_struct, shared_hist, cuda_gradients_and_hessians, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      num_data, is_feature_used_bytree, bin_used);
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramSparseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  int32_t* shared_hist_packed = reinterpret_cast<int32_t*>(shared_hist);
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + blockIdx.x * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_packed + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

template <typename BIN_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data,
  int32_t* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(partition_column_start) * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  int32_t* shared_hist_packed = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    int32_t* shared_hist_ptr = shared_hist_packed + (column_hist_offsets[column_index]);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[static_cast<size_t>(data_index) * num_columns_in_partition + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_ptr + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data,
  int32_t* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + blockIdx.x * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  int32_t* shared_hist_packed = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_packed + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

void CUDAHistogramConstructor::LaunchConstructHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (cuda_row_data_->shared_hist_size() == DP_SHARED_HIST_SIZE && gpu_use_dp_) {
    LaunchConstructHistogramKernelInner<double, DP_SHARED_HIST_SIZE>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->shared_hist_size() == SP_SHARED_HIST_SIZE && !gpu_use_dp_) {
    LaunchConstructHistogramKernelInner<float, SP_SHARED_HIST_SIZE>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else {
    Log::Fatal("Unknown shared histogram size %d", cuda_row_data_->shared_hist_size());
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (cuda_row_data_->bit_type() == 8) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint8_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->bit_type() == 16) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->bit_type() == 32) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint32_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else {
    Log::Fatal("Unknown bit_type = %d", cuda_row_data_->bit_type());
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (cuda_row_data_->row_ptr_bit_type() == 16) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint32_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint64_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else {
    if (!cuda_row_data_->is_sparse()) {
      LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
    } else {
      Log::Fatal("Unknown row_ptr_bit_type = %d", cuda_row_data_->row_ptr_bit_type());
    }
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner1(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (cuda_row_data_->NumLargeBinPartition() == 0) {
    LaunchConstructHistogramKernelInner2<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, PTR_TYPE, false>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else {
    LaunchConstructHistogramKernelInner2<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, PTR_TYPE, true>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE, bool USE_GLOBAL_MEM_BUFFER>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner2(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  int grid_dim_x = 0;
  int grid_dim_y = 0;
  int block_dim_x = 0;
  int block_dim_y = 0;
  CalcConstructHistogramKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, num_data_in_smaller_leaf);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  if (use_quantized_grad_) {
    if (USE_GLOBAL_MEM_BUFFER) {
      if (cuda_row_data_->is_sparse()) {
        if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        } else {
          CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        }
      } else {
        if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory<BIN_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        } else {
          CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory<BIN_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        }
      }
    } else {
      if (cuda_row_data_->is_sparse()) {
        if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramSparseKernel<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else {
          CUDAConstructDiscretizedHistogramSparseKernel<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else {
        if (cuda_row_data_->is_4bit_packed()) {
          if (num_bits_in_histogram_bins <= 16) {
            CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, true, true><<<grid_dim, block_dim, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              reinterpret_cast<const int32_t*>(cuda_gradients_),
              cuda_row_data_->GetBin<BIN_TYPE>(),
              cuda_row_data_->cuda_column_hist_offsets(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              cuda_row_data_->cuda_feature_partition_column_index_offsets(),
              cuda_row_data_->cuda_packed_partition_byte_offsets(),
              num_data_);
          } else {
            CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, false, true><<<grid_dim, block_dim, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              reinterpret_cast<const int32_t*>(cuda_gradients_),
              cuda_row_data_->GetBin<BIN_TYPE>(),
              cuda_row_data_->cuda_column_hist_offsets(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              cuda_row_data_->cuda_feature_partition_column_index_offsets(),
              cuda_row_data_->cuda_packed_partition_byte_offsets(),
              num_data_);
          }
        } else if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            nullptr,
            num_data_);
        } else {
          CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            nullptr,
            num_data_);
        }
      }
    }
  } else {
    if (!USE_GLOBAL_MEM_BUFFER) {
      if (cuda_row_data_->is_sparse()) {
        CUDAConstructHistogramSparseKernel<BIN_TYPE, PTR_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, current_stream()>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->GetBin<BIN_TYPE>(),
          cuda_row_data_->GetRowPtr<PTR_TYPE>(),
          cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          num_data_);
      } else {
        // ====== COMPACT VIEW PATH (feature_fraction sampling honored on GPU) ======
        if (use_compact_view_) {
          // DEBUG: use SOURCE pointers but with my computed compact dims to isolate bug
          const int compact_block_x = max_num_compact_cols_per_partition_;
          const int compact_block_y = NUM_THREADS_PER_BLOCK / std::max(1, compact_block_x);
          const int compact_grid_y = std::max(min_grid_dim_y_,
              ((num_data_in_smaller_leaf + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + std::max(1, compact_block_y) - 1) / std::max(1, compact_block_y));
          dim3 compact_grid_dim(grid_dim_x, compact_grid_y);
          dim3 compact_block_dim(compact_block_x, std::max(1, compact_block_y));
          // After BuildCompactView swap, compact_data_uint8_t_ is whichever buffer is now active.
          // (When use_compact_view_ true and host-mapped path is used, BuildCompactView swaps
          // active_buffer_is_alt_; the "active" buffer for histograms is the OPPOSITE of
          // active_buffer_is_alt_ after the swap, since BuildCompactView fills the "alt"
          // and then flips active_buffer_is_alt_.)
          uint8_t* active_data = compact_data_uint8_t_.RawData();
          if (compact_is_col_major_) {
            CUDAConstructHistogramDenseColMajorKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<compact_grid_dim, compact_block_dim, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              cuda_gradients_, cuda_hessians_,
              reinterpret_cast<const BIN_TYPE*>(active_data),
              compact_column_hist_offsets_.RawData(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              compact_feature_partition_column_index_offsets_.RawData(),
              num_data_);
          } else if (compact_is_4bit_) {
            CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true><<<compact_grid_dim, compact_block_dim, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              cuda_gradients_, cuda_hessians_,
              reinterpret_cast<const BIN_TYPE*>(active_data),
              compact_column_hist_offsets_.RawData(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              compact_feature_partition_column_index_offsets_.RawData(),
              compact_packed_partition_byte_offsets_.RawData(),
              nullptr,
              num_data_);
          } else {
            CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<compact_grid_dim, compact_block_dim, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              cuda_gradients_, cuda_hessians_,
              reinterpret_cast<const BIN_TYPE*>(active_data),
              compact_column_hist_offsets_.RawData(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              compact_feature_partition_column_index_offsets_.RawData(),
              nullptr,
              nullptr,
              num_data_);
          }
        } else if (cuda_row_data_->is_4bit_packed()) {
          CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            cuda_row_data_->cuda_packed_partition_byte_offsets(),
            cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
            num_data_);
        } else {
          CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            nullptr,
            cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
            num_data_);
        }
      }
    } else {
      if (cuda_row_data_->is_sparse()) {
        CUDAConstructHistogramSparseKernel_GlobalMemory<BIN_TYPE, HIST_TYPE, PTR_TYPE><<<grid_dim, block_dim, 0, current_stream()>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->GetBin<BIN_TYPE>(),
          cuda_row_data_->GetRowPtr<PTR_TYPE>(),
          cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          num_data_,
          reinterpret_cast<HIST_TYPE*>(cuda_hist_buffer_.RawData()));
      } else {
        CUDAConstructHistogramDenseKernel_GlobalMemory<BIN_TYPE, HIST_TYPE><<<grid_dim, block_dim, 0, current_stream()>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->GetBin<BIN_TYPE>(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_,
          reinterpret_cast<HIST_TYPE*>(cuda_hist_buffer_.RawData()));
      }
    }
  }
}

__device__ __forceinline__ void SubtractHistogramInner(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cuda_larger_leaf_index = cuda_larger_leaf_splits->leaf_index;
  if (cuda_larger_leaf_index >= 0) {
    const hist_t* smaller_leaf_hist = cuda_smaller_leaf_splits->hist_in_leaf;
    hist_t* larger_leaf_hist = cuda_larger_leaf_splits->hist_in_leaf;
    if (global_thread_index < 2 * num_total_bin) {
      larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
    }
  }
}

__global__ void SubtractHistogramKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits) {
  SubtractHistogramInner(num_total_bin, cuda_smaller_leaf_splits, cuda_larger_leaf_splits);
}

// Batched per-level variant (hybrid growth): blockIdx.y selects the pair.
// bin_used (per-tree feature_fraction bin mask, may be null) skips histogram
// entries of features outside this tree's sample: nothing of this tree reads
// them (the find kernels are feature-masked), so they are dead storage.
__global__ void SubtractHistogramBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  const uint8_t* bin_used) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (bin_used != nullptr &&
      global_thread_index < static_cast<unsigned int>(2 * num_total_bin) &&
      !bin_used[global_thread_index >> 1]) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  SubtractHistogramInner(num_total_bin, desc->smaller_struct, desc->larger_struct);
}

// When fix_feature_index (the index into the need-fix feature list) is -1 it
// comes from blockIdx.x (the standalone fix kernels); the fused small-leaf
// fix+subtract kernel passes it explicitly because its fix blocks start after
// the subtract blocks. When larger_for_subtract is non-null, thread 0 also
// applies the histogram subtraction at the fixed most-frequent-bin entries
// (larger = parent - fixed smaller, the exact arithmetic the standalone
// subtract kernel would perform after the fix kernel).
__device__ __forceinline__ void FixHistogramInner(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  hist_t* shared_mem_buffer,
  const int fix_feature_index = -1,
  const CUDALeafSplitsStruct* larger_for_subtract = nullptr) {
  const unsigned int blockIdx_x = fix_feature_index >= 0 ?
    static_cast<unsigned int>(fix_feature_index) : blockIdx.x;
  const int feature_index = cuda_need_fix_histogram_features[blockIdx_x];
  const uint32_t num_bin_aligned = cuda_need_fix_histogram_features_num_bin_aligned[blockIdx_x];
  const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
  const double leaf_sum_gradients = cuda_smaller_leaf_splits->sum_of_gradients;
  const double leaf_sum_hessians = cuda_smaller_leaf_splits->sum_of_hessians;
  hist_t* feature_hist = cuda_smaller_leaf_splits->hist_in_leaf + feature_hist_offset * 2;
  const unsigned int threadIdx_x = threadIdx.x;
  const uint32_t num_bin = cuda_feature_num_bins[feature_index];
  const uint32_t hist_pos = threadIdx_x << 1;
  const hist_t bin_gradient = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[hist_pos] : 0.0f;
  const hist_t bin_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[hist_pos + 1] : 0.0f;
  const hist_t sum_gradient = ShuffleReduceSum<hist_t>(bin_gradient, shared_mem_buffer, num_bin_aligned);
  const hist_t sum_hessian = ShuffleReduceSum<hist_t>(bin_hessian, shared_mem_buffer, num_bin_aligned);
  if (threadIdx_x == 0) {
    const hist_t fixed_gradient = leaf_sum_gradients - sum_gradient;
    const hist_t fixed_hessian = leaf_sum_hessians - sum_hessian;
    feature_hist[most_freq_bin << 1] = fixed_gradient;
    feature_hist[(most_freq_bin << 1) + 1] = fixed_hessian;
    if (larger_for_subtract != nullptr && larger_for_subtract->leaf_index >= 0) {
      hist_t* larger_feature_hist = larger_for_subtract->hist_in_leaf + feature_hist_offset * 2;
      larger_feature_hist[most_freq_bin << 1] -= fixed_gradient;
      larger_feature_hist[(most_freq_bin << 1) + 1] -= fixed_hessian;
    }
  }
}

__global__ void FixHistogramKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits) {
  __shared__ hist_t shared_mem_buffer[WARPSIZE];
  FixHistogramInner(cuda_feature_num_bins, cuda_feature_hist_offsets,
    cuda_feature_most_freq_bins, cuda_need_fix_histogram_features,
    cuda_need_fix_histogram_features_num_bin_aligned, cuda_smaller_leaf_splits,
    shared_mem_buffer);
}

// Batched per-level variant (hybrid growth): blockIdx.y selects the pair.
// feature_used (per-tree feature_fraction mask, may be null) skips need-fix
// features outside this tree's sample (their bins are dead storage this tree).
__global__ void FixHistogramBatchedKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDAHybridPairDescriptor* pair_descs,
  const int8_t* feature_used) {
  if (feature_used != nullptr &&
      !feature_used[cuda_need_fix_histogram_features[blockIdx.x]]) {
    return;
  }
  __shared__ hist_t shared_mem_buffer[WARPSIZE];
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  FixHistogramInner(cuda_feature_num_bins, cuda_feature_hist_offsets,
    cuda_feature_most_freq_bins, cuda_need_fix_histogram_features,
    cuda_need_fix_histogram_features_num_bin_aligned, desc->smaller_struct,
    shared_mem_buffer);
}

// Fused fix + subtract of the small-leaf level path (hybrid growth,
// non-quantized only): one launch replaces the sequential FixHistogramBatched +
// SubtractHistogramBatched pair. blockIdx.y selects the pair. Blocks with
// blockIdx.x < num_subtract_blocks perform the elementwise larger -= smaller
// subtraction but SKIP the entries flagged in fix_mfb_mask (the most-frequent-
// bin gradient/hessian slots of the need-fix features); the remaining blocks
// (one per need-fix feature) run the most-frequent-bin fix of the smaller leaf
// and apply the subtraction at exactly those skipped entries from the fixed
// values. Every histogram entry is therefore written by exactly one block with
// the identical arithmetic of the sequential launches (bit-identical result);
// the subtraction reads no entry the fix writes and vice versa, so no
// cross-block ordering is needed.
__global__ void FixSubtractHistogramSmallLeafBatchedKernel(
  const int num_total_bin,
  const int num_subtract_blocks,
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const uint8_t* fix_mfb_mask,
  const CUDAHybridPairDescriptor* pair_descs,
  const uint8_t* bin_used,
  const int8_t* feature_used) {
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  if (static_cast<int>(blockIdx.x) < num_subtract_blocks) {
    const CUDALeafSplitsStruct* larger_leaf = desc->larger_struct;
    if (larger_leaf->leaf_index >= 0) {
      const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      // bins of features outside this tree's sample are dead storage: skip
      if (i < static_cast<unsigned int>(2 * num_total_bin) && !fix_mfb_mask[i] &&
          (bin_used == nullptr || bin_used[i >> 1])) {
        larger_leaf->hist_in_leaf[i] -= desc->smaller_struct->hist_in_leaf[i];
      }
    }
  } else {
    if (feature_used != nullptr &&
        !feature_used[cuda_need_fix_histogram_features[
          static_cast<int>(blockIdx.x) - num_subtract_blocks]]) {
      return;  // most-frequent bin of an unused feature: dead storage this tree
    }
    __shared__ hist_t shared_mem_buffer[WARPSIZE];
    FixHistogramInner(cuda_feature_num_bins, cuda_feature_hist_offsets,
      cuda_feature_most_freq_bins, cuda_need_fix_histogram_features,
      cuda_need_fix_histogram_features_num_bin_aligned, desc->smaller_struct,
      shared_mem_buffer, static_cast<int>(blockIdx.x) - num_subtract_blocks,
      desc->larger_struct);
  }
}

template <bool SMALLER_USE_16BIT_HIST, bool LARGER_USE_16BIT_HIST, bool PARENT_USE_16BIT_HIST>
__device__ __forceinline__ void SubtractHistogramDiscretizedInner(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  int32_t* num_bit_change_buffer) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cuda_larger_leaf_index_ref = cuda_larger_leaf_splits->leaf_index;
  if (cuda_larger_leaf_index_ref >= 0) {
    if (PARENT_USE_16BIT_HIST) {
      const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int32_t* larger_leaf_hist = reinterpret_cast<int32_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
      }
    } else if (LARGER_USE_16BIT_HIST) {
      int32_t* buffer = num_bit_change_buffer;
      const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        const int64_t parent_hist_item = larger_leaf_hist[global_thread_index];
        const int32_t smaller_hist_item = smaller_leaf_hist[global_thread_index];
        const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
          static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
        const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
        buffer[global_thread_index] = static_cast<int32_t>(static_cast<int16_t>(larger_hist_item >> 32) << 16) |
          static_cast<int32_t>(larger_hist_item & 0x000000000000ffff);
      }
    } else if (SMALLER_USE_16BIT_HIST) {
        const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
        int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
        if (global_thread_index < num_total_bin) {
          const int64_t parent_hist_item = larger_leaf_hist[global_thread_index];
          const int32_t smaller_hist_item = smaller_leaf_hist[global_thread_index];
          const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
            static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
          const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
          larger_leaf_hist[global_thread_index] = larger_hist_item;
        }
    } else {
      const int64_t* smaller_leaf_hist = reinterpret_cast<const int64_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
      }
    }
  }
}

template <bool SMALLER_USE_16BIT_HIST, bool LARGER_USE_16BIT_HIST, bool PARENT_USE_16BIT_HIST>
__global__ void SubtractHistogramDiscretizedKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  hist_t* num_bit_change_buffer) {
  SubtractHistogramDiscretizedInner<SMALLER_USE_16BIT_HIST, LARGER_USE_16BIT_HIST, PARENT_USE_16BIT_HIST>(
    num_total_bin, cuda_smaller_leaf_splits, cuda_larger_leaf_splits,
    reinterpret_cast<int32_t*>(num_bit_change_buffer));
}

// Batched per-level variant (hybrid growth): blockIdx.y selects the pair. The
// per-pair histogram bit widths choose the arithmetic at runtime (block-uniform
// branches), and each pair that needs the 64->32-bit compaction writes its own
// region of the change buffer (stride num_total_bin int32 entries per pair).
__global__ void SubtractHistogramDiscretizedBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used) {
  // bins of features outside this tree's sample are dead storage: skip (the
  // change-buffer copy kernel skips the same bins, so no stale data is read)
  const unsigned int global_thread_index_gate = threadIdx.x + blockIdx.x * blockDim.x;
  if (bin_used != nullptr &&
      global_thread_index_gate < static_cast<unsigned int>(num_total_bin) &&
      !bin_used[global_thread_index_gate]) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  int32_t* buffer = reinterpret_cast<int32_t*>(num_bit_change_buffer) +
    static_cast<size_t>(blockIdx.y) * static_cast<size_t>(num_total_bin);
  if (desc->parent_num_bits <= 16) {
    SubtractHistogramDiscretizedInner<true, true, true>(
      num_total_bin, desc->smaller_struct, desc->larger_struct, buffer);
  } else if (desc->larger_num_bits <= 16) {
    SubtractHistogramDiscretizedInner<true, true, false>(
      num_total_bin, desc->smaller_struct, desc->larger_struct, buffer);
  } else if (desc->smaller_num_bits <= 16) {
    SubtractHistogramDiscretizedInner<true, false, false>(
      num_total_bin, desc->smaller_struct, desc->larger_struct, buffer);
  } else {
    SubtractHistogramDiscretizedInner<false, false, false>(
      num_total_bin, desc->smaller_struct, desc->larger_struct, buffer);
  }
}

__global__ void CopyChangedNumBitHistogram(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  hist_t* num_bit_change_buffer) {
  int32_t* hist_dst = reinterpret_cast<int32_t*>(cuda_larger_leaf_splits->hist_in_leaf);
  const int32_t* hist_src = reinterpret_cast<const int32_t*>(num_bit_change_buffer);
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_thread_index < static_cast<unsigned int>(num_total_bin)) {
    hist_dst[global_thread_index] = hist_src[global_thread_index];
  }
}

// Batched per-level variant: copies each mixed-bit-width pair's change-buffer
// region back into the larger leaf's histogram. Pairs whose bit widths do not
// need the compaction (or whose larger leaf does not exist) exit immediately.
__global__ void CopyChangedNumBitHistogramBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used) {
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  if (desc->larger_leaf_index < 0 ||
      !(desc->parent_num_bits > 16 && desc->larger_num_bits <= 16)) {
    return;
  }
  int32_t* hist_dst = reinterpret_cast<int32_t*>(desc->larger_struct->hist_in_leaf);
  const int32_t* hist_src = reinterpret_cast<const int32_t*>(num_bit_change_buffer) +
    static_cast<size_t>(blockIdx.y) * static_cast<size_t>(num_total_bin);
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_thread_index < static_cast<unsigned int>(num_total_bin) &&
      (bin_used == nullptr || bin_used[global_thread_index])) {
    hist_dst[global_thread_index] = hist_src[global_thread_index];
  }
}

template <bool USE_16BIT_HIST>
__device__ __forceinline__ void FixHistogramDiscretizedInner(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  int64_t* shared_mem_buffer) {
  const unsigned int blockIdx_x = blockIdx.x;
  const int feature_index = cuda_need_fix_histogram_features[blockIdx_x];
  const uint32_t num_bin_aligned = cuda_need_fix_histogram_features_num_bin_aligned[blockIdx_x];
  const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
  if (USE_16BIT_HIST) {
    const int64_t leaf_sum_gradients_hessians_int64 = cuda_smaller_leaf_splits->sum_of_gradients_hessians;
    const int32_t leaf_sum_gradients_hessians =
      (static_cast<int32_t>(leaf_sum_gradients_hessians_int64 >> 32) << 16) | static_cast<int32_t>(leaf_sum_gradients_hessians_int64 & 0x000000000000ffff);
    int32_t* feature_hist = reinterpret_cast<int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf) + feature_hist_offset;
    const unsigned int threadIdx_x = threadIdx.x;
    const uint32_t num_bin = cuda_feature_num_bins[feature_index];
    const int32_t bin_gradient_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[threadIdx_x] : 0;
    const int32_t sum_gradient_hessian = ShuffleReduceSum<int32_t>(
      bin_gradient_hessian,
      reinterpret_cast<int32_t*>(shared_mem_buffer),
      num_bin_aligned);
    if (threadIdx_x == 0) {
      feature_hist[most_freq_bin] = leaf_sum_gradients_hessians - sum_gradient_hessian;
    }
  } else {
    const int64_t leaf_sum_gradients_hessians = cuda_smaller_leaf_splits->sum_of_gradients_hessians;
    int64_t* feature_hist = reinterpret_cast<int64_t*>(cuda_smaller_leaf_splits->hist_in_leaf) + feature_hist_offset;
    const unsigned int threadIdx_x = threadIdx.x;
    const uint32_t num_bin = cuda_feature_num_bins[feature_index];
    const int64_t bin_gradient_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[threadIdx_x] : 0;
    const int64_t sum_gradient_hessian = ShuffleReduceSum<int64_t>(bin_gradient_hessian, shared_mem_buffer, num_bin_aligned);
    if (threadIdx_x == 0) {
      feature_hist[most_freq_bin] = leaf_sum_gradients_hessians - sum_gradient_hessian;
    }
  }
}

template <bool USE_16BIT_HIST>
__global__ void FixHistogramDiscretizedKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits) {
  __shared__ int64_t shared_mem_buffer[WARPSIZE];
  FixHistogramDiscretizedInner<USE_16BIT_HIST>(
    cuda_feature_num_bins, cuda_feature_hist_offsets, cuda_feature_most_freq_bins,
    cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned,
    cuda_smaller_leaf_splits, shared_mem_buffer);
}

// Batched per-level variant (hybrid growth): blockIdx.y selects the pair; the
// per-pair histogram bit width is a runtime (block-uniform) branch.
__global__ void FixHistogramDiscretizedBatchedKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDAHybridPairDescriptor* pair_descs,
  const int8_t* feature_used) {
  if (feature_used != nullptr &&
      !feature_used[cuda_need_fix_histogram_features[blockIdx.x]]) {
    return;  // most-frequent bin of an unused feature: dead storage this tree
  }
  __shared__ int64_t shared_mem_buffer[WARPSIZE];
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  if (desc->smaller_num_bits <= 16) {
    FixHistogramDiscretizedInner<true>(
      cuda_feature_num_bins, cuda_feature_hist_offsets, cuda_feature_most_freq_bins,
      cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned,
      desc->smaller_struct, shared_mem_buffer);
  } else {
    FixHistogramDiscretizedInner<false>(
      cuda_feature_num_bins, cuda_feature_hist_offsets, cuda_feature_most_freq_bins,
      cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned,
      desc->smaller_struct, shared_mem_buffer);
  }
}

void CUDAHistogramConstructor::LaunchSubtractHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool use_discretized_grad,
  const uint8_t parent_num_bits_in_histogram_bins,
  const uint8_t smaller_num_bits_in_histogram_bins,
  const uint8_t larger_num_bits_in_histogram_bins) {
    if (!use_discretized_grad) {
      const int num_subtract_threads = 2 * num_total_bin_;
      const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
      global_timer.Start("CUDAHistogramConstructor::FixHistogramKernel");
      if (need_fix_histogram_features_.size() > 0) {
        FixHistogramKernel<<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, current_stream()>>>(
          cuda_feature_num_bins_.RawData(),
          cuda_feature_hist_offsets_.RawData(),
          cuda_feature_most_freq_bins_.RawData(),
          cuda_need_fix_histogram_features_.RawData(),
          cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
          cuda_smaller_leaf_splits);
      }
      global_timer.Stop("CUDAHistogramConstructor::FixHistogramKernel");
      global_timer.Start("CUDAHistogramConstructor::SubtractHistogramKernel");
      SubtractHistogramKernel<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, current_stream()>>>(
        num_total_bin_,
        cuda_smaller_leaf_splits,
        cuda_larger_leaf_splits);
      global_timer.Stop("CUDAHistogramConstructor::SubtractHistogramKernel");
    } else {
      const int num_subtract_threads = num_total_bin_;
      const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
      global_timer.Start("CUDAHistogramConstructor::FixHistogramDiscretizedKernel");
      if (need_fix_histogram_features_.size() > 0) {
        if (smaller_num_bits_in_histogram_bins <= 16) {
          FixHistogramDiscretizedKernel<true><<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, current_stream()>>>(
            cuda_feature_num_bins_.RawData(),
            cuda_feature_hist_offsets_.RawData(),
            cuda_feature_most_freq_bins_.RawData(),
            cuda_need_fix_histogram_features_.RawData(),
            cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
            cuda_smaller_leaf_splits);
        } else {
          FixHistogramDiscretizedKernel<false><<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, current_stream()>>>(
            cuda_feature_num_bins_.RawData(),
            cuda_feature_hist_offsets_.RawData(),
            cuda_feature_most_freq_bins_.RawData(),
            cuda_need_fix_histogram_features_.RawData(),
            cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
            cuda_smaller_leaf_splits);
        }
      }
      global_timer.Stop("CUDAHistogramConstructor::FixHistogramDiscretizedKernel");
      global_timer.Start("CUDAHistogramConstructor::SubtractHistogramDiscretizedKernel");
      // Per-pipeline region of the bit-change buffer: sibling pairs of one level run
      // on different pipeline streams concurrently, so the 64->32-bit compaction of
      // two pairs must not share scratch space (the shared buffer was a data race
      // whenever two concurrent pairs both had a >16-bit parent with a <=16-bit
      // larger child). Regions are num_total_bin_ int32 entries per pipeline; the
      // buffer is only ever accessed as int32, so the 4-byte alignment is fine.
      hist_t* change_buffer = reinterpret_cast<hist_t*>(
        reinterpret_cast<int32_t*>(hist_buffer_for_num_bit_change_.RawData()) +
        static_cast<size_t>(active_pipeline_) * static_cast<size_t>(num_total_bin_));
      if (parent_num_bits_in_histogram_bins <= 16) {
        CHECK_LE(smaller_num_bits_in_histogram_bins, 16);
        CHECK_LE(larger_num_bits_in_histogram_bins, 16);
        SubtractHistogramDiscretizedKernel<true, true, true><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, current_stream()>>>(
          num_total_bin_,
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          change_buffer);
      } else if (larger_num_bits_in_histogram_bins <= 16) {
        CHECK_LE(smaller_num_bits_in_histogram_bins, 16);
        SubtractHistogramDiscretizedKernel<true, true, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, current_stream()>>>(
          num_total_bin_,
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          change_buffer);
        CopyChangedNumBitHistogram<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, current_stream()>>>(
          num_total_bin_,
          cuda_larger_leaf_splits,
          change_buffer);
      } else if (smaller_num_bits_in_histogram_bins <= 16) {
        SubtractHistogramDiscretizedKernel<true, false, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, current_stream()>>>(
          num_total_bin_,
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          change_buffer);
      } else {
        SubtractHistogramDiscretizedKernel<false, false, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, current_stream()>>>(
          num_total_bin_,
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          change_buffer);
      }
      global_timer.Stop("CUDAHistogramConstructor::SubtractHistogramDiscretizedKernel");
    }
}

// ---- batched per-level launchers (hybrid growth) ----------------------------------
// All launches go to pipeline_streams_[0] (cuda_stream_) so construct -> fix ->
// subtract are ordered by the stream; the caller records subtract_done_events_[0]
// afterwards for the best split finder to wait on.

void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const data_size_t max_num_data_in_smaller_leaf,
  const data_size_t* level_smaller_num_data) {
  if (cuda_row_data_->shared_hist_size() == DP_SHARED_HIST_SIZE && gpu_use_dp_) {
    LaunchConstructHistogramBatchedKernelInner<double, DP_SHARED_HIST_SIZE>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
  } else if (cuda_row_data_->shared_hist_size() == SP_SHARED_HIST_SIZE && !gpu_use_dp_) {
    LaunchConstructHistogramBatchedKernelInner<float, SP_SHARED_HIST_SIZE>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
  } else {
    Log::Fatal("Unknown shared histogram size %d", cuda_row_data_->shared_hist_size());
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE>
void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const data_size_t max_num_data_in_smaller_leaf,
  const data_size_t* level_smaller_num_data) {
  if (cuda_row_data_->bit_type() == 8) {
    if (gh_interleave_valid_) {
      LaunchConstructHistogramBatchedKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint8_t, true>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
    } else {
      LaunchConstructHistogramBatchedKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint8_t, false>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
    }
  } else if (cuda_row_data_->bit_type() == 16) {
    if (gh_interleave_valid_) {
      LaunchConstructHistogramBatchedKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint16_t, true>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
    } else {
      LaunchConstructHistogramBatchedKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint16_t, false>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
    }
  } else if (cuda_row_data_->bit_type() == 32) {
    if (gh_interleave_valid_) {
      LaunchConstructHistogramBatchedKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint32_t, true>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
    } else {
      LaunchConstructHistogramBatchedKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint32_t, false>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
    }
  } else {
    Log::Fatal("Unknown bit_type = %d", cuda_row_data_->bit_type());
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, bool USE_GH2>
void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const data_size_t max_num_data_in_smaller_leaf,
  const data_size_t* level_smaller_num_data) {
  // dense shared-memory path only (SupportsBatchedLevel() gates the rest)
  int grid_dim_x = 0;
  int grid_dim_y = 0;
  int block_dim_x = 0;
  int block_dim_y = 0;
  CalcConstructHistogramBatchedKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, max_num_data_in_smaller_leaf, num_pairs);
  if (use_compact_view_) {
    // compact column view: same batched kernel, fed with the per-tree compact
    // data/metadata (mirrors the per-pair compact launch). Blocks span the
    // USED columns of a partition; the y sizing formula is the batched one.
    block_dim_x = std::max(1, max_num_compact_cols_per_partition_);
    block_dim_y = std::max(1, NUM_THREADS_PER_BLOCK / block_dim_x);
    grid_dim_y = HybridBatchedConstructGridDimY(
      max_num_data_in_smaller_leaf, num_pairs, block_dim_y, min_grid_dim_y_,
      BatchConstructMinRowsPerThread(), BatchConstructSaturationFloor());
  }
  dim3 grid_dim(grid_dim_x, grid_dim_y, num_pairs);
  dim3 block_dim(block_dim_x, block_dim_y);
  const int* level_dim_y = nullptr;
  const data_size_t* level_sizes_for_kernel = nullptr;
  if (level_smaller_num_data != nullptr) {
    // speculative flow: the grid above was sized from an upper BOUND; the exact
    // row-grouping extent comes from the level's actual sizes. Few-pair levels
    // evaluate the formula inside the construct kernel itself (saves a launch
    // on the per-level critical path); many-pair levels keep the single-block
    // reduction kernel so construct blocks read one precomputed scalar.
    if (num_pairs <= 32) {
      level_sizes_for_kernel = level_smaller_num_data;
    } else {
      LaunchComputeBatchedConstructDimYKernel(level_smaller_num_data, num_pairs, block_dim_y);
      level_dim_y = cuda_hybrid_construct_dim_y_.RawDataReadOnly();
    }
  }
  if (use_quantized_grad_) {
    // quantized training always uses the classic (two-sync) flow, so the exact
    // level sizes are host-known and no device grouping override is needed
    if (cuda_row_data_->is_4bit_packed()) {
      CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
        pair_descs,
        reinterpret_cast<const int32_t*>(cuda_gradients_),
        cuda_row_data_->GetBin<BIN_TYPE>(),
        cuda_row_data_->cuda_column_hist_offsets(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        cuda_row_data_->cuda_feature_partition_column_index_offsets(),
        cuda_row_data_->cuda_packed_partition_byte_offsets(),
        num_data_,
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
    } else {
      CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
        pair_descs,
        reinterpret_cast<const int32_t*>(cuda_gradients_),
        cuda_row_data_->GetBin<BIN_TYPE>(),
        cuda_row_data_->cuda_column_hist_offsets(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        cuda_row_data_->cuda_feature_partition_column_index_offsets(),
        nullptr,
        num_data_,
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
    }
  } else if (use_compact_view_) {
    // compact data holds only the tree's sampled columns, so no per-column
    // feature mask is needed (mirrors the per-pair compact launch). Few-bin
    // datasets take the register-accumulation body (see USE_REG_BINS).
    if (construct_reg_bins_) {
      if (compact_is_4bit_) {
        CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true, true, USE_GH2><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          pair_descs,
          cuda_gradients_, cuda_hessians_,
          USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
          reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()),
          compact_column_hist_offsets_.RawData(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          compact_feature_partition_column_index_offsets_.RawData(),
          compact_packed_partition_byte_offsets_.RawData(),
          nullptr,
          num_data_,
          static_cast<data_size_t>(min_data_in_leaf_),
          min_sum_hessian_in_leaf_,
          level_dim_y,
          level_sizes_for_kernel,
          min_grid_dim_y_,
          BatchConstructMinRowsPerThread(),
          BatchConstructSaturationFloor(),
          SmallLeafConstructEnabled() ? SmallLeafRowThreshold() : 0,
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
      } else {
        CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true, false, USE_GH2><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          pair_descs,
          cuda_gradients_, cuda_hessians_,
          USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
          reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()),
          compact_column_hist_offsets_.RawData(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          compact_feature_partition_column_index_offsets_.RawData(),
          nullptr,
          nullptr,
          num_data_,
          static_cast<data_size_t>(min_data_in_leaf_),
          min_sum_hessian_in_leaf_,
          level_dim_y,
          level_sizes_for_kernel,
          min_grid_dim_y_,
          BatchConstructMinRowsPerThread(),
          BatchConstructSaturationFloor(),
          SmallLeafConstructEnabled() ? SmallLeafRowThreshold() : 0,
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
      }
    } else {
      if (compact_is_4bit_) {
        CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, true, USE_GH2><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          pair_descs,
          cuda_gradients_, cuda_hessians_,
          USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
          reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()),
          compact_column_hist_offsets_.RawData(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          compact_feature_partition_column_index_offsets_.RawData(),
          compact_packed_partition_byte_offsets_.RawData(),
          nullptr,
          num_data_,
          static_cast<data_size_t>(min_data_in_leaf_),
          min_sum_hessian_in_leaf_,
          level_dim_y,
          level_sizes_for_kernel,
          min_grid_dim_y_,
          BatchConstructMinRowsPerThread(),
          BatchConstructSaturationFloor(),
          SmallLeafConstructEnabled() ? SmallLeafRowThreshold() : 0,
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
      } else {
        CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, false, USE_GH2><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          pair_descs,
          cuda_gradients_, cuda_hessians_,
          USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
          reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()),
          compact_column_hist_offsets_.RawData(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          compact_feature_partition_column_index_offsets_.RawData(),
          nullptr,
          nullptr,
          num_data_,
          static_cast<data_size_t>(min_data_in_leaf_),
          min_sum_hessian_in_leaf_,
          level_dim_y,
          level_sizes_for_kernel,
          min_grid_dim_y_,
          BatchConstructMinRowsPerThread(),
          BatchConstructSaturationFloor(),
          SmallLeafConstructEnabled() ? SmallLeafRowThreshold() : 0,
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
      }
    }
  } else if (cuda_row_data_->is_4bit_packed()) {
    CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, true, USE_GH2><<<grid_dim, block_dim, 0, cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
      cuda_row_data_->GetBin<BIN_TYPE>(),
      cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      cuda_row_data_->cuda_packed_partition_byte_offsets(),
      cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
      num_data_,
      static_cast<data_size_t>(min_data_in_leaf_),
      min_sum_hessian_in_leaf_,
      level_dim_y,
      level_sizes_for_kernel,
      min_grid_dim_y_,
      BatchConstructMinRowsPerThread(),
      BatchConstructSaturationFloor(),
      SmallLeafConstructEnabled() ? SmallLeafRowThreshold() : 0,
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
  } else {
    CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, false, USE_GH2><<<grid_dim, block_dim, 0, cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
      cuda_row_data_->GetBin<BIN_TYPE>(),
      cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      nullptr,
      cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
      num_data_,
      static_cast<data_size_t>(min_data_in_leaf_),
      min_sum_hessian_in_leaf_,
      level_dim_y,
      level_sizes_for_kernel,
      min_grid_dim_y_,
      BatchConstructMinRowsPerThread(),
      BatchConstructSaturationFloor(),
      SmallLeafConstructEnabled() ? SmallLeafRowThreshold() : 0,
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
  }
}

void CUDAHistogramConstructor::LaunchFixSubtractHistogramSmallLeafBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs) {
  // block size FIX_HISTOGRAM_BLOCK_SIZE so the fix blocks reduce exactly like
  // the standalone fix kernel (bit-identical); the subtract role is elementwise
  // and block-size invariant
  const int num_subtract_threads = 2 * num_total_bin_;
  const int num_subtract_blocks =
    (num_subtract_threads + FIX_HISTOGRAM_BLOCK_SIZE - 1) / FIX_HISTOGRAM_BLOCK_SIZE;
  const int num_fix_blocks = static_cast<int>(need_fix_histogram_features_.size());
  dim3 grid_dim(num_subtract_blocks + num_fix_blocks, num_pairs);
  FixSubtractHistogramSmallLeafBatchedKernel<<<grid_dim, FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
    num_total_bin_,
    num_subtract_blocks,
    cuda_feature_num_bins_.RawData(),
    cuda_feature_hist_offsets_.RawData(),
    cuda_feature_most_freq_bins_.RawData(),
    cuda_need_fix_histogram_features_.RawData(),
    cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
    cuda_fix_mfb_mask_.RawDataReadOnly(),
    pair_descs,
    any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
    any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr);
}

void CUDAHistogramConstructor::LaunchSubtractHistogramBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const bool any_pair_needs_bit_change_copy) {
  if (!use_quantized_grad_) {
    const int num_subtract_threads = 2 * num_total_bin_;
    const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    if (need_fix_histogram_features_.size() > 0) {
      dim3 fix_grid(static_cast<unsigned int>(need_fix_histogram_features_.size()), num_pairs);
      FixHistogramBatchedKernel<<<fix_grid, FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
        cuda_feature_num_bins_.RawData(),
        cuda_feature_hist_offsets_.RawData(),
        cuda_feature_most_freq_bins_.RawData(),
        cuda_need_fix_histogram_features_.RawData(),
        cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
        pair_descs,
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr);
    }
    dim3 subtract_grid(num_subtract_blocks, num_pairs);
    SubtractHistogramBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      pair_descs,
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
  } else {
    const int num_subtract_threads = num_total_bin_;
    const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    if (need_fix_histogram_features_.size() > 0) {
      dim3 fix_grid(static_cast<unsigned int>(need_fix_histogram_features_.size()), num_pairs);
      FixHistogramDiscretizedBatchedKernel<<<fix_grid, FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
        cuda_feature_num_bins_.RawData(),
        cuda_feature_hist_offsets_.RawData(),
        cuda_feature_most_freq_bins_.RawData(),
        cuda_need_fix_histogram_features_.RawData(),
        cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
        pair_descs,
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr);
    }
    dim3 subtract_grid(num_subtract_blocks, num_pairs);
    SubtractHistogramDiscretizedBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      pair_descs,
      hist_buffer_for_num_bit_change_.RawData(),
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
    if (any_pair_needs_bit_change_copy) {
      CopyChangedNumBitHistogramBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        pair_descs,
        hist_buffer_for_num_bit_change_.RawData(),
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr);
    }
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
