/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_rocm_interop.h>

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include <Falcata/cuda/cuda_driver_shim.hpp>

#include "cuda_meta_batch.hpp"

namespace Falcata {

// cuda_plan key tree_meta_batch: block row y copies segment y of the staged arena to its destination
__global__ void MetaBatchScatterKernel(const uint8_t* __restrict__ staged, const MetaBatchSegments segments) {
  const int s = static_cast<int>(blockIdx.y);
  uint8_t* dst = segments.dst[s];
  const uint8_t* src = staged + segments.src_offset[s];
  const uint32_t bytes = segments.bytes[s];
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes; i += gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

void LaunchMetaBatchScatter(const uint8_t* staged, const MetaBatchSegments& segments) {
  if (segments.count <= 0) {
    return;
  }
  uint32_t max_bytes = 0;
  for (int s = 0; s < segments.count; ++s) {
    max_bytes = std::max(max_bytes, segments.bytes[s]);
  }
  constexpr uint32_t kThreads = 256;
  const uint32_t blocks_x = std::min<uint32_t>(256, (max_bytes + kThreads * 4 - 1) / (kThreads * 4));
  MetaBatchScatterKernel<<<dim3(std::max<uint32_t>(1, blocks_x), static_cast<unsigned int>(segments.count)), kThreads>>>(
    staged, segments);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

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

#if !defined(__HIP_PLATFORM_AMD__)
// Tiled 4-bit fill for column-major sources (stride-1 nibble runs) whose partitions are runs of W consecutive byte
// slots with contiguous destination bytes and stride W; the host checks both (see CompactFill4BitTiledEligible).
// A block owns kFill4BitTiledRows rows of every slot: it stages them into a shared [row][slot] tile with coalesced
// byte loads, then writes each partition's rows as one contiguous run with 16-byte streaming stores.
constexpr int kFill4BitTiledThreads = 256;
constexpr int kFill4BitTiledUnroll = 10;
constexpr int kFill4BitTiledRows = 128;
constexpr size_t kFill4BitTiledSlotBytes = 3 * sizeof(size_t) + sizeof(int);

__host__ __device__ constexpr int Fill4BitTiledPaddedSlots(int num_slots) {
  return num_slots + ((2 - num_slots) & 3);
}

__host__ __device__ constexpr size_t Fill4BitTiledSharedBytes(int num_slots) {
  return static_cast<size_t>(num_slots) * kFill4BitTiledSlotBytes +
         static_cast<size_t>(kFill4BitTiledRows) * static_cast<size_t>(Fill4BitTiledPaddedSlots(num_slots));
}

static_assert(Fill4BitTiledSharedBytes(kFill4BitTiledMaxSlots) <= 48 * 1024,
              "the tiled fill's worst case must fit the default dynamic shared-memory limit");

__global__ void __launch_bounds__(kFill4BitTiledThreads, 4) CUDAFillCompactData4BitTiledKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ compact_data,
  const size_t* __restrict__ bs_src_nib0,
  const size_t* __restrict__ bs_src_nib1,
  const size_t* __restrict__ bs_dst_byte,
  const int* __restrict__ bs_dst_stride,
  const int num_slots,
  const data_size_t num_data) {
  extern __shared__ __align__(16) unsigned char fill_smem[];
  size_t* s_nib0 = reinterpret_cast<size_t*>(fill_smem);
  size_t* s_nib1 = s_nib0 + num_slots;
  size_t* s_dst_byte = s_nib1 + num_slots;
  int* s_dst_stride = reinterpret_cast<int*>(s_dst_byte + num_slots);
  uint8_t* tile = reinterpret_cast<uint8_t*>(s_dst_stride + num_slots);
  constexpr int kHalfRows = kFill4BitTiledRows / 2;
  const size_t kNone = ~static_cast<size_t>(0);
  const int tid = threadIdx.x;
  const int padded_slots = Fill4BitTiledPaddedSlots(num_slots);
  const data_size_t row_start = static_cast<data_size_t>(blockIdx.x) * kFill4BitTiledRows;
  const int rows_valid = static_cast<int>(min(static_cast<data_size_t>(kFill4BitTiledRows), num_data - row_start));
  for (int i = tid; i < num_slots; i += kFill4BitTiledThreads) {
    s_nib0[i] = bs_src_nib0[i];
    s_nib1[i] = bs_src_nib1[i];
    s_dst_byte[i] = bs_dst_byte[i];
    s_dst_stride[i] = bs_dst_stride[i];
  }
  __syncthreads();

  // each source byte holds two consecutive rows of one column
  const int tile_bytes = num_slots * kHalfRows;
  const size_t half_row_start = static_cast<size_t>(row_start >> 1);
  for (int base = tid; base < tile_bytes; base += kFill4BitTiledThreads * kFill4BitTiledUnroll) {
    uint8_t b0[kFill4BitTiledUnroll];
    uint8_t b1[kFill4BitTiledUnroll];
#pragma unroll
    for (int u = 0; u < kFill4BitTiledUnroll; ++u) {
      b0[u] = 0;
      b1[u] = 0;
      const int t = base + u * kFill4BitTiledThreads;
      if (t < tile_bytes) {
        const int i = t / kHalfRows;
        const int j = t % kHalfRows;
        if (row_start + 2 * j < num_data) {
          b0[u] = __ldg(src_data + (s_nib0[i] >> 1) + half_row_start + j);
          if (s_nib1[i] != kNone) b1[u] = __ldg(src_data + (s_nib1[i] >> 1) + half_row_start + j);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < kFill4BitTiledUnroll; ++u) {
      const int t = base + u * kFill4BitTiledThreads;
      if (t < tile_bytes) {
        const int i = t / kHalfRows;
        const int j = t % kHalfRows;
        tile[(2 * j) * padded_slots + i] = static_cast<uint8_t>((b0[u] & 0xf) | ((b1[u] & 0xf) << 4));
        tile[(2 * j + 1) * padded_slots + i] = static_cast<uint8_t>((b0[u] >> 4) | (b1[u] & 0xf0));
      }
    }
  }
  __syncthreads();

  const uintptr_t out_base = reinterpret_cast<uintptr_t>(compact_data);
  for (int s = 0; s < num_slots; s += s_dst_stride[s]) {
    const int width = s_dst_stride[s];
    const size_t run_offset = s_dst_byte[s] + static_cast<size_t>(row_start) * static_cast<size_t>(width);
    const int head = static_cast<int>((out_base + run_offset) & 15);
    const int total = head + rows_valid * width;
    const int num_vec = (total + 15) >> 4;
    uint8_t* aligned = compact_data + run_offset - head;
    const uint8_t* tile_col = tile + s;
    for (int j = tid; j < num_vec; j += kFill4BitTiledThreads) {
      const int off = j << 4;
      if (off >= head && off + 16 <= total) {
        const int q = off - head;
        int row = q / width;
        int m = q - row * width;
        uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          w[k >> 2] |= static_cast<uint32_t>(tile_col[row * padded_slots + m]) << ((k & 3) * 8);
          if (++m == width) {
            m = 0;
            ++row;
          }
        }
        __stcs(reinterpret_cast<uint4*>(aligned + off), make_uint4(w[0], w[1], w[2], w[3]));
      } else {
        for (int k = 0; k < 16; ++k) {
          const int pos = off + k;
          if (pos >= head && pos < total) {
            const int q = pos - head;
            const int row = q / width;
            aligned[pos] = tile_col[row * padded_slots + (q - row * width)];
          }
        }
      }
    }
  }
}

// the one-run tile's start: up to 15 bytes to a 16-byte boundary plus the run's 16-byte phase (< 16)
constexpr size_t kFusedRootRunSlack = 32;

// Shared bytes of the fused fill: the tiled fill's slot metadata and tile, plus the packed first-bin offsets per slot,
// the tile's gradients and the block's histogram (hist_entries cells: one per local bin, or the slot-major table).
size_t FusedRootHistFillSharedBytes(int num_slots, int hist_entries) {
  return Fill4BitTiledSharedBytes(num_slots) + static_cast<size_t>(num_slots) * sizeof(uint32_t) +
         static_cast<size_t>(kFill4BitTiledRows) * sizeof(int32_t) +
         static_cast<size_t>(hist_entries) * sizeof(int32_t) + kFusedRootRunSlack;
}

// Row stride of the slot-major block histogram (cuda_plan key root_hist_slot_major): a multiple of the 32 banks, so
// the cells [nibble][slot] of the 32 consecutive slots a warp adds to fall in 32 distinct banks whatever the nibbles.
__host__ __device__ constexpr int FusedRootHistSlotStride(int num_slots) {
  return (num_slots + 31) & ~31;
}

// twice the tiled fill's block: the histogram limits the fused fill to two blocks per SM, this keeps 1024 threads
constexpr int kFusedRootThreads = 512;

// Tiled 4-bit fill with the root histogram fused in (cuda_plan key fused_root_hist). Every row of every sampled
// column passes through the fill's shared tile, and without bagging the root leaf is exactly all rows, so while a
// tile is staged the block also adds each row's packed grad<<16|hess value into a per-block histogram over the
// compact columns' bins: the 16+16-bit packed int32 accumulation of the quantized construct, under the same
// per-block row bound (tiles_per_block * kFill4BitTiledRows <= 65534 / num_grad_quant_bins). After its last tile
// the block flushes into the root-histogram scratch in the root's format (wrapping int32 adds of the packed sums for
// 16-bit, the construct's int64 conversion otherwise). Integer sums do not depend on grouping, so the scratch holds
// exactly what the root-level construct would add to the root histogram.
// SLOT_MAJOR: the block histogram is laid out [half][nibble][slot] (low then high column of each byte slot, max_span
// nibble rows of FusedRootHistSlotStride cells) instead of one cell per local bin. A warp's lanes take consecutive
// slots of a row, so every atomic hits a distinct bank; with first_bin + nibble the cells of neighbouring slots sit a
// column pair's bin count apart and collide several ways. The flush maps (half, nibble, slot) back to the same local
// bin, so each bin receives the same rows' sum.
template <bool SLOT_MAJOR>
__global__ void __launch_bounds__(kFusedRootThreads, 2) CUDAFillCompactData4BitTiledRootHistKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ compact_data,
  const size_t* __restrict__ bs_src_nib0,
  const size_t* __restrict__ bs_src_nib1,
  const size_t* __restrict__ bs_dst_byte,
  const int* __restrict__ bs_dst_stride,
  const int num_slots,
  const data_size_t num_data,
  const int* __restrict__ slot_first_bin,
  const uint32_t* __restrict__ local_bin_hist_pos,
  const int num_local_bins,
  const int max_span,
  const int tiles_per_block,
  const bool strided_tiles,
  const bool run_copy,
  const bool word_prefetch,
  const bool l2_lead,
  const int32_t* __restrict__ grad_and_hess,
  const bool hist_16bit,
  hist_t* __restrict__ root_hist_scratch) {
  extern __shared__ __align__(16) unsigned char fill_smem[];
  size_t* s_nib0 = reinterpret_cast<size_t*>(fill_smem);
  size_t* s_nib1 = s_nib0 + num_slots;
  size_t* s_dst_byte = s_nib1 + num_slots;
  int* s_dst_stride = reinterpret_cast<int*>(s_dst_byte + num_slots);
  // first local bin of the slot's low column | high column << 16 (0xffff: no high column)
  uint32_t* s_bins = reinterpret_cast<uint32_t*>(s_dst_stride + num_slots);
  int32_t* s_grad = reinterpret_cast<int32_t*>(s_bins + num_slots);
  int32_t* s_hist = s_grad + kFill4BitTiledRows;
  const int hist_stride = FusedRootHistSlotStride(num_slots);
  const int hist_half = max_span * hist_stride;
  const int hist_entries = SLOT_MAJOR ? 2 * hist_half : num_local_bins;
  uint8_t* tile = reinterpret_cast<uint8_t*>(s_hist + hist_entries);
  constexpr int kHalfRows = kFill4BitTiledRows / 2;
  constexpr int kWarps = kFusedRootThreads / 32;
  const size_t kNone = ~static_cast<size_t>(0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int padded_slots = Fill4BitTiledPaddedSlots(num_slots);
  for (int i = tid; i < num_slots; i += kFusedRootThreads) {
    s_nib0[i] = bs_src_nib0[i];
    s_nib1[i] = bs_src_nib1[i];
    s_dst_byte[i] = bs_dst_byte[i];
    s_dst_stride[i] = bs_dst_stride[i];
    const int hi = slot_first_bin[2 * i + 1];
    s_bins[i] = static_cast<uint32_t>(slot_first_bin[2 * i]) | (static_cast<uint32_t>(hi < 0 ? 0xffff : hi) << 16);
  }
  for (int i = tid; i < hist_entries; i += kFusedRootThreads) {
    s_hist[i] = 0;
  }
  const data_size_t num_tiles = (num_data + kFill4BitTiledRows - 1) / kFill4BitTiledRows;
  // contiguous: tiles [blockIdx.x * tiles_per_block, + tiles_per_block); strided: blockIdx.x + k * gridDim.x
  const data_size_t tile_first = strided_tiles ? static_cast<data_size_t>(blockIdx.x)
                                               : static_cast<data_size_t>(blockIdx.x) * tiles_per_block;
  const data_size_t tile_step = strided_tiles ? static_cast<data_size_t>(gridDim.x) : 1;
  const int tile_bytes = num_slots * kHalfRows;
  const uintptr_t out_base = reinterpret_cast<uintptr_t>(compact_data);
  __syncthreads();
  // one run (cuda_plan key root_hist_run_copy): when a single partition spans every slot, a tile's output is one
  // contiguous run of rows * num_slots bytes whose 16-byte phase (run_head) is the same for every tile (128 rows
  // per tile). Staging the tile at row stride num_slots, run_head bytes past a 16-byte boundary, makes it a byte
  // copy of that run, so each aligned output vector is one aligned shared vector load.
  const bool one_run = run_copy && s_dst_stride[0] == num_slots;
  uint8_t* tile_rows = tile;
  int tile_stride = padded_slots;
  int run_head = 0;
  if (one_run) {
    run_head = static_cast<int>((out_base + s_dst_byte[0]) & 15);
    tile_rows = tile + ((16 - (reinterpret_cast<uintptr_t>(tile) & 15)) & 15) + run_head;
    tile_stride = num_slots;
  }
  // word prefetch (cuda_plan key root_hist_prefetch; the host checks every source column starts and ends on a
  // 4-byte boundary): a tile's source is num_slots x 16 aligned 32-bit words per source column (8 rows each), held
  // in registers. The next tile's words are loaded right after the current tile is staged, so they are in flight
  // while the block writes the tile and adds its histogram instead of after the closing barrier. A word holding at
  // least one valid row is loaded whole; its rows past num_data land in tile rows past rows_valid, which are neither
  // written nor histogrammed.
  constexpr int kWordsPerColumn = kFill4BitTiledRows / 8;
  constexpr int kPrefetchWords = kFill4BitTiledMaxSlots * kWordsPerColumn / kFusedRootThreads;
  const int tile_words = num_slots * kWordsPerColumn;
  uint32_t word0[kPrefetchWords];
  uint32_t word1[kPrefetchWords];
  auto load_words = [&](data_size_t tile_index) {
    const data_size_t row_start = tile_index * kFill4BitTiledRows;
    const size_t half_row_start = static_cast<size_t>(row_start >> 1);
#pragma unroll
    for (int u = 0; u < kPrefetchWords; ++u) {
      word0[u] = 0;
      word1[u] = 0;
      const int t = tid + u * kFusedRootThreads;
      if (t < tile_words) {
        const int i = t / kWordsPerColumn;
        const int w = t % kWordsPerColumn;
        if (row_start + 8 * w < num_data) {
          word0[u] = __ldg(reinterpret_cast<const uint32_t*>(src_data + (s_nib0[i] >> 1) + half_row_start) + w);
          if (s_nib1[i] != kNone) {
            word1[u] = __ldg(reinterpret_cast<const uint32_t*>(src_data + (s_nib1[i] >> 1) + half_row_start) + w);
          }
        }
      }
    }
  };
  // L2 lead (cuda_plan key root_hist_l2_prefetch, word path only): a tile's source chunk of every column, cut to
  // the bytes holding rows below num_data (the bytes its word loads read), and its gradients are prefetched into
  // L2 one tile before the register loads of that tile are issued. A chunk is at most 64 bytes from a 4-byte
  // aligned start, so the sectors of its bytes 0, 32 and its last byte cover it.
  auto prefetch_l2 = [&](data_size_t tile_index) {
    const data_size_t row_start = tile_index * kFill4BitTiledRows;
    const int rows = static_cast<int>(min(static_cast<data_size_t>(kFill4BitTiledRows), num_data - row_start));
    const int last_byte = (rows - 1) >> 1;
    const size_t half_row_start = static_cast<size_t>(row_start >> 1);
    for (int t = tid; t < 6 * num_slots; t += kFusedRootThreads) {
      const int c = t / 3;  // slot c >> 1, low (even c) or high (odd c) column
      const int k = t - 3 * c;
      const size_t nib = (c & 1) ? s_nib1[c >> 1] : s_nib0[c >> 1];
      if (nib == kNone) continue;
      const int off = k == 0 ? 0 : (k == 1 ? min(32, last_byte) : last_byte);
      asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(src_data + (nib >> 1) + half_row_start + off)));
    }
    for (int t = tid; 8 * t < rows; t += kFusedRootThreads) {
      asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(grad_and_hess + row_start + 8 * t)));
    }
  };
  if (word_prefetch && tile_first < num_tiles) {
    load_words(tile_first);
    if (l2_lead && tiles_per_block > 1 && tile_first + tile_step < num_tiles) {
      prefetch_l2(tile_first + tile_step);
    }
  }
  for (int step = 0; step < tiles_per_block; ++step) {
    const data_size_t tile_index = tile_first + step * tile_step;
    if (tile_index >= num_tiles) break;  // block-uniform
    const data_size_t row_start = tile_index * kFill4BitTiledRows;
    const int rows_valid = static_cast<int>(min(static_cast<data_size_t>(kFill4BitTiledRows), num_data - row_start));
    if (tid < rows_valid) {
      s_grad[tid] = __ldg(grad_and_hess + row_start + tid);
    }
    if (word_prefetch) {
      // the same tile bytes as the byte staging below: byte q of word w holds rows 8w + 2q (low) and 8w + 2q + 1
#pragma unroll
      for (int u = 0; u < kPrefetchWords; ++u) {
        const int t = tid + u * kFusedRootThreads;
        if (t < tile_words) {
          const int i = t / kWordsPerColumn;
          const int w = t % kWordsPerColumn;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint32_t b0 = (word0[u] >> (8 * q)) & 0xff;
            const uint32_t b1 = (word1[u] >> (8 * q)) & 0xff;
            const int row = 8 * w + 2 * q;
            tile_rows[row * tile_stride + i] = static_cast<uint8_t>((b0 & 0xf) | ((b1 & 0xf) << 4));
            tile_rows[(row + 1) * tile_stride + i] = static_cast<uint8_t>((b0 >> 4) | (b1 & 0xf0));
          }
        }
      }
    }
    // stage the tile exactly as CUDAFillCompactData4BitTiledKernel does
    const size_t half_row_start = static_cast<size_t>(row_start >> 1);
    for (int base = tid; !word_prefetch && base < tile_bytes; base += kFusedRootThreads * kFill4BitTiledUnroll) {
      uint8_t b0[kFill4BitTiledUnroll];
      uint8_t b1[kFill4BitTiledUnroll];
#pragma unroll
      for (int u = 0; u < kFill4BitTiledUnroll; ++u) {
        b0[u] = 0;
        b1[u] = 0;
        const int t = base + u * kFusedRootThreads;
        if (t < tile_bytes) {
          const int i = t / kHalfRows;
          const int j = t % kHalfRows;
          if (row_start + 2 * j < num_data) {
            b0[u] = __ldg(src_data + (s_nib0[i] >> 1) + half_row_start + j);
            if (s_nib1[i] != kNone) b1[u] = __ldg(src_data + (s_nib1[i] >> 1) + half_row_start + j);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kFill4BitTiledUnroll; ++u) {
        const int t = base + u * kFusedRootThreads;
        if (t < tile_bytes) {
          const int i = t / kHalfRows;
          const int j = t % kHalfRows;
          tile_rows[(2 * j) * tile_stride + i] = static_cast<uint8_t>((b0[u] & 0xf) | ((b1[u] & 0xf) << 4));
          tile_rows[(2 * j + 1) * tile_stride + i] = static_cast<uint8_t>((b0[u] >> 4) | (b1[u] & 0xf0));
        }
      }
    }
    __syncthreads();
    if (word_prefetch && step + 1 < tiles_per_block && tile_index + tile_step < num_tiles) {
      load_words(tile_index + tile_step);
      if (l2_lead && step + 2 < tiles_per_block && tile_index + 2 * tile_step < num_tiles) {
        prefetch_l2(tile_index + 2 * tile_step);
      }
    }

    // write the tile exactly as CUDAFillCompactData4BitTiledKernel does
    if (one_run) {
      const size_t run_offset = s_dst_byte[0] + static_cast<size_t>(row_start) * static_cast<size_t>(num_slots);
      const int total = run_head + rows_valid * num_slots;
      const int num_vec = (total + 15) >> 4;
      uint8_t* aligned = compact_data + run_offset - run_head;
      const uint8_t* tile_aligned = tile_rows - run_head;
      for (int j = tid; j < num_vec; j += kFusedRootThreads) {
        const int off = j << 4;
        if (off >= run_head && off + 16 <= total) {
          __stcs(reinterpret_cast<uint4*>(aligned + off), *reinterpret_cast<const uint4*>(tile_aligned + off));
        } else {
          for (int k = 0; k < 16; ++k) {
            const int pos = off + k;
            if (pos >= run_head && pos < total) {
              aligned[pos] = tile_aligned[pos];
            }
          }
        }
      }
    }
    for (int s = 0; !one_run && s < num_slots; s += s_dst_stride[s]) {
      const int width = s_dst_stride[s];
      const size_t run_offset = s_dst_byte[s] + static_cast<size_t>(row_start) * static_cast<size_t>(width);
      const int head = static_cast<int>((out_base + run_offset) & 15);
      const int total = head + rows_valid * width;
      const int num_vec = (total + 15) >> 4;
      uint8_t* aligned = compact_data + run_offset - head;
      const uint8_t* tile_col = tile + s;
      for (int j = tid; j < num_vec; j += kFusedRootThreads) {
        const int off = j << 4;
        if (off >= head && off + 16 <= total) {
          const int q = off - head;
          int row = q / width;
          int m = q - row * width;
          uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            w[k >> 2] |= static_cast<uint32_t>(tile_col[row * padded_slots + m]) << ((k & 3) * 8);
            if (++m == width) {
              m = 0;
              ++row;
            }
          }
          __stcs(reinterpret_cast<uint4*>(aligned + off), make_uint4(w[0], w[1], w[2], w[3]));
        } else {
          for (int k = 0; k < 16; ++k) {
            const int pos = off + k;
            if (pos >= head && pos < total) {
              const int q = pos - head;
              const int row = q / width;
              aligned[pos] = tile_col[row * padded_slots + (q - row * width)];
            }
          }
        }
      }
    }

    // root histogram: one warp per row, lanes across the row's byte slots (distinct columns)
    for (int r = warp; r < rows_valid; r += kWarps) {
      const int32_t grad_hess = s_grad[r];
      const uint8_t* tile_row = tile_rows + r * tile_stride;
      for (int s = lane; s < num_slots; s += 32) {
        const uint32_t b = tile_row[s];
        if (SLOT_MAJOR) {
          // a slot without a high column stages nibble 0 there; the flush skips that cell
          atomicAdd_block(s_hist + (b & 0xf) * hist_stride + s, grad_hess);
          atomicAdd_block(s_hist + hist_half + (b >> 4) * hist_stride + s, grad_hess);
        } else {
          const uint32_t bins = s_bins[s];
          atomicAdd_block(s_hist + (bins & 0xffff) + (b & 0xf), grad_hess);
          if ((bins >> 16) != 0xffff) {
            atomicAdd_block(s_hist + (bins >> 16) + (b >> 4), grad_hess);
          }
        }
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < hist_entries; i += kFusedRootThreads) {
    const int32_t packed_grad_hess = s_hist[i];
    if (packed_grad_hess == 0) continue;  // adding zero is a no-op
    int local_bin = i;
    if (SLOT_MAJOR) {
      // nonzero cells lie in real slots (s < num_slots) at nibbles below their column's span
      const int high = i >= hist_half ? 1 : 0;
      const int cell = i - high * hist_half;
      const int nibble = cell / hist_stride;
      const int s = cell - nibble * hist_stride;
      const uint32_t first = high ? (s_bins[s] >> 16) : (s_bins[s] & 0xffff);
      if (first == 0xffff) continue;  // no high column in this slot
      local_bin = static_cast<int>(first) + nibble;
    }
    const uint32_t pos = __ldg(local_bin_hist_pos + local_bin);
    if (hist_16bit) {
      atomicAdd(reinterpret_cast<int32_t*>(root_hist_scratch) + pos, packed_grad_hess);
    } else {
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd(reinterpret_cast<atomic_add_long_t*>(root_hist_scratch) + pos, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}
#endif  // !defined(__HIP_PLATFORM_AMD__)

// Host wrapper of the fused fill; false (nothing launched) where the tiled kernel does not exist.
bool LaunchFillCompactData4BitTiledRootHistKernel(
  cudaStream_t stream,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* bs_src_nib0,
  const size_t* bs_src_nib1,
  const size_t* bs_dst_byte,
  const int* bs_dst_stride,
  int total_byte_slots,
  data_size_t num_data,
  const int* slot_first_bin,
  const uint32_t* local_bin_hist_pos,
  int num_local_bins,
  int slot_major_span,
  bool strided_tiles,
  bool run_copy,
  bool word_prefetch,
  bool l2_lead,
  int max_rows_per_block,
  const int32_t* grad_and_hess,
  bool hist_16bit,
  hist_t* root_hist_scratch,
  size_t root_hist_scratch_bytes) {
#if !defined(__HIP_PLATFORM_AMD__)
  // the packed per-block row bound, and the default dynamic shared-memory limit (the slot-major table where it
  // was requested and fits, else one cell per local bin)
  const int max_tiles_per_block = std::min(16, max_rows_per_block / kFill4BitTiledRows);
  const size_t slot_major_bytes = FusedRootHistFillSharedBytes(
    total_byte_slots, 2 * slot_major_span * FusedRootHistSlotStride(total_byte_slots));
  const bool slot_major = slot_major_span > 0 && slot_major_bytes <= 48 * 1024;
  const size_t shared_bytes = slot_major ? slot_major_bytes
                                         : FusedRootHistFillSharedBytes(total_byte_slots, num_local_bins);
  if (max_tiles_per_block < 1 || shared_bytes > 48 * 1024) {
    return false;
  }
  const data_size_t num_tiles = (num_data + kFill4BitTiledRows - 1) / kFill4BitTiledRows;
  int tiles_per_block = max_tiles_per_block;
  const double block_rounds = FalcataPlan::Get().root_hist_block_rounds;
  if (FalcataPlan::Get().root_hist_short_blocks && block_rounds >= 1.0) {
    // cuda_plan keys root_hist_short_blocks / root_hist_block_rounds: blocks of equal work start as resident slots
    // free up, so the last ones end up to a block's lifetime apart and the kernel's tail idles about half a block
    // per slot. Size the blocks from the shape (tiles over block_rounds rounds of the blocks resident on the device,
    // by the occupancy API) instead of a fixed count, at least kFusedRootMinTiles: the word path overlaps a tile's
    // loads with the previous tile's work only inside a block, and the block's setup, zeroing and histogram flush
    // stay amortized (1-tile blocks: +37% fill here; 2-4 tiles within 1%). The packed row bound still caps it.
    constexpr int kFusedRootMinTiles = 2;
    // the resident block count, queried once per (device, variant, shared bytes): the launch runs every tree
    struct ResidentQuery {
      int device = -1;
      bool slot_major = false;
      size_t shared_bytes = 0;
      int resident_blocks = 0;
    };
    static thread_local ResidentQuery cached;
    int device = 0;
    CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
    if (cached.device != device || cached.slot_major != slot_major || cached.shared_bytes != shared_bytes) {
      int num_sms = 0;
      int blocks_per_sm = 0;
      CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
      if (slot_major) {
        CUDASUCCESS_OR_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, CUDAFillCompactData4BitTiledRootHistKernel<true>, kFusedRootThreads, shared_bytes));
      } else {
        CUDASUCCESS_OR_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, CUDAFillCompactData4BitTiledRootHistKernel<false>, kFusedRootThreads, shared_bytes));
      }
      cached.device = device;
      cached.slot_major = slot_major;
      cached.shared_bytes = shared_bytes;
      cached.resident_blocks = std::max(1, num_sms * blocks_per_sm);
    }
    const double target_blocks = static_cast<double>(cached.resident_blocks) * block_rounds;
    const int64_t wanted = static_cast<int64_t>(std::ceil(static_cast<double>(num_tiles) / target_blocks));
    tiles_per_block = static_cast<int>(std::min<int64_t>(
      max_tiles_per_block, std::max<int64_t>(std::min(kFusedRootMinTiles, max_tiles_per_block), wanted)));
  }
  CUDASUCCESS_OR_FATAL(cudaMemsetAsync(root_hist_scratch, 0, root_hist_scratch_bytes, stream));
  const int grid = static_cast<int>((num_tiles + tiles_per_block - 1) / tiles_per_block);
  if (slot_major) {
    CUDAFillCompactData4BitTiledRootHistKernel<true><<<grid, kFusedRootThreads, shared_bytes, stream>>>(
      src_data, compact_data, bs_src_nib0, bs_src_nib1, bs_dst_byte, bs_dst_stride, total_byte_slots, num_data,
      slot_first_bin, local_bin_hist_pos, num_local_bins, slot_major_span, tiles_per_block, strided_tiles,
      run_copy, word_prefetch, l2_lead, grad_and_hess,
      hist_16bit, root_hist_scratch);
  } else {
    CUDAFillCompactData4BitTiledRootHistKernel<false><<<grid, kFusedRootThreads, shared_bytes, stream>>>(
      src_data, compact_data, bs_src_nib0, bs_src_nib1, bs_dst_byte, bs_dst_stride, total_byte_slots, num_data,
      slot_first_bin, local_bin_hist_pos, num_local_bins, 0, tiles_per_block, strided_tiles, run_copy,
      word_prefetch, l2_lead, grad_and_hess, hist_16bit, root_hist_scratch);
  }
  return true;
#else
  (void)stream; (void)src_data; (void)compact_data; (void)bs_src_nib0; (void)bs_src_nib1; (void)bs_dst_byte;
  (void)bs_dst_stride; (void)total_byte_slots; (void)num_data; (void)slot_first_bin; (void)local_bin_hist_pos;
  (void)num_local_bins; (void)slot_major_span; (void)strided_tiles; (void)run_copy; (void)word_prefetch; (void)l2_lead;
  (void)max_rows_per_block; (void)grad_and_hess; (void)hist_16bit;
  (void)root_hist_scratch; (void)root_hist_scratch_bytes;
  return false;
#endif
}

// Host wrapper called from cuda_histogram_constructor.cpp. `tiled` is the host's verdict from
// CompactFill4BitTiledEligible; ROCm builds always take the per-cell kernel.
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
  data_size_t num_data,
  bool tiled) {
#if !defined(__HIP_PLATFORM_AMD__)
  if (tiled) {
    const int grid = (num_data + kFill4BitTiledRows - 1) / kFill4BitTiledRows;
    CUDAFillCompactData4BitTiledKernel<<<grid, kFill4BitTiledThreads, Fill4BitTiledSharedBytes(total_byte_slots), stream>>>(
      src_data, compact_data, bs_src_nib0, bs_src_nib1, bs_dst_byte, bs_dst_stride, total_byte_slots, num_data);
    return;
  }
#else
  (void)tiled;
#endif
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

// colmajor_direct: with an odd row count the high nibble of each column's last byte lies past the last row.
// Columns start column_stride bytes apart (column_stride >= column_bytes).
__global__ void ClearColMajorPadNibblesKernel(uint8_t* colmajor, const int num_columns, const size_t column_stride,
                                              const size_t column_bytes) {
  const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c < num_columns) {
    colmajor[static_cast<size_t>(c) * column_stride + column_bytes - 1] &= 0x0f;
  }
}

void LaunchClearColMajorPadNibbles(uint8_t* colmajor, const int num_columns, const size_t column_stride,
                                   const size_t column_bytes) {
  if (num_columns <= 0) return;
  const int block = 256;
  ClearColMajorPadNibblesKernel<<<(num_columns + block - 1) / block, block>>>(colmajor, num_columns, column_stride,
                                                                               column_bytes);
}

// One-time nibble transpose: packed row-major bin matrix -> global
// column-major nibbles (column c occupies nibbles [c*num_data_pad,
// c*num_data_pad + num_data)). Scattered reads, contiguous writes; runs once
// at Init when cuda_plan key colmajor_fill is eligible. The per-tree compact
// fill then reads columns CONTIGUOUSLY (stride-1 nibble metadata) instead of
// dragging every row sector through the memory system (~10x less fill
// traffic at feature_fraction=0.1).
__global__ void CUDATransposeToColMajorNibbleKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ colmajor_data,
  const size_t* __restrict__ col_src_nib_base,
  const int* __restrict__ col_src_stride_nib,
  const int num_columns,
  const data_size_t num_data,
  const size_t num_data_pad) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= num_columns) return;
  const size_t src_base = col_src_nib_base[col];
  const size_t src_stride = static_cast<size_t>(col_src_stride_nib[col]);
  const size_t dst_base = static_cast<size_t>(col) * num_data_pad;
  const data_size_t row_stride = static_cast<data_size_t>(gridDim.y) * static_cast<data_size_t>(blockDim.y);
  // two rows per destination byte: each thread owns even rows and packs pairs
  for (data_size_t r2 = (blockIdx.y * blockDim.y + threadIdx.y) * 2; r2 < num_data; r2 += row_stride * 2) {
    const size_t nib0 = src_base + static_cast<size_t>(r2) * src_stride;
    const uint8_t lo = (src_data[nib0 >> 1] >> ((nib0 & 1) << 2)) & 0xf;
    uint8_t hi = 0;
    if (r2 + 1 < num_data) {
      const size_t nib1 = src_base + static_cast<size_t>(r2 + 1) * src_stride;
      hi = (src_data[nib1 >> 1] >> ((nib1 & 1) << 2)) & 0xf;
    }
    colmajor_data[(dst_base + static_cast<size_t>(r2)) >> 1] = static_cast<uint8_t>(lo | (hi << 4));
  }
}

void LaunchTransposeToColMajorNibbleKernel(
  const uint8_t* src_data,
  uint8_t* colmajor_data,
  const size_t* col_src_nib_base,
  const int* col_src_stride_nib,
  int num_columns,
  data_size_t num_data,
  size_t num_data_pad) {
  const int TX = 32;
  const int TY = 16;
  int grid_y = static_cast<int>(((num_data + 1) / 2 + TY - 1) / TY);
  if (grid_y > 32768) grid_y = 32768;
  dim3 block_dim(TX, TY);
  dim3 grid_dim((num_columns + TX - 1) / TX, grid_y);
  CUDATransposeToColMajorNibbleKernel<<<grid_dim, block_dim>>>(
    src_data, colmajor_data, col_src_nib_base, col_src_stride_nib,
    num_columns, num_data, num_data_pad);
  CUDASUCCESS_OR_FATAL(cudaDeviceSynchronize());
}

// Codec fill (pack_bit3 / pack_radix5 / pack_radix6): transcode the tree's
// sampled columns from the 4-bit source row matrix into CODEC-packed uint32
// words. One thread per (destination word slot, row group): a word slot is a
// (partition, word-within-row) pair covering up to CODEC::kValuesPerWord
// consecutive compact columns; per-column source nibble bases/strides come in
// via col_src_nib_base / col_src_stride_nib (indexed by compact column).
template <class CODEC>
__global__ void CUDAFillCompactCodecKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ compact_data,
  const size_t* __restrict__ col_src_nib_base,
  const int* __restrict__ col_src_stride_nib,
  const size_t* __restrict__ ws_dst_byte,
  const int* __restrict__ ws_dst_stride,
  const int* __restrict__ ws_first_col,
  const uint8_t* __restrict__ ws_num_digits,
  const int total_word_slots,
  const data_size_t num_data) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= total_word_slots) return;
  const size_t dst_byte = ws_dst_byte[slot];
  const size_t dst_stride = static_cast<size_t>(ws_dst_stride[slot]);
  const int first_col = ws_first_col[slot];
  const int ndig = static_cast<int>(ws_num_digits[slot]);
  const data_size_t row_stride = static_cast<data_size_t>(gridDim.y) * static_cast<data_size_t>(blockDim.y);
  for (data_size_t row = blockIdx.y * blockDim.y + threadIdx.y; row < num_data; row += row_stride) {
    uint32_t w = 0;
    uint32_t place = 1;  // running R^d: keeps the radix multiplier in a register
    #pragma unroll 4
    for (int d = 0; d < ndig; ++d) {
      const int col = first_col + d;
      const size_t nib = col_src_nib_base[col] +
        static_cast<size_t>(row) * static_cast<size_t>(col_src_stride_nib[col]);
      const uint32_t v = (src_data[nib >> 1] >> ((nib & 1) << 2)) & 0xfu;
      if constexpr (CODEC::kId == PackCodecId::kBit3x32) {
        w |= v << (3 * d);
      } else {
        w += v * place;
        place *= static_cast<uint32_t>(CODEC::kMaxBins);
      }
    }
    *reinterpret_cast<uint32_t*>(compact_data + dst_byte + static_cast<size_t>(row) * dst_stride) = w;
  }
}

// Host wrapper called from cuda_histogram_constructor.cpp.
void LaunchFillCompactCodecKernel(
  cudaStream_t stream,
  PackCodecId codec,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* col_src_nib_base,
  const int* col_src_stride_nib,
  const size_t* ws_dst_byte,
  const int* ws_dst_stride,
  const int* ws_first_col,
  const uint8_t* ws_num_digits,
  int total_word_slots,
  data_size_t num_data) {
  const int TX = 32;
  const int TY = 32;
  int grid_y = (num_data + TY - 1) / TY;
  if (grid_y > 32768) grid_y = 32768;
  dim3 block_dim(TX, TY);
  dim3 grid_dim((total_word_slots + TX - 1) / TX, grid_y);
  switch (codec) {
    case PackCodecId::kBit3x32:
      CUDAFillCompactCodecKernel<PackBit3x32><<<grid_dim, block_dim, 0, stream>>>(
        src_data, compact_data, col_src_nib_base, col_src_stride_nib,
        ws_dst_byte, ws_dst_stride, ws_first_col, ws_num_digits, total_word_slots, num_data);
      break;
    case PackCodecId::kRadix5x32:
      CUDAFillCompactCodecKernel<PackRadix5x32><<<grid_dim, block_dim, 0, stream>>>(
        src_data, compact_data, col_src_nib_base, col_src_stride_nib,
        ws_dst_byte, ws_dst_stride, ws_first_col, ws_num_digits, total_word_slots, num_data);
      break;
    case PackCodecId::kRadix6x32:
      CUDAFillCompactCodecKernel<PackRadix6x32><<<grid_dim, block_dim, 0, stream>>>(
        src_data, compact_data, col_src_nib_base, col_src_stride_nib,
        ws_dst_byte, ws_dst_stride, ws_first_col, ws_num_digits, total_word_slots, num_data);
      break;
    case PackCodecId::kRadix7x32:
      CUDAFillCompactCodecKernel<PackRadix7x32><<<grid_dim, block_dim, 0, stream>>>(
        src_data, compact_data, col_src_nib_base, col_src_stride_nib,
        ws_dst_byte, ws_dst_stride, ws_first_col, ws_num_digits, total_word_slots, num_data);
      break;
    default:
      break;
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
  // Evict-first streaming loads: bin bytes have no intra-level reuse, and
  // deprioritizing them in L2 leaves room for histogram subtraction re-reads
  // (+8% covtype-deep, +10% year, neutral numerai -- whose hot kernel reads
  // through the pack codecs, not here). Unconditional: measured >= neutral
  // on every shape including fully-L2-resident ones.
  if (IS_4BIT) {
    const uint32_t packed = static_cast<uint32_t>(__ldcs(&row_ptr[column_in_partition >> 1]));
    return (packed >> ((column_in_partition & 1) << 2)) & 0xfu;
  } else {
    return static_cast<uint32_t>(__ldcs(&row_ptr[column_in_partition]));
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
  const bool hist_fp32,
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
  int row_stride = num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + (IS_4BIT ?
    PackedPartitionRows(packed_partition_byte_offsets, blockIdx.x, num_data, &row_stride) :
    static_cast<size_t>(partition_column_start) * num_data);
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
  if (hist_fp32) {
    float* feature_histogram_ptr = reinterpret_cast<float*>(smaller_leaf_splits->hist_in_leaf) + (partition_hist_start << 1);
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      if (bin_used == nullptr || bin_used[partition_hist_start + (i >> 1)]) {
        atomicAdd_system(feature_histogram_ptr + i, static_cast<float>(shared_hist[i]));
      }
    }
  } else {
    hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      if (bin_used == nullptr || bin_used[partition_hist_start + (i >> 1)]) {
        atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
      }
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
  const data_size_t num_data,
  const bool hist_fp32) {
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  ConstructHistogramDenseInner<BIN_TYPE, HIST_TYPE, false, IS_4BIT>(
    smaller_leaf_splits, shared_hist, cuda_gradients, cuda_hessians, nullptr, data,
    column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
    packed_partition_byte_offsets,
    is_feature_used_bytree, num_data, static_cast<int>(gridDim.y * blockDim.y), hist_fp32);
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
  const data_size_t num_data,
  const bool hist_fp32) {
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
  int data_row_stride = num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + (IS_4BIT ?
    PackedPartitionRows(packed_partition_byte_offsets, blockIdx.x, num_data, &data_row_stride) :
    static_cast<size_t>(partition_column_start) * num_data);
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  // column_hist_offsets is PARTITION-RELATIVE (it indexes the per-partition
  // shared histogram in the shared-memory body); the global histogram position
  // additionally needs the partition's own start offset
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t hist_offset = (partition_hist_start + column_hist_offsets[column_index]) << 1;
  hist_t* hist_ptr = smaller_leaf_splits->hist_in_leaf + hist_offset;
  float* hist_ptr32 = reinterpret_cast<float*>(smaller_leaf_splits->hist_in_leaf) + hist_offset;
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
    if (hist_fp32) {
      atomicAdd(hist_ptr32 + pos, static_cast<float>(grad));
      atomicAdd(hist_ptr32 + pos + 1, static_cast<float>(hess));
    } else {
      atomicAdd(hist_ptr + pos, static_cast<hist_t>(grad));
      atomicAdd(hist_ptr + pos + 1, static_cast<hist_t>(hess));
    }
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
  const uint8_t* bin_used,
  const bool hist_fp32,
  const CUDAHybridGraphLoopStateOpt gstate) {
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  // graphs A2: inside the graph loop the grid is frozen at a pow2 bucket of
  // the level shape; pair-blocks beyond the live range exit before any read
  // (the y over-provision is already handled below: the row-grouping extent is
  // device-computed, not grid-derived, and idle row blocks exit in the inner)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.z)) {
    return;
  }
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
      is_feature_used_bytree, num_data, hist_fp32);
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
    // live pair extent: gridDim.z on the exact host grid, the loop state's
    // live split count inside the graph (whose gridDim.z is only a bound)
    const int num_pairs = HybridGraphLivePairCount(gstate, static_cast<int>(gridDim.z));
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
      is_feature_used_bytree, num_data, dim_y, hist_fp32, bin_used);
  } else {
    ConstructHistogramDenseInner<BIN_TYPE, HIST_TYPE, false, IS_4BIT, USE_GH2>(
      smaller_struct, shared_hist, cuda_gradients, cuda_hessians, cuda_gh, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      is_feature_used_bytree, num_data, dim_y, hist_fp32, bin_used);
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

// Deterministic float-mode sparse construct. The atomic kernel above sums a
// bin's gradients in hardware-scheduling order (shared atomics across a
// block's row groups, system atomics across row tiles); on degenerate data
// that jitter decides between a healthy and a runaway boosting basin, so
// identical runs ship different models -- a rare-class multiclass dataset
// trained to logloss 0.013 or 23.6 depending on the run. Here row group y is
// the sole writer of slot row y (a row's entries hit distinct bins: features
// own disjoint bin ranges), which removes the shared atomics, and both
// reductions run in fixed index order, so the histogram is a function of the
// inputs alone. Quantized training never routes here; its integer atomics
// are order-invariant already.
template <typename BIN_TYPE, typename DATA_PTR_TYPE>
__global__ void CUDAConstructHistogramSparseDeterministicKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data,
  const uint32_t slot_stride,  // double elements; >= every partition's item count
  hist_t* tile_partials,       // [gridDim.y][2 * num_total_bin]
  const uint32_t num_total_items) {
  extern __shared__ unsigned char det_smem_raw[];
  double* slot_rows = reinterpret_cast<double*>(det_smem_raw);
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
  for (unsigned int i = thread_idx; i < num_items_in_partition * blockDim.y; i += num_threads_per_block) {
    const unsigned int y = i / num_items_in_partition;
    const unsigned int item = i % num_items_in_partition;
    slot_rows[static_cast<size_t>(y) * slot_stride + item] = 0.0;
  }
  __syncthreads();
  double* my_slot_row = slot_rows + static_cast<size_t>(threadIdx.y) * slot_stride;
  const data_size_t block_start = (static_cast<size_t>(blockIdx.y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx.y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y);
  // One writer per slot row: thread y walks its rows' FULL entry lists, so
  // every bin's addends arrive in row order. (Mapping threadIdx.x to entry
  // positions instead would race: entries at equal positions in different
  // rows belong to different features, so two x threads can hit the same bin
  // of consecutive rows with unsynchronized adds. The x lanes stay idle here;
  // the lost parallelism moves to the tile grid.)
  if (threadIdx.x == 0) {
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
      const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      for (DATA_PTR_TYPE e = row_start; e < row_end; ++e) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[e]);
        const uint32_t pos = bin << 1;
        my_slot_row[pos] += static_cast<double>(grad);
        my_slot_row[pos + 1] += static_cast<double>(hess);
      }
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  // Fixed-order reduce over the block's row groups, then a plain store into
  // this tile's partial: (partition, tile) blocks own disjoint item ranges,
  // so no atomics and no ordering ambiguity remain anywhere in the construct.
  hist_t* tile_row = tile_partials + static_cast<size_t>(blockIdx.y) * num_total_items;
  const uint32_t out_base = partition_hist_start << 1;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    double acc = 0.0;
    for (unsigned int y = 0; y < blockDim.y; ++y) {
      acc += slot_rows[static_cast<size_t>(y) * slot_stride + i];
    }
    tile_row[out_base + i] = acc;
  }
}

// Deterministic float-mode sparse construct for datasets whose widest
// partition's slot row cannot fit the dynamic shared budget: the slot rows
// live in a global scratch slab instead. Ownership and ordering are identical
// to the shared-memory kernel above -- row group y of tile blockIdx.y is the
// sole writer of slot row (blockIdx.y * blockDim.y + y), threadIdx.x == 0
// walks each row's full entry list, and a fixed-order reduce stores this
// tile's partial. Slot positions are GLOBAL (offset by this partition's
// histogram start): with partition-relative positions, different partitions'
// blocks would zero overlapping low ranges of the SAME row -- one partition's
// late zeroing wipes another's accumulated sums, and the reduce reads the
// contamination. Global positions make every partition own a disjoint range
// of every row.
template <typename BIN_TYPE, typename DATA_PTR_TYPE>
__global__ void CUDAConstructHistogramSparseGMDeterministicKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data,
  const uint32_t slot_stride,  // double elements; == the full histogram's item count
  hist_t* slots,               // [tile * blockDim.y + row_group][slot_stride]
  hist_t* tile_partials,       // [gridDim.y][2 * num_total_bin]
  const uint32_t num_total_items) {
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
  hist_t* block_slot_rows = slots + static_cast<size_t>(blockIdx.y) * blockDim.y * slot_stride;
  const uint32_t slot_base = partition_hist_start << 1;
  // Cooperative zero of ALL blockDim.y slot rows this block owns, not just
  // threadIdx.y's: the scratch persists across launches and the accumulate
  // below ADDS into the slots, so every position of every row must be zeroed
  // or the previous leaf's sums leak into this histogram.
  const uint32_t num_zero_items = num_items_in_partition * blockDim.y;
  for (uint32_t i = thread_idx; i < num_zero_items; i += num_threads_per_block) {
    const uint32_t y = i / num_items_in_partition;
    const uint32_t j = i % num_items_in_partition;
    block_slot_rows[static_cast<size_t>(y) * slot_stride + slot_base + j] = 0.0;
  }
  __syncthreads();
  hist_t* my_slot_row = block_slot_rows + static_cast<size_t>(threadIdx.y) * slot_stride;
  const data_size_t block_start = (static_cast<size_t>(blockIdx.y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx.y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y);
  // One writer per slot row; see the shared-memory kernel above for why the
  // x lanes cannot map to entry positions.
  if (threadIdx.x == 0) {
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
      const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      for (DATA_PTR_TYPE e = row_start; e < row_end; ++e) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[e]);
        const uint32_t pos = slot_base + (bin << 1);
        my_slot_row[pos] += static_cast<double>(grad);
        my_slot_row[pos + 1] += static_cast<double>(hess);
      }
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  // Fixed-order reduce over the block's row groups, then a plain store into
  // this tile's partial: (partition, tile) blocks own disjoint item ranges,
  // so no atomics and no ordering ambiguity remain anywhere in the construct.
  hist_t* tile_row = tile_partials + static_cast<size_t>(blockIdx.y) * num_total_items;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    double acc = 0.0;
    for (unsigned int y = 0; y < blockDim.y; ++y) {
      acc += block_slot_rows[static_cast<size_t>(y) * slot_stride + slot_base + i];
    }
    tile_row[slot_base + i] = acc;
  }
}

// Deterministic float-mode dense construct for partitions too wide for shared
// memory (the GlobalMemory variant). Slot row (tile * dy + threadIdx.y) is
// owned exclusively by that row group; within it the x lanes walk a row's
// COLUMNS, and dense columns own disjoint bin ranges, so no atomics are
// needed anywhere and every bin's addends arrive in a fixed order. Slots are
// double: float32 gradients summed in double are exact, which makes the
// histogram bit-equal to the CPU reference instead of noise-shifted -- that
// equality is what makes gain TIES exact and the tie-break deterministic.
// IS_4BIT reads the packed row layout through ReadDenseBin (row stride and
// partition base from packed_partition_byte_offsets, nullptr and unused in
// the 8-bit variant); the bin -> slot arithmetic is unchanged because a
// nibble value still indexes its own column's bin range.
// GRAD_ONLY: vector-leaf gradient plane t >= 1. The hessian is shared across
// targets and every consumer reads it from plane 0, so plane t's hessian cells
// carry no information; skipping their read-modify-write halves the inner
// loop's slot traffic. The zeroed slots still merge into the plane, so the
// cells are defined (zero), just meaningless.
template <typename BIN_TYPE, bool IS_4BIT, bool GRAD_ONLY = false>
__device__ __forceinline__ void ConstructHistogramDenseGMDeterministicInner(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used_bytree,
  const data_size_t num_data,
  const uint32_t slot_stride,  // double elements; >= every partition's item count
  hist_t* slots,               // [tile * dy + row_group][slot_stride]
  const int dy,
  const int dim_y) {           // total slot rows the data is distributed over
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  int row_stride = num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + (IS_4BIT ?
    PackedPartitionRows(packed_partition_byte_offsets, blockIdx.x, num_data, &row_stride) :
    static_cast<size_t>(partition_column_start) * num_data);
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  // Slot positions are GLOBAL (offset by this partition's histogram start):
  // with partition-relative positions, different partitions' blocks would
  // zero overlapping low ranges of the SAME row -- one partition's late
  // zeroing wipes another's accumulated sums, and its merge reads the
  // contamination. Global positions make every partition own a disjoint
  // range of every row.
  hist_t* block_slot_rows = slots + static_cast<size_t>(blockIdx.y) * dy * slot_stride;
  const uint32_t zero_base = partition_hist_start << 1;
  // Cooperative zero of ALL dy slot rows this block owns, not just
  // threadIdx.y's: the scratch persists across launches and the accumulate
  // below ADDS into the slots, so every position of every row must be zeroed
  // or the previous leaf's sums leak into this histogram.
  const uint32_t num_zero_items = num_items_in_partition * static_cast<uint32_t>(dy);
  for (uint32_t i = thread_idx; i < num_zero_items; i += num_threads_per_block) {
    const uint32_t y = i / num_items_in_partition;
    const uint32_t j = i % num_items_in_partition;
    block_slot_rows[static_cast<size_t>(y) * slot_stride + zero_base + j] = 0.0;
  }
  __syncthreads();
  hist_t* my_slot_row = block_slot_rows + static_cast<size_t>(threadIdx.y) * slot_stride;
  const data_size_t block_start = (static_cast<size_t>(blockIdx.y) * dy) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(dy)));
  for (data_size_t inner = static_cast<data_size_t>(threadIdx.y); inner < block_num_data; inner += dy) {
    const data_size_t data_index = data_indices_ref_this_block[inner];
    const score_t grad = cuda_gradients[data_index];
    const score_t hess = GRAD_ONLY ? 0.0f : cuda_hessians[data_index];
    const BIN_TYPE* row = data_ptr + static_cast<size_t>(data_index) * row_stride;
    for (int c = static_cast<int>(threadIdx.x); c < num_columns_in_partition; c += static_cast<int>(blockDim.x)) {
      const int column_index = c + partition_column_start;
      if (is_feature_used_bytree != nullptr && !is_feature_used_bytree[column_index]) {
        continue;
      }
      const uint32_t bin = ReadDenseBin<IS_4BIT>(row, static_cast<unsigned int>(c));
      const uint32_t pos = ((partition_hist_start + column_hist_offsets[column_index]) << 1) + (bin << 1);
      my_slot_row[pos] += static_cast<double>(grad);
      if (!GRAD_ONLY) {
        my_slot_row[pos + 1] += static_cast<double>(hess);
      }
    }
  }
}

template <typename BIN_TYPE, bool IS_4BIT = false, bool GRAD_ONLY = false>
__global__ void CUDAConstructHistogramDenseGMDeterministicKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used_bytree,
  const data_size_t num_data,
  const uint32_t slot_stride,
  hist_t* slots,
  const int dy) {
  ConstructHistogramDenseGMDeterministicInner<BIN_TYPE, IS_4BIT, GRAD_ONLY>(
    smaller_leaf_splits, cuda_gradients, cuda_hessians, data,
    column_hist_offsets, column_hist_offsets_full,
    feature_partition_column_index_offsets, packed_partition_byte_offsets,
    is_feature_used_bytree, num_data, slot_stride, slots, dy,
    static_cast<int>(gridDim.y) * dy);
}

// Device mirror of the batched construct kernels' per-pair skip gates: the
// host-set construct_valid flag plus the min_data/min_hessian early return
// evaluated from the (device-written) pair structs, so the speculative
// single-sync flow -- whose host bounds are not the actual child sizes -- takes
// the identical decision the classic per-leaf host gate would. Block-uniform,
// so the early return is divergence-free. The deterministic batched construct
// AND its merge both call this: a pair whose construct is skipped must also
// keep its (stale-slot) merge from overwriting the leaf histogram.
__device__ __forceinline__ bool HybridPairConstructSkipped(
  const CUDAHybridPairDescriptor* desc,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf) {
  if (!desc->construct_valid) {
    return true;
  }
  const data_size_t num_data_smaller = desc->smaller_struct->num_data_in_leaf;
  const double sum_hessians_smaller = desc->smaller_struct->sum_of_hessians;
  const bool has_larger = desc->larger_struct->leaf_index >= 0;
  const data_size_t num_data_larger = has_larger ? desc->larger_struct->num_data_in_leaf : 0;
  const double sum_hessians_larger = has_larger ? desc->larger_struct->sum_of_hessians : 0.0;
  return (num_data_smaller <= min_data_in_leaf || sum_hessians_smaller <= min_sum_hessian_in_leaf) &&
         (num_data_larger <= min_data_in_leaf || sum_hessians_larger <= min_sum_hessian_in_leaf);
}

// Device replica of the per-pair slot-row layout shared by the deterministic
// batched construct and its merge (the two MUST agree on it exactly):
//   pair_rows = total_slot_rows / live_pairs   (>= dy by the eligibility and
//                                               capture-time dy clamps)
//   tiles     = clamp(ceil(leaf_rows / (kDetRowsPerThread * dy)), 1,
//                     pair_rows / dy)
//   rows_used = tiles * dy                     (rows zeroed AND merged)
// Inputs are device-exact (live pair count from the loop state or the launch
// grid, leaf size from the pair struct), so both kernels derive the identical
// layout with no host round trip -- which is what lets the graph loop freeze
// the launch grids at upper bounds.
__device__ __forceinline__ int DetDensePairRows(const int total_slot_rows,
                                                const int live_pairs) {
  return max(1, total_slot_rows / max(1, live_pairs));
}

__device__ __forceinline__ int DetDensePairTiles(const data_size_t leaf_rows,
                                                 const int pair_rows,
                                                 const int dy) {
  const int tile_cap = max(1, pair_rows / dy);
  const data_size_t denom = static_cast<data_size_t>(kDetRowsPerThread) * dy;
  const int tiles_data = static_cast<int>(
    max(static_cast<data_size_t>(1), (leaf_rows + denom - 1) / denom));
  return min(tiles_data, tile_cap);
}

// Deterministic dense construct for the hybrid LEVEL batch: blockIdx.z selects
// the sibling pair and each pair owns a disjoint pair_rows * slot_stride slab
// of the deterministic slot scratch, so all pairs of a level accumulate
// concurrently with the per-leaf kernel's exact fixed-order math (see
// ConstructHistogramDenseGMDeterministicInner). The batch runs on the single
// batched-level stream (cuda_stream_) with no pipeline concurrency, so the
// WHOLE kNumHistPipelines slab is available to carve into per-pair regions.
// Unlike the atomic batched kernel there is no small-leaf direct body (its
// global float atomics are exactly the order dependence this kernel removes).
// The per-pair layout (pair rows, tile count) is derived on device (see
// DetDensePairRows / DetDensePairTiles): under the graph loop the launch grid
// is a frozen upper bound and the live pair count comes from the loop state;
// on the host path the grid is exact and gstate is null. The double-slot sums
// are exact, so any row grouping produces the identical merged histogram --
// host and graph paths agree bit-for-bit by construction.
template <typename BIN_TYPE, bool IS_4BIT = false>
__global__ void CUDAConstructHistogramDenseGMDeterministicBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used_bytree,
  const data_size_t num_data,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const uint32_t slot_stride,
  hist_t* slots,          // [live_pairs][pair_rows][slot_stride]
  const int total_slot_rows,
  const int dy,
  const CUDAHybridGraphLoopStateOpt gstate) {
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.z)) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (HybridPairConstructSkipped(desc, min_data_in_leaf, min_sum_hessian_in_leaf)) {
    return;
  }
  const int live_pairs = HybridGraphLivePairCount(gstate, static_cast<int>(gridDim.z));
  const int pair_rows = DetDensePairRows(total_slot_rows, live_pairs);
  const int tiles = DetDensePairTiles(desc->smaller_struct->num_data_in_leaf, pair_rows, dy);
  if (static_cast<int>(blockIdx.y) >= tiles) {
    return;  // idle tile (frozen-grid upper bound); exits before the zeroing
  }
  hist_t* pair_slots = slots +
    static_cast<size_t>(blockIdx.z) * static_cast<size_t>(pair_rows) * slot_stride;
  ConstructHistogramDenseGMDeterministicInner<BIN_TYPE, IS_4BIT>(
    desc->smaller_struct, cuda_gradients, cuda_hessians, data,
    column_hist_offsets, column_hist_offsets_full,
    feature_partition_column_index_offsets, packed_partition_byte_offsets,
    is_feature_used_bytree, num_data, slot_stride, pair_slots, dy, tiles * dy);
}

// Fixed-order merge for the dense deterministic construct: one block row per
// partition, threads over its items, summing the (tile, row group) slot rows
// in row-major (tile asc, then row group asc) order and storing into the
// leaf histogram slot.
__global__ void MergeDeterministicDenseHistogramKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const hist_t* slots,
  const uint32_t* column_hist_offsets_full,
  const uint32_t slot_stride,
  const int num_rows) {
  const uint32_t start = column_hist_offsets_full[blockIdx.y];
  const uint32_t items = (column_hist_offsets_full[blockIdx.y + 1] - start) << 1;
  const unsigned int j = threadIdx.x + static_cast<unsigned int>(blockIdx.x) * blockDim.x;
  if (j >= items) {
    return;
  }
  const uint32_t i = (start << 1) + j;  // global slot position
  double acc = 0.0;
  for (int r = 0; r < num_rows; ++r) {
    acc += slots[static_cast<size_t>(r) * slot_stride + i];
  }
  smaller_leaf_splits->hist_in_leaf[i] = acc;
}

// Batched per-level variant: blockIdx.z selects the pair; the per-pair slab
// region and the merged row count are the DetDensePairRows/DetDensePairTiles
// device replica, so the merge reads exactly the rows its construct zeroed
// and accumulated -- on the host path (exact grid, gstate null) and inside
// the graph loop (frozen upper-bound grid, live pair count from the loop
// state) alike. Replicates the construct's per-pair skip gates so a
// gated-out pair's stale slots never overwrite its leaf histogram (the merge
// STORES; only live pairs may write).
__global__ void MergeDeterministicDenseHistogramBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const hist_t* slots,
  const uint32_t* column_hist_offsets_full,
  const uint32_t slot_stride,
  const int total_slot_rows,
  const int dy,
  const int host_num_pairs,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const CUDAHybridGraphLoopStateOpt gstate) {
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.z)) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (HybridPairConstructSkipped(desc, min_data_in_leaf, min_sum_hessian_in_leaf)) {
    return;
  }
  const uint32_t start = column_hist_offsets_full[blockIdx.y];
  const uint32_t items = (column_hist_offsets_full[blockIdx.y + 1] - start) << 1;
  const unsigned int j = threadIdx.x + static_cast<unsigned int>(blockIdx.x) * blockDim.x;
  if (j >= items) {
    return;
  }
  const uint32_t i = (start << 1) + j;  // global slot position
  const int live_pairs = HybridGraphLivePairCount(gstate, host_num_pairs);
  const int pair_rows = DetDensePairRows(total_slot_rows, live_pairs);
  const int num_rows =
    DetDensePairTiles(desc->smaller_struct->num_data_in_leaf, pair_rows, dy) * dy;
  const hist_t* pair_slots = slots +
    static_cast<size_t>(blockIdx.z) * static_cast<size_t>(pair_rows) * slot_stride;
  double acc = 0.0;
  for (int r = 0; r < num_rows; ++r) {
    acc += pair_slots[static_cast<size_t>(r) * slot_stride + i];
  }
  desc->smaller_struct->hist_in_leaf[i] = acc;
}

// Sum the row-tile partials in tile-index order and store: the leaf
// histogram is a fixed-order reduction of fixed-order partials, end to end.
__global__ void MergeDeterministicHistogramKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const hist_t* tile_partials,
  const int num_total_items,
  const int num_tiles) {
  const unsigned int i = threadIdx.x + static_cast<unsigned int>(blockIdx.x) * blockDim.x;
  if (i >= static_cast<unsigned int>(num_total_items)) {
    return;
  }
  double acc = 0.0;
  for (int t = 0; t < num_tiles; ++t) {
    acc += static_cast<double>(tile_partials[static_cast<size_t>(t) * num_total_items + i]);
  }
  smaller_leaf_splits->hist_in_leaf[i] = acc;
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


// Direct small-leaf variant of the discretized construct. For pairs whose
// smaller leaf is tiny, the shared-histogram body pays zero + syncthreads +
// merge proportional to PARTITION BINS regardless of rows -- for a few-hundred
// row leaf that fixed cost dwarfs the row work. Add each row's packed int
// gradient straight to the GLOBAL histogram instead: integer atomics are
// order-invariant and the packed int32 / expanded int64 adds reproduce the
// shared-then-merge sums bit-exactly (same addends, associative), so this is
// a legitimate plan-key path (unlike the float direct body, which is
// permanently disabled -- see SmallLeafRowThreshold()).
template <typename BIN_TYPE, bool USE_16BIT_HIST, class PACK = PackRaw8>
__device__ __forceinline__ void ConstructDiscretizedHistogramDenseDirectInner(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const data_size_t num_data,
  const int dim_y,
  const int8_t* is_feature_used_bytree) {
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx.y) * blockDim.y) * num_data_per_thread;
  if (block_start >= num_data_in_smaller_leaf) {
    return;
  }
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  int row_stride = num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + (PACK::kUsesPackedOffsets ?
    PackedPartitionRows(packed_partition_byte_offsets, blockIdx.x, num_data, &row_stride) :
    static_cast<size_t>(partition_column_start) * num_data);
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx.y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y);
  // identical wide-partition two-column coverage as the shared body
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  const unsigned int col2_local = threadIdx.x + blockDim.x;
  const int column_index2 = static_cast<int>(col2_local) + partition_column_start;
  const bool use1 = threadIdx.x < static_cast<unsigned int>(num_columns_in_partition) &&
      (is_feature_used_bytree == nullptr || is_feature_used_bytree[column_index]);
  const bool use2 = col2_local < static_cast<unsigned int>(num_columns_in_partition) &&
      (is_feature_used_bytree == nullptr || is_feature_used_bytree[column_index2]);
  if (!(use1 || use2)) {
    return;
  }
  const typename PACK::Cursor pack_cursor = PACK::Prepare(use1 ? threadIdx.x : col2_local);
  const typename PACK::Cursor pack_cursor2 = PACK::Prepare(use2 ? col2_local : threadIdx.x);
  const uint32_t hist_base1 = partition_hist_start + column_hist_offsets[use1 ? column_index : column_index2];
  const uint32_t hist_base2 = partition_hist_start + column_hist_offsets[use2 ? column_index2 : column_index];
  int32_t* hist16 = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf);
  atomic_add_long_t* hist64 = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
    const BIN_TYPE* row_base = data_ptr + static_cast<size_t>(data_index) * row_stride;
    const int64_t gh64 = (static_cast<int64_t>(static_cast<int16_t>(grad_and_hess >> 16)) << 32) |
        (static_cast<int64_t>(grad_and_hess & 0x0000ffff));
    if (use1) {
      const uint32_t bin = PACK::ExtractAt(row_base, pack_cursor);
      if (USE_16BIT_HIST) {
        atomicAdd_system(hist16 + hist_base1 + bin, grad_and_hess);
      } else {
        atomicAdd_system(hist64 + hist_base1 + bin, (atomic_add_long_t)gh64);
      }
    }
    if (use2) {
      const uint32_t bin = PACK::ExtractAt(row_base, pack_cursor2);
      if (USE_16BIT_HIST) {
        atomicAdd_system(hist16 + hist_base2 + bin, grad_and_hess);
      } else {
        atomicAdd_system(hist64 + hist_base2 + bin, (atomic_add_long_t)gh64);
      }
    }
    inner_data_index += blockDim.y;
  }
}

// One batch of the row_batch construct: all N gradient loads and bin extractions are issued before the N shared
// atomics. With PREFETCH, the next batch's N row indices (next_src[j * next_step]) are loaded between the gradient
// loads and the extractions.
template <int N, bool PREFETCH, class PACK, typename BIN_TYPE>
__device__ __forceinline__ void ConstructRowBatch(
    const data_size_t (&idx)[N], const int32_t* cuda_gradients_and_hessians, const BIN_TYPE* data_ptr,
    const int row_stride, const typename PACK::Cursor& pack_cursor, int32_t* shared_hist_ptr,
    const data_size_t* next_src = nullptr, const data_size_t next_step = 0, data_size_t* next_idx = nullptr) {
  int32_t g[N];
  uint32_t bins[N];
#pragma unroll
  for (int j = 0; j < N; ++j) g[j] = __ldg(cuda_gradients_and_hessians + idx[j]);
  if (PREFETCH) {
#pragma unroll
    for (int j = 0; j < N; ++j) next_idx[j] = __ldg(next_src + j * next_step);
  }
#pragma unroll
  for (int j = 0; j < N; ++j) bins[j] = PACK::ExtractAt(data_ptr + static_cast<size_t>(idx[j]) * row_stride, pack_cursor);
#pragma unroll
  for (int j = 0; j < N; ++j) atomicAdd_block(shared_hist_ptr + bins[j], g[j]);
}

// Shared body of the discretized dense histogram kernel (see
// ConstructHistogramDenseInner for the shared-memory-passing and early-exit
// rationale).
template <typename BIN_TYPE, bool USE_16BIT_HIST, class PACK = PackRaw8>
__device__ __forceinline__ void ConstructDiscretizedHistogramDenseInner(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  int32_t* shared_hist_packed,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const data_size_t num_data,
  const int dim_y,
  const int8_t* is_feature_used_bytree = nullptr,
  const uint8_t* bin_used = nullptr,
  const bool row_batch = false) {
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
  int row_stride = num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + (PACK::kUsesPackedOffsets ?
    PackedPartitionRows(packed_partition_byte_offsets, blockIdx.x, num_data, &row_stride) :
    static_cast<size_t>(partition_column_start) * num_data);
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
  // Wide partitions (cuda_plan key wide_partitions): a partition may hold up
  // to 2x blockDim.x columns; each thread then covers a second column at
  // threadIdx.x + blockDim.x. Narrow partitions predicate col2 off entirely.
  // The row walk runs ONCE (grad read shared by both columns); integer atomics
  // keep the extra update order-invariant, so results stay bit-identical.
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  const unsigned int col2_local = threadIdx.x + blockDim.x;
  const int column_index2 = static_cast<int>(col2_local) + partition_column_start;
  const bool use1 = threadIdx.x < static_cast<unsigned int>(num_columns_in_partition) &&
      (is_feature_used_bytree == nullptr || is_feature_used_bytree[column_index]);
  const bool use2 = col2_local < static_cast<unsigned int>(num_columns_in_partition) &&
      (is_feature_used_bytree == nullptr || is_feature_used_bytree[column_index2]);
  if (row_batch && num_columns_in_partition <= static_cast<int>(blockDim.x)) {
    // cuda_plan key row_batch, one column per thread: the row loop is latency-bound on the dependent chain
    // row index -> gradient and bin, so each thread issues the loads of 8 (then 4) rows before their atomics
    // and prefetches the next 8 row indices meanwhile. Same rows, same atomics: bit-identical.
    if (use1) {
      const typename PACK::Cursor pack_cursor = PACK::Prepare(threadIdx.x);
      int32_t* shared_hist_ptr = shared_hist_packed + column_hist_offsets[column_index];
      const data_size_t by = static_cast<data_size_t>(blockDim.y);
      data_size_t i = 0;
      if (i + 8 <= num_iteration_this) {
        data_size_t idx[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) idx[j] = __ldg(data_indices_ref_this_block + inner_data_index + j * by);
        for (; i + 8 <= num_iteration_this; i += 8) {
          // without a next batch, re-read the current (valid) positions; the values go unused
          const bool has_next = i + 16 <= num_iteration_this;
          const data_size_t* next_src = data_indices_ref_this_block + inner_data_index + (has_next ? 8 * by : 0);
          data_size_t next_idx[8];
          ConstructRowBatch<8, true, PACK>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, pack_cursor,
                                           shared_hist_ptr, next_src, has_next ? by : 0, next_idx);
#pragma unroll
          for (int j = 0; j < 8; ++j) idx[j] = next_idx[j];
          inner_data_index += 8 * by;
        }
      }
      if (i + 4 <= num_iteration_this) {
        data_size_t idx[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) idx[j] = __ldg(data_indices_ref_this_block + inner_data_index + j * by);
        ConstructRowBatch<4, false, PACK>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, pack_cursor,
                                   shared_hist_ptr);
        inner_data_index += 4 * by;
        i += 4;
      }
      for (; i < num_iteration_this; ++i) {
        const data_size_t idx[1] = {data_indices_ref_this_block[inner_data_index]};
        ConstructRowBatch<1, false, PACK>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, pack_cursor,
                                   shared_hist_ptr);
        inner_data_index += by;
      }
    }
  } else if (use1 || use2) {
    // per-column extraction constants resolved once (register-resident through
    // the row loop; a per-row table lookup runs from thread-local memory)
    const typename PACK::Cursor pack_cursor = PACK::Prepare(use1 ? threadIdx.x : col2_local);
    const typename PACK::Cursor pack_cursor2 = PACK::Prepare(use2 ? col2_local : threadIdx.x);
    int32_t* shared_hist_ptr = shared_hist_packed +
      (column_hist_offsets[use1 ? column_index : column_index2]);
    int32_t* shared_hist_ptr2 = shared_hist_packed +
      (column_hist_offsets[use2 ? column_index2 : column_index]);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const BIN_TYPE* row_base = data_ptr + static_cast<size_t>(data_index) * row_stride;
      if (use1) {
        const uint32_t bin = PACK::ExtractAt(row_base, pack_cursor);
        atomicAdd_block(shared_hist_ptr + bin, grad_and_hess);
      }
      if (use2) {
        const uint32_t bin = PACK::ExtractAt(row_base, pack_cursor2);
        atomicAdd_block(shared_hist_ptr2 + bin, grad_and_hess);
      }
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
      if (row_batch && packed_grad_hess == 0) continue;  // adding zero is a no-op
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      if (bin_used != nullptr && !bin_used[partition_hist_start + i]) {
        continue;
      }
      const int32_t packed_grad_hess = shared_hist_packed[i];
      if (row_batch && packed_grad_hess == 0) continue;  // adding zero is a no-op
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

template <typename BIN_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST, class PACK = PackRaw8>
__global__ void CUDAConstructDiscretizedHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const data_size_t num_data) {
  // packed grad<<16|hess slots. SHARED_HIST_SIZE counts the 16-bit slots of the
  // budget shared with the non-quantized kernels, so the int32 array holds half
  // as many entries in the same bytes, at the alignment its atomics require.
  __shared__ int32_t shared_hist_packed[SHARED_HIST_SIZE / 2];
  ConstructDiscretizedHistogramDenseInner<BIN_TYPE, USE_16BIT_HIST, PACK>(
    smaller_leaf_splits, shared_hist_packed, cuda_gradients_and_hessians, data,
    column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
    packed_partition_byte_offsets,
    num_data, static_cast<int>(gridDim.y * blockDim.y));
}

// Batched per-level variant (hybrid growth): blockIdx.z selects the pair. The
// per-pair histogram bit width is a runtime (block-uniform) branch so pairs with
// 16-bit and 32-bit histograms share a single launch. On the host-launched
// (two-sync) path the bit widths come from the host-written descriptor and the
// row grouping from the exact launch grid, bit-for-bit the previous behavior;
// inside the graph loop (gstate != nullptr) both are derived on-device from the
// child structs the batched apply kernels wrote (exact leaf counts, exact host
// thresholds) and the frozen pow2 grid is only an upper bound.
template <typename BIN_TYPE, int SHARED_HIST_SIZE, class PACK = PackRaw8>
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
  const uint8_t* bin_used,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const data_size_t* level_smaller_num_data,
  const CUDAHybridGraphLoopStateOpt gstate,
  const bool row_batch) {
  // packed grad<<16|hess slots. SHARED_HIST_SIZE counts the 16-bit slots of the
  // budget shared with the non-quantized kernels, so the int32 array holds half
  // as many entries in the same bytes, at the alignment its atomics require.
  __shared__ int32_t shared_hist_packed[SHARED_HIST_SIZE / 2];
  // graphs A2 idle-block guard (pow2-frozen grid; see the non-quantized kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.z)) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (!desc->construct_valid) {
    return;
  }
  const CUDALeafSplitsStruct* smaller_struct = desc->smaller_struct;
  // device mirror of the host min_data/min_hessian construct gate: a no-op on
  // the host-launched path (construct_valid already encodes it from identical
  // values), required inside the graph loop where the controller-written
  // descriptors carry construct_valid == 1
  const data_size_t num_data_smaller = smaller_struct->num_data_in_leaf;
  const double sum_hessians_smaller = smaller_struct->sum_of_hessians;
  const CUDALeafSplitsStruct* larger_struct = desc->larger_struct;
  const bool has_larger = larger_struct->leaf_index >= 0;
  const data_size_t num_data_larger = has_larger ? larger_struct->num_data_in_leaf : 0;
  const double sum_hessians_larger = has_larger ? larger_struct->sum_of_hessians : 0.0;
  if ((num_data_smaller <= min_data_in_leaf || sum_hessians_smaller <= min_sum_hessian_in_leaf) &&
      (num_data_larger <= min_data_in_leaf || sum_hessians_larger <= min_sum_hessian_in_leaf)) {
    return;
  }
  // effective row-grouping extent: the launch grid on the host-launched exact
  // grid; inside the graph the device replica of the host sizing (incl. the
  // packed int32 shared-histogram overflow guard) evaluated from the level's
  // actual smaller-child sizes (block-uniform, <= live pair count loads)
  int dim_y;
  if (level_smaller_num_data == nullptr) {
    dim_y = static_cast<int>(gridDim.y * blockDim.y);
  } else {
    data_size_t max_num_data = 0;
    const int num_pairs = HybridGraphLivePairCount(gstate, static_cast<int>(gridDim.z));
    for (int i = 0; i < num_pairs; ++i) {
      const data_size_t n = level_smaller_num_data[i];
      if (n > max_num_data) {
        max_num_data = n;
      }
    }
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
    dim_y = HybridBatchedConstructGridDimYQuant(
      max_num_data, num_pairs, static_cast<int>(blockDim.y),
      gstate->construct_min_grid_dim_y, gstate->construct_min_rows_per_thread,
      gstate->construct_saturation_floor, gstate->num_grad_quant_bins) *
      static_cast<int>(blockDim.y);
#else
    dim_y = static_cast<int>(gridDim.y * blockDim.y);
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  }
  const uint8_t smaller_num_bits = HybridGraphActive(gstate) ?
    HybridGraphQuantHistBits(gstate, num_data_smaller) : desc->smaller_num_bits;
  if (smaller_num_bits <= 16) {
    ConstructDiscretizedHistogramDenseInner<BIN_TYPE, true, PACK>(
      smaller_struct, shared_hist_packed, cuda_gradients_and_hessians, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      num_data, dim_y, is_feature_used_bytree, bin_used, row_batch);
  } else {
    ConstructDiscretizedHistogramDenseInner<BIN_TYPE, false, PACK>(
      smaller_struct, shared_hist_packed, cuda_gradients_and_hessians, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      num_data, dim_y, is_feature_used_bytree, bin_used, row_batch);
  }
}



// One batch of the pair-joint construct: as ConstructRowBatch, with the byte of the thread's column pair as the
// value and the cell (lo * hi_span + hi) of the pair's joint histogram as the bin.
template <int N, bool PREFETCH>
__device__ __forceinline__ void ConstructPairRowBatch(
    const data_size_t (&idx)[N], const int32_t* cuda_gradients_and_hessians, const uint8_t* data_ptr,
    const int row_stride, const int byte_idx, const uint32_t hi_span, const uint32_t hi_mask, int32_t* joint,
    const data_size_t* next_src = nullptr, const data_size_t next_step = 0, data_size_t* next_idx = nullptr) {
  int32_t g[N];
  uint32_t cells[N];
#pragma unroll
  for (int j = 0; j < N; ++j) g[j] = __ldg(cuda_gradients_and_hessians + idx[j]);
  if (PREFETCH) {
#pragma unroll
    for (int j = 0; j < N; ++j) next_idx[j] = __ldg(next_src + j * next_step);
  }
#pragma unroll
  for (int j = 0; j < N; ++j) {
    const uint32_t v = static_cast<uint32_t>(__ldcs(data_ptr + static_cast<size_t>(idx[j]) * row_stride + byte_idx));
    cells[j] = (v & 0xfu) * hi_span + ((v >> 4) & hi_mask);
  }
#pragma unroll
  for (int j = 0; j < N; ++j) atomicAdd_block(joint + cells[j], g[j]);
}

// Pair-joint body of the discretized construct on the 4-bit compact view (cuda_plan key pair_hist). Thread x owns
// byte x of the partition's packed row: compact columns 2x (low nibble) and 2x+1 (high nibble). A row adds its
// packed gradient once, to cell lo * span(hi) + hi of the byte's joint table, instead of once per column. After
// the row loop each column's bin is the sum of its joint row / column, i.e. the same packed int32 sum of the same
// rows the per-column kernel accumulates (wrapping integer adds are order-invariant), flushed by the same rule.
// Rows are grouped exactly as in ConstructDiscretizedHistogramDenseInner (same dim_y, same per-block extents).
// column_meta: [T] partition-relative hist offsets, [T, 2T) value spans, [2T, 3T) joint-table offsets.
// min_rows_per_thread (cuda_plan key level_row_blocks; 0 = off): a leaf gets at least this many rows per thread,
// so the small leaves of a many-pair level fill a few whole blocks instead of thinly spreading over every block
// row of the grid (sized for the level's largest leaf), each paying a full table zero and flush for a few rows.
// The host passes the largest leaf's own rows per thread, which the grid's overflow guard already bounds, so no
// block holds more rows than the largest leaf's blocks do.
// DIRECT (cuda_plan key all_rows_direct): the leaf holds every row of the matrix, so its index list is a
// permutation of 0 .. num_data - 1 and position p is read as row p, without the index gather. Every block keeps
// its row count (the overflow guard is unchanged) and the grid adds up the same rows; wrapping integer sums do
// not depend on which block adds which row, so the histogram is the same.
template <bool USE_16BIT_HIST, bool DIRECT, int kB>
__device__ __forceinline__ void ConstructDiscretizedHistogramPairJointInner(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  int32_t* shared_joint,
  const int32_t* cuda_gradients_and_hessians,
  const uint8_t* data,
  const uint32_t* column_meta,
  const int num_compact_columns,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const data_size_t num_data,
  const int dim_y,
  const int whole_row_partitions,
  const uint32_t whole_row_joint,
  const data_size_t min_rows_per_thread,
  const int block_y) {
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = max((num_data_in_smaller_leaf + dim_y - 1) / dim_y, min_rows_per_thread);
  const size_t block_start_wide = (static_cast<size_t>(block_y) * blockDim.y) * static_cast<size_t>(num_data_per_thread);
  if (block_start_wide >= static_cast<size_t>(num_data_in_smaller_leaf)) {
    return;
  }
  const data_size_t block_start = static_cast<data_size_t>(block_start_wide);
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  // whole_row_partitions > 0 (cuda_plan key pair_hist_rows, row-interleaved view): the block covers every
  // partition's bytes of its rows, thread x owning byte x of the full row; its partition is the one whose byte run
  // holds x, and the joint tables of all partitions are laid out back to back ([3T, 4T) of column_meta)
  int part = static_cast<int>(blockIdx.x);
  int part_byte = static_cast<int>(threadIdx.x);
  if (whole_row_partitions > 0) {
    part = 0;
    while (part + 1 < whole_row_partitions && part_byte >= packed_partition_byte_offsets[part + 1]) ++part;
    if (part > 0) part_byte -= packed_partition_byte_offsets[part];
  }
  const int partition_column_start = feature_partition_column_index_offsets[part];
  const int partition_column_end = feature_partition_column_index_offsets[part + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  int row_stride = 0;
  const uint8_t* data_ptr = data + (whole_row_partitions > 0 ? 0 :
      PackedPartitionRows(packed_partition_byte_offsets, blockIdx.x, num_data, &row_stride));
  if (whole_row_partitions > 0) row_stride = -packed_partition_byte_offsets[0];
  const uint32_t partition_hist_start = column_hist_offsets_full[part];
  const uint32_t* spans = column_meta + num_compact_columns;
  const uint32_t* joint_offsets = spans + num_compact_columns + (whole_row_partitions > 0 ? num_compact_columns : 0);
  uint32_t num_joint = 0;
  if (whole_row_partitions > 0) {
    num_joint = whole_row_joint;
  } else if (num_columns_in_partition > 0) {
    const int last_lo = partition_column_start + ((num_columns_in_partition - 1) & ~1);
    const uint32_t last_hi_span = last_lo + 1 < partition_column_end ? spans[last_lo + 1] : 1u;
    num_joint = joint_offsets[last_lo] + spans[last_lo] * last_hi_span;
  }
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_joint; i += num_threads_per_block) {
    shared_joint[i] = 0;
  }
  __syncthreads();
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  const data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx.y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y);
  const int byte_idx = static_cast<int>(threadIdx.x);
  const int lo_col = partition_column_start + 2 * part_byte;
  const bool active = (whole_row_partitions == 0 || byte_idx < row_stride) && lo_col < partition_column_end;
  const bool has_hi = lo_col + 1 < partition_column_end;
  const uint32_t lo_span = active ? spans[lo_col] : 0u;
  const uint32_t hi_span = has_hi ? spans[lo_col + 1] : 1u;
  int32_t* joint = shared_joint + (active ? joint_offsets[lo_col] : 0u);
  if (active) {
    const uint32_t hi_mask = has_hi ? 0xfu : 0u;
    const data_size_t by = static_cast<data_size_t>(blockDim.y);
    data_size_t i = 0;
    // kB rows in flight per thread (16 in the default build; the per-column kernel uses 8): with half as many
    // threads per row, the gathered levels are bound by the latency of the index -> gradient / bin chain, and the
    // 16-row batch measured -11% against 8 (cuda_plan key pair_capped_rows: a register-capped build, see below)
    if (DIRECT) {
      for (; i + kB <= num_iteration_this; i += kB) {
        data_size_t idx[kB];
#pragma unroll
        for (int j = 0; j < kB; ++j) idx[j] = block_start + inner_data_index + j * by;
        ConstructPairRowBatch<kB, false>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, byte_idx, hi_span,
                                         hi_mask, joint);
        inner_data_index += kB * by;
      }
    } else if (i + kB <= num_iteration_this) {
      data_size_t idx[kB];
#pragma unroll
      for (int j = 0; j < kB; ++j) idx[j] = __ldg(data_indices_ref_this_block + inner_data_index + j * by);
      for (; i + kB <= num_iteration_this; i += kB) {
        // without a next batch, re-read the current (valid) positions; the values go unused
        const bool has_next = i + 2 * kB <= num_iteration_this;
        const data_size_t* next_src = data_indices_ref_this_block + inner_data_index + (has_next ? kB * by : 0);
        data_size_t next_idx[kB];
        ConstructPairRowBatch<kB, true>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, byte_idx, hi_span,
                                        hi_mask, joint, next_src, has_next ? by : 0, next_idx);
#pragma unroll
        for (int j = 0; j < kB; ++j) idx[j] = next_idx[j];
        inner_data_index += kB * by;
      }
    }
    if (kB > 16 && i + 16 <= num_iteration_this) {
      data_size_t idx[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        idx[j] = DIRECT ? block_start + inner_data_index + j * by :
                          __ldg(data_indices_ref_this_block + inner_data_index + j * by);
      }
      ConstructPairRowBatch<16, false>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, byte_idx, hi_span,
                                       hi_mask, joint);
      inner_data_index += 16 * by;
      i += 16;
    }
    if (kB > 8 && i + 8 <= num_iteration_this) {
      data_size_t idx[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        idx[j] = DIRECT ? block_start + inner_data_index + j * by :
                          __ldg(data_indices_ref_this_block + inner_data_index + j * by);
      }
      ConstructPairRowBatch<8, false>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, byte_idx, hi_span,
                                      hi_mask, joint);
      inner_data_index += 8 * by;
      i += 8;
    }
    if (kB > 4 && i + 4 <= num_iteration_this) {
      data_size_t idx[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        idx[j] = DIRECT ? block_start + inner_data_index + j * by :
                          __ldg(data_indices_ref_this_block + inner_data_index + j * by);
      }
      ConstructPairRowBatch<4, false>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, byte_idx, hi_span,
                                      hi_mask, joint);
      inner_data_index += 4 * by;
      i += 4;
    }
    for (; i < num_iteration_this; ++i) {
      const data_size_t idx[1] = {DIRECT ? block_start + inner_data_index :
                                           data_indices_ref_this_block[inner_data_index]};
      ConstructPairRowBatch<1, false>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, byte_idx, hi_span,
                                      hi_mask, joint);
      inner_data_index += by;
    }
  }
  __syncthreads();
  if (!active) {
    return;
  }
  // marginals: low column bin b = joint row b, high column bin b = joint column b; empty bins are skipped as in
  // the per-column flush (adding zero is a no-op)
  const uint32_t num_marginal = lo_span + (has_hi ? hi_span : 0u);
  for (uint32_t m = threadIdx.y; m < num_marginal; m += blockDim.y) {
    const bool is_lo = m < lo_span;
    const uint32_t b = is_lo ? m : m - lo_span;
    const uint32_t count = is_lo ? hi_span : lo_span;
    const uint32_t stride = is_lo ? 1u : hi_span;
    const int32_t* cell = joint + (is_lo ? b * hi_span : b);
    uint32_t sum = 0;
    for (uint32_t k = 0; k < count; ++k) {
      sum += static_cast<uint32_t>(cell[k * stride]);
    }
    const int32_t packed_grad_hess = static_cast<int32_t>(sum);
    if (packed_grad_hess == 0) continue;
    const uint32_t hist_pos = partition_hist_start + column_meta[is_lo ? lo_col : lo_col + 1] + b;
    if (USE_16BIT_HIST) {
      atomicAdd(reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + hist_pos, packed_grad_hess);
    } else {
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd(reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + hist_pos, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

// Host-launched (two-sync) batched pair-joint construct: the prologue of CUDAConstructDiscretizedHistogramDenseBatchedKernel
// without the graph-loop branches (the host takes this kernel only outside graph capture and with exact level sizes).
// Dynamic shared memory: the largest partition joint table.
// ALL_ROWS (cuda_plan key all_rows_direct; the host takes it for a single-pair level whose leaf holds every row):
// a separate instantiation, so the code of the hot per-level kernel is untouched; it still checks the leaf size.
// cuda_plan key pair_block_map: num_pairs > 0 when the launch holds only the block rows with rows, start[num_pairs]
// of them in y (x and blockDim unchanged): block row b belongs to the pair p with start[p] <= b < start[p + 1] (pairs
// without rows have equal starts) and is that pair's block row b - start[p] of the level grid (grid_y rows per pair,
// num_pairs pairs) it stands for.
// num_pairs == 0: blockIdx.z is the pair and blockIdx.y the block row. Passed by value (kernel parameter space,
// ~2 KB); starts fit 16 bits because a launch holds at most 65535 block rows.
struct PairJointBlockPrefix {
  static constexpr int kMaxPairs = 1024;
  int num_pairs;
  int grid_y;
  uint16_t start[kMaxPairs + 1];
};

#define FALCATA_PAIR_JOINT_BATCHED_PARAMS \
  const CUDAHybridPairDescriptor* pair_descs, \
  const int32_t* cuda_gradients_and_hessians, \
  const uint8_t* data, \
  const uint32_t* column_meta, \
  const int num_compact_columns, \
  const uint32_t* column_hist_offsets_full, \
  const int* feature_partition_column_index_offsets, \
  const int* packed_partition_byte_offsets, \
  const data_size_t num_data, \
  const data_size_t min_data_in_leaf, \
  const double min_sum_hessian_in_leaf, \
  const int per_pair_min_grid_dim_y, \
  const int min_rows_per_thread, \
  const int saturation_floor_total, \
  const int num_grad_quant_bins, \
  const int whole_row_partitions, \
  const uint32_t whole_row_joint, \
  const data_size_t level_min_rows_per_thread, \
  const PairJointBlockPrefix block_prefix
#define FALCATA_PAIR_JOINT_BATCHED_ARGS \
  pair_descs, cuda_gradients_and_hessians, data, column_meta, num_compact_columns, column_hist_offsets_full, \
  feature_partition_column_index_offsets, packed_partition_byte_offsets, num_data, min_data_in_leaf, \
  min_sum_hessian_in_leaf, per_pair_min_grid_dim_y, min_rows_per_thread, saturation_floor_total, \
  num_grad_quant_bins, whole_row_partitions, whole_row_joint, level_min_rows_per_thread, block_prefix
template <bool ALL_ROWS, int BATCH>
__device__ __forceinline__ void PairJointBatchedBody(FALCATA_PAIR_JOINT_BATCHED_PARAMS) {
  extern __shared__ int32_t shared_joint[];
  // cuda_plan key pair_block_map: the pair and block row this block stands for (see PairJointBlockPrefix)
  unsigned int pair_index = blockIdx.z;
  int block_y = static_cast<int>(blockIdx.y);
  int level_grid_y = static_cast<int>(gridDim.y);
  int level_num_pairs = static_cast<int>(gridDim.z);
  if (block_prefix.num_pairs > 0) {
    int lo = 0;
    int hi = block_prefix.num_pairs - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (static_cast<unsigned int>(block_prefix.start[mid]) <= blockIdx.y) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    pair_index = static_cast<unsigned int>(lo);
    block_y = static_cast<int>(blockIdx.y - block_prefix.start[lo]);
    level_grid_y = block_prefix.grid_y;
    level_num_pairs = block_prefix.num_pairs;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  if (!desc->construct_valid) {
    return;
  }
  const CUDALeafSplitsStruct* smaller_struct = desc->smaller_struct;
  const data_size_t num_data_smaller = smaller_struct->num_data_in_leaf;
  const double sum_hessians_smaller = smaller_struct->sum_of_hessians;
  const CUDALeafSplitsStruct* larger_struct = desc->larger_struct;
  const bool has_larger = larger_struct->leaf_index >= 0;
  const data_size_t num_data_larger = has_larger ? larger_struct->num_data_in_leaf : 0;
  const double sum_hessians_larger = has_larger ? larger_struct->sum_of_hessians : 0.0;
  if ((num_data_smaller <= min_data_in_leaf || sum_hessians_smaller <= min_sum_hessian_in_leaf) &&
      (num_data_larger <= min_data_in_leaf || sum_hessians_larger <= min_sum_hessian_in_leaf)) {
    return;
  }
  // cuda_plan key per_pair_rows (per_pair_min_grid_dim_y > 0): the grid's y extent is the level formula at the
  // level's LARGEST smaller leaf; each pair instead groups its rows by the same formula at its OWN size (never more
  // blocks, as the formula is monotone in the size), so a small leaf of a deep level is not cut into many blocks of
  // a few rows each paying the full joint-table zeroing and flush. The formula keeps the packed-cell row cap.
  int pair_grid_y = level_grid_y;
  if (per_pair_min_grid_dim_y > 0) {
    const int y = HybridBatchedConstructGridDimYQuant(
      num_data_smaller, level_num_pairs, static_cast<int>(blockDim.y), per_pair_min_grid_dim_y,
      min_rows_per_thread, saturation_floor_total, num_grad_quant_bins);
    pair_grid_y = max(1, min(pair_grid_y, y));
    if (block_y >= pair_grid_y) {
      return;
    }
  }
  const int dim_y = pair_grid_y * static_cast<int>(blockDim.y);
  // a leaf of num_data distinct rows of the num_data-row matrix holds every row (see the inner DIRECT case)
  const bool direct = ALL_ROWS && num_data_smaller == num_data;
#define FALCATA_PAIR_JOINT_INNER(B16, DIRECT) \
    ConstructDiscretizedHistogramPairJointInner<B16, DIRECT, BATCH>( \
      smaller_struct, shared_joint, cuda_gradients_and_hessians, data, column_meta, num_compact_columns, \
      column_hist_offsets_full, feature_partition_column_index_offsets, packed_partition_byte_offsets, \
      num_data, dim_y, whole_row_partitions, whole_row_joint, level_min_rows_per_thread, block_y)
  if (desc->smaller_num_bits <= 16) {
    if (direct) {
      FALCATA_PAIR_JOINT_INNER(true, ALL_ROWS);
    } else {
      FALCATA_PAIR_JOINT_INNER(true, false);
    }
  } else {
    if (direct) {
      FALCATA_PAIR_JOINT_INNER(false, ALL_ROWS);
    } else {
      FALCATA_PAIR_JOINT_INNER(false, false);
    }
  }
#undef FALCATA_PAIR_JOINT_INNER
}

template <bool ALL_ROWS>
__global__ void CUDAConstructDiscretizedHistogramPairJointBatchedKernel(FALCATA_PAIR_JOINT_BATCHED_PARAMS) {
  PairJointBatchedBody<ALL_ROWS, 16>(FALCATA_PAIR_JOINT_BATCHED_ARGS);
}

// cuda_plan key pair_capped_rows: the same kernel built at most MAXREG registers per thread with BATCH rows in
// flight per thread. A whole-row block of the default build (64 registers) runs one block per SM; at a lower
// register count two or three shorter whole-row blocks fit the register file (the joint tables' shared memory
// permitting), so more warps are resident. The host takes it only where the occupancy API gives it strictly more
// resident warps than the default build (see PairJointCappedRows). Same rows, same integer sums: bit-identical.
// __maxnreg__ needs CUDA 12.4; an older toolkit builds the kernel uncapped and the host never takes it.
#if CUDART_VERSION >= 12040
#define FALCATA_PAIR_JOINT_CAPPED_BUILD 1
#define FALCATA_PAIR_JOINT_MAXNREG(n) __maxnreg__(n)
#else
#define FALCATA_PAIR_JOINT_CAPPED_BUILD 0
#define FALCATA_PAIR_JOINT_MAXNREG(n)
#endif
template <bool ALL_ROWS, int BATCH, int MAXREG>
__global__ void FALCATA_PAIR_JOINT_MAXNREG(MAXREG)
CUDAConstructDiscretizedHistogramPairJointCappedKernel(FALCATA_PAIR_JOINT_BATCHED_PARAMS) {
  PairJointBatchedBody<ALL_ROWS, BATCH>(FALCATA_PAIR_JOINT_BATCHED_ARGS);
}
#undef FALCATA_PAIR_JOINT_MAXNREG
#undef FALCATA_PAIR_JOINT_BATCHED_ARGS
#undef FALCATA_PAIR_JOINT_BATCHED_PARAMS

// The default pair-joint build's limits on the current device: the largest block both instantiations can launch
// (register-limited; the whole-row block of pair_hist_rows can exceed the per-partition shapes this kernel was
// otherwise launched with), and the warps one SM holds by its register file at the kernels' register count (the
// larger of the two instantiations, allocated per warp in 256-register units) and by its warp limit. Memoised per
// device, like the capped build's choice (PairJointCappedRows): a process can train on devices whose register file,
// warp limit or kernel image differ, and the device is whichever one is current at the launch.
struct PairJointLimits {
  bool ready = false;
  int max_threads = 0;
  int warps_by_regs = 0;
  int warps_per_sm = 0;
};
static const PairJointLimits& PairJointDeviceLimits() {
  static thread_local std::vector<PairJointLimits> by_device;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  if (static_cast<size_t>(device) >= by_device.size()) {
    by_device.resize(static_cast<size_t>(device) + 1);
  }
  PairJointLimits& limits = by_device[device];
  if (!limits.ready) {
    cudaFuncAttributes attr_gather, attr_direct;
    CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attr_gather, CUDAConstructDiscretizedHistogramPairJointBatchedKernel<false>));
    CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attr_direct, CUDAConstructDiscretizedHistogramPairJointBatchedKernel<true>));
    int regs_per_sm = 0;
    int threads_per_sm = 0;
    CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, device));
    CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    const int regs = std::max(1, std::max(attr_gather.numRegs, attr_direct.numRegs));
    const int regs_per_warp = (regs * 32 + 255) / 256 * 256;
    limits.warps_by_regs = std::max(1, regs_per_sm / regs_per_warp);
    limits.warps_per_sm = std::max(1, threads_per_sm / 32);
    limits.max_threads = std::min(attr_gather.maxThreadsPerBlock, attr_direct.maxThreadsPerBlock);
    limits.ready = true;
  }
  return limits;
}

// Largest block both pair-joint instantiations can launch on the current device.
static int PairJointMaxThreadsPerBlock() {
  return PairJointDeviceLimits().max_threads;
}

// Warps of a pair-joint block of `threads` threads resident on one SM of the current device: whole blocks, as many as
// the register file and the SM's warp limit allow (PairJointDeviceLimits).
static int PairJointResidentWarps(const int threads) {
  const PairJointLimits& limits = PairJointDeviceLimits();
  const int block_warps = std::max(1, (threads + 31) / 32);
  return std::min(limits.warps_by_regs / block_warps, limits.warps_per_sm / block_warps) * block_warps;
}

using PairJointKernelFn = void (*)(const CUDAHybridPairDescriptor*, const int32_t*, const uint8_t*, const uint32_t*,
                                   int, const uint32_t*, const int*, const int*, data_size_t, data_size_t, double, int,
                                   int, int, int, int, uint32_t, data_size_t, PairJointBlockPrefix);

// cuda_plan key pair_capped_rows: the register-capped build. 48 registers fit three 12-warp blocks in a 64K register
// file. Rows in flight per thread: 15 for the per-level (gathering) instantiation and 12 for the root's direct-read
// one, each the deepest batch measured that compiles to 48 registers without spilling (15 and 14 beat 12 and 16 at
// 48-56 registers; 8, 10 and 13 were slower).
struct PairJointCappedBuild {
  PairJointKernelFn gather;
  PairJointKernelFn direct;
};
static PairJointCappedBuild PairJointCapped() {
  return {CUDAConstructDiscretizedHistogramPairJointCappedKernel<false, 15, 48>,
          CUDAConstructDiscretizedHistogramPairJointCappedKernel<true, 12, 48>};
}

// resident warps per SM of a pair-joint block of `threads` threads and `smem_bytes` dynamic shared memory, the
// fewer of the two instantiations, by the occupancy API (registers, shared memory, warp and block limits)
static int PairJointOccupancyWarps(PairJointKernelFn gather, PairJointKernelFn direct, const int threads,
                                   const size_t smem_bytes) {
  int blocks_gather = 0;
  int blocks_direct = 0;
  CUDASUCCESS_OR_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_gather, gather, threads, smem_bytes));
  CUDASUCCESS_OR_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_direct, direct, threads, smem_bytes));
  return std::min(blocks_gather, blocks_direct) * ((threads + 31) / 32);
}

// cuda_plan key pair_capped_rows: the whole-row block height of the register-capped build that keeps the most warps
// resident per SM (ties keep the taller block: fewer blocks, less zeroing and flushing), or 0 when that is not
// strictly more than the default build keeps at its own height `default_y`. Heights stay within the packed-cell
// row cap and both instantiations' max threads. For a chosen height the capped build's shared-memory carveout is
// set to hold just its resident blocks' joint tables: the driver otherwise takes the largest carveout the occupancy
// could use (here 100 KB for 56 KB of tables), shrinking the L1 that serves the gathers to a quarter; a shape whose
// tables need more than 64% of the SM's shared memory keeps the default build. The carveout is a hint and never
// limits a launch. Memoised per device and launch shape (the queries cost host time).
static int PairJointCappedRows(const int row_bytes, const size_t smem_bytes, const int default_y,
                               const int num_grad_quant_bins) {
  if (!FALCATA_PAIR_JOINT_CAPPED_BUILD) {
    return 0;  // built without the register cap (CUDA < 12.4): it would not raise occupancy
  }
  struct Entry {
    int device = -1;
    int row_bytes = -1;
    size_t smem_bytes = 0;
    int default_y = 0;
    int bins = 0;
    int y = 0;
  };
  static thread_local Entry memo;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  if (memo.device == device && memo.row_bytes == row_bytes && memo.smem_bytes == smem_bytes &&
      memo.default_y == default_y && memo.bins == num_grad_quant_bins) {
    return memo.y;
  }
  const PairJointCappedBuild capped = PairJointCapped();
  // cuda_plan key shape_memo: every shape's choice is kept (the joint-table bytes change with the column sample, so
  // the one-entry memo above missed at nearly every tree); a hit re-sets the capped build's carveout only when it
  // is not the one this shape chose (the attribute is per kernel, shared by all shapes)
  struct ShapeEntry {
    Entry key;
    int percent = cudaSharedmemCarveoutDefault;
  };
  static thread_local std::vector<ShapeEntry> shapes;
  static thread_local int set_device = -1;
  static thread_local int set_percent = cudaSharedmemCarveoutDefault;
  const bool shape_memo = FalcataPlan::Get().shape_memo;
  if (shape_memo) {
    for (const ShapeEntry& e : shapes) {
      if (e.key.device == device && e.key.row_bytes == row_bytes && e.key.smem_bytes == smem_bytes &&
          e.key.default_y == default_y && e.key.bins == num_grad_quant_bins) {
        if (e.key.y > 0 && (set_device != device || set_percent != e.percent)) {
          CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(capped.gather, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                    e.percent));
          CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(capped.direct, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                    e.percent));
          set_device = device;
          set_percent = e.percent;
        }
        memo = e.key;
        return e.key.y;
      }
    }
  }
  int chosen_percent = cudaSharedmemCarveoutDefault;
  // occupancy at the default carveout (a previous shape's carveout must not bound this one's choice)
  CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(capped.gather, cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutDefault));
  CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(capped.direct, cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutDefault));
  cudaFuncAttributes attr_gather, attr_direct;
  CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attr_gather, capped.gather));
  CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attr_direct, capped.direct));
  const int max_threads = std::min(1024, std::min(attr_gather.maxThreadsPerBlock, attr_direct.maxThreadsPerBlock));
  const int default_warps = PairJointOccupancyWarps(CUDAConstructDiscretizedHistogramPairJointBatchedKernel<false>,
                                                    CUDAConstructDiscretizedHistogramPairJointBatchedKernel<true>,
                                                    row_bytes * default_y, smem_bytes);
  int best_y = 0;
  int best_warps = default_warps;
  for (int c = 1; row_bytes * c <= max_threads && HybridQuantConstructBlockDimY(c, num_grad_quant_bins) == c; ++c) {
    const int warps = PairJointOccupancyWarps(capped.gather, capped.direct, row_bytes * c, smem_bytes);
    if (warps > best_warps || (best_y > 0 && warps == best_warps)) {
      best_warps = warps;
      best_y = c;
    }
  }
  if (best_y > 0) {
    int smem_per_sm = 0;
    int reserved = 0;
    CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, device));
    const int blocks = std::max(1, best_warps / ((row_bytes * best_y + 31) / 32));
    const int64_t need = static_cast<int64_t>(blocks) * (static_cast<int64_t>(smem_bytes) + reserved);
    const int percent = static_cast<int>(std::min<int64_t>(100, (need * 100 + std::max(1, smem_per_sm) - 1) /
                                                                std::max(1, smem_per_sm)));
    // the extra warps pay off only while the gathers keep at least half of the unified L1: at most 64% of the SM's
    // shared memory for the tables (the 64 KB tier of 100 KB on sm_120); past it the carveout jumps to the full
    // 100 KB, L1 falls to a quarter, and the capped build measured slower in every such shape (3 blocks of 22 KB
    // tables -0.7%, 4 blocks of 18 KB at 40 registers -4.5%), so the default build is kept
    if (percent > 64) {
      best_y = 0;
    } else {
      CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(capped.gather, cudaFuncAttributePreferredSharedMemoryCarveout, percent));
      CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(capped.direct, cudaFuncAttributePreferredSharedMemoryCarveout, percent));
      chosen_percent = percent;
    }
  }
  memo = {device, row_bytes, smem_bytes, default_y, num_grad_quant_bins, best_y};
  set_device = device;
  set_percent = chosen_percent;
  if (shape_memo) {
    if (shapes.size() >= 256) shapes.clear();
    shapes.push_back({memo, chosen_percent});
  }
  return best_y;
}

// Dedicated all-small-level variant: launched by the host INSTEAD of the
// shared-histogram batched kernel when the level's largest smaller-leaf is
// under the small-leaf threshold (deep-tree tail levels). Keeping it a
// separate kernel leaves the hot kernel's register footprint untouched --
// an in-kernel branch measurably slowed the never-taken numerai path.
template <typename BIN_TYPE, class PACK = PackRaw8>
__global__ void CUDAConstructDiscretizedHistogramDenseSmallLeafBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const data_size_t num_data,
  const int8_t* is_feature_used_bytree,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf) {
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (!desc->construct_valid) {
    return;
  }
  const CUDALeafSplitsStruct* smaller_struct = desc->smaller_struct;
  const data_size_t num_data_smaller = smaller_struct->num_data_in_leaf;
  const double sum_hessians_smaller = smaller_struct->sum_of_hessians;
  const CUDALeafSplitsStruct* larger_struct = desc->larger_struct;
  const bool has_larger = larger_struct->leaf_index >= 0;
  const data_size_t num_data_larger = has_larger ? larger_struct->num_data_in_leaf : 0;
  const double sum_hessians_larger = has_larger ? larger_struct->sum_of_hessians : 0.0;
  if ((num_data_smaller <= min_data_in_leaf || sum_hessians_smaller <= min_sum_hessian_in_leaf) &&
      (num_data_larger <= min_data_in_leaf || sum_hessians_larger <= min_sum_hessian_in_leaf)) {
    return;
  }
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  if (desc->smaller_num_bits <= 16) {
    ConstructDiscretizedHistogramDenseDirectInner<BIN_TYPE, true, PACK>(
      smaller_struct, cuda_gradients_and_hessians, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets, num_data, dim_y, is_feature_used_bytree);
  } else {
    ConstructDiscretizedHistogramDenseDirectInner<BIN_TYPE, false, PACK>(
      smaller_struct, cuda_gradients_and_hessians, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets, num_data, dim_y, is_feature_used_bytree);
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
  // packed grad<<16|hess slots. SHARED_HIST_SIZE counts the 16-bit slots of the
  // budget shared with the non-quantized kernels, so the int32 array holds half
  // as many entries in the same bytes, at the alignment its atomics require.
  __shared__ int32_t shared_hist_packed[SHARED_HIST_SIZE / 2];
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
  if (colmajor_direct_ && use_quantized_grad_) {
    // the quantized per-leaf construct (classic loop, hybrid per-pair fallback, leaf-wise tail) has no compact-view
    // branch: it reads every column of the row-major layout. The tree learner picks the mask regime where its trees
    // take this path; elsewhere (an ablation key, the aggressive tail) the compact regime fills the full view here.
    EnsureFullView("the per-leaf quantized construct");
  }
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
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        } else {
          CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            RowMajorBin<BIN_TYPE>(),
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
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        } else {
          CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory<BIN_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            RowMajorBin<BIN_TYPE>(),
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
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else {
          CUDAConstructDiscretizedHistogramSparseKernel<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else {
        if (cuda_row_data_->is_4bit_packed()) {
          if (num_bits_in_histogram_bins <= 16) {
            CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, true, PackNibble4><<<grid_dim, block_dim, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              reinterpret_cast<const int32_t*>(cuda_gradients_),
              RowMajorBin<BIN_TYPE>(),
              cuda_row_data_->cuda_column_hist_offsets(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              cuda_row_data_->cuda_feature_partition_column_index_offsets(),
              cuda_row_data_->cuda_packed_partition_byte_offsets(),
              num_data_);
          } else {
            CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, false, PackNibble4><<<grid_dim, block_dim, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              reinterpret_cast<const int32_t*>(cuda_gradients_),
              RowMajorBin<BIN_TYPE>(),
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
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            nullptr,
            num_data_);
        } else {
          CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            RowMajorBin<BIN_TYPE>(),
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
        // Deterministic float construct: slot rows sized by the widest
        // partition's item count. Rows that fit the dynamic shared budget use
        // the shared-memory kernel (faster); wider rows move to the global
        // slab variant, so the order-dependent atomic kernel only remains for
        // shapes with no det scratch at all.
        const std::vector<uint32_t>& part_offsets = cuda_row_data_->host_partition_hist_offsets();
        uint32_t slot_stride = 0;
        for (size_t p = 1; p < part_offsets.size(); ++p) {
          slot_stride = std::max(slot_stride, (part_offsets[p] - part_offsets[p - 1]) << 1);
        }
        // Double slot rows: float32 gradients summed in double are exact and
        // order-free, matching the CPU reference bit for bit.
        const size_t smem_per_row = static_cast<size_t>(slot_stride) * sizeof(double);
        constexpr bool kDetSparseConstructEnabled = true;
        if (kDetSparseConstructEnabled && det_tile_alloc_ > 0 && smem_per_row > 0 && smem_per_row <= kDetFloatSharedBudget) {
          const int det_dy = std::max(1, std::min<int>(block_dim.y, static_cast<int>(kDetFloatSharedBudget / smem_per_row)));
          // One writer per row group leaves the x lanes idle, so buy back
          // parallelism with a finer row tiling than the atomic kernel's.
          int det_grid_y = (num_data_in_smaller_leaf + kDetRowsPerThread * det_dy - 1) / (kDetRowsPerThread * det_dy);
          det_grid_y = std::max(det_grid_y, min_grid_dim_y_);
          det_grid_y = std::min(std::min(det_grid_y, static_cast<int>(kDetTileCap)), det_tile_alloc_);
          dim3 det_grid(grid_dim_x, static_cast<unsigned int>(std::max(det_grid_y, 1)));
          dim3 det_block(block_dim_x, static_cast<unsigned int>(det_dy));
          // Pipeline-private scratch region (pipelined pair-constructs run on
          // different pipeline streams; a shared region would race).
          hist_t* det_tiles = cuda_det_tile_partials_.RawData() +
            static_cast<size_t>(active_pipeline_) * det_tile_alloc_ * (2 * num_total_bin_);
          CUDAConstructHistogramSparseDeterministicKernel<BIN_TYPE, PTR_TYPE>
            <<<det_grid, det_block, static_cast<size_t>(det_dy) * smem_per_row, current_stream()>>>(
              cuda_smaller_leaf_splits,
              cuda_gradients_, cuda_hessians_,
              RowMajorBin<BIN_TYPE>(),
              cuda_row_data_->GetRowPtr<PTR_TYPE>(),
              cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              num_data_,
              slot_stride,
              det_tiles,
              static_cast<uint32_t>(2 * num_total_bin_));
          const int num_items_total = 2 * num_total_bin_;
          const int merge_threads = 256;
          const int merge_blocks = (num_items_total + merge_threads - 1) / merge_threads;
          MergeDeterministicHistogramKernel<<<merge_blocks, merge_threads, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            det_tiles,
            num_items_total,
            std::max(det_grid_y, 1));
        } else if (kDetSparseConstructEnabled && det_tile_alloc_ > 0 && smem_per_row > 0) {
          // Wide-partition fallback: one slot row exceeds the shared budget,
          // so the rows live in the dense det construct's global slab instead
          // (slot positions GLOBAL per partition; see
          // CUDAConstructHistogramSparseGMDeterministicKernel). The slab is
          // sized for every non-quantized dataset with a nonempty partition
          // table (cuda_histogram_constructor.cpp), this branch is
          // non-quantized-only, and smem_per_row > 0 proves the table is
          // nonempty, so the scratch is always present here; its stride spans
          // the full histogram, which is exactly what global slot positions
          // and the 2 * num_total_bin_ tile rows require.
          CHECK_GT(det_dense_dy_, 0);
          CHECK_EQ(det_dense_slot_stride_, static_cast<uint32_t>(2 * num_total_bin_));
          // block_dim_x * dy must stay within the 1024-thread block budget or
          // the launch fails with InvalidConfiguration -- which, unchecked,
          // is a SILENT no-op construct (see the dense det launch below).
          const int det_dy = std::max(1, std::min(det_dense_dy_, 1024 / std::max(1, block_dim_x)));
          int det_grid_y = (num_data_in_smaller_leaf + kDetRowsPerThread * det_dy - 1) / (kDetRowsPerThread * det_dy);
          det_grid_y = std::max(det_grid_y, min_grid_dim_y_);
          det_grid_y = std::min(std::min(det_grid_y, det_dense_tile_cap_), det_tile_alloc_);
          dim3 det_grid(grid_dim_x, static_cast<unsigned int>(std::max(det_grid_y, 1)));
          dim3 det_block(block_dim_x, static_cast<unsigned int>(det_dy));
          // Pipeline-private scratch regions (pipelined pair-constructs run
          // on different pipeline streams; shared regions would race).
          hist_t* det_slots = cuda_det_dense_slots_.RawData() +
            static_cast<size_t>(active_pipeline_) * det_dense_tile_cap_ * det_dense_dy_ * det_dense_slot_stride_;
          hist_t* det_tiles = cuda_det_tile_partials_.RawData() +
            static_cast<size_t>(active_pipeline_) * det_tile_alloc_ * (2 * num_total_bin_);
          CUDAConstructHistogramSparseGMDeterministicKernel<BIN_TYPE, PTR_TYPE>
            <<<det_grid, det_block, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              cuda_gradients_, cuda_hessians_,
              RowMajorBin<BIN_TYPE>(),
              cuda_row_data_->GetRowPtr<PTR_TYPE>(),
              cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              num_data_,
              det_dense_slot_stride_,
              det_slots,
              det_tiles,
              static_cast<uint32_t>(2 * num_total_bin_));
          const int num_items_total = 2 * num_total_bin_;
          const int merge_threads = 256;
          const int merge_blocks = (num_items_total + merge_threads - 1) / merge_threads;
          MergeDeterministicHistogramKernel<<<merge_blocks, merge_threads, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            det_tiles,
            num_items_total,
            std::max(det_grid_y, 1));
          CUDASUCCESS_OR_FATAL(cudaGetLastError());
        } else {
          CUDAConstructHistogramSparseKernel<BIN_TYPE, PTR_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
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
          if (det_dense_dy_ > 0 && !hist_fp32_) {
            // Deterministic float construct over the compact view: the
            // compact offsets share the det kernel's convention, so slot
            // positions land in the same global ranges as the uncompacted
            // path; each partition's FULL global range is zeroed while
            // accumulation touches only the sampled columns' subranges, so
            // unsampled columns merge as zeros -- dead storage the finder
            // never reads this tree.
            const DetDenseSource compact_source{
              active_data,
              compact_column_hist_offsets_.RawData(),
              compact_feature_partition_column_index_offsets_.RawData(),
              compact_is_4bit_ ? compact_packed_partition_byte_offsets_.RawData() : nullptr,
              nullptr,
              compact_is_4bit_};
            LaunchConstructHistogramDenseDeterministic<BIN_TYPE>(
              cuda_smaller_leaf_splits, num_data_in_smaller_leaf,
              grid_dim_x, compact_block_x, &compact_source);
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
              num_data_,
              hist_fp32_);
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
              num_data_,
              hist_fp32_);
          }
        } else if (det_dense_dy_ > 0 && !hist_fp32_) {
          // Deterministic float construct: double slot rows cannot fit the
          // shared budget at these partition widths, so this routes through
          // the global-slot kernel (see CUDAConstructHistogramDenseGMDeterministicKernel).
          // Tested BEFORE is_4bit_packed: the det kernel serves both row
          // layouts, so the order-dependent float atomics only remain for
          // shapes it does not cover (hist_fp32_ -- the det merge stores
          // hist_t doubles while that mode reinterprets the leaf histogram
          // as float pairs).
          LaunchConstructHistogramDenseDeterministic<BIN_TYPE>(
            cuda_smaller_leaf_splits, num_data_in_smaller_leaf, grid_dim_x, block_dim_x);
        } else if (cuda_row_data_->is_4bit_packed()) {
          CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            cuda_row_data_->cuda_packed_partition_byte_offsets(),
            cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
            num_data_,
            hist_fp32_);
        } else {
          CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            nullptr,
            cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
            num_data_,
            hist_fp32_);
        }
      }
    } else {
      if (cuda_row_data_->is_sparse()) {
        if (det_tile_alloc_ > 0 && det_dense_dy_ > 0 && num_total_bin_ > 0) {
          // Deterministic float construct: a large-bin partition put this
          // dataset on the GlobalMemory arm, so a shared slot row can never
          // fit and the global slab serves directly (slot positions GLOBAL
          // per partition; see CUDAConstructHistogramSparseGMDeterministicKernel
          // and the non-GM sparse arm for the launch-shape rationale).
          CHECK_EQ(det_dense_slot_stride_, static_cast<uint32_t>(2 * num_total_bin_));
          // block_dim_x * dy capped at the 1024-thread block budget; see the
          // non-GM arm for the silent-no-op failure mode.
          const int det_dy = std::max(1, std::min(det_dense_dy_, 1024 / std::max(1, block_dim_x)));
          int det_grid_y = (num_data_in_smaller_leaf + kDetRowsPerThread * det_dy - 1) / (kDetRowsPerThread * det_dy);
          det_grid_y = std::max(det_grid_y, min_grid_dim_y_);
          det_grid_y = std::min(std::min(det_grid_y, det_dense_tile_cap_), det_tile_alloc_);
          dim3 det_grid(grid_dim_x, static_cast<unsigned int>(std::max(det_grid_y, 1)));
          dim3 det_block(block_dim_x, static_cast<unsigned int>(det_dy));
          // Pipeline-private scratch regions (pipelined pair-constructs run
          // on different pipeline streams; shared regions would race).
          hist_t* det_slots = cuda_det_dense_slots_.RawData() +
            static_cast<size_t>(active_pipeline_) * det_dense_tile_cap_ * det_dense_dy_ * det_dense_slot_stride_;
          hist_t* det_tiles = cuda_det_tile_partials_.RawData() +
            static_cast<size_t>(active_pipeline_) * det_tile_alloc_ * (2 * num_total_bin_);
          CUDAConstructHistogramSparseGMDeterministicKernel<BIN_TYPE, PTR_TYPE>
            <<<det_grid, det_block, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              cuda_gradients_, cuda_hessians_,
              RowMajorBin<BIN_TYPE>(),
              cuda_row_data_->GetRowPtr<PTR_TYPE>(),
              cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              num_data_,
              det_dense_slot_stride_,
              det_slots,
              det_tiles,
              static_cast<uint32_t>(2 * num_total_bin_));
          const int num_items_total = 2 * num_total_bin_;
          const int merge_threads = 256;
          const int merge_blocks = (num_items_total + merge_threads - 1) / merge_threads;
          MergeDeterministicHistogramKernel<<<merge_blocks, merge_threads, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            det_tiles,
            num_items_total,
            std::max(det_grid_y, 1));
          CUDASUCCESS_OR_FATAL(cudaGetLastError());
        } else {
          CUDAConstructHistogramSparseKernel_GlobalMemory<BIN_TYPE, HIST_TYPE, PTR_TYPE><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            reinterpret_cast<HIST_TYPE*>(cuda_hist_buffer_.RawData()));
        }
      } else if (det_dense_dy_ > 0 && !hist_fp32_) {
        // Deterministic float construct for wide partitions (double slots in
        // global scratch; see CUDAConstructHistogramDenseGMDeterministicKernel).
        // fp32-pair histograms excluded; see the non-GM arm.
        LaunchConstructHistogramDenseDeterministic<BIN_TYPE>(
          cuda_smaller_leaf_splits, num_data_in_smaller_leaf, grid_dim_x, block_dim_x);
      } else {
        CUDAConstructHistogramDenseKernel_GlobalMemory<BIN_TYPE, HIST_TYPE><<<grid_dim, block_dim, 0, current_stream()>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          RowMajorBin<BIN_TYPE>(),
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
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool hist_fp32) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cuda_larger_leaf_index = cuda_larger_leaf_splits->leaf_index;
  if (cuda_larger_leaf_index >= 0) {
    if (global_thread_index < 2 * num_total_bin) {
      if (hist_fp32) {
        const float* smaller_leaf_hist = reinterpret_cast<const float*>(cuda_smaller_leaf_splits->hist_in_leaf);
        float* larger_leaf_hist = reinterpret_cast<float*>(cuda_larger_leaf_splits->hist_in_leaf);
        larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
      } else {
        const hist_t* smaller_leaf_hist = cuda_smaller_leaf_splits->hist_in_leaf;
        hist_t* larger_leaf_hist = cuda_larger_leaf_splits->hist_in_leaf;
        larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
      }
    }
  }
}

__global__ void SubtractHistogramKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool hist_fp32) {
  SubtractHistogramInner(num_total_bin, cuda_smaller_leaf_splits, cuda_larger_leaf_splits, hist_fp32);
}

// Batched per-level variant (hybrid growth): blockIdx.y selects the pair.
// bin_used (per-tree feature_fraction bin mask, may be null) skips histogram
// entries of features outside this tree's sample: nothing of this tree reads
// them (the find kernels are feature-masked), so they are dead storage.
__global__ void SubtractHistogramBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  const uint8_t* bin_used,
  const bool hist_fp32,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2: the graph-frozen grid is a pow2 bucket of the live pair count;
  // blocks beyond the live range exit before any read (stale descriptors)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (bin_used != nullptr &&
      global_thread_index < static_cast<unsigned int>(2 * num_total_bin) &&
      !bin_used[global_thread_index >> 1]) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  SubtractHistogramInner(num_total_bin, desc->smaller_struct, desc->larger_struct, hist_fp32);
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
  const bool hist_fp32,
  const int fix_feature_index = -1,
  const CUDALeafSplitsStruct* larger_for_subtract = nullptr) {
  const unsigned int blockIdx_x = fix_feature_index >= 0 ?
    static_cast<unsigned int>(fix_feature_index) : blockIdx.x;
  const int feature_index = cuda_need_fix_histogram_features[blockIdx_x];
  const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
  const double leaf_sum_gradients = cuda_smaller_leaf_splits->sum_of_gradients;
  const double leaf_sum_hessians = cuda_smaller_leaf_splits->sum_of_hessians;
  hist_t* feature_hist = cuda_smaller_leaf_splits->hist_in_leaf + feature_hist_offset * 2;
  float* feature_hist32 = reinterpret_cast<float*>(cuda_smaller_leaf_splits->hist_in_leaf) + feature_hist_offset * 2;
  const unsigned int threadIdx_x = threadIdx.x;
  const uint32_t num_bin = cuda_feature_num_bins[feature_index];
  // Sequential accumulation in ascending bin order: CPU's FixHistogram sums
  // the non-mfb bins in bin order, and the reconstructed most-frequent bin
  // must be bit-equal to CPU's or every threshold prefix crossing it inherits
  // the ulp difference. Bins are STAGED into shared memory cooperatively
  // (coalesced) so thread 0's order-exact fold reads shared, not a dependent
  // chain of global loads; chunking keeps it exact for any bin count. (The
  // discretized fix keeps its parallel reduction: integer sums are
  // order-invariant. shared_mem_buffer stays a parameter for that variant's
  // shared call signature.)
  (void)shared_mem_buffer;
  constexpr uint32_t kFixStageChunk = 256;
  __shared__ double fix_stage[2 * kFixStageChunk];
  hist_t sum_gradient = 0.0f;
  hist_t sum_hessian = 0.0f;
  for (uint32_t chunk_start = 0; chunk_start < num_bin; chunk_start += kFixStageChunk) {
    const uint32_t chunk = min(kFixStageChunk, num_bin - chunk_start);
    __syncthreads();
    for (uint32_t i = threadIdx_x; i < (chunk << 1); i += blockDim.x) {
      const uint32_t pos = (chunk_start << 1) + i;
      fix_stage[i] = hist_fp32 ? static_cast<hist_t>(feature_hist32[pos]) : feature_hist[pos];
    }
    __syncthreads();
    if (threadIdx_x == 0) {
      for (uint32_t b = 0; b < chunk; ++b) {
        if (chunk_start + b != most_freq_bin) {
          sum_gradient += fix_stage[b << 1];
          sum_hessian += fix_stage[(b << 1) + 1];
        }
      }
    }
  }
  if (threadIdx_x == 0) {
    const hist_t fixed_gradient = leaf_sum_gradients - sum_gradient;
    const hist_t fixed_hessian = leaf_sum_hessians - sum_hessian;
    if (hist_fp32) {
      feature_hist32[most_freq_bin << 1] = static_cast<float>(fixed_gradient);
      feature_hist32[(most_freq_bin << 1) + 1] = static_cast<float>(fixed_hessian);
      if (larger_for_subtract != nullptr && larger_for_subtract->leaf_index >= 0) {
        float* larger_feature_hist = reinterpret_cast<float*>(larger_for_subtract->hist_in_leaf) + feature_hist_offset * 2;
        larger_feature_hist[most_freq_bin << 1] -= static_cast<float>(fixed_gradient);
        larger_feature_hist[(most_freq_bin << 1) + 1] -= static_cast<float>(fixed_hessian);
      }
    } else {
      feature_hist[most_freq_bin << 1] = fixed_gradient;
      feature_hist[(most_freq_bin << 1) + 1] = fixed_hessian;
      if (larger_for_subtract != nullptr && larger_for_subtract->leaf_index >= 0) {
        hist_t* larger_feature_hist = larger_for_subtract->hist_in_leaf + feature_hist_offset * 2;
        larger_feature_hist[most_freq_bin << 1] -= fixed_gradient;
        larger_feature_hist[(most_freq_bin << 1) + 1] -= fixed_hessian;
      }
    }
  }
}

__global__ void FixHistogramKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const bool hist_fp32) {
  __shared__ hist_t shared_mem_buffer[WARPSIZE];
  FixHistogramInner(cuda_feature_num_bins, cuda_feature_hist_offsets,
    cuda_feature_most_freq_bins, cuda_need_fix_histogram_features,
    cuda_need_fix_histogram_features_num_bin_aligned, cuda_smaller_leaf_splits,
    shared_mem_buffer, hist_fp32);
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
  const int8_t* feature_used,
  const bool hist_fp32,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the subtract kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  if (feature_used != nullptr &&
      !feature_used[cuda_need_fix_histogram_features[blockIdx.x]]) {
    return;
  }
  __shared__ hist_t shared_mem_buffer[WARPSIZE];
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  FixHistogramInner(cuda_feature_num_bins, cuda_feature_hist_offsets,
    cuda_feature_most_freq_bins, cuda_need_fix_histogram_features,
    cuda_need_fix_histogram_features_num_bin_aligned, desc->smaller_struct,
    shared_mem_buffer, hist_fp32);
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
  const int8_t* feature_used,
  const bool hist_fp32,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the subtract kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  if (static_cast<int>(blockIdx.x) < num_subtract_blocks) {
    const CUDALeafSplitsStruct* larger_leaf = desc->larger_struct;
    if (larger_leaf->leaf_index >= 0) {
      const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      // bins of features outside this tree's sample are dead storage: skip
      if (i < static_cast<unsigned int>(2 * num_total_bin) && !fix_mfb_mask[i] &&
          (bin_used == nullptr || bin_used[i >> 1])) {
        if (hist_fp32) {
          reinterpret_cast<float*>(larger_leaf->hist_in_leaf)[i] -=
            reinterpret_cast<const float*>(desc->smaller_struct->hist_in_leaf)[i];
        } else {
          larger_leaf->hist_in_leaf[i] -= desc->smaller_struct->hist_in_leaf[i];
        }
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
      shared_mem_buffer, hist_fp32, static_cast<int>(blockIdx.x) - num_subtract_blocks,
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
// Host-launched path: bit widths from the host-written descriptor (bit-for-bit
// the previous behavior); graph loop: derived on-device from the child structs'
// exact leaf counts (parent == smaller + larger) with the host thresholds.
__global__ void SubtractHistogramDiscretizedBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
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
  uint8_t parent_num_bits = desc->parent_num_bits;
  uint8_t smaller_num_bits = desc->smaller_num_bits;
  uint8_t larger_num_bits = desc->larger_num_bits;
  if (HybridGraphActive(gstate)) {
    const data_size_t num_data_smaller = desc->smaller_struct->num_data_in_leaf;
    const data_size_t num_data_larger = desc->larger_struct->leaf_index >= 0 ?
      desc->larger_struct->num_data_in_leaf : 0;
    smaller_num_bits = HybridGraphQuantHistBits(gstate, num_data_smaller);
    larger_num_bits = HybridGraphQuantHistBits(gstate, num_data_larger);
    // the parent's leaf bit width was set from its (pre-split) row count,
    // which the split partitions exactly into the two children
    parent_num_bits = HybridGraphQuantHistBits(gstate, num_data_smaller + num_data_larger);
  }
  if (parent_num_bits <= 16) {
    SubtractHistogramDiscretizedInner<true, true, true>(
      num_total_bin, desc->smaller_struct, desc->larger_struct, buffer);
  } else if (larger_num_bits <= 16) {
    SubtractHistogramDiscretizedInner<true, true, false>(
      num_total_bin, desc->smaller_struct, desc->larger_struct, buffer);
  } else if (smaller_num_bits <= 16) {
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
// Inside the graph loop the node is always captured (the host launches it only
// on bit-change levels) and the per-pair device-derived bit widths gate it.
__global__ void CopyChangedNumBitHistogramBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  uint8_t parent_num_bits = desc->parent_num_bits;
  uint8_t larger_num_bits = desc->larger_num_bits;
  int larger_leaf_index = desc->larger_leaf_index;
  if (HybridGraphActive(gstate)) {
    larger_leaf_index = desc->larger_struct->leaf_index;
    const data_size_t num_data_smaller = desc->smaller_struct->num_data_in_leaf;
    const data_size_t num_data_larger = larger_leaf_index >= 0 ?
      desc->larger_struct->num_data_in_leaf : 0;
    larger_num_bits = HybridGraphQuantHistBits(gstate, num_data_larger);
    parent_num_bits = HybridGraphQuantHistBits(gstate, num_data_smaller + num_data_larger);
  }
  if (larger_leaf_index < 0 ||
      !(parent_num_bits > 16 && larger_num_bits <= 16)) {
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
    // block-strided: one-bin-per-thread missed bins >= blockDim.x (see the
    // non-discretized FixHistogramInner for the failure mode)
    int32_t bin_gradient_hessian = 0;
    for (uint32_t bin = threadIdx_x; bin < num_bin; bin += blockDim.x) {
      if (bin != most_freq_bin) {
        bin_gradient_hessian += feature_hist[bin];
      }
    }
    const uint32_t reduce_len = num_bin_aligned < blockDim.x ? num_bin_aligned : blockDim.x;
    const int32_t sum_gradient_hessian = ShuffleReduceSum<int32_t>(
      bin_gradient_hessian,
      reinterpret_cast<int32_t*>(shared_mem_buffer),
      reduce_len);
    if (threadIdx_x == 0) {
      feature_hist[most_freq_bin] = leaf_sum_gradients_hessians - sum_gradient_hessian;
    }
  } else {
    const int64_t leaf_sum_gradients_hessians = cuda_smaller_leaf_splits->sum_of_gradients_hessians;
    int64_t* feature_hist = reinterpret_cast<int64_t*>(cuda_smaller_leaf_splits->hist_in_leaf) + feature_hist_offset;
    const unsigned int threadIdx_x = threadIdx.x;
    const uint32_t num_bin = cuda_feature_num_bins[feature_index];
    // block-strided: one-bin-per-thread missed bins >= blockDim.x (see the
    // non-discretized FixHistogramInner for the failure mode)
    int64_t bin_gradient_hessian = 0;
    for (uint32_t bin = threadIdx_x; bin < num_bin; bin += blockDim.x) {
      if (bin != most_freq_bin) {
        bin_gradient_hessian += feature_hist[bin];
      }
    }
    const uint32_t reduce_len = num_bin_aligned < blockDim.x ? num_bin_aligned : blockDim.x;
    const int64_t sum_gradient_hessian = ShuffleReduceSum<int64_t>(bin_gradient_hessian, shared_mem_buffer, reduce_len);
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
// per-pair histogram bit width is a runtime (block-uniform) branch (host
// descriptor on the host-launched path, device-derived inside the graph loop).
__global__ void FixHistogramDiscretizedBatchedKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDAHybridPairDescriptor* pair_descs,
  const int8_t* feature_used,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  if (feature_used != nullptr &&
      !feature_used[cuda_need_fix_histogram_features[blockIdx.x]]) {
    return;  // most-frequent bin of an unused feature: dead storage this tree
  }
  __shared__ int64_t shared_mem_buffer[WARPSIZE];
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  const uint8_t smaller_num_bits = HybridGraphActive(gstate) ?
    HybridGraphQuantHistBits(gstate, desc->smaller_struct->num_data_in_leaf) :
    desc->smaller_num_bits;
  if (smaller_num_bits <= 16) {
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

// cuda_plan key fix_subtract_fused: one sampled feature (at most kFixSubtractMaxSpan bins) of one pair, for a pair whose
// bit widths type the smaller leaf's bins the same way in the fix and in the subtract (SUB_CASE: 0 parent <= 16 bits,
// 1 parent > 16 and larger <= 16 (into the change buffer), 2 only the smaller <= 16, 3 none; the fix types them int32
// iff the smaller is <= 16 bits, i.e. in cases 0-2). Every bin of the feature is loaded first; the fix (as
// FixHistogramDiscretizedInner: the leaf total minus the wrapping sum of the other bins below num_bin) replaces the
// most-frequent bin in registers and in the smaller histogram, and the subtract (as SubtractHistogramDiscretizedInner)
// uses that value. No other thread reads or writes this feature's bins.
constexpr uint32_t kFixSubtractMaxSpan = 8;
// Returns whether it did the change-buffer copy (case 1 with copy_changed, which is block-uniform: every thread of
// the block calls this with the same SUB_CASE, larger_exists and copy_changed): from the values it just wrote to the
// buffer, after a barrier behind every subtract read of the pair's 32-bit view.
template <int SUB_CASE>
__device__ __forceinline__ bool FixSubtractUsedFeature(
  const uint2 item, const CUDALeafSplitsStruct* smaller, const CUDALeafSplitsStruct* larger,
  const bool larger_exists, const bool copy_changed, int32_t* buffer) {
  typedef typename std::conditional<SUB_CASE == 3, int64_t, int32_t>::type SmallerT;
  typedef typename std::conditional<SUB_CASE == 0, int32_t, int64_t>::type LargerT;
  typedef typename std::conditional<SUB_CASE == 3, uint64_t, uint32_t>::type SumT;
  const uint32_t span = item.y & 0xffu;
  const bool need_fix = (item.y >> 24) != 0;
  const uint32_t num_bin = (item.y >> 8) & 0xffu;
  const uint32_t most_freq_bin = (item.y >> 16) & 0xffu;
  SmallerT* smaller_hist = reinterpret_cast<SmallerT*>(smaller->hist_in_leaf) + item.x;
  const int64_t leaf_sum_gradients_hessians_int64 = smaller->sum_of_gradients_hessians;
  SmallerT s[kFixSubtractMaxSpan];
  LargerT l[kFixSubtractMaxSpan];
#pragma unroll
  for (uint32_t k = 0; k < kFixSubtractMaxSpan; ++k) {
    s[k] = k < span ? smaller_hist[k] : 0;
  }
  if (larger_exists) {
    const LargerT* larger_hist = reinterpret_cast<const LargerT*>(larger->hist_in_leaf) + item.x;
#pragma unroll
    for (uint32_t k = 0; k < kFixSubtractMaxSpan; ++k) {
      l[k] = k < span ? larger_hist[k] : 0;
    }
  }
  if (need_fix) {
    SumT sum_gradient_hessian = 0;
#pragma unroll
    for (uint32_t k = 0; k < kFixSubtractMaxSpan; ++k) {
      if (k < num_bin && k != most_freq_bin) {
        sum_gradient_hessian += static_cast<SumT>(s[k]);
      }
    }
    SmallerT leaf_sum_gradients_hessians;
    if (SUB_CASE == 3) {
      leaf_sum_gradients_hessians = static_cast<SmallerT>(leaf_sum_gradients_hessians_int64);
    } else {
      leaf_sum_gradients_hessians = static_cast<SmallerT>(
        (static_cast<int32_t>(leaf_sum_gradients_hessians_int64 >> 32) << 16) | static_cast<int32_t>(leaf_sum_gradients_hessians_int64 & 0x000000000000ffff));
    }
    const SmallerT fixed = static_cast<SmallerT>(static_cast<SumT>(leaf_sum_gradients_hessians) - sum_gradient_hessian);
#pragma unroll
    for (uint32_t k = 0; k < kFixSubtractMaxSpan; ++k) {
      if (k == most_freq_bin) s[k] = fixed;
    }
    smaller_hist[most_freq_bin] = fixed;
  }
  if (!larger_exists) {
    return false;
  }
  if (SUB_CASE == 1) {
    int32_t* out = buffer + item.x;
    int32_t o[kFixSubtractMaxSpan];
#pragma unroll
    for (uint32_t k = 0; k < kFixSubtractMaxSpan; ++k) {
      const int64_t parent_hist_item = static_cast<int64_t>(l[k]);
      const int32_t smaller_hist_item = static_cast<int32_t>(s[k]);
      const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
        static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
      const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
      o[k] = static_cast<int32_t>(static_cast<int16_t>(larger_hist_item >> 32) << 16) |
        static_cast<int32_t>(larger_hist_item & 0x000000000000ffff);
      if (k < span) {
        out[k] = o[k];
      }
    }
    if (copy_changed) {
      __syncthreads();
      int32_t* hist_dst = reinterpret_cast<int32_t*>(larger->hist_in_leaf) + item.x;
#pragma unroll
      for (uint32_t k = 0; k < kFixSubtractMaxSpan; ++k) {
        if (k < span) {
          hist_dst[k] = o[k];
        }
      }
      return true;
    }
  } else {
    LargerT* larger_hist = reinterpret_cast<LargerT*>(larger->hist_in_leaf) + item.x;
#pragma unroll
    for (uint32_t k = 0; k < kFixSubtractMaxSpan; ++k) {
      if (k < span) {
        if (SUB_CASE == 2) {
          const int64_t parent_hist_item = static_cast<int64_t>(l[k]);
          const int32_t smaller_hist_item = static_cast<int32_t>(s[k]);
          const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
            static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
          larger_hist[k] = static_cast<LargerT>(parent_hist_item - smaller_hist_item_int64);
        } else {
          // cases 0 and 3: larger -= smaller in the shared type
          LargerT v = l[k];
          v -= static_cast<LargerT>(s[k]);
          larger_hist[k] = v;
        }
      }
    }
  }
  return false;
}

// cuda_plan key fix_subtract_fused (quantized, host-launched level, feature sample active): block = pair, thread =
// sampled feature (blockDim >= the number of sampled features). FixHistogramDiscretizedBatchedKernel for the pair's
// sampled features that need a fix, SubtractHistogramDiscretizedBatchedKernel for their bins and
// CopyChangedNumBitHistogramBatchedKernel's copy of them, in one launch: the same integer expressions on the same
// inputs under the same conditions. Every fix of the pair is written before any of its subtracts reads it (within a
// thread when the bit widths type the smaller leaf's bins alike in both; else in two phases with a barrier between
// them, as across the two kernels), and the copy into the larger leaf's 16-bit view (which overlaps other bins of
// its 32-bit view) starts after a barrier behind every subtract read of the pair. Pairs touch disjoint histograms
// and change-buffer regions. used_feature_info: two words per sampled feature (see host_used_feature_info_).
__global__ void __launch_bounds__(512) FixSubtractUsedFeaturesBatchedKernel(
  const uint32_t* used_feature_info,
  const int num_used_features,
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer) {
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.x;
  const CUDALeafSplitsStruct* smaller = desc->smaller_struct;
  const CUDALeafSplitsStruct* larger = desc->larger_struct;
  const uint8_t parent_num_bits = desc->parent_num_bits;
  const uint8_t smaller_num_bits = desc->smaller_num_bits;
  const uint8_t larger_num_bits = desc->larger_num_bits;
  const int sub_case = parent_num_bits <= 16 ? 0 : (larger_num_bits <= 16 ? 1 : (smaller_num_bits <= 16 ? 2 : 3));
  // the subtract's gate (the larger struct) and the copy's (the descriptor), as in the two kernels
  const bool larger_exists = larger->leaf_index >= 0;
  const bool copy_changed = desc->larger_leaf_index >= 0 && parent_num_bits > 16 && larger_num_bits <= 16;
  const int u = static_cast<int>(threadIdx.x);
  const uint2 item = u < num_used_features ? reinterpret_cast<const uint2*>(used_feature_info)[u] : make_uint2(0u, 0u);
  int32_t* buffer = reinterpret_cast<int32_t*>(num_bit_change_buffer) +
    static_cast<size_t>(blockIdx.x) * static_cast<size_t>(num_total_bin);
  const uint32_t span = item.y & 0xffu;  // 0 for threads past the sampled features
  bool copied = false;  // block-uniform
  if ((smaller_num_bits <= 16) == (sub_case != 3)) {
    switch (sub_case) {
      case 0: copied = FixSubtractUsedFeature<0>(item, smaller, larger, larger_exists, copy_changed, buffer); break;
      case 1: copied = FixSubtractUsedFeature<1>(item, smaller, larger, larger_exists, copy_changed, buffer); break;
      case 2: copied = FixSubtractUsedFeature<2>(item, smaller, larger, larger_exists, copy_changed, buffer); break;
      default: copied = FixSubtractUsedFeature<3>(item, smaller, larger, larger_exists, copy_changed, buffer); break;
    }
  } else {
    // the fix and the subtract type the smaller leaf's bins differently: the two phases of the original kernels
    if ((item.y >> 24) != 0) {
      const uint32_t num_bin = (item.y >> 8) & 0xffu;
      const uint32_t most_freq_bin = (item.y >> 16) & 0xffu;
      if (smaller_num_bits <= 16) {
        const int64_t leaf_sum_gradients_hessians_int64 = smaller->sum_of_gradients_hessians;
        const int32_t leaf_sum_gradients_hessians =
          (static_cast<int32_t>(leaf_sum_gradients_hessians_int64 >> 32) << 16) | static_cast<int32_t>(leaf_sum_gradients_hessians_int64 & 0x000000000000ffff);
        int32_t* feature_hist = reinterpret_cast<int32_t*>(smaller->hist_in_leaf) + item.x;
        uint32_t sum_gradient_hessian = 0;  // the block reduction's wrapping int32 sum
        for (uint32_t bin = 0; bin < num_bin; ++bin) {
          if (bin != most_freq_bin) {
            sum_gradient_hessian += static_cast<uint32_t>(feature_hist[bin]);
          }
        }
        feature_hist[most_freq_bin] = static_cast<int32_t>(
          static_cast<uint32_t>(leaf_sum_gradients_hessians) - sum_gradient_hessian);
      } else {
        const int64_t leaf_sum_gradients_hessians = smaller->sum_of_gradients_hessians;
        int64_t* feature_hist = reinterpret_cast<int64_t*>(smaller->hist_in_leaf) + item.x;
        uint64_t sum_gradient_hessian = 0;  // the block reduction's wrapping int64 sum
        for (uint32_t bin = 0; bin < num_bin; ++bin) {
          if (bin != most_freq_bin) {
            sum_gradient_hessian += static_cast<uint64_t>(feature_hist[bin]);
          }
        }
        feature_hist[most_freq_bin] = static_cast<int64_t>(
          static_cast<uint64_t>(leaf_sum_gradients_hessians) - sum_gradient_hessian);
      }
    }
    __syncthreads();
    if (larger_exists) {
      const uint32_t bin_start = item.x;
      const uint32_t bin_end = item.x + span;
      if (sub_case == 0) {
        const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(smaller->hist_in_leaf);
        int32_t* larger_leaf_hist = reinterpret_cast<int32_t*>(larger->hist_in_leaf);
        for (uint32_t i = bin_start; i < bin_end; ++i) {
          larger_leaf_hist[i] -= smaller_leaf_hist[i];
        }
      } else if (sub_case == 1) {
        const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(smaller->hist_in_leaf);
        const int64_t* larger_leaf_hist = reinterpret_cast<const int64_t*>(larger->hist_in_leaf);
        for (uint32_t i = bin_start; i < bin_end; ++i) {
          const int64_t parent_hist_item = larger_leaf_hist[i];
          const int32_t smaller_hist_item = smaller_leaf_hist[i];
          const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
            static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
          const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
          buffer[i] = static_cast<int32_t>(static_cast<int16_t>(larger_hist_item >> 32) << 16) |
            static_cast<int32_t>(larger_hist_item & 0x000000000000ffff);
        }
      } else if (sub_case == 2) {
        const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(smaller->hist_in_leaf);
        int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(larger->hist_in_leaf);
        for (uint32_t i = bin_start; i < bin_end; ++i) {
          const int64_t parent_hist_item = larger_leaf_hist[i];
          const int32_t smaller_hist_item = smaller_leaf_hist[i];
          const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
            static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
          const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
          larger_leaf_hist[i] = larger_hist_item;
        }
      } else {
        const int64_t* smaller_leaf_hist = reinterpret_cast<const int64_t*>(smaller->hist_in_leaf);
        int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(larger->hist_in_leaf);
        for (uint32_t i = bin_start; i < bin_end; ++i) {
          larger_leaf_hist[i] -= smaller_leaf_hist[i];
        }
      }
    }
  }
  if (copy_changed && !copied) {
    // CopyChangedNumBitHistogramBatchedKernel: this thread's bins of the change buffer (its own writes above, or
    // what the buffer held if the subtract was gated off) into the larger leaf's 16-bit view, after every subtract
    // read of the pair's 32-bit view
    __syncthreads();
    int32_t* hist_dst = reinterpret_cast<int32_t*>(larger->hist_in_leaf);
    for (uint32_t i = item.x; i < item.x + span; ++i) {
      hist_dst[i] = buffer[i];
    }
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
          cuda_smaller_leaf_splits,
          hist_fp32_);
      }
      global_timer.Stop("CUDAHistogramConstructor::FixHistogramKernel");
      global_timer.Start("CUDAHistogramConstructor::SubtractHistogramKernel");
      SubtractHistogramKernel<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, current_stream()>>>(
        num_total_bin_,
        cuda_smaller_leaf_splits,
        cuda_larger_leaf_splits,
        hist_fp32_);
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

// Shared per-leaf deterministic dense construct+merge launch (the non-GM and
// GM per-leaf arms route here identically).
// The block budget caps the row groups: block_dim_x follows the widest
// partition's column count, and block_dim_x * dy must stay within 1024 threads
// or the launch fails with InvalidConfiguration -- which, unchecked, is a
// SILENT no-op construct: the histogram stays zero and FixHistogram turns it
// into a degenerate everything-in-the-mfb-bin shape that gates out every split.
template <typename BIN_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramDenseDeterministic(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const int grid_dim_x,
  const int block_dim_x,
  const DetDenseSource* source) {
  const int det_dy = std::max(1, std::min(det_dense_dy_, 1024 / std::max(1, block_dim_x)));
  int det_grid_y = (num_data_in_smaller_leaf + kDetRowsPerThread * det_dy - 1) / (kDetRowsPerThread * det_dy);
  det_grid_y = std::min(std::min(det_grid_y, det_dense_tile_cap_), det_tile_alloc_);
  dim3 det_grid(grid_dim_x, static_cast<unsigned int>(std::max(det_grid_y, 1)));
  dim3 det_block(block_dim_x, static_cast<unsigned int>(det_dy));
  // Pipeline-private scratch region (pipelined pair-constructs run on
  // different pipeline streams; a shared region would race).
  hist_t* det_slots = cuda_det_dense_slots_.RawData() +
    static_cast<size_t>(active_pipeline_) * det_dense_tile_cap_ * det_dense_dy_ * det_dense_slot_stride_;
  const bool det_packed_4bit = source != nullptr ?
    source->is_4bit : cuda_row_data_->is_4bit_packed();
  auto det_kernel = grad_only_plane_ ?
    (det_packed_4bit ?
      &CUDAConstructHistogramDenseGMDeterministicKernel<BIN_TYPE, true, true> :
      &CUDAConstructHistogramDenseGMDeterministicKernel<BIN_TYPE, false, true>) :
    (det_packed_4bit ?
      &CUDAConstructHistogramDenseGMDeterministicKernel<BIN_TYPE, true, false> :
      &CUDAConstructHistogramDenseGMDeterministicKernel<BIN_TYPE, false, false>);
  det_kernel<<<det_grid, det_block, 0, current_stream()>>>(
      cuda_smaller_leaf_splits,
      cuda_gradients_, cuda_hessians_,
      source != nullptr ? static_cast<const BIN_TYPE*>(source->data)
                        : RowMajorBin<BIN_TYPE>(),
      source != nullptr ? source->column_hist_offsets
                        : cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      source != nullptr ? source->partition_column_offsets
                        : cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      source != nullptr ? source->packed_byte_offsets :
        (det_packed_4bit ? cuda_row_data_->cuda_packed_partition_byte_offsets() : nullptr),
      source != nullptr ? source->is_feature_used :
        (cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr),
      num_data_,
      det_dense_slot_stride_,
      det_slots,
      det_dy);
  const int merge_threads = 256;
  dim3 merge_grid((det_dense_slot_stride_ + merge_threads - 1) / merge_threads, grid_dim_x);
  MergeDeterministicDenseHistogramKernel<<<merge_grid, merge_threads, 0, current_stream()>>>(
    cuda_smaller_leaf_splits,
    det_slots,
    cuda_row_data_->cuda_partition_hist_offsets(),
    det_dense_slot_stride_,
    std::max(det_grid_y, 1) * det_dy);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

// Eligibility of the deterministic BATCHED dense construct for a level:
// mirrors the per-leaf det arm's shape gates (non-quantized, dense global-slot
// scratch engaged, no fp32-pair histograms, no compact column view -- the
// per-leaf det path does not serve compact either, so batched coverage equals
// per-leaf coverage), plus two batch-specific ones:
//  - hybrid_graph_capture_gstate_ != nullptr: the graph loop keeps the atomic
//    kernel. The captured construct node's frozen pow2 grid and baked pointer
//    parameters are incompatible with a two-kernel construct+merge (the merge
//    would need its own controller node role, resized and gated per level);
//    until that lands, the captured body stays the single atomic kernel
//    (order-invariant for quantized training, quality-parity for non-quant).
//  - the level's pairs must each get at least one slot row out of the full
//    deterministic slab (batched levels run on the single batched stream, so
//    all kNumHistPipelines regions are available to carve up), capped at
//    kDetDenseBatchedPairCap. Determinism is level-global -- one atomic level
//    would break run-to-run identity of the whole model -- so the cap is
//    sized to cover every level of the deepest supported prefixes rather than
//    being a small tuning constant.
bool CUDAHistogramConstructor::DetDenseBatchedEligible(const int num_pairs) const {
  if (use_quantized_grad_ || det_dense_dy_ <= 0 || hist_fp32_ || use_compact_view_) {
    return false;
  }
  if (hybrid_graph_capture_gstate_ != nullptr || !det_batched_allowed_) {
    // hybrid_graph_capture_gstate_: the graph loop's det construct is
    // captured explicitly (CaptureHybridGraphDetConstructMerge), never
    // through this launcher chain -- an atomic body capture must stay atomic.
    // det_batched_allowed_: one tree must keep ONE ownership model of the
    // histogram storage end to end. Mixing graph-replayed atomic construct
    // levels with host det construct+merge levels raced in the 2026-08-19
    // lattice (intermittent illegal access, imbalanced/nonquant); the
    // learner therefore allows host batched det only when the graph prefix
    // is off OR the graph itself runs the det nodes (DetDenseGraphEligible).
    return false;
  }
  const int total_slot_rows = det_dense_total_slot_rows();
  return num_pairs > 0 &&
         num_pairs <= std::min(total_slot_rows, kDetDenseBatchedPairCap);
}

bool CUDAHistogramConstructor::DetDenseGraphEligible(const int max_level_pairs) const {
  if (!FalcataPlan::Get().graph_det) {
    // opt-in (cuda_plan=auto,graph_det:on): by default the graph loop keeps
    // the atomic construct -- the det merge costs ~4x on shallow non-quant
    // shapes, and the default trades that determinism for speed. graph_det
    // gates BOTH the capture and the host-det-alongside-graph allowance
    // (SetDetBatchedAllowed), so a default run stays atomic end to end.
    return false;
  }
  if (use_quantized_grad_ || det_dense_dy_ <= 0 || hist_fp32_ || use_compact_view_) {
    return false;
  }
  const int total_slot_rows = det_dense_total_slot_rows();
  return max_level_pairs > 0 &&
         max_level_pairs <= std::min(total_slot_rows, kDetDenseBatchedPairCap);
}

// Deterministic dense construct+merge for a whole hybrid level: one construct
// launch (pair axis = blockIdx.z) and one merge launch on cuda_stream_. Each
// pair owns floor(total_rows / num_pairs) slot rows of the full deterministic
// slab; dy and the tile grid are clamped so every pair's used rows fit its
// region. max_num_data_in_smaller_leaf may be an upper BOUND (speculative
// single-sync flow): grid sizing from a bound is safe because the kernel reads
// the actual leaf sizes from the device structs and the double-slot sums are
// exact under any row grouping.
template <typename BIN_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramDenseBatchedDeterministic(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const data_size_t max_num_data_in_smaller_leaf,
  const int grid_dim_x,
  const int block_dim_x) {
  const int total_slot_rows = kNumHistPipelines * det_dense_tile_cap_ * det_dense_dy_;
  const int pair_rows = std::max(1, total_slot_rows / num_pairs);
  // block_dim_x * dy within the 1024-thread block budget (see the per-leaf
  // launcher for the silent-no-op failure mode of an oversized block)
  int det_dy = std::max(1, std::min(det_dense_dy_, 1024 / std::max(1, block_dim_x)));
  det_dy = std::min(det_dy, pair_rows);
  const int pair_tiles = std::max(1, pair_rows / det_dy);
  int det_grid_y = (max_num_data_in_smaller_leaf + kDetRowsPerThread * det_dy - 1) /
    (kDetRowsPerThread * det_dy);
  det_grid_y = std::min(std::max(det_grid_y, 1), pair_tiles);
  dim3 det_grid(grid_dim_x, static_cast<unsigned int>(det_grid_y),
                static_cast<unsigned int>(num_pairs));
  dim3 det_block(block_dim_x, static_cast<unsigned int>(det_dy));
  const bool det_packed_4bit = cuda_row_data_->is_4bit_packed();
  auto det_kernel = det_packed_4bit ?
    &CUDAConstructHistogramDenseGMDeterministicBatchedKernel<BIN_TYPE, true> :
    &CUDAConstructHistogramDenseGMDeterministicBatchedKernel<BIN_TYPE, false>;
  det_kernel<<<det_grid, det_block, 0, cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      RowMajorBin<BIN_TYPE>(),
      cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      det_packed_4bit ? cuda_row_data_->cuda_packed_partition_byte_offsets() : nullptr,
      cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
      num_data_,
      static_cast<data_size_t>(min_data_in_leaf_),
      min_sum_hessian_in_leaf_,
      det_dense_slot_stride_,
      cuda_det_dense_slots_.RawData(),
      total_slot_rows,
      det_dy,
      nullptr);
  const int merge_threads = 256;
  dim3 merge_grid((det_dense_slot_stride_ + merge_threads - 1) / merge_threads,
                  grid_dim_x, static_cast<unsigned int>(num_pairs));
  MergeDeterministicDenseHistogramBatchedKernel<<<merge_grid, merge_threads, 0, cuda_stream_>>>(
    pair_descs,
    cuda_det_dense_slots_.RawData(),
    cuda_row_data_->cuda_partition_hist_offsets(),
    det_dense_slot_stride_,
    total_slot_rows,
    det_dy,
    num_pairs,
    static_cast<data_size_t>(min_data_in_leaf_),
    min_sum_hessian_in_leaf_,
    nullptr);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

// Root level with the fused fill (cuda_plan key fused_root_hist): adds the scratch the fill accumulated into the
// root leaf's histogram, under the gate of CUDAConstructDiscretizedHistogramDenseBatchedKernel (pair 0 is the root;
// the bit width matches the scratch, the host checked it). Each entry has one writer and receives the same integer
// sum the construct's atomics would add.
__global__ void ApplyFusedRootHistogramKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const hist_t* root_hist_scratch,
  const int num_total_bin,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf) {
  const CUDAHybridPairDescriptor* desc = pair_descs;
  if (!desc->construct_valid) {
    return;
  }
  const CUDALeafSplitsStruct* smaller_struct = desc->smaller_struct;
  const data_size_t num_data_smaller = smaller_struct->num_data_in_leaf;
  const double sum_hessians_smaller = smaller_struct->sum_of_hessians;
  const CUDALeafSplitsStruct* larger_struct = desc->larger_struct;
  const bool has_larger = larger_struct->leaf_index >= 0;
  const data_size_t num_data_larger = has_larger ? larger_struct->num_data_in_leaf : 0;
  const double sum_hessians_larger = has_larger ? larger_struct->sum_of_hessians : 0.0;
  if ((num_data_smaller <= min_data_in_leaf || sum_hessians_smaller <= min_sum_hessian_in_leaf) &&
      (num_data_larger <= min_data_in_leaf || sum_hessians_larger <= min_sum_hessian_in_leaf)) {
    return;
  }
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= num_total_bin) {
    return;
  }
  if (desc->smaller_num_bits <= 16) {
    const uint32_t v = reinterpret_cast<const uint32_t*>(root_hist_scratch)[i];
    if (v != 0) {
      uint32_t* hist = reinterpret_cast<uint32_t*>(smaller_struct->hist_in_leaf);
      hist[i] += v;
    }
  } else {
    const atomic_add_long_t v = reinterpret_cast<const atomic_add_long_t*>(root_hist_scratch)[i];
    if (v != 0) {
      atomic_add_long_t* hist = reinterpret_cast<atomic_add_long_t*>(smaller_struct->hist_in_leaf);
      hist[i] += v;
    }
  }
}

void CUDAHistogramConstructor::LaunchApplyFusedRootHistogram(const CUDAHybridPairDescriptor* pair_descs) {
  const int block = 256;
  const int grid = (num_total_bin_ + block - 1) / block;
  ApplyFusedRootHistogramKernel<<<grid, block, 0, cuda_stream_>>>(
    pair_descs, fused_root_scratch_.RawDataReadOnly(), num_total_bin_,
    static_cast<data_size_t>(min_data_in_leaf_), min_sum_hessian_in_leaf_);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const data_size_t max_num_data_in_smaller_leaf,
  const data_size_t* level_smaller_num_data) {
  // One-time NVRTC JIT self-check (no-op unless cuda_plan=auto,construct_jit:on). Runs
  // off the first construct; never affects the trained model. The self-test
  // compiles + launches + copies on the stream, which is illegal mid graph
  // capture ("operation not permitted when stream is capturing"); if the first
  // construct is captured (multiclass graph loop with JIT force-enabled), defer
  // the self-test to the first NON-capturing construct. The JIT declines under
  // graph capture anyway, so deferring never loses a live launch.
  if (!construct_jit_selftest_done_ && construct_jit_allowed_ && CUDAConstructJIT::Enabled()) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cuda_stream_, &cap) == cudaSuccess &&
        cap == cudaStreamCaptureStatusNone) {
      RunConstructJITSelfTest();
    }
  } else if (!construct_jit_selftest_done_ && !construct_jit_allowed_) {
    // JIT not allowed for this run (plan off, or auto-gate: < 300 rounds):
    // mark done directly. (Historically this called RunConstructJITSelfTest
    // relying on its !Enabled() short-circuit -- with construct_jit now
    // default-ON that would run the full compile+launch self-test, including
    // MID GRAPH CAPTURE, which is illegal.)
    construct_jit_selftest_done_ = true;
  }
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
    // USED columns of a partition; the y sizing formula is the batched one
    // INCLUDING the quantized overflow guard: the compact block_dim_y is
    // recomputed here (wider than the generic one when few columns are used),
    // so the rows-per-block cap (65534/bins, guarding the packed 16+16-bit
    // shared-histogram partials) must be re-applied for it. Sizing with the
    // plain formula overflowed the int16 partials on large leaves at high
    // quant_bins (the "fixedpoint stops after 44 trees" production
    // corruption: quant_bins=64, 6.8M rows, feature_fraction 0.1);
    // num_grad_quant_bins == 0 (non-quantized) reduces to the plain formula.
    {
      const int cc = std::max(1, max_num_compact_cols_per_partition_);
      block_dim_x = cc > NUM_THREADS_PER_BLOCK ? (cc + 1) / 2 : cc;
    }
    block_dim_y = std::max(1, NUM_THREADS_PER_BLOCK / block_dim_x);
    block_dim_y = HybridQuantConstructBlockDimY(
      block_dim_y, use_quantized_grad_ ? num_grad_quant_bins_ : 0);
    grid_dim_y = HybridBatchedConstructGridDimYQuant(
      max_num_data_in_smaller_leaf, num_pairs, block_dim_y, min_grid_dim_y_,
      BatchConstructMinRowsPerThread(), BatchConstructSaturationFloor(),
      use_quantized_grad_ ? num_grad_quant_bins_ : 0);
  }
  dim3 grid_dim(grid_dim_x, grid_dim_y, num_pairs);
  dim3 block_dim(block_dim_x, block_dim_y);
  const bool det_batched = DetDenseBatchedEligible(num_pairs);
  const int* level_dim_y = nullptr;
  const data_size_t* level_sizes_for_kernel = nullptr;
  if (level_smaller_num_data != nullptr && !det_batched) {
    // speculative flow: the grid above was sized from an upper BOUND; the exact
    // row-grouping extent comes from the level's actual sizes. Few-pair levels
    // evaluate the formula inside the construct kernel itself (saves a launch
    // on the per-level critical path); many-pair levels keep the single-block
    // reduction kernel so construct blocks read one precomputed scalar.
    // (The deterministic batched construct needs neither: its exact double
    // sums are row-grouping invariant, so the launched grid extent is used.)
    if (num_pairs <= 32) {
      level_sizes_for_kernel = level_smaller_num_data;
    } else {
      LaunchComputeBatchedConstructDimYKernel(level_smaller_num_data, num_pairs, block_dim_y);
      level_dim_y = cuda_hybrid_construct_dim_y_.RawDataReadOnly();
    }
  }
  if (use_quantized_grad_) {
    // All-small level fast path: when even the level's LARGEST smaller-leaf is
    // under the small-leaf threshold (deep-tree tail levels), launch the
    // dedicated direct kernel -- no shared zero/sync/merge, and zero register
    // impact on the regular hot kernel. Bit-identical (order-invariant integer
    // atomics; the lattice quant fingerprints pin it). Skipped during graph
    // capture (the captured body must be the general kernel).
    const data_size_t small_leaf_rows =
      static_cast<data_size_t>(FalcataPlan::Get().quant_small_leaf_rows);
    if (SmallLeafConstructEnabled() && small_leaf_rows > 0 &&
        hybrid_graph_capture_gstate_ == nullptr &&
        max_num_data_in_smaller_leaf <= small_leaf_rows) {
      const int8_t* feat_used =
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr;
      if (use_compact_view_) {
#define FALCATA_LAUNCH_SMALL_LEAF_QUANT(PACKT, DATA, COLOFF, PARTOFF, BYTEOFF, FUSED) \
        CUDAConstructDiscretizedHistogramDenseSmallLeafBatchedKernel<BIN_TYPE, PACKT><<<grid_dim, block_dim, 0, cuda_stream_>>>( \
          pair_descs, \
          reinterpret_cast<const int32_t*>(cuda_gradients_), \
          DATA, COLOFF, cuda_row_data_->cuda_partition_hist_offsets(), PARTOFF, BYTEOFF, \
          num_data_, FUSED, \
          static_cast<data_size_t>(min_data_in_leaf_), min_sum_hessian_in_leaf_)
        if (compact_is_4bit_) {
          switch (compact_codec_) {
            case PackCodecId::kBit3x32:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackBit3x32, reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
            case PackCodecId::kRadix5x32:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRadix5x32, reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
            case PackCodecId::kRadix6x32:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRadix6x32, reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
            case PackCodecId::kRadix7x32:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRadix7x32, reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
            default:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackNibble4, reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
          }
        } else {
          FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRaw8, reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), nullptr, nullptr);
        }
      } else if (cuda_row_data_->is_4bit_packed()) {
        FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackNibble4, RowMajorBin<BIN_TYPE>(), cuda_row_data_->cuda_column_hist_offsets(), cuda_row_data_->cuda_feature_partition_column_index_offsets(), cuda_row_data_->cuda_packed_partition_byte_offsets(), feat_used);
      } else {
        FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRaw8, RowMajorBin<BIN_TYPE>(), cuda_row_data_->cuda_column_hist_offsets(), cuda_row_data_->cuda_feature_partition_column_index_offsets(), nullptr, feat_used);
      }
#undef FALCATA_LAUNCH_SMALL_LEAF_QUANT
      return;
    }
    // classic (two-sync) flow: exact host-known level sizes and descriptor bit
    // widths (level_smaller_num_data == nullptr, gstate == nullptr). Inside the
    // graph loop both are derived on-device (see the kernel comment); the
    // level-dim-y precompute kernel is never used here (the graph body always
    // takes the inline formula path, captured with num_pairs == 1).
    //
    // compact quant view (FALCATA_CONSTRUCT_COMPACT_QUANT): feed the SAME
    // discretized kernel the per-tree compact bin matrix + compact metadata so
    // only the sampled columns are gathered/accumulated. The compact matrix
    // holds only used columns, so no per-column feature mask (is_feature_used /
    // bin_used) is needed; the compact per-column absolute hist offsets and the
    // shared partition hist offsets preserve every used bin's global position ->
    // bit-identical histograms (integer atomics are order-invariant).
    if (use_compact_view_) {
      // JIT live fast path (cuda_plan=auto,construct_jit:on): a validated shape-
      // specialized construct_jit_batched replaces the AOT kernel for BOTH the
      // 8-bit and 4-bit-packed compact-quant shapes. Declines (default OFF /
      // unavailable / unvalidated / graph capture / speculative flow) -> AOT.
      if (compact_codec_ == PackCodecId::kNibble4 &&
          TryLaunchConstructJITBatchedCompactQuant(
              grid_dim, block_dim, pair_descs, level_smaller_num_data,
              static_cast<int>(SHARED_HIST_SIZE), sizeof(BIN_TYPE))) {
        // launched the JIT kernel (nibble layout only; codec views use AOT)
        DiagConstructKernel("jit");
      } else if (compact_is_4bit_ && compact_codec_ == PackCodecId::kNibble4 && FalcataPlan::Get().pair_hist &&
                 compact_pair_joint_max_ > 0 &&
                 compact_pair_joint_max_ * sizeof(int32_t) <= 48 * 1024 &&
                 level_smaller_num_data == nullptr && hybrid_graph_capture_gstate_ == nullptr &&
                 (std::max(1, max_num_compact_cols_per_partition_) + 1) / 2 * static_cast<int>(block_dim.y) <=
                   PairJointMaxThreadsPerBlock()) {
        // cuda_plan key pair_hist: one thread per packed byte (two compact columns), same grid and block rows
        DiagConstructKernel("pair_hist");
        const int cc = std::max(1, max_num_compact_cols_per_partition_);
        // cuda_plan key pair_hist_rows: on the row-interleaved view one block covers whole rows (every partition's
        // bytes), when all joint tables fit 48 KB together and the row fits the block
        const bool whole_rows = FalcataPlan::Get().pair_hist_rows && compact_row_interleave_ &&
            compact_pair_joint_total_ > 0 && compact_pair_joint_total_ * sizeof(int32_t) <= 48 * 1024 &&
            compact_row_bytes_ > 0 &&
            compact_row_bytes_ * static_cast<int>(block_dim.y) <= std::min(1024, PairJointMaxThreadsPerBlock());
        // cuda_plan key pair_block_rows: a whole-row block takes the row count (from the generic per-partition
        // block's up to the largest launchable one) that keeps the most warps resident per SM, by the kernel's
        // register count, instead of the generic block's rows; ties keep the fewer rows. The grid y is re-derived
        // for that block by the same formula, so the packed-cell rows-per-block guard holds for it.
        int pair_y = static_cast<int>(block_dim.y);
        int pair_grid_y = static_cast<int>(grid_dim.y);
        if (whole_rows && FalcataPlan::Get().pair_block_rows) {
          const int max_threads = std::min(1024, PairJointMaxThreadsPerBlock());
          int y = pair_y;
          int best_warps = PairJointResidentWarps(compact_row_bytes_ * pair_y);
          for (int c = pair_y + 1; compact_row_bytes_ * c <= max_threads &&
               HybridQuantConstructBlockDimY(c, use_quantized_grad_ ? num_grad_quant_bins_ : 0) == c; ++c) {
            const int warps = PairJointResidentWarps(compact_row_bytes_ * c);
            if (warps > best_warps) {
              best_warps = warps;
              y = c;
            }
          }
          if (y != pair_y) {
            pair_y = y;
            pair_grid_y = HybridBatchedConstructGridDimYQuant(
              max_num_data_in_smaller_leaf, num_pairs, pair_y, min_grid_dim_y_,
              BatchConstructMinRowsPerThread(), BatchConstructSaturationFloor(),
              use_quantized_grad_ ? num_grad_quant_bins_ : 0);
          }
        }
        // cuda_plan key pair_capped_rows: the register-capped build at the height that keeps strictly more warps
        // resident (shorter blocks, two or three per SM); the grid y is re-derived for that height by the same
        // formula and packed-cell guard
        const size_t pair_smem_bytes =
          static_cast<size_t>(whole_rows ? compact_pair_joint_total_ : compact_pair_joint_max_) * sizeof(int32_t);
        int capped_y = whole_rows && FalcataPlan::Get().pair_capped_rows ?
          PairJointCappedRows(compact_row_bytes_, pair_smem_bytes, pair_y, use_quantized_grad_ ? num_grad_quant_bins_ : 0) : 0;
        if (capped_y > 0) {
          const int capped_grid_y = HybridBatchedConstructGridDimYQuant(
            max_num_data_in_smaller_leaf, num_pairs, capped_y, min_grid_dim_y_,
            BatchConstructMinRowsPerThread(), BatchConstructSaturationFloor(),
            use_quantized_grad_ ? num_grad_quant_bins_ : 0);
          // shorter blocks need more of them: keep the default shape if the grid would pass CUDA's y limit
          if (capped_grid_y <= 65535) {
            pair_y = capped_y;
            pair_grid_y = capped_grid_y;
          } else {
            capped_y = 0;
          }
        }
        const dim3 pair_grid_dim(whole_rows ? 1 : grid_dim.x, pair_grid_y, grid_dim.z);
        const dim3 pair_block_dim(whole_rows ? compact_row_bytes_ : (cc + 1) / 2, pair_y);
        // level_row_blocks: every leaf at the largest leaf's rows per thread (within the grid's overflow guard)
        const int64_t pair_dim_y = static_cast<int64_t>(pair_grid_y) * pair_y;
        const data_size_t pair_min_rows_per_thread = FalcataPlan::Get().level_row_blocks && num_pairs > 1 ?
          static_cast<data_size_t>((static_cast<int64_t>(max_num_data_in_smaller_leaf) + pair_dim_y - 1) / pair_dim_y) : 0;
        // all_rows_direct: the level's one leaf holds every row (the root without bagging)
        const bool all_rows = FalcataPlan::Get().all_rows_direct && num_pairs == 1 &&
          max_num_data_in_smaller_leaf == num_data_;
        // cuda_plan key pair_block_map: launch only the blocks with rows. Each pair's count follows the kernel's own
        // per-pair sizing (per_pair_rows' formula at the pair's size, level_row_blocks' rows-per-thread floor, then
        // the blocks whose first row is inside the leaf), from the host copy of the leaf counts the kernel reads
        PairJointBlockPrefix block_prefix{};  // num_pairs 0: the plain grid
        dim3 launch_grid_dim = pair_grid_dim;
        if (FalcataPlan::Get().pair_block_map && num_pairs > 1 &&
            num_pairs <= PairJointBlockPrefix::kMaxPairs && level_host_pair_descs_ != nullptr &&
            level_host_num_pairs_ == num_pairs) {
          // the kernel arguments below: per_pair_rows' grid floor and its rows-per-thread floor
          const int per_pair_min_grid = FalcataPlan::Get().per_pair_rows ? min_grid_dim_y_ : 0;
          const int per_pair_min_rows = BatchConstructMinRowsPerThread() * (whole_rows ? static_cast<int>(grid_dim.x) : 1);
          int64_t total = 0;
          for (int p = 0; p < num_pairs; ++p) {
            block_prefix.start[p] = static_cast<uint16_t>(std::min<int64_t>(total, 65535));
            const CUDAHybridPairDescriptor& d = level_host_pair_descs_[p];
            const data_size_t n = d.num_data_in_smaller_leaf;
            if (!d.construct_valid || n <= 0) {
              continue;
            }
            int pgy = pair_grid_y;
            if (per_pair_min_grid > 0) {
              const int yy = HybridBatchedConstructGridDimYQuant(
                n, num_pairs, pair_y, per_pair_min_grid, per_pair_min_rows, BatchConstructSaturationFloor(),
                use_quantized_grad_ ? num_grad_quant_bins_ : 0);
              pgy = std::max(1, std::min(pgy, yy));
            }
            const data_size_t dim_y = pgy * pair_y;
            const data_size_t rows_per_thread = std::max((n + dim_y - 1) / dim_y, pair_min_rows_per_thread);
            const int64_t block_rows = static_cast<int64_t>(pair_y) * rows_per_thread;
            total += std::min(static_cast<int64_t>(pgy), (static_cast<int64_t>(n) + block_rows - 1) / block_rows);
          }
          block_prefix.start[num_pairs] = static_cast<uint16_t>(std::min<int64_t>(total, 65535));
          if (total > 0 && total <= 65535) {
            block_prefix.num_pairs = num_pairs;
            block_prefix.grid_y = pair_grid_y;
            launch_grid_dim = dim3(pair_grid_dim.x, static_cast<unsigned int>(total), 1);
          }
        }
#define FALCATA_LAUNCH_PAIR_JOINT(KERNEL) \
        KERNEL<<<launch_grid_dim, pair_block_dim, pair_smem_bytes, cuda_stream_>>>( \
          pair_descs, \
          reinterpret_cast<const int32_t*>(cuda_gradients_), \
          compact_data_uint8_t_.RawData(), \
          compact_column_hist_offsets_.RawData(), \
          num_compact_columns_, \
          cuda_row_data_->cuda_partition_hist_offsets(), \
          compact_feature_partition_column_index_offsets_.RawData(), \
          compact_packed_partition_byte_offsets_.RawData(), \
          num_data_, \
          static_cast<data_size_t>(min_data_in_leaf_), \
          min_sum_hessian_in_leaf_, \
          FalcataPlan::Get().per_pair_rows ? min_grid_dim_y_ : 0, \
          BatchConstructMinRowsPerThread() * (whole_rows ? static_cast<int>(grid_dim.x) : 1), \
          BatchConstructSaturationFloor(), \
          use_quantized_grad_ ? num_grad_quant_bins_ : 0, \
          whole_rows ? static_cast<int>(grid_dim.x) : 0, \
          whole_rows ? static_cast<uint32_t>(compact_pair_joint_total_) : 0u, \
          pair_min_rows_per_thread, \
          block_prefix)
        // a whole-row block does the work of grid_dim.x partition blocks and zeroes / flushes all their joint
        // tables, so its per_pair_rows rows-per-thread floor scales by the partitions it covers to keep the same
        // share of fixed per-block cost (a larger floor only lowers the formula, so the launched grid still bounds it)
        if (capped_y > 0) {
          const PairJointCappedBuild capped = PairJointCapped();
          const PairJointKernelFn kernel = all_rows ? capped.direct : capped.gather;
          FALCATA_LAUNCH_PAIR_JOINT(kernel);
          // a new launch shape: a rejected launch must not leave the histogram unbuilt silently
          CUDASUCCESS_OR_FATAL(cudaPeekAtLastError());
        } else if (all_rows) {
          FALCATA_LAUNCH_PAIR_JOINT(CUDAConstructDiscretizedHistogramPairJointBatchedKernel<true>);
        } else {
          FALCATA_LAUNCH_PAIR_JOINT(CUDAConstructDiscretizedHistogramPairJointBatchedKernel<false>);
        }
#undef FALCATA_LAUNCH_PAIR_JOINT
      } else if (compact_is_4bit_) {
        DiagConstructKernel(FalcataPlan::Get().row_batch ? "row_batch" : "unbatched");
#define FALCATA_LAUNCH_BATCHED_COMPACT_QUANT(PACKT) \
        CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, SHARED_HIST_SIZE, PACKT><<<grid_dim, block_dim, 0, cuda_stream_>>>( \
          pair_descs, \
          reinterpret_cast<const int32_t*>(cuda_gradients_), \
          reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()), \
          compact_column_hist_offsets_.RawData(), \
          cuda_row_data_->cuda_partition_hist_offsets(), \
          compact_feature_partition_column_index_offsets_.RawData(), \
          compact_packed_partition_byte_offsets_.RawData(), \
          num_data_, \
          nullptr, \
          nullptr, \
          static_cast<data_size_t>(min_data_in_leaf_), \
          min_sum_hessian_in_leaf_, \
          level_smaller_num_data, \
          hybrid_graph_capture_gstate_, FalcataPlan::Get().row_batch)
        switch (compact_codec_) {
          case PackCodecId::kBit3x32:
            FALCATA_LAUNCH_BATCHED_COMPACT_QUANT(PackBit3x32);
            break;
          case PackCodecId::kRadix5x32:
            FALCATA_LAUNCH_BATCHED_COMPACT_QUANT(PackRadix5x32);
            break;
          case PackCodecId::kRadix6x32:
            FALCATA_LAUNCH_BATCHED_COMPACT_QUANT(PackRadix6x32);
            break;
          case PackCodecId::kRadix7x32:
            FALCATA_LAUNCH_BATCHED_COMPACT_QUANT(PackRadix7x32);
            break;
          default:
            FALCATA_LAUNCH_BATCHED_COMPACT_QUANT(PackNibble4);
            break;
        }
#undef FALCATA_LAUNCH_BATCHED_COMPACT_QUANT
      } else {
        DiagConstructKernel(FalcataPlan::Get().row_batch ? "row_batch" : "unbatched");
        CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          pair_descs,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          reinterpret_cast<const BIN_TYPE*>(compact_data_uint8_t_.RawData()),
          compact_column_hist_offsets_.RawData(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          compact_feature_partition_column_index_offsets_.RawData(),
          nullptr,
          num_data_,
          nullptr,
          nullptr,
          static_cast<data_size_t>(min_data_in_leaf_),
          min_sum_hessian_in_leaf_,
          level_smaller_num_data,
          hybrid_graph_capture_gstate_, FalcataPlan::Get().row_batch);
      }
    } else if (cuda_row_data_->is_4bit_packed()) {
      if (TryLaunchConstructJITBatchedRowDataQuant(
              grid_dim, block_dim, pair_descs, level_smaller_num_data,
              static_cast<int>(SHARED_HIST_SIZE), sizeof(BIN_TYPE), true)) {
        // JIT launched (validated module; mask-free shape)
        DiagConstructKernel("jit");
      } else {
        DiagConstructKernel(FalcataPlan::Get().row_batch ? "row_batch" : "unbatched");
        CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, SHARED_HIST_SIZE, PackNibble4><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          pair_descs,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          RowMajorBin<BIN_TYPE>(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          cuda_row_data_->cuda_packed_partition_byte_offsets(),
          num_data_,
          any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
          static_cast<data_size_t>(min_data_in_leaf_),
          min_sum_hessian_in_leaf_,
          level_smaller_num_data,
          hybrid_graph_capture_gstate_, FalcataPlan::Get().row_batch);
      }
    } else {
      if (TryLaunchConstructJITBatchedRowDataQuant(
              grid_dim, block_dim, pair_descs, level_smaller_num_data,
              static_cast<int>(SHARED_HIST_SIZE), sizeof(BIN_TYPE), false)) {
        DiagConstructKernel("jit");
        return;  // JIT launched (validated module; mask-free shape)
      }
      DiagConstructKernel(FalcataPlan::Get().row_batch ? "row_batch" : "unbatched");
      CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
        pair_descs,
        reinterpret_cast<const int32_t*>(cuda_gradients_),
        RowMajorBin<BIN_TYPE>(),
        cuda_row_data_->cuda_column_hist_offsets(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        cuda_row_data_->cuda_feature_partition_column_index_offsets(),
        nullptr,
        num_data_,
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
        static_cast<data_size_t>(min_data_in_leaf_),
        min_sum_hessian_in_leaf_,
        level_smaller_num_data,
        hybrid_graph_capture_gstate_, FalcataPlan::Get().row_batch);
    }
  } else if (det_batched) {
    // Deterministic float construct for the level batch: the same fixed-order
    // double-slot math as the per-leaf det arm, one construct + one merge
    // launch covering every pair (see DetDenseBatchedEligible for the shape
    // gates and why graph capture keeps the atomic kernel below).
    LaunchConstructHistogramDenseBatchedDeterministic<BIN_TYPE>(
      pair_descs, num_pairs, max_num_data_in_smaller_leaf, grid_dim_x, block_dim_x);
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
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
          hist_fp32_,
          hybrid_graph_capture_gstate_);
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
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
          hist_fp32_,
          hybrid_graph_capture_gstate_);
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
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
          hist_fp32_,
          hybrid_graph_capture_gstate_);
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
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
          hist_fp32_,
          hybrid_graph_capture_gstate_);
      }
    }
  } else if (cuda_row_data_->is_4bit_packed()) {
    CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, true, USE_GH2><<<grid_dim, block_dim, 0, cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
      RowMajorBin<BIN_TYPE>(),
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
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
      hist_fp32_,
      hybrid_graph_capture_gstate_);
  } else {
    CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, false, USE_GH2><<<grid_dim, block_dim, 0, cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
      RowMajorBin<BIN_TYPE>(),
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
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
      hist_fp32_,
      hybrid_graph_capture_gstate_);
  }
}

void CUDAHistogramConstructor::LaunchFixSubtractHistogramSmallLeafBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const CUDAHybridGraphLoopStateOpt gstate) {
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
    any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
    hist_fp32_,
    gstate);
}

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
template <typename BIN_TYPE>
void CUDAHistogramConstructor::CaptureHybridGraphDetConstructMergeInner(
    const CUDAHybridPairDescriptor* pair_descs,
    const CUDAHybridGraphLoopState* gstate,
    const int block_dim_x,
    const int grid_dim_x,
    const int dy,
    std::vector<cudaGraphNode_t>* nodes,
    std::vector<int>* roles,
    std::vector<int>* role_static_x) {
  const int total_slot_rows = det_dense_total_slot_rows();
  const bool det_packed_4bit = cuda_row_data_->is_4bit_packed();
  auto det_kernel = det_packed_4bit ?
    &CUDAConstructHistogramDenseGMDeterministicBatchedKernel<BIN_TYPE, true> :
    &CUDAConstructHistogramDenseGMDeterministicBatchedKernel<BIN_TYPE, false>;
  // placeholder grids: the controller resizes both nodes per level (exact
  // tile extent for the construct, pow2 pair bucket for both)
  dim3 det_grid(static_cast<unsigned int>(grid_dim_x), 1, 1);
  dim3 det_block(static_cast<unsigned int>(block_dim_x), static_cast<unsigned int>(dy));
  det_kernel<<<det_grid, det_block, 0, cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      RowMajorBin<BIN_TYPE>(),
      cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      det_packed_4bit ? cuda_row_data_->cuda_packed_partition_byte_offsets() : nullptr,
      cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
      num_data_,
      static_cast<data_size_t>(min_data_in_leaf_),
      min_sum_hessian_in_leaf_,
      det_dense_slot_stride_,
      cuda_det_dense_slots_.RawData(),
      total_slot_rows,
      dy,
      gstate);
  if (!AppendCapturedNode(cuda_stream_, nodes)) return;
  roles->push_back(kHybridGraphNodeConstructDet);
  role_static_x->push_back(dy);  // the controller's tile formula needs dy
  const int merge_threads = 256;
  const int merge_blocks_x =
    (static_cast<int>(det_dense_slot_stride_) + merge_threads - 1) / merge_threads;
  dim3 merge_grid(static_cast<unsigned int>(merge_blocks_x),
                  static_cast<unsigned int>(grid_dim_x), 1);
  MergeDeterministicDenseHistogramBatchedKernel<<<merge_grid, merge_threads, 0, cuda_stream_>>>(
    pair_descs,
    cuda_det_dense_slots_.RawData(),
    cuda_row_data_->cuda_partition_hist_offsets(),
    det_dense_slot_stride_,
    total_slot_rows,
    dy,
    /*host_num_pairs=*/1,
    static_cast<data_size_t>(min_data_in_leaf_),
    min_sum_hessian_in_leaf_,
    gstate);
  if (!AppendCapturedNode(cuda_stream_, nodes)) return;
  roles->push_back(kHybridGraphNodeConstructDetMerge);
  role_static_x->push_back(merge_blocks_x);
}

void CUDAHistogramConstructor::CaptureHybridGraphDetConstructMerge(
    const CUDAHybridPairDescriptor* pair_descs,
    const CUDAHybridGraphLoopState* gstate,
    const int max_level_pairs,
    std::vector<cudaGraphNode_t>* nodes,
    std::vector<int>* roles,
    std::vector<int>* role_static_x) {
  int grid_dim_x = 0, grid_dim_y = 0, block_dim_x = 0, block_dim_y = 0;
  CalcConstructHistogramBatchedKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, 1, 1);
  // dy under the 1024-thread block budget (mirrors the host launcher), then
  // clamped to the widest level's slab carve so the frozen block's dy rows
  // always fit one pair region (DetDensePairTiles assumes dy <= pair_rows)
  int dy = std::max(1, std::min(det_dense_dy_, 1024 / std::max(1, block_dim_x)));
  dy = std::min(dy, std::max(1, det_dense_total_slot_rows() / std::max(1, max_level_pairs)));
  if (cuda_row_data_->bit_type() == 8) {
    CaptureHybridGraphDetConstructMergeInner<uint8_t>(
      pair_descs, gstate, block_dim_x, grid_dim_x, dy, nodes, roles, role_static_x);
  } else if (cuda_row_data_->bit_type() == 16) {
    CaptureHybridGraphDetConstructMergeInner<uint16_t>(
      pair_descs, gstate, block_dim_x, grid_dim_x, dy, nodes, roles, role_static_x);
  } else {
    CaptureHybridGraphDetConstructMergeInner<uint32_t>(
      pair_descs, gstate, block_dim_x, grid_dim_x, dy, nodes, roles, role_static_x);
  }
}

void CUDAHistogramConstructor::CaptureHybridGraphSearchKernels(
    const CUDAHybridPairDescriptor* pair_descs,
    const data_size_t* level_smaller_num_data,
    const CUDAHybridGraphLoopState* gstate,
    const int det_max_level_pairs,
    std::vector<cudaGraphNode_t>* nodes,
    std::vector<int>* roles,
    std::vector<int>* role_static_x) {
  // graphs L1 body capture: construct + fix/subtract with PLACEHOLDER grids;
  // the controller resizes them per level. num_pairs == 1 freezes the construct
  // kernel's inline row-grouping path (level sizes read on-device, extent from
  // the loop state's live pair count), which computes bit-identical grouping to
  // the host's per-level sizing (quantized: including the packed int32
  // shared-histogram overflow guard).
  // graphs A2: the captured construct kernel reads the live pair count from
  // the loop state (its frozen grid is only a pow2 upper bound)
  if (det_max_level_pairs > 0) {
    // deterministic runs: the level's construct is the det construct + merge
    // node pair; every per-level extent the host launcher would compute is
    // derived on device (DetDensePairRows/DetDensePairTiles) or resized by
    // the controller, so determinism survives the frozen graph body
    CaptureHybridGraphDetConstructMerge(pair_descs, gstate, det_max_level_pairs,
                                        nodes, roles, role_static_x);
  } else {
    hybrid_graph_capture_gstate_ = gstate;
    LaunchConstructHistogramBatchedKernel(pair_descs, 1, 1, level_smaller_num_data);
    hybrid_graph_capture_gstate_ = nullptr;
    if (!AppendCapturedNode(cuda_stream_, nodes)) return;
    roles->push_back(kHybridGraphNodeConstruct);
    role_static_x->push_back(0);
  }
  if (use_quantized_grad_) {
    // mirror of LaunchSubtractHistogramBatchedKernel's quantized branch, one
    // collected node per launch; the copy node is always captured (per-pair
    // device-derived bit widths gate it inside the kernel)
    const int num_subtract_threads = num_total_bin_;
    const int num_subtract_blocks =
      (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    if (need_fix_histogram_features_.size() > 0) {
      dim3 fix_grid(static_cast<unsigned int>(need_fix_histogram_features_.size()), 1);
      FixHistogramDiscretizedBatchedKernel<<<fix_grid, FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
        cuda_feature_num_bins_.RawData(),
        cuda_feature_hist_offsets_.RawData(),
        cuda_feature_most_freq_bins_.RawData(),
        cuda_need_fix_histogram_features_.RawData(),
        cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
        pair_descs,
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
        gstate);
      if (!AppendCapturedNode(cuda_stream_, nodes)) return;
      roles->push_back(kHybridGraphNodeSearchPairY);
      role_static_x->push_back(static_cast<int>(need_fix_histogram_features_.size()));
    }
    dim3 subtract_grid(num_subtract_blocks, 1);
    SubtractHistogramDiscretizedBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      pair_descs,
      hist_buffer_for_num_bit_change_.RawData(),
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
      gstate);
    if (!AppendCapturedNode(cuda_stream_, nodes)) return;
    roles->push_back(kHybridGraphNodeSearchPairY);
    role_static_x->push_back(num_subtract_blocks);
    CopyChangedNumBitHistogramBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      pair_descs,
      hist_buffer_for_num_bit_change_.RawData(),
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
      gstate);
    if (!AppendCapturedNode(cuda_stream_, nodes)) return;
    roles->push_back(kHybridGraphNodeSearchPairY);
    role_static_x->push_back(num_subtract_blocks);
    CUDASUCCESS_OR_FATAL(cudaEventRecord(subtract_done_events_[0], cuda_stream_));
    return;
  }
  if (SmallLeafConstructEnabled()) {
    LaunchFixSubtractHistogramSmallLeafBatchedKernel(pair_descs, 1, gstate);
    if (!AppendCapturedNode(cuda_stream_, nodes)) return;
    roles->push_back(kHybridGraphNodeSearchPairY);
    const int num_subtract_threads = 2 * num_total_bin_;
    const int num_subtract_blocks =
      (num_subtract_threads + FIX_HISTOGRAM_BLOCK_SIZE - 1) / FIX_HISTOGRAM_BLOCK_SIZE;
    role_static_x->push_back(num_subtract_blocks + static_cast<int>(need_fix_histogram_features_.size()));
  } else {
    // mirror of LaunchSubtractHistogramBatchedKernel's non-quantized branch,
    // one collected node per launch
    const int num_subtract_threads = 2 * num_total_bin_;
    const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    if (need_fix_histogram_features_.size() > 0) {
      dim3 fix_grid(static_cast<unsigned int>(need_fix_histogram_features_.size()), 1);
      FixHistogramBatchedKernel<<<fix_grid, FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
        cuda_feature_num_bins_.RawData(),
        cuda_feature_hist_offsets_.RawData(),
        cuda_feature_most_freq_bins_.RawData(),
        cuda_need_fix_histogram_features_.RawData(),
        cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
        pair_descs,
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
        hist_fp32_,
        gstate);
      if (!AppendCapturedNode(cuda_stream_, nodes)) return;
      roles->push_back(kHybridGraphNodeSearchPairY);
      role_static_x->push_back(static_cast<int>(need_fix_histogram_features_.size()));
    }
    dim3 subtract_grid(num_subtract_blocks, 1);
    SubtractHistogramBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      pair_descs,
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
      hist_fp32_,
      gstate);
    if (!AppendCapturedNode(cuda_stream_, nodes)) return;
    roles->push_back(kHybridGraphNodeSearchPairY);
    role_static_x->push_back(num_subtract_blocks);
  }
  CUDASUCCESS_OR_FATAL(cudaEventRecord(subtract_done_events_[0], cuda_stream_));
}

void CUDAHistogramConstructor::HybridGraphConstructDims(int* grid_x, int* block_dim_y) const {
  int grid_dim_x = 0, grid_dim_y = 0, block_dim_x = 0, block_dim_y_local = 0;
  const_cast<CUDAHistogramConstructor*>(this)->CalcConstructHistogramBatchedKernelDim(
    &grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y_local, 1, 1);
  if (use_compact_view_) {
    // mirror of LaunchConstructHistogramBatchedKernelInner0's compact override
    // (per-tree shape: the graph key includes it, one instance per shape)
    {
      const int cc = std::max(1, max_num_compact_cols_per_partition_);
      block_dim_x = cc > NUM_THREADS_PER_BLOCK ? (cc + 1) / 2 : cc;
    }
    block_dim_y_local = std::max(1, NUM_THREADS_PER_BLOCK / block_dim_x);
  }
  *grid_x = grid_dim_x;
  *block_dim_y = block_dim_y_local;
}
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

void CUDAHistogramConstructor::LaunchSubtractHistogramBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const bool any_pair_needs_bit_change_copy,
  const CUDAHybridGraphLoopStateOpt gstate) {
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
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
        hist_fp32_,
        nullptr);
    }
    dim3 subtract_grid(num_subtract_blocks, num_pairs);
    SubtractHistogramBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      pair_descs,
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
      hist_fp32_,
      nullptr);
  } else {
    const int num_subtract_threads = num_total_bin_;
    const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    dim3 subtract_grid(num_subtract_blocks, num_pairs);
    bool fused_fix_subtract = false;
    if (gstate == nullptr && any_feature_unused_bytree_ && used_feature_info_ok_ &&
        FalcataPlan::Get().fix_subtract_fused) {
      // cuda_plan key fix_subtract_fused: fix + subtract of the sampled features, one block per pair
      const int threads = std::max(32, (num_used_feature_info_ + 31) / 32 * 32);  // <= 512 (host gate)
      FixSubtractUsedFeaturesBatchedKernel<<<num_pairs, threads, 0, cuda_stream_>>>(
        cuda_used_feature_info_.RawDataReadOnly(),
        num_used_feature_info_,
        num_total_bin_,
        pair_descs,
        hist_buffer_for_num_bit_change_.RawData());
      fused_fix_subtract = true;  // the kernel also did CopyChangedNumBitHistogramBatchedKernel's copies
    } else {
      if (need_fix_histogram_features_.size() > 0) {
        dim3 fix_grid(static_cast<unsigned int>(need_fix_histogram_features_.size()), num_pairs);
        FixHistogramDiscretizedBatchedKernel<<<fix_grid, FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
          cuda_feature_num_bins_.RawData(),
          cuda_feature_hist_offsets_.RawData(),
          cuda_feature_most_freq_bins_.RawData(),
          cuda_need_fix_histogram_features_.RawData(),
          cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
          pair_descs,
          any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
          gstate);
      }
      SubtractHistogramDiscretizedBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        pair_descs,
        hist_buffer_for_num_bit_change_.RawData(),
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
        gstate);
    }
    // graph capture always includes the copy node (the device-derived per-pair
    // bit widths gate it); the host path launches it only on bit-change levels
    if ((any_pair_needs_bit_change_copy && !fused_fix_subtract) || gstate != nullptr) {
      CopyChangedNumBitHistogramBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        pair_descs,
        hist_buffer_for_num_bit_change_.RawData(),
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
        gstate);
    }
  }
}

// ============================================================================
// NVRTC construct-JIT one-time self-test. Proves the JIT pipeline end-to-end:
// NVRTC compile -> cuModuleLoadData -> cuLaunchKernel -> bit-identical histogram
// vs a host reference, on a tiny synthetic single-partition low-bin shape (the
// numerai regime). No-op unless cuda_plan=auto,construct_jit:on. It never touches the
// trained model -- it only validates that the specialized kernel this build would
// JIT reproduces the reference bins exactly (integer atomics order-invariant), so
// the JIT can be trusted as a perf-only fast path.
// ============================================================================
bool CUDAHistogramConstructor::RunConstructJITSelfTest() {
  if (construct_jit_selftest_done_) return true;
  construct_jit_selftest_done_ = true;
  if (!CUDAConstructJIT::Enabled() || !CUDAConstructJIT::Available()) return false;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  // Validate the LIVE batched kernel (the one the dispatch launches) for each
  // packing separately -- the 4-bit and 8-bit bin reads are different code.
  bool any = false;
  any |= RunConstructJITSelfTestShape(false, prop.major, prop.minor);
  any |= RunConstructJITSelfTestShape(true, prop.major, prop.minor);
  return any;
}

// One self-test shape: compile + launch construct_jit_batched on a synthetic
// single-partition single-pair histogram and confirm bit-identity vs a host
// reference. Arms the live path for this packing on success.
bool CUDAHistogramConstructor::RunConstructJITSelfTestShape(bool is_4bit, int sm_major, int sm_minor) {
  const int slot = is_4bit ? 1 : 0;
  const int kCols = 8;
  const int kBins = is_4bit ? 4 : 6;  // 4-bit values live in [0,15]; keep < 16
  const int kRows = 1024;
  ConstructJITShapeKey key;
  key.bins = kBins;
  key.num_partitions = 1;
  key.cols_per_partition = kCols;
  key.shared_hist_size = static_cast<int>(SP_SHARED_HIST_SIZE);
  key.use_16bit_hist = 1;
  key.is_4bit = is_4bit ? 1 : 0;
  key.sm_major = sm_major;
  key.sm_minor = sm_minor;

  double compile_ms = 0.0;
  void* fn = construct_jit_.GetOrCompile(key, &compile_ms);
  if (fn == nullptr) return false;
  void* fn_batched = construct_jit_.GetBatchedFunc(key);
  if (fn_batched == nullptr) return false;
  CUfunction func = reinterpret_cast<CUfunction>(fn_batched);

  // 8-bit: row byte layout data[row*kCols + col]; 4-bit: packed nibbles,
  // row byte layout data[row*packed_width + (col>>1)] nibble (col&1).
  const int packed_width = (kCols + 1) >> 1;
  const int row_width = is_4bit ? packed_width : kCols;
  std::vector<uint8_t> h_data(static_cast<size_t>(kRows) * row_width, 0);
  std::vector<int32_t> h_gh(kRows);
  std::vector<data_size_t> h_indices(kRows);
  std::vector<uint32_t> h_col_off(kCols);
  std::vector<uint32_t> h_part_hist_off(2);
  std::vector<int> h_part_col_off(2);
  std::vector<int> h_packed_off(2);
  for (int c = 0; c < kCols; ++c) h_col_off[c] = static_cast<uint32_t>(c * kBins);
  h_part_hist_off[0] = 0;
  h_part_hist_off[1] = static_cast<uint32_t>(kCols * kBins);
  h_part_col_off[0] = 0;
  h_part_col_off[1] = kCols;
  h_packed_off[0] = 0;
  h_packed_off[1] = packed_width;
  std::vector<uint8_t> h_bins(static_cast<size_t>(kRows) * kCols);
  for (int r = 0; r < kRows; ++r) {
    h_indices[r] = r;
    const int16_t g = static_cast<int16_t>((r % 7) - 3);
    const int16_t h = static_cast<int16_t>(1);
    h_gh[r] = (static_cast<int32_t>(g) << 16) | (static_cast<int32_t>(h) & 0xffff);
    for (int c = 0; c < kCols; ++c) {
      const uint8_t bin = static_cast<uint8_t>((r + c) % kBins);
      h_bins[static_cast<size_t>(r) * kCols + c] = bin;
      if (is_4bit) {
        uint8_t& byte = h_data[static_cast<size_t>(r) * row_width + (c >> 1)];
        byte = static_cast<uint8_t>((byte & ~(0xf << ((c & 1) << 2))) | (bin << ((c & 1) << 2)));
      } else {
        h_data[static_cast<size_t>(r) * row_width + c] = bin;
      }
    }
  }

  const int total_bins = kCols * kBins;
  std::vector<int32_t> ref(total_bins, 0);
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      ref[c * kBins + h_bins[static_cast<size_t>(r) * kCols + c]] += h_gh[r];
    }
  }

  CUDAVector<uint8_t> d_data(h_data.size());
  CUDAVector<int32_t> d_gh(kRows);
  CUDAVector<data_size_t> d_indices(kRows);
  CUDAVector<uint32_t> d_col_off(kCols);
  CUDAVector<uint32_t> d_part_hist_off(2);
  CUDAVector<int> d_part_col_off(2);
  CUDAVector<int> d_packed_off(2);
  CUDAVector<int32_t> d_hist(total_bins);
  CopyFromHostToCUDADevice<uint8_t>(d_data.RawData(), h_data.data(), h_data.size(), __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int32_t>(d_gh.RawData(), h_gh.data(), kRows, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<data_size_t>(d_indices.RawData(), h_indices.data(), kRows, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<uint32_t>(d_col_off.RawData(), h_col_off.data(), kCols, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<uint32_t>(d_part_hist_off.RawData(), h_part_hist_off.data(), 2, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(d_part_col_off.RawData(), h_part_col_off.data(), 2, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(d_packed_off.RawData(), h_packed_off.data(), 2, __FILE__, __LINE__);
  cudaMemset(d_hist.RawData(), 0, total_bins * sizeof(int32_t));

  CUDALeafSplitsStruct h_smaller;
  memset(&h_smaller, 0, sizeof(h_smaller));
  h_smaller.leaf_index = 0;
  h_smaller.num_data_in_leaf = kRows;
  h_smaller.sum_of_hessians = static_cast<double>(kRows);
  h_smaller.data_indices_in_leaf = d_indices.RawData();
  h_smaller.hist_in_leaf = reinterpret_cast<hist_t*>(d_hist.RawData());
  CUDALeafSplitsStruct h_larger;
  memset(&h_larger, 0, sizeof(h_larger));
  h_larger.leaf_index = -1;  // no larger sibling
  CUDAVector<CUDALeafSplitsStruct> d_structs(2);
  CopyFromHostToCUDADevice<CUDALeafSplitsStruct>(d_structs.RawData(), &h_smaller, 1, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<CUDALeafSplitsStruct>(d_structs.RawData() + 1, &h_larger, 1, __FILE__, __LINE__);

  CUDAHybridPairDescriptor h_desc;
  memset(&h_desc, 0, sizeof(h_desc));
  h_desc.smaller_struct = d_structs.RawData();
  h_desc.larger_struct = d_structs.RawData() + 1;
  h_desc.construct_valid = 1;
  h_desc.smaller_num_bits = 16;  // 16-bit hist path
  CUDAVector<CUDAHybridPairDescriptor> d_desc(1);
  CopyFromHostToCUDADevice<CUDAHybridPairDescriptor>(d_desc.RawData(), &h_desc, 1, __FILE__, __LINE__);

  const int block_x = kCols;
  const int block_y = NUM_THREADS_PER_BLOCK / block_x;
  const int grid_y = 4;
  const CUDAHybridPairDescriptor* desc_ptr = d_desc.RawData();
  const int32_t* gh_ptr = d_gh.RawData();
  const uint8_t* data_ptr = d_data.RawData();
  const uint32_t* col_off_ptr = d_col_off.RawData();
  const uint32_t* part_hist_ptr = d_part_hist_off.RawData();
  const int* part_col_ptr = d_part_col_off.RawData();
  const int* packed_off_ptr = d_packed_off.RawData();
  data_size_t num_data_arg = kRows;
  data_size_t min_data_arg = 0;
  double min_hess_arg = 0.0;
  void* args[] = {&desc_ptr, &gh_ptr, &data_ptr, &col_off_ptr, &part_hist_ptr,
                  &part_col_ptr, &packed_off_ptr, &num_data_arg, &min_data_arg, &min_hess_arg};
  const CUresult lr = cuLaunchKernel(func, key.num_partitions, grid_y, 1,
                                     block_x, block_y, 1, 0, nullptr, args, nullptr);
  if (lr != CUDA_SUCCESS) {
    Log::Warning("CUDAConstructJIT self-test (%s): cuLaunchKernel failed (%d)",
                 is_4bit ? "4bit" : "8bit", static_cast<int>(lr));
    construct_jit_.SetValidated(key, false);
    return false;
  }
  cudaDeviceSynchronize();

  std::vector<int32_t> got(total_bins, 0);
  CopyFromCUDADeviceToHost<int32_t>(got.data(), d_hist.RawData(), total_bins, __FILE__, __LINE__);
  bool ok = true;
  for (int i = 0; i < total_bins && ok; ++i) {
    if (got[i] != ref[i]) ok = false;
  }
  construct_jit_.SetValidated(key, ok);
  if (ok) {
    construct_jit_live_key_[slot] = key;
    construct_jit_live_ready_[slot] = (construct_jit_.GetBatchedIfValidated(key) != nullptr);
    Log::Info("CUDAConstructJIT self-test PASSED %s (compile %.1f ms): batched kernel "
              "bit-identical to reference; live path %s", is_4bit ? "4bit" : "8bit", compile_ms,
              construct_jit_live_ready_[slot] ? "ARMED" : "unavailable");
  } else {
    Log::Warning("CUDAConstructJIT self-test FAILED %s bit-identity; that packing stays AOT",
                 is_4bit ? "4bit" : "8bit");
  }
  return ok;
}


// Live JIT batched construct for the NON-COMPACT quant dense path (the
// covtype/year/higgs-class shapes: no feature sampling, so the per-tree
// feature/bin masks are null and the mask-free JIT body is exact). Reuses the
// module the self-test validated -- the batched body bakes only
// (SHARED_HIST_SIZE, IS_4BIT, sm), none of the dataset shape.
bool CUDAHistogramConstructor::TryLaunchConstructJITBatchedRowDataQuant(
    const dim3& grid_dim, const dim3& block_dim,
    const CUDAHybridPairDescriptor* pair_descs,
    const data_size_t* level_smaller_num_data,
    int shared_hist_size, size_t bin_type_bytes, bool is_4bit) {
  const int slot = is_4bit ? 1 : 0;
  if (!construct_jit_live_ready_[slot]) return false;
  if (hybrid_graph_capture_gstate_ != nullptr) return false;
  if (level_smaller_num_data != nullptr) return false;
  if (bin_type_bytes != 1) return false;
  if (any_feature_unused_bytree_) return false;  // JIT body has no mask support
  // wide partitions map two columns per thread; the JIT body maps one
  if (cuda_row_data_->max_num_column_per_partition() > static_cast<int>(block_dim.x)) return false;
  if (shared_hist_size != construct_jit_live_key_[slot].shared_hist_size) return false;
  void* fn = construct_jit_.GetBatchedIfValidated(construct_jit_live_key_[slot]);
  if (fn == nullptr) return false;
  CUfunction func = reinterpret_cast<CUfunction>(fn);

  const int32_t* gh_ptr = reinterpret_cast<const int32_t*>(cuda_gradients_);
  const uint8_t* data_ptr = RowMajorBin<uint8_t>();
  const uint32_t* col_off_ptr = cuda_row_data_->cuda_column_hist_offsets();
  const uint32_t* part_hist_ptr = cuda_row_data_->cuda_partition_hist_offsets();
  const int* part_col_ptr = cuda_row_data_->cuda_feature_partition_column_index_offsets();
  const int* packed_off_ptr = is_4bit ?
      cuda_row_data_->cuda_packed_partition_byte_offsets() : nullptr;
  data_size_t num_data_arg = num_data_;
  data_size_t min_data_arg = static_cast<data_size_t>(min_data_in_leaf_);
  double min_hess_arg = min_sum_hessian_in_leaf_;
  void* args[] = {&pair_descs, &gh_ptr, &data_ptr, &col_off_ptr, &part_hist_ptr,
                  &part_col_ptr, &packed_off_ptr, &num_data_arg, &min_data_arg, &min_hess_arg};
  const CUresult lr = cuLaunchKernel(
      func, grid_dim.x, grid_dim.y, grid_dim.z,
      block_dim.x, block_dim.y, block_dim.z, 0, cuda_stream_, args, nullptr);
  if (lr != CUDA_SUCCESS) {
    Log::Warning("CUDAConstructJIT rowdata launch failed (%d); disabling live JIT, using AOT",
                 static_cast<int>(lr));
    construct_jit_live_ready_[slot] = false;
    return false;
  }
  return true;
}

// Live JIT batched construct launch (see the header). Bit-identical to the AOT
// compact-quant kernel; a thin cuLaunchKernel with the same argument pack.
bool CUDAHistogramConstructor::TryLaunchConstructJITBatchedCompactQuant(
    const dim3& grid_dim, const dim3& block_dim,
    const CUDAHybridPairDescriptor* pair_descs,
    const data_size_t* level_smaller_num_data,
    int shared_hist_size, size_t bin_type_bytes) {
  // The JIT kernel's HybridPairDescriptor / LeafSplits mirror these host structs
  // field-for-field; guard the offsets the kernel reads via cuLaunchKernel. The
  // JIT struct uses the same member order + default C++ alignment, so matching
  // offsets here guarantees the padding matches too.
  static_assert(offsetof(CUDAHybridPairDescriptor, smaller_struct) == 0,
                "JIT pair-descriptor ABI drift: smaller_struct must be first");
  static_assert(offsetof(CUDAHybridPairDescriptor, larger_struct) == sizeof(void*),
                "JIT pair-descriptor ABI drift: larger_struct offset changed");
  static_assert(offsetof(CUDAHybridPairDescriptor, construct_valid) ==
                    2 * sizeof(void*) + 4 * sizeof(int32_t),
                "JIT pair-descriptor ABI drift: construct_valid offset changed");
  // construct_valid, smaller_valid, larger_valid, parent_num_bits then smaller_num_bits.
  static_assert(offsetof(CUDAHybridPairDescriptor, smaller_num_bits) ==
                    2 * sizeof(void*) + 4 * sizeof(int32_t) + 4,
                "JIT pair-descriptor ABI drift: smaller_num_bits offset changed");
  // int leaf_index (+4 pad to 8) then 2 doubles + int64 before num_data_in_leaf.
  static_assert(offsetof(CUDALeafSplitsStruct, num_data_in_leaf) ==
                    2 * sizeof(double) + sizeof(int64_t) + sizeof(int64_t),
                "JIT LeafSplits ABI drift: num_data_in_leaf offset changed");
  static_assert(offsetof(CUDALeafSplitsStruct, sum_of_hessians) == 2 * sizeof(int64_t),
                "JIT LeafSplits ABI drift: sum_of_hessians offset changed");
  // Scope guard: only the non-graph, host-launched, uint8, SP shared-hist path
  // was validated. Graph capture derives dim_y / bit widths on-device (different
  // ABI); speculative flow passes level sizes; both fall to AOT.
  const int slot = compact_is_4bit_ ? 1 : 0;
  if (!construct_jit_live_ready_[slot]) return false;
  if (hybrid_graph_capture_gstate_ != nullptr) return false;
  if (level_smaller_num_data != nullptr) return false;
  if (bin_type_bytes != 1) return false;
  if (shared_hist_size != construct_jit_live_key_[slot].shared_hist_size) return false;
  void* fn = construct_jit_.GetBatchedIfValidated(construct_jit_live_key_[slot]);
  if (fn == nullptr) return false;
  CUfunction func = reinterpret_cast<CUfunction>(fn);

  const int32_t* gh_ptr = reinterpret_cast<const int32_t*>(cuda_gradients_);
  const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(compact_data_uint8_t_.RawData());
  const uint32_t* col_off_ptr = compact_column_hist_offsets_.RawData();
  const uint32_t* part_hist_ptr = cuda_row_data_->cuda_partition_hist_offsets();
  const int* part_col_ptr = compact_feature_partition_column_index_offsets_.RawData();
  // packed byte offsets only read by the 4-bit kernel; the 8-bit kernel ignores it.
  const int* packed_off_ptr = compact_is_4bit_ ?
      compact_packed_partition_byte_offsets_.RawData() : nullptr;
  data_size_t num_data_arg = num_data_;
  data_size_t min_data_arg = static_cast<data_size_t>(min_data_in_leaf_);
  double min_hess_arg = min_sum_hessian_in_leaf_;
  void* args[] = {&pair_descs, &gh_ptr, &data_ptr, &col_off_ptr, &part_hist_ptr,
                  &part_col_ptr, &packed_off_ptr, &num_data_arg, &min_data_arg, &min_hess_arg};
  const CUresult lr = cuLaunchKernel(
      func, grid_dim.x, grid_dim.y, grid_dim.z,
      block_dim.x, block_dim.y, block_dim.z, 0, cuda_stream_, args, nullptr);
  if (lr != CUDA_SUCCESS) {
    Log::Warning("CUDAConstructJIT live launch failed (%d); disabling live JIT, using AOT",
                 static_cast<int>(lr));
    construct_jit_live_ready_[slot] = false;
    return false;
  }
  return true;
}

}  // namespace Falcata

#endif  // USE_CUDA
