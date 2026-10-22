/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"
#include "cuda_device_levels.hpp"

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_rocm_interop.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuda_meta_batch.hpp"
#include "cuda_nonquant_fixed.cuh"
#include "cuda_pdl_launch.hpp"

#include "cuda_histogram_constructor_kernels.cuh"

namespace Falcata {

// Instantiated in cuda_histogram_constructor_bin*.cu.
extern template void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0<double, DP_SHARED_HIST_SIZE, uint8_t>(const CUDALeafSplitsStruct*, const data_size_t, const uint8_t);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0<float, SP_SHARED_HIST_SIZE, uint8_t>(const CUDALeafSplitsStruct*, const data_size_t, const uint8_t);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<double, DP_SHARED_HIST_SIZE, uint8_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<double, DP_SHARED_HIST_SIZE, uint8_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<float, SP_SHARED_HIST_SIZE, uint8_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<float, SP_SHARED_HIST_SIZE, uint8_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, DP_SHARED_HIST_SIZE, uint8_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, DP_SHARED_HIST_SIZE, uint8_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, SP_SHARED_HIST_SIZE, uint8_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, SP_SHARED_HIST_SIZE, uint8_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0<double, DP_SHARED_HIST_SIZE, uint16_t>(const CUDALeafSplitsStruct*, const data_size_t, const uint8_t);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0<float, SP_SHARED_HIST_SIZE, uint16_t>(const CUDALeafSplitsStruct*, const data_size_t, const uint8_t);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<double, DP_SHARED_HIST_SIZE, uint16_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<double, DP_SHARED_HIST_SIZE, uint16_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<float, SP_SHARED_HIST_SIZE, uint16_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<float, SP_SHARED_HIST_SIZE, uint16_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, DP_SHARED_HIST_SIZE, uint16_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, DP_SHARED_HIST_SIZE, uint16_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, SP_SHARED_HIST_SIZE, uint16_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, SP_SHARED_HIST_SIZE, uint16_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0<double, DP_SHARED_HIST_SIZE, uint32_t>(const CUDALeafSplitsStruct*, const data_size_t, const uint8_t);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0<float, SP_SHARED_HIST_SIZE, uint32_t>(const CUDALeafSplitsStruct*, const data_size_t, const uint8_t);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<double, DP_SHARED_HIST_SIZE, uint32_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<double, DP_SHARED_HIST_SIZE, uint32_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<float, SP_SHARED_HIST_SIZE, uint32_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<float, SP_SHARED_HIST_SIZE, uint32_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, DP_SHARED_HIST_SIZE, uint32_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, DP_SHARED_HIST_SIZE, uint32_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, SP_SHARED_HIST_SIZE, uint32_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
extern template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, SP_SHARED_HIST_SIZE, uint32_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);

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

__global__ void InterleaveGradHessKernel(
  const score_t* __restrict__ gradients,
  const score_t* __restrict__ hessians,
  float2* __restrict__ gradients_hessians,
  const data_size_t num_data,
  uint32_t* absmax_bits) {
  const data_size_t i = static_cast<data_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float g = 0.0f, h = 0.0f;
  if (i < num_data) {
    g = static_cast<float>(gradients[i]);
    h = static_cast<float>(hessians[i]);
    gradients_hessians[i] = make_float2(g, h);
  }
  if (absmax_bits != nullptr) {
    float mg = fabsf(g), mh = fabsf(h);
    for (int off = 16; off > 0; off >>= 1) {
      mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, off));
      mh = fmaxf(mh, __shfl_xor_sync(0xffffffffu, mh, off));
    }
    if ((threadIdx.x & 31) == 0) {
      atomicMax(absmax_bits, __float_as_uint(mg));
      atomicMax(absmax_bits + 1, __float_as_uint(mh));
    }
  }
}

void LaunchInterleaveGradHessKernel(
  const score_t* gradients,
  const score_t* hessians,
  float2* gradients_hessians,
  data_size_t num_data,
  uint32_t* absmax_bits) {
  const int block_size = 1024;
  const int num_blocks = static_cast<int>((num_data + block_size - 1) / block_size);
  InterleaveGradHessKernel<<<num_blocks, block_size>>>(
    gradients, hessians, gradients_hessians, num_data, absmax_bits);
}

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

size_t FusedRootHistFillSharedBytes(int num_slots, int hist_entries) {
  return Fill4BitTiledSharedBytes(num_slots) + static_cast<size_t>(num_slots) * sizeof(uint32_t) +
         static_cast<size_t>(kFill4BitTiledRows) * sizeof(int32_t) +
         static_cast<size_t>(hist_entries) * sizeof(int32_t) + kFusedRootRunSlack;
}

#endif  // !defined(__HIP_PLATFORM_AMD__)
DeviceLaunchLimits CurrentDeviceLaunchLimits() {
  struct Entry {
    int device;
    DeviceLaunchLimits limits;
  };
  static thread_local std::vector<Entry> entries;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  for (const Entry& e : entries) {
    if (e.device == device) return e.limits;
  }
  DeviceLaunchLimits limits;
  int shared_bytes = 0;
  CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&shared_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
  limits.shared_bytes_per_block = static_cast<size_t>(shared_bytes);
  CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&limits.max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
  CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&limits.max_grid_dim_y, cudaDevAttrMaxGridDimY, device));
  CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&limits.warp_size, cudaDevAttrWarpSize, device));
  entries.push_back({device, limits});
  return limits;
}

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
  size_t root_hist_scratch_bytes,
  uint32_t* code_view,
  const uint8_t* code_table,
  int code_row_words,
  int code_bits,
  bool* code_view_written) {
  *code_view_written = false;
#if !defined(__HIP_PLATFORM_AMD__)
  // the packed per-block row bound, and the default dynamic shared-memory limit (the slot-major table where it
  // was requested and fits, else one cell per local bin): the device's, as the launch does not opt in to more
  const int max_tiles_per_block = std::min(16, max_rows_per_block / kFill4BitTiledRows);
  const size_t max_shared_bytes = CurrentDeviceLaunchLimits().shared_bytes_per_block;
  const size_t slot_major_bytes = FusedRootHistFillSharedBytes(
    total_byte_slots, 2 * slot_major_span * FusedRootHistSlotStride(total_byte_slots));
  const bool slot_major = slot_major_span > 0 && slot_major_bytes <= max_shared_bytes;
  const size_t base_shared_bytes = slot_major ? slot_major_bytes
                                              : FusedRootHistFillSharedBytes(total_byte_slots, num_local_bins);
  if (max_tiles_per_block < 1 || base_shared_bytes > max_shared_bytes) {
    return false;
  }
  // cuda_plan key pair_code5: the 5-bit code pack reads up to 15 bytes past a row's last slot (four aligned words),
  // kept inside the block's shared memory; where that slack does not fit, the fill writes the nibble view instead.
  // cuda_plan key pair_code_wide: the 6-bit code's scalar pack reads no slot past the row and needs no slack.
  const bool code5 = code_view != nullptr && (code_bits == 6 || code_bits == kCodeRadix ||
                                              (code_bits == 5 && base_shared_bytes + 16 <= max_shared_bytes));
  // cuda_plan key pair_code_wide_regs: the 6-bit / mixed-radix register pack reads up to 15 bytes past a row's last
  // slot (four aligned words for a thread's two words): the same 16-byte slack, or the generic scalar pack where it
  // does not fit
  const bool code_wide_regs = code5 && (code_bits == 6 || code_bits == kCodeRadix) &&
                              FalcataPlan::Get().pair_code_wide_regs && base_shared_bytes + 16 <= max_shared_bytes;
  const size_t shared_bytes = base_shared_bytes + ((code5 && code_bits == 5) || code_wide_regs ? 16 : 0);
  using FusedFillKernel = void (*)(const uint8_t*, uint8_t*, const size_t*, const size_t*, const size_t*, const int*,
                                   int, data_size_t, const int*, const uint32_t*, int, int, int, bool, bool, bool, bool,
                                   const int32_t*, bool, hist_t*, uint32_t*, const uint8_t*, int, bool, bool, bool);
  const int fill_code_bits = code5 ? code_bits : 0;
  const FusedFillKernel kernel = slot_major ?
    (fill_code_bits == kCodeRadix ? CUDAFillCompactData4BitTiledRootHistKernel<true, kCodeRadix> :
     fill_code_bits == 6 ? CUDAFillCompactData4BitTiledRootHistKernel<true, 6> :
     fill_code_bits == 5 ? CUDAFillCompactData4BitTiledRootHistKernel<true, 5> :
                           CUDAFillCompactData4BitTiledRootHistKernel<true, 0>) :
    (fill_code_bits == kCodeRadix ? CUDAFillCompactData4BitTiledRootHistKernel<false, kCodeRadix> :
     fill_code_bits == 6 ? CUDAFillCompactData4BitTiledRootHistKernel<false, 6> :
     fill_code_bits == 5 ? CUDAFillCompactData4BitTiledRootHistKernel<false, 5> :
                           CUDAFillCompactData4BitTiledRootHistKernel<false, 0>);
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
      FusedFillKernel kernel = nullptr;
      size_t shared_bytes = 0;
      int resident_blocks = 0;
    };
    static thread_local ResidentQuery cached;
    int device = 0;
    CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
    if (cached.device != device || cached.kernel != kernel || cached.shared_bytes != shared_bytes) {
      int num_sms = 0;
      int blocks_per_sm = 0;
      CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
      CUDASUCCESS_OR_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, kFusedRootThreads, shared_bytes));
      cached.device = device;
      cached.kernel = kernel;
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
  kernel<<<grid, kFusedRootThreads, shared_bytes, stream>>>(
    src_data, compact_data, bs_src_nib0, bs_src_nib1, bs_dst_byte, bs_dst_stride, total_byte_slots, num_data,
    slot_first_bin, local_bin_hist_pos, num_local_bins, slot_major ? slot_major_span : 0, tiles_per_block,
    strided_tiles, run_copy, word_prefetch, l2_lead, grad_and_hess, hist_16bit, root_hist_scratch,
    code5 ? code_view : nullptr, code_table, code_row_words, FalcataPlan::Get().pair_code5_pack2,
    !FalcataPlan::Get().pair_code5_l2_store, code_wide_regs);
  if (code5) {
    // a new variant: a rejected launch must not leave the code view (and the root histogram) unwritten silently
    CUDASUCCESS_OR_FATAL(cudaPeekAtLastError());
  }
  *code_view_written = code5;
  return true;
#else
  (void)stream; (void)src_data; (void)compact_data; (void)bs_src_nib0; (void)bs_src_nib1; (void)bs_dst_byte;
  (void)bs_dst_stride; (void)total_byte_slots; (void)num_data; (void)slot_first_bin; (void)local_bin_hist_pos;
  (void)num_local_bins; (void)slot_major_span; (void)strided_tiles; (void)run_copy; (void)word_prefetch; (void)l2_lead;
  (void)max_rows_per_block; (void)grad_and_hess; (void)hist_16bit;
  (void)root_hist_scratch; (void)root_hist_scratch_bytes; (void)code_view; (void)code_table; (void)code_row_words;
  (void)code_bits;
  return false;
#endif
}

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

__global__ void ZeroHistSlotsKernel(hist_t* hist, const int* slots, const size_t slot_size) {
  uint4* const slot = reinterpret_cast<uint4*>(hist + static_cast<size_t>(slots[blockIdx.y]) * slot_size);
  const size_t num_words = slot_size / 2;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_words;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    slot[i] = zero;
  }
}

void CUDAHistogramConstructor::LaunchZeroHistSlotsKernel(const std::vector<int>& slots, const size_t slot_size) {
  static_assert(sizeof(hist_t) == 8, "ZeroHistSlotsKernel zeroes slots of doubles as 16-byte words");
  const int num_slots = static_cast<int>(slots.size());
  if (cuda_zero_slots_.Size() < slots.size()) {
    // every slot of a level at most: one allocation (a growth would synchronize the device)
    cuda_zero_slots_.Resize(std::max(slots.size(), static_cast<size_t>(num_leaves_) + 2));
  }
  // pageable source: the copy is staged before it returns, so the caller's vector may change at once; the legacy
  // stream orders it before the kernel, as the memsets it replaces were ordered before the next construct
  CopyFromHostToCUDADeviceAsync<int>(cuda_zero_slots_.RawData(), slots.data(), slots.size(), 0, __FILE__, __LINE__);
  constexpr int kThreads = 256;
  const size_t num_words = slot_size / 2;
  const int blocks_x = static_cast<int>(std::max<size_t>(1, std::min<size_t>(64, (num_words + kThreads - 1) / kThreads)));
  // grid.y holds at most 65535 slots: more go in further launches over the rest of the list
  constexpr int kMaxSlotsPerLaunch = 65535;
  for (int first = 0; first < num_slots; first += kMaxSlotsPerLaunch) {
    const int count = std::min(kMaxSlotsPerLaunch, num_slots - first);
    ZeroHistSlotsKernel<<<dim3(blocks_x, count), kThreads, 0, 0>>>(cuda_hist_.RawData(),
                                                                   cuda_zero_slots_.RawData() + first, slot_size);
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

bool CUDAHistogramConstructor::PairJointServesView() const {
  const FalcataPlan& plan = FalcataPlan::Get();
  if (!plan.pair_hist || !use_quantized_grad_) {
    return false;
  }
  if (plan.pack_bit3 || plan.pack_radix5 || plan.pack_radix6 || plan.pack_radix7 || plan.graph_quant) {
    return false;
  }
  const std::vector<int>& part_cols = cuda_row_data_->host_feature_partition_column_index_offsets();
  const std::vector<uint32_t>& col_hist_offsets = cuda_row_data_->host_column_hist_offsets();
  const std::vector<uint32_t>& part_hist_offsets = cuda_row_data_->host_partition_hist_offsets();
  const int num_columns = part_cols.back();
  if (num_columns <= 0 || part_hist_offsets.size() < part_cols.size()) {
    return false;
  }
  const double share = static_cast<double>(SampledColumns()) / static_cast<double>(num_columns);
  double max_cells = 0.0;
  int max_sampled = 1;
  int max_part_columns = 0;
  for (size_t p = 0; p + 1 < part_cols.size(); ++p) {
    const int n = part_cols[p + 1] - part_cols[p];
    if (n <= 0) {
      continue;
    }
    const uint32_t part_span = part_hist_offsets[p + 1] - part_hist_offsets[p];
    uint64_t joint = 0;
    for (int c = part_cols[p]; c < part_cols[p + 1]; c += 2) {
      uint32_t spans[2] = {1u, 1u};
      for (int k = 0; k < 2 && c + k < part_cols[p + 1]; ++k) {
        const int col = c + k;
        const uint32_t end = col + 1 < part_cols[p + 1] ? col_hist_offsets[col + 1] : part_span;
        spans[k] = end - col_hist_offsets[col];
        if (spans[k] == 0 || spans[k] > static_cast<uint32_t>(PackNibble4::kMaxBins)) {
          return false;  // not a nibble view: the pair-joint construct never runs on it
        }
      }
      const uint32_t stride = spans[0] * spans[1];
      joint += stride + ((plan.pair_pad && (stride & 1u) == 0) ? 1u : 0u);
    }
    const double mean = share * n;
    const double spread = num_columns > 1 ?
      std::sqrt(std::max(0.0, mean * (1.0 - share) * (num_columns - n) / (num_columns - 1.0))) : 0.0;
    const double sampled = std::min(static_cast<double>(n), mean + 2.0 * spread);
    const int bytes_full = (n + 1) / 2;
    const int bytes_sampled = static_cast<int>(std::ceil(sampled / 2.0));
    max_cells = std::max(max_cells, static_cast<double>(joint) * bytes_sampled / bytes_full);
    max_sampled = std::max(max_sampled, static_cast<int>(std::ceil(sampled)));
    max_part_columns = std::max(max_part_columns, n);
  }
  // the construct's block for the compact view (the batched launch's sizing)
  const int block_dim_x = max_sampled > NUM_THREADS_PER_BLOCK ? (max_sampled + 1) / 2 : max_sampled;
  const int block_dim_y = HybridQuantConstructBlockDimY(std::max(1, NUM_THREADS_PER_BLOCK / block_dim_x),
                                                        num_grad_quant_bins_);
  const int threads = (max_sampled + 1) / 2 * block_dim_y;
  const bool served = max_cells * sizeof(int32_t) <= 48.0 * 1024 && threads <= PairJointMaxThreadsPerBlock();
  const char* fmt = "colmajor_direct: pair-joint construct for the compact view: %d partitions of up to %d columns, "
                    "%d sampled (mean + 2 sd), joint table %.1f KB (limit 48), block %d threads (limit %d): %s";
  if (FalcataDebug().diag) {
    Log::Info(fmt, static_cast<int>(part_cols.size()) - 1, max_part_columns, max_sampled,
              max_cells * sizeof(int32_t) / 1024.0, threads, PairJointMaxThreadsPerBlock(),
              served ? "served" : "not served");
  } else {
    Log::Debug(fmt, static_cast<int>(part_cols.size()) - 1, max_part_columns, max_sampled,
               max_cells * sizeof(int32_t) / 1024.0, threads, PairJointMaxThreadsPerBlock(),
               served ? "served" : "not served");
  }
  return served;
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

__global__ void SubtractHistogramKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool hist_fp32) {
  SubtractHistogramInner(num_total_bin, cuda_smaller_leaf_splits, cuda_larger_leaf_splits, hist_fp32);
}

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

__global__ void FixSubtractHistogramSmallLeafBatchedKernel(
  const int num_total_bin,
  const int num_subtract_blocks,
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const int num_need_fix,
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
    // features outside this tree's sample are skipped inside (their most-
    // frequent bin is dead storage this tree)
    FixHistogramGroupInner(cuda_feature_num_bins, cuda_feature_hist_offsets,
      cuda_feature_most_freq_bins, cuda_need_fix_histogram_features, num_need_fix,
      (static_cast<int>(blockIdx.x) - num_subtract_blocks) * kFixGroupFeatures,
      feature_used, desc->smaller_struct, desc->larger_struct, hist_fp32);
  }
}

__global__ void SubtractHistogramDiscretizedBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const CUDAHybridGraphLoopStateOpt gstate,
  const CUDADeviceLevelState* dls) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y) || DeviceLevelBeyondLivePairs(dls, blockIdx.y)) {
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

__global__ void CopyChangedNumBitHistogramBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const CUDAHybridGraphLoopStateOpt gstate,
  const CUDADeviceLevelState* dls) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the subtract's change buffer is complete and visible
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y) || DeviceLevelBeyondLivePairs(dls, blockIdx.y)) {
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

__global__ void FixHistogramDiscretizedBatchedKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDAHybridPairDescriptor* pair_descs,
  const int8_t* feature_used,
  const CUDAHybridGraphLoopStateOpt gstate,
  const CUDADeviceLevelState* dls) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y) || DeviceLevelBeyondLivePairs(dls, blockIdx.y)) {
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
      desc->smaller_struct, shared_mem_buffer, blockIdx.x);
  } else {
    FixHistogramDiscretizedInner<false>(
      cuda_feature_num_bins, cuda_feature_hist_offsets, cuda_feature_most_freq_bins,
      cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned,
      desc->smaller_struct, shared_mem_buffer, blockIdx.x);
  }
}

__global__ void FixSubtractDiscretizedMaskedBatchedKernel(
  const int num_total_bin,
  const int num_subtract_blocks,
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const uint8_t* fix_mfb_mask,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const int8_t* feature_used,
  const CUDAHybridGraphLoopStateOpt gstate,
  const CUDADeviceLevelState* dls) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the construct's histograms are complete and visible
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next level-chain kernel may get resident
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y) || DeviceLevelBeyondLivePairs(dls, blockIdx.y)) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  const bool larger_exists = desc->larger_struct->leaf_index >= 0;
  uint8_t parent_num_bits = desc->parent_num_bits;
  uint8_t smaller_num_bits = desc->smaller_num_bits;
  uint8_t larger_num_bits = desc->larger_num_bits;
  if (HybridGraphActive(gstate)) {
    // as SubtractHistogramDiscretizedBatchedKernel (the fix kernel derives the smaller's width the same way)
    const data_size_t num_data_smaller = desc->smaller_struct->num_data_in_leaf;
    const data_size_t num_data_larger = larger_exists ? desc->larger_struct->num_data_in_leaf : 0;
    smaller_num_bits = HybridGraphQuantHistBits(gstate, num_data_smaller);
    larger_num_bits = HybridGraphQuantHistBits(gstate, num_data_larger);
    parent_num_bits = HybridGraphQuantHistBits(gstate, num_data_smaller + num_data_larger);
  }
  FixSubtractDiscretizedMaskedPair(
    num_total_bin, num_subtract_blocks, cuda_feature_num_bins, cuda_feature_hist_offsets, cuda_feature_most_freq_bins,
    cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned, fix_mfb_mask,
    desc->smaller_struct, desc->larger_struct, larger_exists, parent_num_bits, smaller_num_bits, larger_num_bits,
    reinterpret_cast<int32_t*>(num_bit_change_buffer) + static_cast<size_t>(blockIdx.y) * static_cast<size_t>(num_total_bin),
    bin_used, feature_used);
}

// cuda_plan key device_tree_live_grid: the complete device tree's level fix/subtract/copy with work proportional to
// the level's live pairs. The batched kernels above launch one block row per pair of the level's worst case
// (2^level) and the rows past the live pairs exit; on wide histograms (num_total_bin / block x 2^level) those are
// millions of blocks per level. These take a bounded grid y and stride over the live pairs (dls->live_pairs, read
// from device memory at kernel start); the copy applies the batched kernel's per-pair bit-width gate. Each
// (pair, bin) element is computed by the same code as in the batched kernels (blockIdx.x keeps its role), so the
// results are bit-identical; only which block row handles a pair changes.
__global__ void FixSubtractDiscretizedMaskedLiveKernel(
  const int num_total_bin,
  const int num_subtract_blocks,
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const uint8_t* fix_mfb_mask,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const int8_t* feature_used,
  const CUDADeviceLevelState* dls) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the construct's histograms are complete and visible
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next level-chain kernel may get resident
  const unsigned int live_pairs = static_cast<unsigned int>(dls->live_pairs);
  for (unsigned int p = blockIdx.y; p < live_pairs; p += gridDim.y) {
    const CUDAHybridPairDescriptor* desc = pair_descs + p;
    FixSubtractDiscretizedMaskedPair(
      num_total_bin, num_subtract_blocks, cuda_feature_num_bins, cuda_feature_hist_offsets,
      cuda_feature_most_freq_bins, cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned,
      fix_mfb_mask, desc->smaller_struct, desc->larger_struct, desc->larger_struct->leaf_index >= 0,
      desc->parent_num_bits, desc->smaller_num_bits, desc->larger_num_bits,
      reinterpret_cast<int32_t*>(num_bit_change_buffer) + static_cast<size_t>(p) * static_cast<size_t>(num_total_bin),
      bin_used, feature_used);
    __syncthreads();  // a fix block's shared reduction scratch is reused by its next pair
  }
}

__global__ void SubtractHistogramDiscretizedLiveKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const CUDADeviceLevelState* dls) {
  const unsigned int global_thread_index_gate = threadIdx.x + blockIdx.x * blockDim.x;
  if (bin_used != nullptr && global_thread_index_gate < static_cast<unsigned int>(num_total_bin) &&
      !bin_used[global_thread_index_gate]) {
    return;  // as SubtractHistogramDiscretizedBatchedKernel: a dead bin for every pair (no barrier below)
  }
  const unsigned int live_pairs = static_cast<unsigned int>(dls->live_pairs);
  for (unsigned int p = blockIdx.y; p < live_pairs; p += gridDim.y) {
    const CUDAHybridPairDescriptor* desc = pair_descs + p;
    int32_t* buffer = reinterpret_cast<int32_t*>(num_bit_change_buffer) +
      static_cast<size_t>(p) * static_cast<size_t>(num_total_bin);
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
}

__global__ void CopyChangedNumBitHistogramLiveKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const CUDADeviceLevelState* dls) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the subtract's change buffer is complete and visible
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_thread_index >= static_cast<unsigned int>(num_total_bin) ||
      (bin_used != nullptr && !bin_used[global_thread_index])) {
    return;
  }
  // CopyChangedNumBitHistogramBatchedKernel's per-pair gate: a level without a bit-width change is one bounded
  // launch whose rows only read the live pairs' descriptors
  const unsigned int live_pairs = static_cast<unsigned int>(dls->live_pairs);
  for (unsigned int p = blockIdx.y; p < live_pairs; p += gridDim.y) {
    const CUDAHybridPairDescriptor* desc = pair_descs + p;
    if (desc->larger_leaf_index < 0 || !(desc->parent_num_bits > 16 && desc->larger_num_bits <= 16)) {
      continue;
    }
    int32_t* hist_dst = reinterpret_cast<int32_t*>(desc->larger_struct->hist_in_leaf);
    const int32_t* hist_src = reinterpret_cast<const int32_t*>(num_bit_change_buffer) +
      static_cast<size_t>(p) * static_cast<size_t>(num_total_bin);
    hist_dst[global_thread_index] = hist_src[global_thread_index];
  }
}

__global__ void FixSubtractDiscretizedMaskedKernel(
  const int num_total_bin,
  const int num_subtract_blocks,
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const uint8_t* fix_mfb_mask,
  const CUDALeafSplitsStruct* smaller_struct,
  const CUDALeafSplitsStruct* larger_struct,
  const uint8_t parent_num_bits,
  const uint8_t smaller_num_bits,
  const uint8_t larger_num_bits,
  hist_t* num_bit_change_buffer) {
  FixSubtractDiscretizedMaskedPair(
    num_total_bin, num_subtract_blocks, cuda_feature_num_bins, cuda_feature_hist_offsets, cuda_feature_most_freq_bins,
    cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned, fix_mfb_mask,
    smaller_struct, larger_struct, larger_struct->leaf_index >= 0, parent_num_bits, smaller_num_bits, larger_num_bits,
    reinterpret_cast<int32_t*>(num_bit_change_buffer), nullptr, nullptr);
}

__global__ void __launch_bounds__(512) FixSubtractUsedFeaturesBatchedKernel(
  const uint32_t* used_feature_info,
  const int num_used_features,
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const CUDADeviceLevelState* dls) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the construct's histograms are complete and visible
  if (DeviceLevelBeyondLivePairs(dls, blockIdx.x)) {
    return;  // cuda_plan key device_levels: a worst-case launch's pair past the level's live pairs
  }
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
      if (FixSubtractMaskedEnabled()) {
        // cuda_plan key fix_subtract_masked: fix + subtract in one launch (this pipeline's bit-change region, as below)
        hist_t* change_buffer = reinterpret_cast<hist_t*>(
          reinterpret_cast<int32_t*>(hist_buffer_for_num_bit_change_.RawData()) +
          static_cast<size_t>(active_pipeline_) * static_cast<size_t>(num_total_bin_));
        if (parent_num_bits_in_histogram_bins <= 16) {
          CHECK_LE(smaller_num_bits_in_histogram_bins, 16);
          CHECK_LE(larger_num_bits_in_histogram_bins, 16);
        } else if (larger_num_bits_in_histogram_bins <= 16) {
          CHECK_LE(smaller_num_bits_in_histogram_bins, 16);
        }
        FixSubtractDiscretizedMaskedKernel<<<FixSubtractMaskedGridX(), fix_subtract_masked_block_, 0, current_stream()>>>(
          num_total_bin_,
          FixSubtractMaskedSubtractBlocks(),
          cuda_feature_num_bins_.RawData(),
          cuda_feature_hist_offsets_.RawData(),
          cuda_feature_most_freq_bins_.RawData(),
          cuda_need_fix_histogram_features_.RawData(),
          cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
          cuda_fix_mfb_mask_.RawDataReadOnly(),
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          parent_num_bits_in_histogram_bins,
          smaller_num_bits_in_histogram_bins,
          larger_num_bits_in_histogram_bins,
          change_buffer);
        CUDASUCCESS_OR_FATAL(cudaPeekAtLastError());  // a new launch shape: its launch errors surface
        if (parent_num_bits_in_histogram_bins > 16 && larger_num_bits_in_histogram_bins <= 16) {
          CopyChangedNumBitHistogram<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, current_stream()>>>(
            num_total_bin_,
            cuda_larger_leaf_splits,
            change_buffer);
        }
        return;
      }
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
  count_scratch_dirty_ = true;  // cuda_plan key forest_count_hist: the det slab is also the count scratch
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
        column_used_bytree_device(),
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
  // A device without room for the fixed table uses the deterministic classic loop, never an atomic graph.
  if (NonquantClassicFallback() || !FalcataPlan::Get().graph_det) {
    // opt-in (cuda_plan=auto,graph_det:on): by default the graph loop does not
    // capture the det construct+merge (~4x on shallow non-quant shapes); under
    // nonquant_deterministic it captures the fixed-point construct + convert
    // instead, with the key off the float atomics. graph_det gates BOTH the
    // capture and the host-det-alongside-graph allowance (SetDetBatchedAllowed),
    // so a run keeps one histogram ownership model end to end.
    return false;
  }
  if (use_quantized_grad_ || det_dense_dy_ <= 0 || hist_fp32_ || use_compact_view_) {
    return false;
  }
  const int total_slot_rows = det_dense_total_slot_rows();
  return max_level_pairs > 0 &&
         max_level_pairs <= std::min(total_slot_rows, kDetDenseBatchedPairCap);
}

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
  count_scratch_dirty_ = true;  // cuda_plan key forest_count_hist: the det slab is also the count scratch
  det_kernel<<<det_grid, det_block, 0, cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      RowMajorBin<BIN_TYPE>(),
      cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      det_packed_4bit ? cuda_row_data_->cuda_packed_partition_byte_offsets() : nullptr,
      column_used_bytree_device(),
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

namespace {

// The count construct's dynamic shared table (up to the widest partition's bins x 12 bytes) may exceed the 48 KB a
// kernel gets without opting in: raised once per (kernel, device) to the largest size asked for. A process-wide
// registry under a mutex.
template <typename KERNEL>
void EnsureCountKernelSharedBytes(KERNEL kernel, const size_t bytes) {
  if (bytes <= 46 * 1024) {
    return;  // with the kernel's few static bytes, within the 48 KB every kernel gets
  }
  static std::mutex mutex;
  static std::map<std::pair<const void*, int>, size_t> raised;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  const std::pair<const void*, int> key(reinterpret_cast<const void*>(kernel), device);
  std::lock_guard<std::mutex> lock(mutex);
  auto it = raised.find(key);
  if (it != raised.end() && it->second >= bytes) {
    return;
  }
  CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(bytes)));
  raised[key] = bytes;
}

// Row tiles of the count construct. A tile flushes its partition's whole shared table to the scratch, so it takes at
// least 8x as many (row, column) adds as its partition has bins (and 4 rows per row lane); per-value counts (MODE 3)
// cap a tile at a field's capacity. The grid's y extent is the largest leaf's tile count, at most about 32 blocks per
// SM over the launch's partitions and pairs (but never fewer tiles than the field cap needs).
struct CountTiling {
  data_size_t rows_min;
  int max_tiles;
};

CountTiling CountTiles(const data_size_t max_rows, const int dy, const int lanes_columns, const uint32_t partition_bins,
                       const int blocks_per_tile, const int num_sms, const int field_bits) {
  const data_size_t field_cap = field_bits > 0 && field_bits < 31 ?
    (static_cast<data_size_t>(1) << field_bits) - 1 : std::numeric_limits<data_size_t>::max();
  data_size_t rows_min = std::max<data_size_t>(4 * dy,
    static_cast<data_size_t>((8 * static_cast<int64_t>(partition_bins)) / std::max(1, lanes_columns)));
  rows_min = std::max<data_size_t>(1, std::min(rows_min, field_cap));
  const int by_rows = static_cast<int>(std::max<data_size_t>(1, (max_rows + rows_min - 1) / rows_min));
  const int by_device = std::max(1, (32 * std::max(1, num_sms)) / std::max(1, blocks_per_tile));
  int tiles = std::min(by_rows, by_device);
  if (field_bits > 0) {
    tiles = std::max(tiles, static_cast<int>((max_rows + field_cap - 1) / field_cap));
  }
  return {rows_min, std::max(1, std::min(tiles, 65535))};
}

}  // namespace

size_t CUDAHistogramConstructor::CountScratchRowBytes(const bool wide) const {
  const size_t bins = static_cast<size_t>(num_total_bin_);
  return wide ? bins * sizeof(uint64_t) + ((bins + 1) / 2) * sizeof(uint64_t) : bins * sizeof(uint64_t);
}

size_t CUDAHistogramConstructor::CountSharedBytes(const bool wide) const {
  return static_cast<size_t>(count_max_partition_bins_) * (wide ? sizeof(uint64_t) + sizeof(uint32_t) : sizeof(uint64_t));
}

uint64_t* CUDAHistogramConstructor::CountScratch(int* rows) {
  uint64_t* base = reinterpret_cast<uint64_t*>(cuda_det_dense_slots_.RawData());
  const size_t slab_bytes = cuda_det_dense_slots_.Size() * sizeof(hist_t);
  *rows = static_cast<int>(slab_bytes / CountScratchRowBytes(count_codes_->wide));
  if (count_scratch_dirty_) {
    // the deterministic construct wrote its slot rows here since the last zeroing: the count scratch starts from
    // zero (on the legacy default stream: the blocking histogram streams order after it)
    CUDASUCCESS_OR_FATAL(cudaMemsetAsync(base, 0, slab_bytes, 0));
    count_scratch_dirty_ = false;
  }
  return base;
}

template <typename BIN_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramDenseCount(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const int grid_dim_x,
  const int block_dim_x,
  const DetDenseSource* source) {
  const CUDACountHistCodes& cc = *count_codes_;
  int rows = 0;
  uint64_t* scratch_base = CountScratch(&rows);
  CHECK_GE(rows, kNumHistPipelines);
  // pipeline-private scratch row: pipelined pair constructs run on different streams
  uint64_t* scratch = scratch_base + static_cast<size_t>(active_pipeline_) * (CountScratchRowBytes(cc.wide) / sizeof(uint64_t));
  uint32_t* scratch_n = reinterpret_cast<uint32_t*>(scratch + num_total_bin_);
  const int lanes = std::max(1, std::min(block_dim_x, kCountMaxLanes));
  const int dy = std::max(1, std::min(kCountDyCap, 1024 / lanes));
  const CountTiling tiling = CountTiles(num_data_in_smaller_leaf, dy, block_dim_x, count_max_partition_bins_,
                                        grid_dim_x, count_num_sms_, cc.field_bits);
  const dim3 grid(grid_dim_x, static_cast<unsigned int>(tiling.max_tiles));
  const dim3 block(lanes, static_cast<unsigned int>(dy));
  const size_t smem = CountSharedBytes(cc.wide && cc.field_bits == 0);
  const bool is_4bit = source != nullptr ? source->is_4bit : cuda_row_data_->is_4bit_packed();
  const BIN_TYPE* data = source != nullptr ? static_cast<const BIN_TYPE*>(source->data) : RowMajorBin<BIN_TYPE>();
  const uint32_t* column_offsets = source != nullptr ? source->column_hist_offsets : cuda_row_data_->cuda_column_hist_offsets();
  const int* partition_columns = source != nullptr ? source->partition_column_offsets :
    cuda_row_data_->cuda_feature_partition_column_index_offsets();
  const int* packed_offsets = source != nullptr ? source->packed_byte_offsets :
    (is_4bit ? cuda_row_data_->cuda_packed_partition_byte_offsets() : nullptr);
  const int8_t* feature_used = source != nullptr ? source->is_feature_used :
    column_used_bytree_device();
  const int mode = cc.field_bits > 0 ? 3 : (cc.wide ? 2 : (cc.identity ? 0 : 1));
#define FALCATA_COUNT_LEAF_LAUNCH(IS4, MODE) \
  do { \
    auto kernel = &CUDAConstructHistogramDenseCountKernel<BIN_TYPE, IS4, MODE>; \
    EnsureCountKernelSharedBytes(kernel, smem); \
    kernel<<<grid, block, smem, current_stream()>>>(cuda_smaller_leaf_splits, cc.codes.RawData(), \
      cc.values.RawData(), data, column_offsets, cuda_row_data_->cuda_partition_hist_offsets(), partition_columns, \
      packed_offsets, feature_used, num_data_, scratch, scratch_n, tiling.rows_min, cc.field_bits, cc.num_values); \
  } while (0)
  if (is_4bit) {
    switch (mode) {
      case 0: FALCATA_COUNT_LEAF_LAUNCH(true, 0); break;
      case 1: FALCATA_COUNT_LEAF_LAUNCH(true, 1); break;
      case 2: FALCATA_COUNT_LEAF_LAUNCH(true, 2); break;
      default: FALCATA_COUNT_LEAF_LAUNCH(true, 3); break;
    }
  } else {
    switch (mode) {
      case 0: FALCATA_COUNT_LEAF_LAUNCH(false, 0); break;
      case 1: FALCATA_COUNT_LEAF_LAUNCH(false, 1); break;
      case 2: FALCATA_COUNT_LEAF_LAUNCH(false, 2); break;
      default: FALCATA_COUNT_LEAF_LAUNCH(false, 3); break;
    }
  }
#undef FALCATA_COUNT_LEAF_LAUNCH
  const int convert_threads = 256;
  const int convert_blocks = (num_total_bin_ + convert_threads - 1) / convert_threads;
  if (cc.wide) {
    ConvertCountHistogramKernel<true><<<convert_blocks, convert_threads, 0, current_stream()>>>(
      cuda_smaller_leaf_splits, scratch, scratch_n, static_cast<uint32_t>(num_total_bin_), cc.grad_base, cc.grad_unit,
      cc.hessian);
  } else {
    ConvertCountHistogramKernel<false><<<convert_blocks, convert_threads, 0, current_stream()>>>(
      cuda_smaller_leaf_splits, scratch, scratch_n, static_cast<uint32_t>(num_total_bin_), cc.grad_base, cc.grad_unit,
      cc.hessian);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

bool CUDAHistogramConstructor::CountBatchedEligible() const {
  // the deterministic batched arm's gates (DetDenseBatchedEligible) without its slot-slab pair cap: the count scratch
  // needs one row per pair and the launcher chunks the level by the rows the slab holds
  // (the compact view too: the per-leaf deterministic arm serves it, the batched one leaves it to the float atomics)
  return count_codes_ != nullptr && !use_quantized_grad_ && det_dense_dy_ > 0 && !hist_fp32_ &&
         hybrid_graph_capture_gstate_ == nullptr && det_batched_allowed_;
}

template <typename BIN_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramDenseBatchedCount(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const data_size_t max_num_data_in_smaller_leaf,
  const int grid_dim_x,
  const int block_dim_x) {
  const CUDACountHistCodes& cc = *count_codes_;
  int rows = 0;
  uint64_t* scratch = CountScratch(&rows);
  CHECK_GE(rows, 1);
  const size_t row_words = CountScratchRowBytes(cc.wide) / sizeof(uint64_t);
  // the tree's compact view (its sampled columns; offsets of the same convention, global bin positions) or the full
  // row data with the per-tree feature mask, as the per-leaf arms read them
  const bool compact = use_compact_view_;
  const int columns = compact ? std::max(1, max_num_compact_cols_per_partition_) : block_dim_x;
  const int lanes = std::max(1, std::min(columns, kCountMaxLanes));
  const int dy = std::max(1, std::min(kCountDyCap, 1024 / lanes));
  const dim3 block(lanes, static_cast<unsigned int>(dy));
  const size_t smem = CountSharedBytes(cc.wide && cc.field_bits == 0);
  const bool is_4bit = compact ? compact_is_4bit_ : cuda_row_data_->is_4bit_packed();
  const BIN_TYPE* data = compact ? reinterpret_cast<const BIN_TYPE*>(NibbleViewData()) : RowMajorBin<BIN_TYPE>();
  const uint32_t* column_offsets = compact ? compact_column_hist_offsets_.RawData() :
    cuda_row_data_->cuda_column_hist_offsets();
  const int* partition_columns = compact ? compact_feature_partition_column_index_offsets_.RawData() :
    cuda_row_data_->cuda_feature_partition_column_index_offsets();
  const int* packed_offsets = !is_4bit ? nullptr : (compact ? compact_packed_partition_byte_offsets_.RawData() :
    cuda_row_data_->cuda_packed_partition_byte_offsets());
  const int8_t* feature_used = compact ? nullptr :
    column_used_bytree_device();
  const int mode = cc.field_bits > 0 ? 3 : (cc.wide ? 2 : (cc.identity ? 0 : 1));
  const int convert_threads = 256;
  const int convert_blocks = (num_total_bin_ + convert_threads - 1) / convert_threads;
  for (int first = 0; first < num_pairs; first += std::min(rows, 65535)) {
    const int chunk = std::min(std::min(rows, 65535), num_pairs - first);
    const CUDAHybridPairDescriptor* descs = pair_descs + first;
    const CountTiling tiling = CountTiles(max_num_data_in_smaller_leaf, dy, columns, count_max_partition_bins_,
                                          grid_dim_x * chunk, count_num_sms_, cc.field_bits);
    const dim3 grid(grid_dim_x, static_cast<unsigned int>(tiling.max_tiles), static_cast<unsigned int>(chunk));
#define FALCATA_COUNT_BATCHED_LAUNCH(IS4, MODE) \
    do { \
      auto kernel = &CUDAConstructHistogramDenseCountBatchedKernel<BIN_TYPE, IS4, MODE>; \
      EnsureCountKernelSharedBytes(kernel, smem); \
      kernel<<<grid, block, smem, cuda_stream_>>>(descs, cc.codes.RawData(), cc.values.RawData(), \
        data, column_offsets, cuda_row_data_->cuda_partition_hist_offsets(), partition_columns, \
        packed_offsets, feature_used, num_data_, \
        static_cast<data_size_t>(min_data_in_leaf_), min_sum_hessian_in_leaf_, scratch, row_words, \
        static_cast<uint32_t>(num_total_bin_), tiling.rows_min, cc.field_bits, cc.num_values); \
    } while (0)
    if (is_4bit) {
      switch (mode) {
        case 0: FALCATA_COUNT_BATCHED_LAUNCH(true, 0); break;
        case 1: FALCATA_COUNT_BATCHED_LAUNCH(true, 1); break;
        case 2: FALCATA_COUNT_BATCHED_LAUNCH(true, 2); break;
        default: FALCATA_COUNT_BATCHED_LAUNCH(true, 3); break;
      }
    } else {
      switch (mode) {
        case 0: FALCATA_COUNT_BATCHED_LAUNCH(false, 0); break;
        case 1: FALCATA_COUNT_BATCHED_LAUNCH(false, 1); break;
        case 2: FALCATA_COUNT_BATCHED_LAUNCH(false, 2); break;
        default: FALCATA_COUNT_BATCHED_LAUNCH(false, 3); break;
      }
    }
#undef FALCATA_COUNT_BATCHED_LAUNCH
    const dim3 convert_grid(convert_blocks, 1, static_cast<unsigned int>(chunk));
    if (cc.wide) {
      ConvertCountHistogramBatchedKernel<true><<<convert_grid, convert_threads, 0, cuda_stream_>>>(
        descs, scratch, row_words, static_cast<uint32_t>(num_total_bin_), cc.grad_base, cc.grad_unit, cc.hessian,
        static_cast<data_size_t>(min_data_in_leaf_), min_sum_hessian_in_leaf_);
    } else {
      ConvertCountHistogramBatchedKernel<false><<<convert_grid, convert_threads, 0, cuda_stream_>>>(
        descs, scratch, row_words, static_cast<uint32_t>(num_total_bin_), cc.grad_base, cc.grad_unit, cc.hessian,
        static_cast<data_size_t>(min_data_in_leaf_), min_sum_hessian_in_leaf_);
    }
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

__global__ void ConvertFixedHistBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_items,
  const double* inv_scale,
  const uint8_t* bin_used,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const CUDAHybridGraphLoopStateOpt gstate) {
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.y;
  if (HybridPairConstructSkipped(desc, min_data_in_leaf, min_sum_hessian_in_leaf)) {
    return;
  }
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i >= num_items || (bin_used != nullptr && !bin_used[i >> 1])) {
    return;
  }
  hist_t* hist = desc->smaller_struct->hist_in_leaf;
  const int64_t word = reinterpret_cast<const int64_t*>(hist)[i];
  hist[i] = static_cast<hist_t>(word) * inv_scale[i & 1];
}

int CUDAHistogramConstructor::LaunchConvertFixedHistBatched(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  CUDAHybridGraphLoopStateOpt gstate) {
  constexpr int kThreads = 256;
  const int num_items = 2 * num_total_bin_;
  const int blocks_x = (num_items + kThreads - 1) / kThreads;
  ConvertFixedHistBatchedKernel<<<dim3(blocks_x, num_pairs), kThreads, 0, cuda_stream_>>>(
    pair_descs, num_items, nq_inv_scale_.RawDataReadOnly(),
    any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
    static_cast<data_size_t>(min_data_in_leaf_), min_sum_hessian_in_leaf_, gstate);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
  return blocks_x;
}

void CUDAHistogramConstructor::InitNonquantFixed() {
  nq_fixed_smem_bytes_ = 0;
  nq_fixed_smem_set_.clear();
  nq_fixed_supported_ = true;
  if (use_quantized_grad_ || !FalcataPlan::Get().nonquant_deterministic) {
    return;
  }
  const std::vector<uint32_t>& offsets = cuda_row_data_->host_partition_hist_offsets();
  uint32_t max_bins = 1;
  for (size_t p = 1; p < offsets.size(); ++p) {
    max_bins = std::max(max_bins, offsets[p] - offsets[p - 1]);
  }
  // All fixed specializations share the same static allocations. Query a compiled fixed instantiation instead
  // of assuming the compiler/architecture keeps their size constant (the histogram itself is dynamic shared memory).
  cudaFuncAttributes attrs;
  CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attrs,
    CUDAConstructHistogramDenseBatchedKernel<uint8_t, nq_fixed_t, SP_SHARED_HIST_SIZE, false, false, true>));
  const size_t need = static_cast<size_t>(max_bins) * 2 * sizeof(nq_fixed_t) + attrs.sharedSizeBytes;
  int shared_limit = 0;
  const int device = gpu_device_id_ < 0 ? 0 : gpu_device_id_;
  if (cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) {
    shared_limit = 48 * 1024;
  }
  // Testing-only fault injection, deliberately absent from cuda_plan: exercise the 64 KiB-device routing on CI GPUs.
  const char* force_64k = std::getenv("FALCATA_TEST_FORCE_NONQUANT_64K");
  if (force_64k != nullptr && std::strcmp(force_64k, "1") == 0) {
    shared_limit = std::min(shared_limit, 64 * 1024);
    Log::Info("TEST ONLY: nonquant shared limit 65536 bytes, classic fallback=%d",
              static_cast<int>(need > static_cast<size_t>(shared_limit)));
  }
  nq_fixed_supported_ = need <= static_cast<size_t>(shared_limit);
  if (!nq_fixed_supported_) {
    static bool warned = false;  // once per process, across datasets and resets
    if (!warned) {
      warned = true;
      Log::Warning("nonquant_deterministic: the fixed-point histogram construct needs %zu bytes of shared memory per "
                   "block (widest partition %u bins), this GPU gives %d: using the deterministic classic CUDA "
                   "training loop for this dataset",
                   need, max_bins, shared_limit);
    }
  }
}

size_t CUDAHistogramConstructor::NqFixedSmem(const void* kernel) {
  if (!nq_fixed_launch_) {
    return 0;
  }
  if (nq_fixed_smem_bytes_ == 0) {
    // the widest partition's items (gradient + hessian per bin) x 8 bytes; the compact view's partitions are
    // subsets of the source partitions
    const std::vector<uint32_t>& offsets = cuda_row_data_->host_partition_hist_offsets();
    uint32_t max_bins = 1;
    for (size_t p = 1; p < offsets.size(); ++p) {
      max_bins = std::max(max_bins, offsets[p] - offsets[p - 1]);
    }
    nq_fixed_smem_bytes_ = static_cast<size_t>(max_bins) * 2 * sizeof(nq_fixed_t);
  }
  if (std::find(nq_fixed_smem_set_.begin(), nq_fixed_smem_set_.end(), kernel) == nq_fixed_smem_set_.end()) {
    CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(nq_fixed_smem_bytes_)));
    nq_fixed_smem_set_.push_back(kernel);
  }
  return nq_fixed_smem_bytes_;
}

__global__ void NonquantFixedAbsMaxKernel(const score_t* gradients, const score_t* hessians, const data_size_t n,
                                          uint32_t* absmax_bits) {
  float mg = 0.0f, mh = 0.0f;
  for (data_size_t i = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x); i < n;
       i += static_cast<data_size_t>(blockDim.x * gridDim.x)) {
    mg = fmaxf(mg, fabsf(gradients[i]));
    mh = fmaxf(mh, fabsf(hessians[i]));
  }
  for (int off = 16; off > 0; off >>= 1) {
    mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, off));
    mh = fmaxf(mh, __shfl_xor_sync(0xffffffffu, mh, off));
  }
  if ((threadIdx.x & 31) == 0) {
    atomicMax(absmax_bits, __float_as_uint(mg));
    atomicMax(absmax_bits + 1, __float_as_uint(mh));
  }
}

__global__ void NonquantFixedScaleKernel(const uint32_t* absmax_bits, const int lg_n, int* scale,
                                         double* inv_scale) {
  for (int k = 0; k < 2; ++k) {
    const int e = NonquantFixedExponent(absmax_bits[k], lg_n);
    scale[k] = e;
    inv_scale[k] = ldexp(1.0, -e);
  }
}

void LaunchNonquantFixedScale(const score_t* gradients, const score_t* hessians, const data_size_t n,
                              const bool absmax_ready, uint32_t* absmax_bits, int* scale, double* inv_scale) {
  int lg_n = 0;
  while (lg_n < 62 && (static_cast<int64_t>(1) << lg_n) < static_cast<int64_t>(std::max<data_size_t>(n, 1))) {
    ++lg_n;
  }
  // legacy default stream, like the gradient interleave: ordered before the construct streams
  if (!absmax_ready) {  // (zeroed by the caller)
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(1024, (static_cast<int64_t>(n) + kThreads - 1) / kThreads));
    NonquantFixedAbsMaxKernel<<<std::max(blocks, 1), kThreads, 0, 0>>>(gradients, hessians, n, absmax_bits);
  }
  NonquantFixedScaleKernel<<<1, 1, 0, 0>>>(absmax_bits, lg_n, scale, inv_scale);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

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
  // cuda_plan key nonquant_deterministic: the level goes through the fixed-point instantiation; Inner0 still picks the
  // count or deterministic arm where they serve, and reports whether its fixed-point kernel built the level
  // (nq_fixed_kernel_launched_): only then are the words converted in place (on the host path here; the graph capture
  // appends the convert as its own node, CaptureHybridGraphSearchKernels)
  nq_fixed_kernel_launched_ = false;
  if (!use_quantized_grad_ && NonquantFixedActive()) {
    nq_fixed_launch_ = true;
    if (cuda_row_data_->shared_hist_size() == DP_SHARED_HIST_SIZE) {
      LaunchConstructHistogramBatchedKernelInner<nq_fixed_t, DP_SHARED_HIST_SIZE>(
        pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
    } else if (cuda_row_data_->shared_hist_size() == SP_SHARED_HIST_SIZE) {
      LaunchConstructHistogramBatchedKernelInner<nq_fixed_t, SP_SHARED_HIST_SIZE>(
        pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
    } else {
      Log::Fatal("Unknown shared histogram size %d", cuda_row_data_->shared_hist_size());
    }
    nq_fixed_launch_ = false;
    if (nq_fixed_kernel_launched_ && hybrid_graph_capture_gstate_ == nullptr) {
      LaunchConvertFixedHistBatched(pair_descs, num_pairs, nullptr);
    }
    return;
  }
  if (cuda_row_data_->shared_hist_size() == DP_SHARED_HIST_SIZE && gpu_use_dp_) {
    LaunchConstructHistogramBatchedKernelInner<double, DP_SHARED_HIST_SIZE>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
  } else if (cuda_row_data_->shared_hist_size() == SP_SHARED_HIST_SIZE && !gpu_use_dp_) {
    LaunchConstructHistogramBatchedKernelInner<float, SP_SHARED_HIST_SIZE>(pair_descs, num_pairs, max_num_data_in_smaller_leaf, level_smaller_num_data);
  } else {
    Log::Fatal("Unknown shared histogram size %d", cuda_row_data_->shared_hist_size());
  }
}

bool CUDAHistogramConstructor::DeviceLevelConstructSupported() {
  // the dense batched kernel and the pair-joint family both take the device block map
  return use_quantized_grad_ && SupportsBatchedLevel();
}

void CUDAHistogramConstructor::DeviceLevelConstructShape(CUDADeviceLevelConstructShape* shape) {
  std::memset(shape, 0, sizeof(*shape));
  // LaunchConstructHistogramBatchedKernelInner0's tree-constant decisions for a multi-pair level that holds fewer
  // rows than the matrix (a device level: not the root)
  int grid_dim_x = 0;
  int grid_dim_y = 0;
  int block_dim_x = 0;
  int block_dim_y = 0;
  CalcConstructHistogramBatchedKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, 1, 2);
  const int cc = std::max(1, max_num_compact_cols_per_partition_);
  if (use_compact_view_) {
    block_dim_x = cc > NUM_THREADS_PER_BLOCK ? (cc + 1) / 2 : cc;
    block_dim_y = std::max(1, NUM_THREADS_PER_BLOCK / block_dim_x);
    block_dim_y = HybridQuantConstructBlockDimY(block_dim_y, use_quantized_grad_ ? num_grad_quant_bins_ : 0);
  }
  shape->block_y = block_dim_y;
  shape->grid_x = grid_dim_x;
  shape->block_x = block_dim_x;
  const int bins = use_quantized_grad_ ? num_grad_quant_bins_ : 0;
  const FalcataPlan& plan = FalcataPlan::Get();
  if (!(use_compact_view_ && compact_is_4bit_ && compact_codec_ == PackCodecId::kNibble4 && plan.pair_hist &&
        compact_pair_joint_max_ > 0 && compact_pair_joint_max_ * sizeof(int32_t) <= 48 * 1024 &&
        (cc + 1) / 2 * block_dim_y <= PairJointMaxThreadsPerBlock())) {
    return;  // the dense batched kernel
  }
  shape->pair_mode = 1;
  const bool whole_rows = plan.pair_hist_rows && compact_row_interleave_ &&
      compact_pair_joint_total_ > 0 && compact_pair_joint_total_ * sizeof(int32_t) <= 48 * 1024 &&
      compact_row_bytes_ > 0 && compact_row_bytes_ * block_dim_y <= std::min(1024, PairJointMaxThreadsPerBlock());
  int pair_y = block_dim_y;
  if (whole_rows && plan.pair_block_rows) {
    const int max_threads = std::min(1024, PairJointMaxThreadsPerBlock());
    int y = pair_y;
    int best_warps = PairJointResidentWarps(compact_row_bytes_ * pair_y);
    for (int c = pair_y + 1; compact_row_bytes_ * c <= max_threads && HybridQuantConstructBlockDimY(c, bins) == c; ++c) {
      const int warps = PairJointResidentWarps(compact_row_bytes_ * c);
      if (warps > best_warps) {
        best_warps = warps;
        y = c;
      }
    }
    pair_y = y;
  }
  const size_t pair_smem_bytes =
    static_cast<size_t>(whole_rows ? compact_pair_joint_total_ : compact_pair_joint_max_) * sizeof(int32_t);
  // (the host keeps the default shape where the capped grid passes CUDA's y limit; a device level strides its
  // block rows over a fixed launch, so it has no such limit)
  const int capped_y = whole_rows && plan.pair_capped_rows ?
    PairJointCappedRows(compact_row_bytes_, pair_smem_bytes, pair_y, bins) : 0;
  if (capped_y > 0) {
    pair_y = capped_y;
  }
  const size_t code5_flat_smem_bytes = pair_smem_bytes + Code5DummyCells(code5_bits_) * sizeof(int32_t);
  const bool code5_flat = plan.pair_code5_flat && code5_flat_smem_bytes <= CurrentDeviceLaunchLimits().shared_bytes_per_block;
  const size_t code5_word_smem_bytes = code5_flat ? code5_flat_smem_bytes : pair_smem_bytes;
  const bool code5_tall = plan.pair_code5_slots && plan.pair_code5_tall;
  const bool code5_epilogue_flat = code5_flat && plan.pair_code5_epilogue;
  int code5_word_rows = 0;
  if (whole_rows && code5_valid_ && plan.pair_code5 && plan.pair_code5_words &&
      Code5RowWords(compact_row_bytes_, code5_bits_) == code5_row_words_) {
    const PairJointCode5WordShape word_shape = PairJointCode5WordShapeFor(
      code5_row_words_, code5_word_smem_bytes, bins, code5_flat, code5_tall, code5_epilogue_flat, code5_bits_);
    code5_word_rows = word_shape.rows;
    if (code5_word_rows > 0) {
      pair_y = code5_word_rows;
      shape->kernel_kind = 4;
      shape->block_x = code5_row_words_;
      shape->smem_bytes = static_cast<int>(code5_word_smem_bytes);
      shape->code5_flat = code5_flat ? 1 : 0;
      shape->code5_epilogue_flat = code5_epilogue_flat ? 1 : 0;
      if (plan.pair_code5_slots && word_shape.device_slots > 0) {
        shape->slot = 1;
        shape->slot_device_slots = word_shape.device_slots;
        shape->slot_min_rows = std::max(1, BatchConstructMinRowsPerThread());
        shape->slot_max_rows = bins > 0 ? HybridQuantConstructMaxRowsPerThread(bins, code5_word_rows) : 1 << 30;
      }
    }
  }
  if (code5_word_rows == 0) {
    shape->block_x = whole_rows ? compact_row_bytes_ : (cc + 1) / 2;
    shape->smem_bytes = static_cast<int>(pair_smem_bytes);
    const PairJointKernelFn code5_kernel = capped_y > 0 ? PairJointCapped().gather_code5 :
      CUDAConstructDiscretizedHistogramPairJointBatchedKernel<false, true>;
    const bool code5 = whole_rows && code5_valid_ && plan.pair_code5 && code5_bits_ == 5 &&
      Code5RowWords(compact_row_bytes_) == code5_row_words_ &&
      shape->block_x * pair_y <= PairJointCode5MaxThreads(code5_kernel);
    shape->kernel_kind = code5 ? (capped_y > 0 ? 3 : 2) : (capped_y > 0 ? 1 : 0);
  }
  shape->block_y = pair_y;
  shape->grid_x = whole_rows ? 1 : grid_dim_x;
  shape->per_pair_min_grid = shape->slot != 0 ? 0 : (plan.per_pair_rows ? min_grid_dim_y_ : 0);
  shape->per_pair_min_rows = BatchConstructMinRowsPerThread() * (whole_rows ? grid_dim_x : 1);
  shape->whole_row_partitions = whole_rows ? grid_dim_x : 0;
  shape->whole_row_joint = whole_rows ? static_cast<uint32_t>(compact_pair_joint_total_) : 0u;
}

bool CUDAHistogramConstructor::DeviceLevelGraphCapturable() {
  if (!nibble_pending_) {
    return true;
  }
  // a pending nibble view is materialized by its first reader: only the code-view constructs, which read the code
  // view itself, launch nothing else
  CUDADeviceLevelConstructShape shape;
  DeviceLevelConstructShape(&shape);
  return shape.pair_mode == 1 && shape.kernel_kind >= 2;
}

void CUDAHistogramConstructor::LaunchDeviceLevelPairJoint(const CUDAHybridPairDescriptor* pair_descs) {
  // cuda_plan key device_levels: a device level's pair-joint construct (the pair_hist branch of
  // LaunchConstructHistogramBatchedKernelInner0 with the device's block map, grid y and rows floor)
  CUDADeviceLevelConstructShape sh;
  DeviceLevelConstructShape(&sh);
  CHECK_EQ(sh.pair_mode, 1);
  const PairJointKernelFn kernel =
    sh.kernel_kind == 4 ? PairJointCode5WordKernelDL(sh.code5_flat != 0, sh.code5_epilogue_flat != 0, code5_bits_) :
    sh.kernel_kind == 3 ? PairJointCappedDL().gather_code5 :
    sh.kernel_kind == 2 ? CUDAConstructDiscretizedHistogramPairJointBatchedKernel<false, true, true> :
    sh.kernel_kind == 1 ? PairJointCappedDL().gather :
    CUDAConstructDiscretizedHistogramPairJointBatchedKernel<false, false, true>;
  DiagConstructKernel(sh.kernel_kind == 4 ? "pair_code5_words (device levels)" : "pair_hist (device levels)");
  const uint8_t* data = sh.kernel_kind >= 2 ? compact_data_uint8_t_.RawDataReadOnly() : NibbleViewData();
  const int grid_x = std::max(1, sh.grid_x);
  int grid_y = std::max(1, std::min(65535, (DeviceLevelConstructTargetBlocks() + grid_x - 1) / grid_x));
  if (device_level_whole_tree_ && sh.kernel_kind == 4 && sh.whole_row_partitions > 0 && grid_x == 1 &&
      sh.slot != 0 && sh.slot_device_slots > 0) {
    // Workers claim virtual block rows until exhausted; row grouping and flushes use the device map, not grid_y.
    // The slotted whole-tree code-word path needs no more physical workers than its known resident capacity.
    grid_y = std::min(grid_y, sh.slot_device_slots);
  }
  kernel<<<dim3(grid_x, grid_y, 1), dim3(sh.block_x, sh.block_y), static_cast<size_t>(sh.smem_bytes), cuda_stream_>>>(
    pair_descs,
    reinterpret_cast<const int32_t*>(cuda_gradients_),
    data,
    compact_column_hist_offsets_.RawData(),
    num_compact_columns_,
    cuda_row_data_->cuda_partition_hist_offsets(),
    compact_feature_partition_column_index_offsets_.RawData(),
    compact_packed_partition_byte_offsets_.RawData(),
    num_data_,
    static_cast<data_size_t>(min_data_in_leaf_),
    min_sum_hessian_in_leaf_,
    sh.per_pair_min_grid,
    sh.per_pair_min_rows,
    BatchConstructSaturationFloor(),
    use_quantized_grad_ ? num_grad_quant_bins_ : 0,
    sh.whole_row_partitions,
    sh.whole_row_joint,
    static_cast<data_size_t>(0),
    PairJointBlockPrefix{},
    device_level_state_);
  CUDASUCCESS_OR_FATAL(cudaPeekAtLastError());  // a new launch shape: its launch errors surface
}

int CUDAHistogramConstructor::DeviceLevelConstructTargetBlocks() {
  static const int target = []() {
    int device = 0;
    int num_sms = 0;
    CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
    CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    return std::max(1, num_sms) * 4;
  }();
  return target;
}

void CUDAHistogramConstructor::LaunchFixSubtractDiscretizedMaskedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const CUDAHybridGraphLoopStateOpt gstate) {
  dim3 grid_dim(FixSubtractMaskedGridX(), num_pairs);
  // cuda_plan key pdl_level_chain: a programmatic dependent of the construct launched just before on this stream
  // (host launches only: a graph capture keeps the ordinary node)
  LaunchLevelChainKernel(gstate == nullptr && PdlLevelChainEnabled(), FixSubtractDiscretizedMaskedBatchedKernel,
    grid_dim, dim3(fix_subtract_masked_block_), 0, cuda_stream_,
    num_total_bin_,
    FixSubtractMaskedSubtractBlocks(),
    cuda_feature_num_bins_.RawData(),
    cuda_feature_hist_offsets_.RawData(),
    cuda_feature_most_freq_bins_.RawData(),
    cuda_need_fix_histogram_features_.RawData(),
    cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
    cuda_fix_mfb_mask_.RawDataReadOnly(),
    pair_descs,
    hist_buffer_for_num_bit_change_.RawData(),
    any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
    any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
    gstate, device_level_state_);
  CUDASUCCESS_OR_FATAL(cudaPeekAtLastError());  // a new launch shape: its launch errors surface
}

void CUDAHistogramConstructor::LaunchFixSubtractHistogramSmallLeafBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // the subtract role is elementwise and block-size invariant; the fix role
  // stages one bin entry per thread (FixHistogramGroupInner)
  static_assert(2 * kFixGroupChunk * kFixGroupFeatures == FIX_HISTOGRAM_BLOCK_SIZE,
                "FixHistogramGroupInner stages one element per thread of the block");
  const int num_subtract_blocks = FixSubtractSmallLeafSubtractBlocks();
  dim3 grid_dim(FixSubtractSmallLeafGridX(), num_pairs);
  FixSubtractHistogramSmallLeafBatchedKernel<<<grid_dim, FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
    num_total_bin_,
    num_subtract_blocks,
    cuda_feature_num_bins_.RawData(),
    cuda_feature_hist_offsets_.RawData(),
    cuda_feature_most_freq_bins_.RawData(),
    cuda_need_fix_histogram_features_.RawData(),
    static_cast<int>(need_fix_histogram_features_.size()),
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
  count_scratch_dirty_ = true;  // cuda_plan key forest_count_hist: the det slab is also the count scratch
  det_kernel<<<det_grid, det_block, 0, cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      RowMajorBin<BIN_TYPE>(),
      cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      det_packed_4bit ? cuda_row_data_->cuda_packed_partition_byte_offsets() : nullptr,
      column_used_bytree_device(),
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
    if (nq_fixed_kernel_launched_) {
      // nonquant_deterministic: the construct above accumulated in fixed point (LaunchConstructHistogramBatchedKernel);
      // its in-place convert is a pair-on-y node (blocks over the items on x, the live-pair guard inside)
      const int convert_blocks_x = LaunchConvertFixedHistBatched(pair_descs, 1, gstate);
      if (!AppendCapturedNode(cuda_stream_, nodes)) return;
      roles->push_back(kHybridGraphNodeSearchPairY);
      role_static_x->push_back(convert_blocks_x);
    }
  }
  if (use_quantized_grad_) {
    // mirror of LaunchSubtractHistogramBatchedKernel's quantized branch, one
    // collected node per launch; the copy node is always captured (per-pair
    // device-derived bit widths gate it inside the kernel)
    const int num_subtract_threads = num_total_bin_;
    const int num_subtract_blocks =
      (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    dim3 subtract_grid(num_subtract_blocks, 1);
    if (FixSubtractMaskedEnabled()) {
      LaunchFixSubtractDiscretizedMaskedKernel(pair_descs, 1, gstate);
      if (!AppendCapturedNode(cuda_stream_, nodes)) return;
      roles->push_back(kHybridGraphNodeSearchPairY);
      role_static_x->push_back(FixSubtractMaskedGridX());
    } else {
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
        gstate, nullptr);
      if (!AppendCapturedNode(cuda_stream_, nodes)) return;
      roles->push_back(kHybridGraphNodeSearchPairY);
      role_static_x->push_back(static_cast<int>(need_fix_histogram_features_.size()));
    }
    SubtractHistogramDiscretizedBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      pair_descs,
      hist_buffer_for_num_bit_change_.RawData(),
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
      gstate, nullptr);
    if (!AppendCapturedNode(cuda_stream_, nodes)) return;
    roles->push_back(kHybridGraphNodeSearchPairY);
    role_static_x->push_back(num_subtract_blocks);
    }
    CopyChangedNumBitHistogramBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
      num_total_bin_,
      pair_descs,
      hist_buffer_for_num_bit_change_.RawData(),
      any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
      gstate, nullptr);
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
    role_static_x->push_back(FixSubtractSmallLeafGridX());
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
    block_dim_y_local = std::max(1, BatchedConstructThreads() / block_dim_x);
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
    // cuda_plan key device_levels: a device-driven level's pairs are guarded by the device's live count, and its
    // bit-change copies are gated per pair inside the copy kernel (the host does not know whether any applies)
    const CUDADeviceLevelState* dls = device_level_state_;
    const bool bit_change_copy = any_pair_needs_bit_change_copy || dls != nullptr;
    const int num_subtract_threads = num_total_bin_;
    const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
    dim3 subtract_grid(num_subtract_blocks, num_pairs);
    bool fused_fix_subtract = false;
    // cuda_plan key device_tree_live_grid: a complete device tree's level (worst-case num_pairs, live count on the
    // device) runs the fix/subtract and the bit-change copy over its live pairs with a bounded grid, where the
    // worst-case launch exceeds that grid's block budget (narrower levels keep the batched kernels unchanged)
    const bool live_tree = device_level_whole_tree_ && gstate == nullptr;
    const char* fix_route = "batched-masked";  // diag only: which kernels this level enqueued
    int fix_grid_x = FixSubtractMaskedGridX();
    if (gstate == nullptr && any_feature_unused_bytree_ && used_feature_info_ok_ &&
        FalcataPlan::Get().fix_subtract_fused) {
      // cuda_plan key fix_subtract_fused: fix + subtract of the sampled features, one block per pair
      const int threads = std::max(32, (num_used_feature_info_ + 31) / 32 * 32);  // <= 512 (host gate)
      // cuda_plan key pdl_level_chain: a programmatic dependent of the construct launched just before
      LaunchLevelChainKernel(PdlLevelChainEnabled(), FixSubtractUsedFeaturesBatchedKernel, dim3(num_pairs),
        dim3(threads), 0, cuda_stream_,
        cuda_used_feature_info_.RawDataReadOnly(),
        num_used_feature_info_,
        num_total_bin_,
        pair_descs,
        hist_buffer_for_num_bit_change_.RawData(),
        dls);
      fused_fix_subtract = true;  // the kernel also did CopyChangedNumBitHistogramBatchedKernel's copies
      fix_route = "fused";
      fix_grid_x = 1;
    } else if (FixSubtractMaskedEnabled() && live_tree && LiveGridPays(FixSubtractMaskedGridX(), num_pairs)) {
      fix_route = "live-masked";
      const dim3 grid(FixSubtractMaskedGridX(), LiveGridRows(FixSubtractMaskedGridX(), num_pairs));
      LaunchLevelChainKernel(PdlLevelChainEnabled(), FixSubtractDiscretizedMaskedLiveKernel,
        grid, dim3(fix_subtract_masked_block_), 0, cuda_stream_,
        num_total_bin_,
        FixSubtractMaskedSubtractBlocks(),
        cuda_feature_num_bins_.RawData(),
        cuda_feature_hist_offsets_.RawData(),
        cuda_feature_most_freq_bins_.RawData(),
        cuda_need_fix_histogram_features_.RawData(),
        cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
        cuda_fix_mfb_mask_.RawDataReadOnly(),
        pair_descs,
        hist_buffer_for_num_bit_change_.RawData(),
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
        any_feature_unused_bytree_ ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
        dls);
    } else if (FixSubtractMaskedEnabled()) {
      LaunchFixSubtractDiscretizedMaskedKernel(pair_descs, num_pairs, gstate);
    } else {
      fix_grid_x = num_subtract_blocks;
      fix_route = "batched-subtract";
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
          gstate, dls);
      }
      if (live_tree && LiveGridPays(num_subtract_blocks, num_pairs)) {
        fix_route = "live-subtract";
        const dim3 grid(num_subtract_blocks, LiveGridRows(num_subtract_blocks, num_pairs));
        SubtractHistogramDiscretizedLiveKernel<<<grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
          num_total_bin_,
          pair_descs,
          hist_buffer_for_num_bit_change_.RawData(),
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
          dls);
      } else {
        SubtractHistogramDiscretizedBatchedKernel<<<subtract_grid, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
          num_total_bin_,
          pair_descs,
          hist_buffer_for_num_bit_change_.RawData(),
          any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
          gstate, dls);
      }
    }
    const char* copy_route = fused_fix_subtract ? "fused" : "batched";
    if (live_tree && !fused_fix_subtract && LiveGridPays(num_subtract_blocks, num_pairs)) {
      copy_route = "live";
      // the bit-change copies over the live pairs (a level without any: one bounded launch reading descriptors)
      const dim3 grid(num_subtract_blocks, LiveGridRows(num_subtract_blocks, num_pairs));
      LaunchLevelChainKernel(PdlLevelChainEnabled(), CopyChangedNumBitHistogramLiveKernel,
        grid, dim3(SUBTRACT_BLOCK_SIZE), 0, cuda_stream_,
        num_total_bin_,
        pair_descs,
        hist_buffer_for_num_bit_change_.RawData(),
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
        dls);
    } else if ((bit_change_copy && !fused_fix_subtract) || gstate != nullptr) {
      // graph capture always includes the copy node (the device-derived per-pair
      // bit widths gate it); the host path launches it only on bit-change levels
      // cuda_plan key pdl_level_chain: a programmatic dependent of the subtract launched just before (host launches)
      LaunchLevelChainKernel(gstate == nullptr && PdlLevelChainEnabled(), CopyChangedNumBitHistogramBatchedKernel,
        subtract_grid, dim3(SUBTRACT_BLOCK_SIZE), 0, cuda_stream_,
        num_total_bin_,
        pair_descs,
        hist_buffer_for_num_bit_change_.RawData(),
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
        gstate, dls);
    }
    if (live_tree && FalcataDebug().diag) {
      // once per enqueued level (direct enqueues and captures; a graph replay enqueues nothing): the routes taken and
      // the inputs of the live-grid decision (live iff grid_x * pairs > budget)
      fprintf(stderr, "[hybrid-diag] device_tree_live_grid: pairs=%d budget=%" PRId64 " fix_subtract=%s grid_x=%d "
              "copy=%s copy_grid_x=%d\n", num_pairs, LiveGridBudget(), fix_route, fix_grid_x, copy_route,
              num_subtract_blocks);
    }
  }
}

// Called from the dispatch trees in cuda_histogram_constructor_bin*.cu.
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseDeterministic<uint8_t>(const CUDALeafSplitsStruct*, const data_size_t, const int, const int, const DetDenseSource*);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseDeterministic<uint16_t>(const CUDALeafSplitsStruct*, const data_size_t, const int, const int, const DetDenseSource*);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseDeterministic<uint32_t>(const CUDALeafSplitsStruct*, const data_size_t, const int, const int, const DetDenseSource*);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseCount<uint8_t>(const CUDALeafSplitsStruct*, const data_size_t, const int, const int, const DetDenseSource*);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseCount<uint16_t>(const CUDALeafSplitsStruct*, const data_size_t, const int, const int, const DetDenseSource*);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseCount<uint32_t>(const CUDALeafSplitsStruct*, const data_size_t, const int, const int, const DetDenseSource*);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseBatchedDeterministic<uint8_t>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const int, const int);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseBatchedDeterministic<uint16_t>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const int, const int);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseBatchedDeterministic<uint32_t>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const int, const int);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseBatchedCount<uint8_t>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const int, const int);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseBatchedCount<uint16_t>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const int, const int);
template void CUDAHistogramConstructor::LaunchConstructHistogramDenseBatchedCount<uint32_t>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const int, const int);

}  // namespace Falcata

#endif  // USE_CUDA
