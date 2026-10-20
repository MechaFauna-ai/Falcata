/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_KERNELS_CUH_
#define FALCATA_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_KERNELS_CUH_

#ifdef USE_CUDA


#include "cuda_histogram_constructor.hpp"
#include "cuda_device_levels.hpp"

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_rocm_interop.h>

#include <algorithm>
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

namespace Falcata {

__global__ void MetaBatchScatterKernel(const uint8_t* __restrict__ staged, const MetaBatchSegments segments);

void LaunchMetaBatchScatter(const uint8_t* staged, const MetaBatchSegments& segments);

__global__ void CUDAFillCompactDataKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ compact_data,
  const size_t* __restrict__ slot_src_byte,
  const int* __restrict__ slot_src_stride,
  const size_t* __restrict__ slot_dst_byte,
  const int* __restrict__ slot_dst_stride,
  const int total_compact_cols,
  const data_size_t num_data,
  uint8_t* __restrict__ colmajor_out);

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
    bool src_is_4bit);

__global__ void InterleaveGradHessKernel(
  const score_t* __restrict__ gradients,
  const score_t* __restrict__ hessians,
  float2* __restrict__ gradients_hessians,
  const data_size_t num_data,
  uint32_t* absmax_bits);

void LaunchInterleaveGradHessKernel(
  const score_t* gradients,
  const score_t* hessians,
  float2* gradients_hessians,
  data_size_t num_data,
  uint32_t* absmax_bits);

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
  uint8_t* colmajor_out);

__global__ void CUDAFillCompactData4BitKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ compact_data,
  const size_t* __restrict__ bs_src_nib0,
  const size_t* __restrict__ bs_src_nib1,
  const int* __restrict__ bs_src_stride_nib,
  const size_t* __restrict__ bs_dst_byte,
  const int* __restrict__ bs_dst_stride,
  const int total_byte_slots,
  const data_size_t num_data);

#if !defined(__HIP_PLATFORM_AMD__)
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
  const data_size_t num_data);

constexpr size_t kFusedRootRunSlack = 32;

size_t FusedRootHistFillSharedBytes(int num_slots, int hist_entries);

__host__ __device__ constexpr int FusedRootHistSlotStride(int num_slots) {
  return (num_slots + 31) & ~31;
}

constexpr int kFusedRootThreads = 512;

__device__ __forceinline__ uint32_t Code5Dot4(const uint32_t a, const uint32_t b, const uint32_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  uint32_t sum = c;
#pragma unroll
  for (int i = 0; i < 4; ++i) sum += ((a >> (8 * i)) & 0xffu) * ((b >> (8 * i)) & 0xffu);
  return sum;
#endif
}

template <bool HI_BYTES>
__device__ __forceinline__ uint32_t Code5Dot2(const uint32_t a, const uint32_t b, const uint32_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return HI_BYTES ? __dp2a_hi(a, b, c) : __dp2a_lo(a, b, c);
#else
  const uint32_t bb = HI_BYTES ? b >> 16 : b;
  return c + (a & 0xffffu) * (bb & 0xffu) + (a >> 16) * ((bb >> 8) & 0xffu);
#endif
}

__device__ __forceinline__ uint32_t Code5Funnel(const uint32_t lo, const uint32_t hi, const uint32_t shift) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 320
  return __funnelshift_r(lo, hi, shift);
#else
  return shift == 0 ? lo : ((lo >> shift) | (hi << (32u - shift)));
#endif
}

template <int kCodesPerWord, int kCodeShift>
__device__ __forceinline__ uint32_t Code5BitsPackWord(const uint8_t* tile_row, const uint8_t* __restrict__ code_table,
                                                      const int num_slots, const int k) {
  constexpr uint32_t kCodeMask = (1u << kCodeShift) - 1u;
  uint32_t w = 0;
#pragma unroll
  for (int j = 0; j < kCodesPerWord; ++j) {
    const int slot = kCodesPerWord * k + j;
    if (slot < num_slots) {
      const uint32_t b = tile_row[slot];
      const uint32_t e = __ldg(code_table + slot);
      const uint32_t code = (b & 0xfu) * (e & 31u) + ((e & 32u) != 0 ? (b >> 4) : 0u);
      w |= (code & kCodeMask) << (kCodeShift * j);
    }
  }
  return w;
}

__device__ __forceinline__ uint32_t Code5RadixPackWord(const uint8_t* tile_row, const uint8_t* __restrict__ code_table,
                                                       const int num_slots, const int k) {
  uint32_t c[6];
  uint32_t n[6];
#pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int slot = 6 * k + j;
    c[j] = 0;
    n[j] = 1;
    if (slot < num_slots) {
      const uint32_t b = tile_row[slot];
      const uint32_t en = __ldg(reinterpret_cast<const uint16_t*>(code_table) + slot);
      const uint32_t e = en & 0xffu;
      c[j] = (b & 0xfu) * (e & 31u) + ((e & 32u) != 0 ? (b >> 4) : 0u);
      n[j] = en >> 8;
    }
  }
  const uint32_t lo_half = c[0] + n[0] * (c[1] + n[1] * c[2]);
  const uint32_t hi_half = c[3] + n[3] * (c[4] + n[4] * c[5]);
  return lo_half | (hi_half << 16);
}

template <int CODE_BITS, int B0>
__device__ __forceinline__ void Code5WideCoefs(const uint8_t* __restrict__ code_table, const int num_slots,
                                               const int slot0, uint32_t (&lo_coef)[4], uint32_t (&hi_coef)[4]) {
  constexpr bool kRadix = CODE_BITS == kCodeRadix;
  static_assert(!kRadix || (B0 & 1) == 0, "a radix word starts on an even byte");
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    lo_coef[g] = 0;
    hi_coef[g] = 0;
  }
  uint32_t place = 1;
#pragma unroll
  for (int j = 0; j < Code5CodesPerWord(CODE_BITS); ++j) {
    // slot j's group and its 16-bit lane in the group's coefficient pair (the byte lane of the nibble word)
    const int g = kRadix ? (j < 2 ? 0 : j == 2 ? 1 : j == 3 ? 2 : 3) : ((B0 & 1) + j) / 2;
    const int lane16 = (B0 + j) & 1;
    if (kRadix ? j == 3 : lane16 == 0) {
      place = 1;
    }
    const int slot = slot0 + j;
    uint32_t count = kRadix ? 1u : 64u;
    if (slot < num_slots) {
      const uint32_t e = kRadix ?
        static_cast<uint32_t>(__ldg(reinterpret_cast<const uint16_t*>(code_table) + slot)) :
        static_cast<uint32_t>(__ldg(code_table + slot));
      lo_coef[g] |= (((e & 31u) * place) & 0xffffu) << (16 * lane16);
      hi_coef[g] |= ((e & 32u) != 0 ? place : 0u) << (16 * lane16);
      if (kRadix) {
        count = e >> 8;
      }
    }
    place *= count;
  }
}

template <int P>
__device__ __forceinline__ uint32_t Code5WideGroupDot(const uint32_t* lx, const uint32_t* hx, const uint32_t lc,
                                                      const uint32_t hc, const uint32_t c) {
  return Code5Dot2<((P >> 1) & 1) != 0>(hc, hx[P >> 2], Code5Dot2<((P >> 1) & 1) != 0>(lc, lx[P >> 2], c));
}

template <int CODE_BITS, int B0>
__device__ __forceinline__ uint32_t Code5WideWordDot(const uint32_t* lx, const uint32_t* hx,
                                                     const uint32_t (&lc)[4], const uint32_t (&hc)[4]) {
  if (CODE_BITS == kCodeRadix) {
    const uint32_t s012 = Code5WideGroupDot<B0 + 2>(lx, hx, lc[1], hc[1], Code5WideGroupDot<B0>(lx, hx, lc[0], hc[0], 0u));
    const uint32_t s345 = Code5WideGroupDot<B0 + 4>(lx, hx, lc[3], hc[3],
                                                    Code5WideGroupDot<B0 + 2>(lx, hx, lc[2], hc[2], 0u));
    return s012 | (s345 << 16);
  }
  constexpr int p0 = B0 & ~1;
  const uint32_t s0 = Code5WideGroupDot<p0>(lx, hx, lc[0], hc[0], 0u);
  const uint32_t s1 = Code5WideGroupDot<p0 + 2>(lx, hx, lc[1], hc[1], 0u);
  const uint32_t s2 = Code5WideGroupDot<p0 + 4>(lx, hx, lc[2], hc[2], 0u);
  return s0 | (s1 << (6 * (p0 + 2 - B0))) | (s2 << (6 * (p0 + 4 - B0)));
}

template <bool SLOT_MAJOR, int CODE_BITS>
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
  hist_t* __restrict__ root_hist_scratch,
  uint32_t* __restrict__ code_view,
  const uint8_t* __restrict__ code_table,
  const int code_row_words,
  const bool code_pack2,
  const bool code_evict_first,
  const bool code_wide_regs) {
  constexpr bool CODE5 = CODE_BITS > 0;
  constexpr int kCodesPerWord = CODE_BITS > 0 ? Code5CodesPerWord(CODE_BITS) : 1;
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

    // write the tile exactly as CUDAFillCompactData4BitTiledKernel does (CODE5: the code view replaces it; the host
    // writes the nibble view later if a reader needs it)
    if (one_run && !CODE5) {
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
    for (int s = 0; !CODE5 && !one_run && s < num_slots; s += s_dst_stride[s]) {
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

    if (CODE_BITS == 5 && code_pack2 && code_row_words / 2 <= kFusedRootThreads) {
      // cuda_plan key pair_code5_pack2: the block's first whole rows' worth of threads: thread tid always packs words
      // 2m and 2m + 1 (m = tid % (words / 2)) of its rows, with their twelve slots' multipliers in registers, and
      // stores them as one 8-byte pair (code_row_words is a multiple of 8); the remaining threads (fewer than a row's)
      // skip the pack
      uint32_t* code_out = code_view + static_cast<size_t>(row_start) * static_cast<size_t>(code_row_words);
      const int half_words = code_row_words / 2;
      const int m = tid % half_words;
      // slot 12m + 6w + j's code c_j = lo * span(hi) + hi (< 32) at bits 5j of word 2m + w: codes 2p and 2p + 1 as
      // one dot product of the bytes' low nibbles with (span(hi)_2p, 32 span(hi)_2p+1) plus their high nibbles with
      // (has_hi, 32 has_hi), = c_2p + 32 c_2p+1, at the pair's byte lanes of the 32-bit source word holding it (bytes
      // 6w + 2p and 6w + 2p + 1 of the twelve share one: 6w + 2p is even); the host keeps 32 span(hi) of an odd slot
      // below 256; all 0 for a slot past num_slots, whose byte is not used
      uint32_t lo_mul[6];
      uint32_t hi_mul[6];
#pragma unroll
      for (int wp = 0; wp < 6; ++wp) {
        lo_mul[wp] = 0;
        hi_mul[wp] = 0;
        const int b0 = 6 * (wp / 3) + 2 * (wp % 3);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int slot = 12 * m + b0 + q;
          const uint32_t e = slot < num_slots ? __ldg(code_table + slot) : 0u;
          const int byte_lane = (b0 & 3) + q;
          lo_mul[wp] |= ((e & 31u) << (5 * q)) << (8 * byte_lane);
          hi_mul[wp] |= ((e & 32u) != 0 ? (1u << (5 * q)) : 0u) << (8 * byte_lane);
        }
      }
      const int rows_per_pass = kFusedRootThreads / half_words;
      const int tile_row_offset = static_cast<int>(tile_rows - fill_smem);
      for (int r = tid < rows_per_pass * half_words ? tid / half_words : rows_valid; r < rows_valid;
           r += rows_per_pass) {
        uint2 w = make_uint2(0u, 0u);
        if (12 * m < num_slots) {
          // the twelve bytes from the four aligned 32-bit words holding them (a row starts on any byte when num_slots
          // is odd); the bytes past the row's last slot are read (up to 15, within the block's shared slack) but
          // multiplied by 0
          // (addressed from the shared array's 16-byte aligned base, so the loads stay shared-memory loads)
          const int o = tile_row_offset + r * tile_stride + 12 * m;
          const uint32_t* a = reinterpret_cast<const uint32_t*>(fill_smem) + (o >> 2);
          const uint32_t shift = static_cast<uint32_t>(o & 3) * 8u;
          const uint32_t a1 = a[1];
          const uint32_t a2 = a[2];
          uint32_t lx[3];
          uint32_t hx[3];
          lx[0] = Code5Funnel(a[0], a1, shift);
          lx[1] = Code5Funnel(a1, a2, shift);
          lx[2] = Code5Funnel(a2, a[3], shift);
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            hx[i] = (lx[i] >> 4) & 0x0f0f0f0fu;
            lx[i] &= 0x0f0f0f0fu;
          }
          uint32_t c[6];
#pragma unroll
          for (int wp = 0; wp < 6; ++wp) {
            const int src = (6 * (wp / 3) + 2 * (wp % 3)) >> 2;
            c[wp] = Code5Dot4(hx[src], hi_mul[wp], Code5Dot4(lx[src], lo_mul[wp], 0u));
          }
          w.x = c[0] | (c[1] << 10) | (c[2] << 20);
          w.y = c[3] | (c[4] << 10) | (c[5] << 20);
        }
        // cuda_plan key pair_code5_l2_store: a plain store, not evict-first, keeps the view's last written rows in L2
        // for the first level's construct
        uint2* const dst = reinterpret_cast<uint2*>(code_out + r * code_row_words + 2 * m);
        if (code_evict_first) {
          __stcs(dst, w);
        } else {
          *dst = w;
        }
      }
    } else if (CODE_BITS == 5 && kFusedRootThreads % code_row_words == 0) {
      // one word per thread where the block's threads cover whole rows: thread tid always packs word
      // tid % code_row_words, with its six slots' multipliers in registers
      uint32_t* code_out = code_view + static_cast<size_t>(row_start) * static_cast<size_t>(code_row_words);
      const int k = tid % code_row_words;
      // slot 6k + j's code c_j = lo * span(hi) + hi (< 32) at bits 5j: codes 2p and 2p + 1 as one dot product of
      // the bytes' low nibbles with (span(hi)_2p, 32 span(hi)_2p+1) plus their high nibbles with (has_hi, 32 has_hi),
      // = c_2p + 32 c_2p+1, at the pair's byte lanes of the word read (the host keeps 32 span(hi) of an odd slot below
      // 256); all 0 for a slot past num_slots, whose byte is not used
      uint32_t lo_mul[3];
      uint32_t hi_mul[3];
#pragma unroll
      for (int p = 0; p < 3; ++p) {
        lo_mul[p] = 0;
        hi_mul[p] = 0;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int slot = 6 * k + 2 * p + q;
          const uint32_t e = slot < num_slots ? __ldg(code_table + slot) : 0u;
          const int byte_lane = (p == 1 ? 2 : 0) + q;
          lo_mul[p] |= ((e & 31u) << (5 * q)) << (8 * byte_lane);
          hi_mul[p] |= ((e & 32u) != 0 ? (1u << (5 * q)) : 0u) << (8 * byte_lane);
        }
      }
      const int rows_per_pass = kFusedRootThreads / code_row_words;
      for (int r = tid / code_row_words; r < rows_valid; r += rows_per_pass) {
        uint32_t w = 0;
        if (6 * k < num_slots) {
          // the word's six bytes from the three aligned 32-bit words holding them (a row starts on any byte when
          // num_slots is odd); the bytes past the row's last slot are read (within the block's shared slack) but
          // multiplied by 0
          const uintptr_t o = reinterpret_cast<uintptr_t>(tile_rows + r * tile_stride + 6 * k);
          const uint32_t* a = reinterpret_cast<const uint32_t*>(o & ~static_cast<uintptr_t>(3));
          const uint32_t shift = static_cast<uint32_t>(o & 3) * 8u;
          const uint32_t a1 = a[1];
          const uint32_t x0 = Code5Funnel(a[0], a1, shift);
          const uint32_t x1 = Code5Funnel(a1, a[2], shift);
          const uint32_t l0 = x0 & 0x0f0f0f0fu;
          const uint32_t h0 = (x0 >> 4) & 0x0f0f0f0fu;
          const uint32_t l1 = x1 & 0x0f0fu;
          const uint32_t h1 = (x1 >> 4) & 0x0f0fu;
          const uint32_t c01 = Code5Dot4(h0, hi_mul[0], Code5Dot4(l0, lo_mul[0], 0u));
          const uint32_t c23 = Code5Dot4(h0, hi_mul[1], Code5Dot4(l0, lo_mul[1], 0u));
          const uint32_t c45 = Code5Dot4(h1, hi_mul[2], Code5Dot4(l1, lo_mul[2], 0u));
          w = c01 | (c23 << 10) | (c45 << 20);
        }
        if (code_evict_first) {
          __stcs(code_out + r * code_row_words + k, w);
        } else {
          code_out[r * code_row_words + k] = w;
        }
      }
    } else if ((CODE_BITS == 6 || CODE_BITS == kCodeRadix) && code_wide_regs && code_pack2 &&
               code_row_words / 2 <= kFusedRootThreads) {
      // cuda_plan keys pair_code_wide_regs, pair_code5_pack2: two words per thread for the 6-bit and the mixed-radix
      // code: thread tid always packs words 2m and 2m + 1 (m = tid % (words / 2)) of its rows (the block's first whole
      // rows' worth of threads; the rest skip the pack), their ten or twelve slots from the four aligned shared words
      // holding them, one 8-byte store (code_row_words is a multiple of kCode5RowWordAlign, even). Either word is
      // linear in its slots' nibbles: slot j's cell lo * span(hi) + hi times the slot's place value. The word is
      // summed in groups of 16-bit coefficients, two slots per dp2a (the byte lanes 0-1 or 2-3 of the nibble words),
      // each group shifted into place:
      //   6 bits:      slots (0, 1) at bit 0, (2, 3) at bit 12, 4 at bit 24, coefficients x (1, 64) (a word starting
      //                on an odd byte, the second of a pair: 0 at bit 0, (1, 2) at bit 6, (3, 4) at bit 18)
      //   mixed radix: slots (0, 1) and 2 at bit 0 (x 1, n0, n0 n1), 3 and (4, 5) at bit 16 (x 1, n3, n3 n4)
      // a slot's coefficients are span(hi) x place (its low nibble) and has_hi x place (its high nibble), from the
      // code table once per thread. Every cell is below its slot's count (and below 64) and every radix half below
      // 2^16, so no group carries into the next: the generic pack's word. A radix low-nibble coefficient span(hi) x
      // place is at most 2^16 (span(lo) x span(hi) x place <= the half's cell product), equal only where span(lo) is 1,
      // whose low nibble is always 0: kept mod 2^16, its product is still exact. A slot past the row has coefficients
      // 0 (cell 0, count 1); its byte is read (within the block's 16-byte shared slack) but not used. With
      // pair_code5_pack2 off the generic pack below writes the wide views
      const int tile_row_offset = static_cast<int>(tile_rows - fill_smem);
      uint32_t* code_out = code_view + static_cast<size_t>(row_start) * static_cast<size_t>(code_row_words);
      constexpr int kPairSlots = 2 * kCodesPerWord;
      const int half_words = code_row_words / 2;
      const int m = tid % half_words;
      uint32_t lc0[4], hc0[4], lc1[4], hc1[4];
      Code5WideCoefs<CODE_BITS, 0>(code_table, num_slots, kPairSlots * m, lc0, hc0);
      Code5WideCoefs<CODE_BITS, kCodesPerWord>(code_table, num_slots, kPairSlots * m + kCodesPerWord, lc1, hc1);
      const int rows_per_pass = kFusedRootThreads / half_words;
      for (int r = tid < rows_per_pass * half_words ? tid / half_words : rows_valid; r < rows_valid;
           r += rows_per_pass) {
        uint2 w = make_uint2(0u, 0u);
        if (kPairSlots * m < num_slots) {
          const int o = tile_row_offset + r * tile_stride + kPairSlots * m;
          const uint32_t* a = reinterpret_cast<const uint32_t*>(fill_smem) + (o >> 2);
          const uint32_t shift = static_cast<uint32_t>(o & 3) * 8u;
          const uint32_t a1 = a[1];
          const uint32_t a2 = a[2];
          uint32_t lx[3];
          uint32_t hx[3];
          lx[0] = Code5Funnel(a[0], a1, shift);
          lx[1] = Code5Funnel(a1, a2, shift);
          lx[2] = Code5Funnel(a2, a[3], shift);
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            hx[i] = (lx[i] >> 4) & 0x0f0f0f0fu;
            lx[i] &= 0x0f0f0f0fu;
          }
          w.x = Code5WideWordDot<CODE_BITS, 0>(lx, hx, lc0, hc0);
          w.y = Code5WideWordDot<CODE_BITS, kCodesPerWord>(lx, hx, lc1, hc1);
        }
        uint2* const dst = reinterpret_cast<uint2*>(code_out + r * code_row_words + 2 * m);
        if (code_evict_first) {
          __stcs(dst, w);
        } else {
          *dst = w;
        }
      }
    } else if (CODE5) {
      // the tile's rows as code words: consecutive threads write consecutive words of the tile's contiguous run
      // (kCodesPerWord codes of CODE_BITS bits per word; the only pack of the 6-bit code, cuda_plan key
      // pair_code_wide)
      uint32_t* code_out = code_view + static_cast<size_t>(row_start) * static_cast<size_t>(code_row_words);
      const int code_words = rows_valid * code_row_words;
      constexpr int kCodeShift = CODE_BITS > 0 && CODE_BITS < 32 ? CODE_BITS : 1;  // (kCodeRadix: unused)
      for (int t = tid; t < code_words; t += kFusedRootThreads) {
        const int r = t / code_row_words;
        const int k = t - r * code_row_words;
        const uint8_t* tile_row = tile_rows + r * tile_stride;
        // (cuda_plan key pair_code_radix: six cells per word in mixed radix)
        const uint32_t w = CODE_BITS == kCodeRadix ?
          Code5RadixPackWord(tile_row, code_table, num_slots, k) :
          Code5BitsPackWord<kCodesPerWord, kCodeShift>(tile_row, code_table, num_slots, k);
        if (code_evict_first) {
          __stcs(code_out + t, w);
        } else {
          code_out[t] = w;
        }
      }
    }

    // root histogram: one warp per row, lanes across the row's byte slots (distinct columns)
    for (int r = warp; r < rows_valid; r += kWarps) {
      const int32_t grad_hess = s_grad[r];
      const uint8_t* tile_row = tile_rows + r * tile_stride;
      if (SLOT_MAJOR) {
        // a slot without a high column stages nibble 0 there; the flush skips that cell. A lane's slots
        // s = lane + 32 i have the same nibble-0 cells (s_hist + s, + hist_half for the high nibble) in every row, so
        // a slot's cells are those plus nibble * hist_stride cells: one multiply-add per cell. A lane has at most
        // kFill4BitTiledMaxSlots / 32 slots: the loop is unrolled whole (no remainder blocks for a lane's few slots)
        unsigned char* lo_cell = reinterpret_cast<unsigned char*>(s_hist + lane);
        unsigned char* hi_cell = reinterpret_cast<unsigned char*>(s_hist + hist_half + lane);
        const uint32_t cell_stride_bytes = static_cast<uint32_t>(hist_stride) * sizeof(int32_t);
#pragma unroll
        for (int i = 0; i < kFill4BitTiledMaxSlots / 32; ++i) {
          const int s = lane + 32 * i;
          if (s >= num_slots) break;
          const uint32_t b = tile_row[s];
          atomicAdd_block(reinterpret_cast<int32_t*>(lo_cell + 32 * sizeof(int32_t) * i + (b & 0xf) * cell_stride_bytes),
                          grad_hess);
          atomicAdd_block(reinterpret_cast<int32_t*>(hi_cell + 32 * sizeof(int32_t) * i + (b >> 4) * cell_stride_bytes),
                          grad_hess);
        }
      } else {
        for (int s = lane; s < num_slots; s += 32) {
          const uint32_t b = tile_row[s];
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
DeviceLaunchLimits CurrentDeviceLaunchLimits();

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
  bool* code_view_written);

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
  bool tiled);

__global__ void ClearColMajorPadNibblesKernel(uint8_t* colmajor, const int num_columns, const size_t column_stride,
                                              const size_t column_bytes);

void LaunchClearColMajorPadNibbles(uint8_t* colmajor, const int num_columns, const size_t column_stride,
                                   const size_t column_bytes);

__global__ void CUDATransposeToColMajorNibbleKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ colmajor_data,
  const size_t* __restrict__ col_src_nib_base,
  const int* __restrict__ col_src_stride_nib,
  const int num_columns,
  const data_size_t num_data,
  const size_t num_data_pad);

void LaunchTransposeToColMajorNibbleKernel(
  const uint8_t* src_data,
  uint8_t* colmajor_data,
  const size_t* col_src_nib_base,
  const int* col_src_stride_nib,
  int num_columns,
  data_size_t num_data,
  size_t num_data_pad);

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
  data_size_t num_data);

#define kRegHistMaxBins (8)

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

template <typename HIST_TYPE>
__device__ __forceinline__ HIST_TYPE HistAddend(const score_t v, const int /*exponent*/) {
  return static_cast<HIST_TYPE>(v);
}

template <>
__device__ __forceinline__ nq_fixed_t HistAddend<nq_fixed_t>(const score_t v, const int exponent) {
  return static_cast<nq_fixed_t>(NonquantFixedAddend(v, exponent));
}

template <typename HIST_TYPE>
__device__ __forceinline__ void HistSharedAdd(HIST_TYPE* dst, const HIST_TYPE v) {
  atomicAdd_block(dst, v);
}

template <>
__device__ __forceinline__ void HistSharedAdd<nq_fixed_t>(nq_fixed_t* dst, const nq_fixed_t v) {
  uint32_t* const words = reinterpret_cast<uint32_t*>(dst);
  const uint32_t lo = static_cast<uint32_t>(v);
  const uint32_t old = atomicAdd_block(words, lo);
  const uint32_t hi = static_cast<uint32_t>(v >> 32) + (static_cast<uint32_t>(old + lo) < old ? 1u : 0u);
  if (hi != 0u) {
    atomicAdd_block(words + 1, hi);
  }
}

template <typename HIST_TYPE>
__device__ __forceinline__ void HistFlushAdd(hist_t* dst, const HIST_TYPE v) {
  atomicAdd_system(dst, v);
}

template <>
__device__ __forceinline__ void HistFlushAdd<nq_fixed_t>(hist_t* dst, const nq_fixed_t v) {
  atomicAdd(reinterpret_cast<nq_fixed_t*>(dst), v);
}

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
  const uint8_t* bin_used = nullptr,
  const int* fixed_scale = nullptr) {
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
  // the fixed-point accumulator's power-of-two scalings (gradient, hessian) of this tree
  int scale_g = 0, scale_h = 0;
  if (std::is_same<HIST_TYPE, nq_fixed_t>::value) {
    scale_g = fixed_scale[0];
    scale_h = fixed_scale[1];
  }
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
            reg_grad[b] += HistAddend<HIST_TYPE>(grad, scale_g);
            reg_hess[b] += HistAddend<HIST_TYPE>(hess, scale_h);
          }
        }
      }
#pragma unroll
      for (int b = 0; b < kRegHistMaxBins; ++b) {
        // (0, 0) sums are no-ops; skipping them saves most of the flush atomics
        if (reg_grad[b] != 0.0f || reg_hess[b] != 0.0f) {
          HistSharedAdd<HIST_TYPE>(shared_hist_ptr + (b << 1), reg_grad[b]);
          HistSharedAdd<HIST_TYPE>(shared_hist_ptr + (b << 1) + 1, reg_hess[b]);
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
        HistSharedAdd<HIST_TYPE>(pos_ptr, HistAddend<HIST_TYPE>(grad, scale_g));
        HistSharedAdd<HIST_TYPE>(pos_ptr + 1, HistAddend<HIST_TYPE>(hess, scale_h));
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
        HistFlushAdd<HIST_TYPE>(feature_histogram_ptr + i, shared_hist[i]);
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

__global__ void ComputeBatchedConstructDimYKernel(
  const data_size_t* level_smaller_num_data,
  const int num_pairs,
  const int block_dim_y,
  const int min_grid_dim_y,
  const int min_rows_per_thread,
  const int saturation_floor_total,
  int* out_dim_y);

__global__ void ZeroHistSlotsKernel(hist_t* hist, const int* slots, const size_t slot_size);

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

template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE, bool USE_REG_BINS, bool IS_4BIT, bool USE_GH2>
__device__ __forceinline__ void ConstructHistogramDenseBatchedBody(
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
  const CUDAHybridGraphLoopStateOpt gstate,
  const int* fixed_scale) {
  // the fixed-point table (8 bytes per item over the float table's partitions: up to 96 KB) is dynamic shared
  // memory sized to the widest partition (NqFixedSmem); the float / double tables stay static
  constexpr bool kFixed = std::is_same<HIST_TYPE, nq_fixed_t>::value;
  __shared__ HIST_TYPE static_hist[kFixed ? 1 : SHARED_HIST_SIZE];
  extern __shared__ __align__(16) unsigned char nq_fixed_dyn_smem[];
  HIST_TYPE* shared_hist = kFixed ? reinterpret_cast<HIST_TYPE*>(nq_fixed_dyn_smem) : static_hist;
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
  // (never under the fixed-point accumulator: the direct body adds hist_t values)
  if (!kFixed && num_data_smaller <= small_leaf_threshold) {
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
      is_feature_used_bytree, num_data, dim_y, hist_fp32, bin_used, fixed_scale);
  } else {
    ConstructHistogramDenseInner<BIN_TYPE, HIST_TYPE, false, IS_4BIT, USE_GH2>(
      smaller_struct, shared_hist, cuda_gradients, cuda_hessians, cuda_gh, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      is_feature_used_bytree, num_data, dim_y, hist_fp32, bin_used, fixed_scale);
  }
}

template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE, bool USE_REG_BINS = false, bool IS_4BIT = false, bool USE_GH2 = false, typename std::enable_if<!std::is_same<HIST_TYPE, nq_fixed_t>::value, int>::type = 0>
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
  const CUDAHybridGraphLoopStateOpt gstate,
  const int* fixed_scale) {
  ConstructHistogramDenseBatchedBody<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, USE_REG_BINS, IS_4BIT, USE_GH2>(
    pair_descs, cuda_gradients, cuda_hessians, cuda_gh, data, column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets, packed_partition_byte_offsets, is_feature_used_bytree, num_data, min_data_in_leaf, min_sum_hessian_in_leaf, level_dim_y, level_smaller_num_data, min_grid_dim_y, min_rows_per_thread, saturation_floor_total, small_leaf_threshold, bin_used, hist_fp32, gstate, fixed_scale);
}

template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE, bool USE_REG_BINS = false, bool IS_4BIT = false, bool USE_GH2 = false, typename std::enable_if<std::is_same<HIST_TYPE, nq_fixed_t>::value, int>::type = 0>
__global__ void __launch_bounds__(2 * NUM_THREADS_PER_BLOCK) CUDAConstructHistogramDenseBatchedKernel(
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
  const CUDAHybridGraphLoopStateOpt gstate,
  const int* fixed_scale) {
  ConstructHistogramDenseBatchedBody<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, USE_REG_BINS, IS_4BIT, USE_GH2>(
    pair_descs, cuda_gradients, cuda_hessians, cuda_gh, data, column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets, packed_partition_byte_offsets, is_feature_used_bytree, num_data, min_data_in_leaf, min_sum_hessian_in_leaf, level_dim_y, level_smaller_num_data, min_grid_dim_y, min_rows_per_thread, saturation_floor_total, small_leaf_threshold, bin_used, hist_fp32, gstate, fixed_scale);
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
    // block-strided over the row's elements: blockDim.x is halved for partitions wider than
    // NUM_THREADS_PER_BLOCK (CalcConstructHistogramKernelDim), so a row can hold more elements than threads
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      for (DATA_PTR_TYPE e = threadIdx.x; e < row_size; e += blockDim.x) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + e]);
        const uint32_t pos = bin << 1;
        HIST_TYPE* pos_ptr = shared_hist + pos;
        atomicAdd_block(pos_ptr, grad);
        atomicAdd_block(pos_ptr + 1, hess);
      }
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

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

__global__ void MergeDeterministicDenseHistogramKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const hist_t* slots,
  const uint32_t* column_hist_offsets_full,
  const uint32_t slot_stride,
  const int num_rows);

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
  const CUDAHybridGraphLoopStateOpt gstate);

__global__ void MergeDeterministicHistogramKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const hist_t* tile_partials,
  const int num_total_items,
  const int num_tiles);

inline constexpr int kCountDyCap = 32;      // row lanes per block (lanes * dy <= 1024)

inline constexpr int kCountMaxLanes = 128;  // column lanes per block: wider partitions give a lane several columns

inline constexpr int kCountMaxFieldValues = 6;

template <typename BIN_TYPE, bool IS_4BIT, int MODE>
__device__ __forceinline__ void ConstructHistogramDenseCountInner(
  const CUDALeafSplitsStruct* leaf,
  const uint8_t* codes,
  const uint64_t* code_values,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* partition_hist_offsets,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used,
  const data_size_t num_data,
  uint64_t* scratch,         // this leaf's scratch row: num_total_bin words
  uint32_t* scratch_n,       // MODE 2, 3: this leaf's 32-bit counts
  const data_size_t rows_min,  // rows per tile at least (MODE 3: at most 2^field_bits - 1 as well; see the launchers)
  const int field_bits,
  const int num_values) {
  extern __shared__ uint64_t count_shared[];
  const data_size_t leaf_rows = leaf->num_data_in_leaf;
  const int dy = static_cast<int>(blockDim.y);
  // the grid's y extent was sized for the largest leaf of the launch (MODE 3: so that no tile exceeds a field)
  const int tiles = min(static_cast<int>(gridDim.y), static_cast<int>(max(static_cast<data_size_t>(1),
    (leaf_rows + rows_min - 1) / rows_min)));
  if (static_cast<int>(blockIdx.y) >= tiles) {
    return;  // block-uniform
  }
  const data_size_t tile_rows = (leaf_rows + tiles - 1) / tiles;
  if (MODE == 3 && tile_rows >= (static_cast<data_size_t>(1) << field_bits)) {
    // the launcher sized the grid so that no tile holds more rows than a field counts (a host bound on the leaf
    // sizes): a violation would carry one value's count into the next, so stop loudly instead
    __trap();
  }
  const data_size_t row_begin = static_cast<data_size_t>(blockIdx.y) * tile_rows;
  const data_size_t row_end = min(leaf_rows, row_begin + tile_rows);
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  int row_stride = num_columns_in_partition;
  const BIN_TYPE* data_ptr = data + (IS_4BIT ?
    PackedPartitionRows(packed_partition_byte_offsets, blockIdx.x, num_data, &row_stride) :
    static_cast<size_t>(partition_column_start) * num_data);
  const uint32_t partition_hist_start = partition_hist_offsets[blockIdx.x];
  const uint32_t partition_bins = partition_hist_offsets[blockIdx.x + 1] - partition_hist_start;
  uint32_t* shared_n = reinterpret_cast<uint32_t*>(count_shared + partition_bins);
  const unsigned int num_threads = blockDim.x * blockDim.y;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (uint32_t i = thread_idx; i < partition_bins; i += num_threads) {
    count_shared[i] = 0;
    if (MODE == 2) {
      shared_n[i] = 0;
    }
  }
  // MODE 3: the per-value addends of the flush
  __shared__ uint64_t field_values[kCountMaxFieldValues];
  if (MODE == 3 && thread_idx < static_cast<unsigned int>(num_values)) {
    field_values[thread_idx] = code_values[thread_idx];
  }
  __syncthreads();
  const data_size_t* data_indices = leaf->data_indices_in_leaf;
  for (data_size_t r = row_begin + static_cast<data_size_t>(threadIdx.y); r < row_end; r += dy) {
    const data_size_t data_index = data_indices[r];
    const uint32_t code = static_cast<uint32_t>(codes[data_index]);
    const uint64_t t = (MODE == 0 || MODE == 3) ? static_cast<uint64_t>(code) : __ldg(code_values + code);
    const uint64_t packed_add = MODE == 3 ? (1ull << (field_bits * code)) : ((t << 32) | 1ull);
    const BIN_TYPE* row = data_ptr + static_cast<size_t>(data_index) * row_stride;
    for (int c = static_cast<int>(threadIdx.x); c < num_columns_in_partition; c += static_cast<int>(blockDim.x)) {
      const int column_index = c + partition_column_start;
      if (is_feature_used != nullptr && !is_feature_used[column_index]) {
        continue;
      }
      const uint32_t pos = column_hist_offsets[column_index] + ReadDenseBin<IS_4BIT>(row, static_cast<unsigned int>(c));
      if (MODE == 2) {
        atomicAdd(reinterpret_cast<unsigned long long*>(count_shared + pos), static_cast<unsigned long long>(t));  // NOLINT
        atomicAdd(shared_n + pos, 1u);
      } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(count_shared + pos), static_cast<unsigned long long>(packed_add));  // NOLINT
      }
    }
  }
  __syncthreads();
  for (uint32_t i = thread_idx; i < partition_bins; i += num_threads) {
    if (MODE == 3) {
      const uint64_t v = count_shared[i];
      if (v != 0) {
        const uint64_t mask = (1ull << field_bits) - 1ull;
        uint32_t n = 0;
        uint64_t sum = 0;
        for (int j = 0; j < num_values; ++j) {
          const uint64_t cnt = (v >> (field_bits * j)) & mask;
          n += static_cast<uint32_t>(cnt);
          sum += cnt * field_values[j];
        }
        atomicAdd(scratch_n + partition_hist_start + i, n);
        if (sum != 0) {
          atomicAdd(reinterpret_cast<unsigned long long*>(scratch + partition_hist_start + i), static_cast<unsigned long long>(sum));  // NOLINT
        }
      }
    } else if (MODE == 2) {
      const uint32_t n = shared_n[i];
      if (n != 0) {
        atomicAdd(scratch_n + partition_hist_start + i, n);
        const uint64_t sum = count_shared[i];
        if (sum != 0) {
          atomicAdd(reinterpret_cast<unsigned long long*>(scratch + partition_hist_start + i), static_cast<unsigned long long>(sum));  // NOLINT
        }
      }
    } else {
      const uint64_t v = count_shared[i];
      if (v != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(scratch + partition_hist_start + i), static_cast<unsigned long long>(v));  // NOLINT
      }
    }
  }
}

template <typename BIN_TYPE, bool IS_4BIT, int MODE>
__global__ void CUDAConstructHistogramDenseCountKernel(
  const CUDALeafSplitsStruct* leaf,
  const uint8_t* codes,
  const uint64_t* code_values,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* partition_hist_offsets,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used,
  const data_size_t num_data,
  uint64_t* scratch,
  uint32_t* scratch_n,
  const data_size_t rows_min,
  const int field_bits,
  const int num_values) {
  ConstructHistogramDenseCountInner<BIN_TYPE, IS_4BIT, MODE>(
    leaf, codes, code_values, data, column_hist_offsets, partition_hist_offsets,
    feature_partition_column_index_offsets, packed_partition_byte_offsets, is_feature_used, num_data,
    scratch, scratch_n, rows_min, field_bits, num_values);
}

template <typename BIN_TYPE, bool IS_4BIT, int MODE>
__global__ void CUDAConstructHistogramDenseCountBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const uint8_t* codes,
  const uint64_t* code_values,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* partition_hist_offsets,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int8_t* is_feature_used,
  const data_size_t num_data,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  uint64_t* scratch,
  const size_t scratch_row_words,
  const uint32_t num_total_bin,
  const data_size_t rows_min,
  const int field_bits,
  const int num_values) {
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (HybridPairConstructSkipped(desc, min_data_in_leaf, min_sum_hessian_in_leaf)) {
    return;
  }
  uint64_t* row = scratch + static_cast<size_t>(blockIdx.z) * scratch_row_words;
  ConstructHistogramDenseCountInner<BIN_TYPE, IS_4BIT, MODE>(
    desc->smaller_struct, codes, code_values, data, column_hist_offsets, partition_hist_offsets,
    feature_partition_column_index_offsets, packed_partition_byte_offsets, is_feature_used, num_data,
    row, reinterpret_cast<uint32_t*>(row + num_total_bin), rows_min, field_bits, num_values);
}

template <bool WIDE>
__device__ __forceinline__ void ConvertCountBin(const uint32_t i, uint64_t* scratch, uint32_t* scratch_n,
                                                const double grad_base, const double grad_unit, const double hessian,
                                                hist_t* hist) {
  uint64_t sum;
  uint32_t n;
  if (WIDE) {
    n = scratch_n[i];
    sum = scratch[i];
    if (n != 0) {
      scratch_n[i] = 0;
      scratch[i] = 0;
    }
  } else {
    const uint64_t v = scratch[i];
    n = static_cast<uint32_t>(v & 0xffffffffull);
    sum = v >> 32;
    if (v != 0) {
      scratch[i] = 0;
    }
  }
  const double dn = static_cast<double>(n);
  hist[i << 1] = __dadd_rn(__dmul_rn(grad_base, dn), __dmul_rn(grad_unit, static_cast<double>(sum)));
  hist[(i << 1) + 1] = __dmul_rn(hessian, dn);
}

template <bool WIDE>
__global__ void ConvertCountHistogramKernel(
  const CUDALeafSplitsStruct* leaf,
  uint64_t* scratch,
  uint32_t* scratch_n,
  const uint32_t num_total_bin,
  const double grad_base,
  const double grad_unit,
  const double hessian) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_total_bin) {
    ConvertCountBin<WIDE>(i, scratch, scratch_n, grad_base, grad_unit, hessian, leaf->hist_in_leaf);
  }
}

template <bool WIDE>
__global__ void ConvertCountHistogramBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  uint64_t* scratch,
  const size_t scratch_row_words,
  const uint32_t num_total_bin,
  const double grad_base,
  const double grad_unit,
  const double hessian,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf) {
  const CUDAHybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (HybridPairConstructSkipped(desc, min_data_in_leaf, min_sum_hessian_in_leaf)) {
    return;
  }
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_total_bin) {
    uint64_t* row = scratch + static_cast<size_t>(blockIdx.z) * scratch_row_words;
    ConvertCountBin<WIDE>(i, row, reinterpret_cast<uint32_t*>(row + num_total_bin), grad_base, grad_unit, hessian,
                          desc->smaller_struct->hist_in_leaf);
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
    // block-strided over the row's elements: blockDim.x is halved for partitions wider than
    // NUM_THREADS_PER_BLOCK (CalcConstructHistogramKernelDim), so a row can hold more elements than threads
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      for (DATA_PTR_TYPE e = threadIdx.x; e < row_size; e += blockDim.x) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + e]);
        const uint32_t pos = bin << 1;
        HIST_TYPE* pos_ptr = shared_hist + pos;
        atomicAdd_block(pos_ptr, grad);
        atomicAdd_block(pos_ptr + 1, hess);
      }
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

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
  const bool row_batch = false,
  const data_size_t min_rows_per_thread = 0,
  const int block_y = -1) {
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  // min_rows_per_thread (cuda_plan key dense_level_row_blocks; 0 = off): as level_row_blocks in the pair-joint
  // body, a small leaf of a many-pair level fills a few whole blocks instead of a few rows in every block row
  const data_size_t num_data_per_thread = max((num_data_in_smaller_leaf + dim_y - 1) / dim_y, min_rows_per_thread);
  // the block row in the level grid (block_y >= 0: a block-mapped launch's, see BlockPrefix)
  const unsigned int blockIdx_y = block_y >= 0 ? static_cast<unsigned int>(block_y) : blockIdx.y;
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

template <int MAX_PAIRS>
struct BlockPrefix {
  static constexpr int kMaxPairs = MAX_PAIRS;
  int num_pairs;
  int grid_y;
  uint16_t start[kMaxPairs + 1];
};

using PairJointBlockPrefix = BlockPrefix<1024>;

using DenseBlockPrefix = BlockPrefix<256>;

template <int MAX_PAIRS>
__device__ __forceinline__ void BlockPrefixLookup(const BlockPrefix<MAX_PAIRS>& block_prefix,
                                                  unsigned int* pair_index, int* block_y) {
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
  *pair_index = static_cast<unsigned int>(lo);
  *block_y = static_cast<int>(blockIdx.y - block_prefix.start[lo]);
}

template <typename BIN_TYPE, int SHARED_HIST_SIZE, class PACK = PackRaw8, bool DL = false>
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
  const bool row_batch,
  const data_size_t host_min_rows_per_thread,
  const bool level_row_blocks,
  const DenseBlockPrefix block_prefix,
  const CUDADeviceLevelState* dls) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next level-chain kernel may get resident
  // packed grad<<16|hess slots. SHARED_HIST_SIZE counts the 16-bit slots of the
  // budget shared with the non-quantized kernels, so the int32 array holds half
  // as many entries in the same bytes, at the alignment its atomics require.
  __shared__ int32_t shared_hist_packed[SHARED_HIST_SIZE / 2];
  if constexpr (DL) {
    // cuda_plan key device_levels: the level's block map, grid y and rows-per-thread floor from the device
    // (DeviceLevelPairsKernel: dense_block_map / dense_level_row_blocks' host math on the host's descriptors); the
    // launch's y extent is a fixed count of block rows that strides over the map's. A mapped block row belongs to
    // a pair whose construct is valid and runs exactly the rows the host-launched block row would.
    const int total = dls->construct_total_blocks;
    // A stopped level has no work. Do not claim rows or inspect stale launch geometry.
    if (total == 0) {
      return;
    }
    const int dim_y = dls->construct_grid_y * static_cast<int>(blockDim.y);
    const data_size_t min_rows_per_thread = dls->construct_min_rows;
    // the map's block rows are claimed one at a time (their row counts differ: a static stride balances worse); the
    // claimed row goes to the block through the shared table's first slot, read before the inner body zeroes it
    // (one counter per launch column: every column of a block row is its own block)
    int* claim = gridDim.x <= kDeviceLevelClaimColumns ?
      const_cast<CUDADeviceLevelState*>(dls)->construct_next_block + blockIdx.x : nullptr;
    for (int k = 0;; ++k) {
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_hist_packed[0] = claim != nullptr ? atomicAdd(claim, 1) :
          static_cast<int>(blockIdx.y) + k * static_cast<int>(gridDim.y);
      }
      __syncthreads();
      const int vb = shared_hist_packed[0];
      __syncthreads();
      if (vb >= total) {
        break;
      }
      const int pair = DeviceLevelBlockPair(dls, vb);
      const int block_y = vb - dls->construct_block_start[pair];
      const CUDAHybridPairDescriptor* desc = pair_descs + pair;
      if (desc->smaller_num_bits <= 16) {
        ConstructDiscretizedHistogramDenseInner<BIN_TYPE, true, PACK>(
          desc->smaller_struct, shared_hist_packed, cuda_gradients_and_hessians, data,
          column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
          packed_partition_byte_offsets,
          num_data, dim_y, is_feature_used_bytree, bin_used, row_batch, min_rows_per_thread, block_y);
      } else {
        ConstructDiscretizedHistogramDenseInner<BIN_TYPE, false, PACK>(
          desc->smaller_struct, shared_hist_packed, cuda_gradients_and_hessians, data,
          column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
          packed_partition_byte_offsets,
          num_data, dim_y, is_feature_used_bytree, bin_used, row_batch, min_rows_per_thread, block_y);
      }
      __syncthreads();  // the next mapped block row reuses the shared table
    }
    return;
  }
  // cuda_plan key dense_block_map (classic flow only): the pair and block row this block stands for
  unsigned int pair_index = blockIdx.z;
  int block_y = static_cast<int>(blockIdx.y);
  int level_grid_y = static_cast<int>(gridDim.y);
  if (block_prefix.num_pairs > 0) {
    BlockPrefixLookup(block_prefix, &pair_index, &block_y);
    level_grid_y = block_prefix.grid_y;
  }
  // graphs A2 idle-block guard (pow2-frozen grid; see the non-quantized kernel)
  if (HybridGraphBeyondLiveSplits(gstate, pair_index)) {
    return;
  }
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
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
  // cuda_plan key dense_level_row_blocks: every leaf at the level's largest leaf's rows per thread, which the
  // grid's overflow guard bounds (the host's value in the classic flow, the device's replica below otherwise)
  data_size_t min_rows_per_thread = host_min_rows_per_thread;
  if (level_smaller_num_data == nullptr) {
    dim_y = level_grid_y * static_cast<int>(blockDim.y);
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
    min_rows_per_thread = level_row_blocks && num_pairs > 1 ?
      DenseLevelRowsPerThreadFloor(max_num_data, dim_y, gstate->construct_min_rows_per_thread) : 0;
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
      num_data, dim_y, is_feature_used_bytree, bin_used, row_batch, min_rows_per_thread, block_y);
  } else {
    ConstructDiscretizedHistogramDenseInner<BIN_TYPE, false, PACK>(
      smaller_struct, shared_hist_packed, cuda_gradients_and_hessians, data,
      column_hist_offsets, column_hist_offsets_full, feature_partition_column_index_offsets,
      packed_partition_byte_offsets,
      num_data, dim_y, is_feature_used_bytree, bin_used, row_batch, min_rows_per_thread, block_y);
  }
}

template <int N, bool PREFETCH, bool CODE5 = false>
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
    if (CODE5) {
      const uint32_t w = __ldcs(reinterpret_cast<const uint32_t*>(data_ptr + static_cast<size_t>(idx[j]) * row_stride));
      cells[j] = (w >> hi_span) & 31u;
    } else {
      const uint32_t v = static_cast<uint32_t>(__ldcs(data_ptr + static_cast<size_t>(idx[j]) * row_stride + byte_idx));
      cells[j] = (v & 0xfu) * hi_span + ((v >> 4) & hi_mask);
    }
  }
#pragma unroll
  for (int j = 0; j < N; ++j) atomicAdd_block(joint + cells[j], g[j]);
}

template <bool USE_16BIT_HIST, bool DIRECT, int kB, bool CODE5 = false>
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
  // CODE5: this byte's code word of row 0, the code row's bytes as the stride, the code's bit offset for the span
  int row_byte = byte_idx;
  uint32_t row_hi = hi_span;
  if (CODE5) {
    data_ptr = data + 4 * (byte_idx / 6);
    row_stride = 4 * Code5RowWords(row_stride);
    row_byte = 0;
    row_hi = static_cast<uint32_t>(5 * (byte_idx % 6));
  }
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
        ConstructPairRowBatch<kB, false, CODE5>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, row_byte, row_hi,
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
        ConstructPairRowBatch<kB, true, CODE5>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, row_byte, row_hi,
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
      ConstructPairRowBatch<16, false, CODE5>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, row_byte, row_hi,
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
      ConstructPairRowBatch<8, false, CODE5>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, row_byte, row_hi,
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
      ConstructPairRowBatch<4, false, CODE5>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, row_byte, row_hi,
                                      hi_mask, joint);
      inner_data_index += 4 * by;
      i += 4;
    }
    for (; i < num_iteration_this; ++i) {
      const data_size_t idx[1] = {DIRECT ? block_start + inner_data_index :
                                           data_indices_ref_this_block[inner_data_index]};
      ConstructPairRowBatch<1, false, CODE5>(idx, cuda_gradients_and_hessians, data_ptr, row_stride, row_byte, row_hi,
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

__host__ __device__ constexpr uint32_t Code5DummyCells(const int code_bits) {
  return code_bits == kCodeRadix ? 64u : 1u << code_bits;
}

struct Code5Radix {
  uint32_t n[4];
  uint32_t magic[4];
};

template <bool FLAT, int CODE_BITS>
__device__ __forceinline__ void Code5WordAdd(int32_t* shared_joint,
                                             const uint32_t (&joint_base)[Code5CodesPerWord(CODE_BITS)],
                                             const uint32_t valid_mask, const uint32_t w, const int32_t g,
                                             const Code5Radix& radix) {
  constexpr int kCodesPerWord = Code5CodesPerWord(CODE_BITS);
  constexpr uint32_t kCodeMask = (1u << (CODE_BITS == kCodeRadix ? 6 : CODE_BITS)) - 1u;
  if (CODE_BITS == kCodeRadix) {
    // cuda_plan key pair_code_radix: the mixed-radix digits of the two 16-bit halves, exact quotients by multiply
    const uint32_t lo = w & 0xffffu;
    const uint32_t hi = w >> 16;
    const uint32_t lo1 = __umulhi(lo << 10, radix.magic[0]);
    const uint32_t hi1 = __umulhi(hi << 10, radix.magic[2]);
    const uint32_t lo2 = __umulhi(lo1 << 10, radix.magic[1]);
    const uint32_t hi2 = __umulhi(hi1 << 10, radix.magic[3]);
    uint32_t c[6];
    c[0] = lo - radix.n[0] * lo1;
    c[1] = lo1 - radix.n[1] * lo2;
    c[2] = lo2;
    c[3] = hi - radix.n[2] * hi1;
    c[4] = hi1 - radix.n[3] * hi2;
    c[5] = hi2;
    if (FLAT) {
      char* const joint_bytes = reinterpret_cast<char*>(shared_joint);
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        atomicAdd_block(reinterpret_cast<int32_t*>(joint_bytes + (joint_base[j] + (c[j] << 2))), g);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        if ((valid_mask >> j) & 1u) {
          atomicAdd_block(shared_joint + joint_base[j] + c[j], g);
        }
      }
    }
    return;
  }
  if (FLAT) {
    char* const joint_bytes = reinterpret_cast<char*>(shared_joint);
#pragma unroll
    for (int j = 0; j < kCodesPerWord; ++j) {
      const uint32_t cell_bytes = (j == 0 ? (w << 2) : (w >> (CODE_BITS * j - 2))) & (kCodeMask << 2);
      atomicAdd_block(reinterpret_cast<int32_t*>(joint_bytes + (joint_base[j] + cell_bytes)), g);
    }
  } else {
#pragma unroll
    for (int j = 0; j < kCodesPerWord; ++j) {
      if ((valid_mask >> j) & 1u) {
        atomicAdd_block(shared_joint + joint_base[j] + ((w >> (CODE_BITS * j)) & kCodeMask), g);
      }
    }
  }
}

template <bool USE_16BIT_HIST, int kB, bool FLAT, bool EPILOGUE_FLAT, int CODE_BITS>
__device__ __forceinline__ void ConstructDiscretizedHistogramPairJointCode5WordInner(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  int32_t* shared_joint,
  const int32_t* cuda_gradients_and_hessians,
  const uint8_t* data,
  const uint32_t* column_meta,
  const int num_compact_columns,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int* packed_partition_byte_offsets,
  const int dim_y,
  const int whole_row_partitions,
  const uint32_t whole_row_joint,
  const data_size_t min_rows_per_thread,
  const int block_y) {
  constexpr int kCodesPerWord = Code5CodesPerWord(CODE_BITS);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = max((num_data_in_smaller_leaf + dim_y - 1) / dim_y, min_rows_per_thread);
  const size_t block_start_wide = (static_cast<size_t>(block_y) * blockDim.y) * static_cast<size_t>(num_data_per_thread);
  if (block_start_wide >= static_cast<size_t>(num_data_in_smaller_leaf)) {
    return;
  }
  const data_size_t block_start = static_cast<data_size_t>(block_start_wide);
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int row_bytes = -packed_partition_byte_offsets[0];
  const int row_words = Code5RowWords(row_bytes, CODE_BITS);
  const uint32_t* spans = column_meta + num_compact_columns;
  const uint32_t* joint_offsets = spans + 2 * num_compact_columns;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < whole_row_joint; i += num_threads_per_block) {
    shared_joint[i] = 0;
  }
  // the thread's kCodesPerWord byte slots: the partition whose byte run holds the byte, its low column, its joint table
  const int k = static_cast<int>(threadIdx.x);
  uint32_t joint_base[kCodesPerWord];
  uint32_t valid_mask = 0;
  // cuda_plan key pair_code_radix: the slots' cell counts as the fill computed them (code_table: lo span x hi span
  // of a slot with columns, 1 for a slot past the row)
  uint32_t slot_cells[kCodesPerWord];
#pragma unroll
  for (int j = 0; j < kCodesPerWord; ++j) {
    joint_base[j] = 0;
    slot_cells[j] = 1;
    const int b = kCodesPerWord * k + j;
    if (b < row_bytes) {
      int part = 0;
      while (part + 1 < whole_row_partitions && b >= packed_partition_byte_offsets[part + 1]) ++part;
      const int part_byte = part > 0 ? b - packed_partition_byte_offsets[part] : b;
      const int lo_col = feature_partition_column_index_offsets[part] + 2 * part_byte;
      if (lo_col < feature_partition_column_index_offsets[part + 1]) {
        joint_base[j] = joint_offsets[lo_col];
        valid_mask |= 1u << j;
        if (CODE_BITS == kCodeRadix) {
          const bool has_hi = lo_col + 1 < feature_partition_column_index_offsets[part + 1];
          slot_cells[j] = spans[lo_col] * (has_hi ? spans[lo_col + 1] : 1u);
        }
      }
    }
    if (FLAT) {
      // byte offsets; a slot without columns adds into the dummy cells past the tables
      joint_base[j] = static_cast<uint32_t>(sizeof(int32_t)) *
                      (((valid_mask >> j) & 1u) ? joint_base[j] : whole_row_joint);
    }
  }
  Code5Radix radix = {};
  if (CODE_BITS == kCodeRadix) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int d = 0; d < 2; ++d) {
        radix.n[2 * h + d] = slot_cells[(3 * h + d) % kCodesPerWord];
        radix.magic[2 * h + d] = Code5RadixMagic(radix.n[2 * h + d]);
      }
    }
  }
  __syncthreads();
  const data_size_t* data_indices_ref_this_block = smaller_leaf_splits->data_indices_in_leaf + block_start;
  const data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx.y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y);
  const uint32_t* code_ptr = reinterpret_cast<const uint32_t*>(data) + k;
  if (valid_mask != 0) {
    const data_size_t by = static_cast<data_size_t>(blockDim.y);
    data_size_t i = 0;
    if (i + kB <= num_iteration_this) {
      data_size_t idx[kB];
#pragma unroll
      for (int j = 0; j < kB; ++j) idx[j] = __ldg(data_indices_ref_this_block + inner_data_index + j * by);
      for (; i + kB <= num_iteration_this; i += kB) {
        int32_t g[kB];
        uint32_t w[kB];
#pragma unroll
        for (int j = 0; j < kB; ++j) g[j] = __ldg(cuda_gradients_and_hessians + idx[j]);
#pragma unroll
        for (int j = 0; j < kB; ++j) w[j] = __ldcs(code_ptr + static_cast<size_t>(idx[j]) * row_words);
        // the next batch's positions (without one, the current valid positions again; the values go unused)
        const bool has_next = i + 2 * kB <= num_iteration_this;
        const data_size_t* next_src = data_indices_ref_this_block + inner_data_index + (has_next ? kB * by : 0);
#pragma unroll
        for (int j = 0; j < kB; ++j) idx[j] = __ldg(next_src + j * (has_next ? by : 0));
#pragma unroll
        for (int j = 0; j < kB; ++j) Code5WordAdd<FLAT, CODE_BITS>(shared_joint, joint_base, valid_mask, w[j], g[j], radix);
        inner_data_index += kB * by;
      }
    }
    if (FLAT && i < num_iteration_this) {
      // the last (fewer than kB) rows as one batch: a position past them repeats the first and adds a zero gradient
      // (adding 0 changes no cell), so the tail costs one round trip instead of one per row
      const data_size_t rest = num_iteration_this - i;
      data_size_t idx[kB];
#pragma unroll
      for (int j = 0; j < kB; ++j) {
        idx[j] = __ldg(data_indices_ref_this_block + inner_data_index + (j < rest ? j * by : 0));
      }
      int32_t g[kB];
      uint32_t w[kB];
#pragma unroll
      for (int j = 0; j < kB; ++j) g[j] = j < rest ? __ldg(cuda_gradients_and_hessians + idx[j]) : 0;
#pragma unroll
      for (int j = 0; j < kB; ++j) w[j] = __ldcs(code_ptr + static_cast<size_t>(idx[j]) * row_words);
#pragma unroll
      for (int j = 0; j < kB; ++j) Code5WordAdd<FLAT, CODE_BITS>(shared_joint, joint_base, valid_mask, w[j], g[j], radix);
      i = num_iteration_this;
    }
    for (; i < num_iteration_this; ++i) {
      const data_size_t row = __ldg(data_indices_ref_this_block + inner_data_index);
      const int32_t g = __ldg(cuda_gradients_and_hessians + row);
      const uint32_t w = __ldcs(code_ptr + static_cast<size_t>(row) * row_words);
      Code5WordAdd<FLAT, CODE_BITS>(shared_joint, joint_base, valid_mask, w, g, radix);
      inner_data_index += by;
    }
  }
  __syncthreads();
  // marginals of each of the thread's bytes, by the per-byte rule of ConstructDiscretizedHistogramPairJointInner.
  // EPILOGUE_FLAT (cuda_plan key pair_code5_epilogue, a template bool: a runtime branch here would keep both
  // code paths' live ranges in one register allocation for no gain): one flat threadIdx.y-strided item index over
  // every (slot, marginal) pair instead of 6 serial per-slot passes, where most of blockDim.y's warps have no
  // marginal to add for a given slot (few bins per feature). Same cells, same per-cell wrapping sum, same
  // (slot, marginal) visitation order, only a different thread owns each item: bit-identical.
  if (EPILOGUE_FLAT) {
    // pass 1: the per-slot item counts only, to get the flat range and its per-slot offsets (selected below by
    // compile-time-unrolled comparison, never by a runtime index into a register array, which would spill to
    // local memory). The other per-slot fields are not kept here: pass 2 recomputes them for the one winning slot.
    uint32_t slot_offset[kCodesPerWord];
    uint32_t total_marginal = 0;
#pragma unroll
    for (int j = 0; j < kCodesPerWord; ++j) {
      slot_offset[j] = total_marginal;
      if (((valid_mask >> j) & 1u) == 0) continue;
      const int b = kCodesPerWord * k + j;
      int part = 0;
      while (part + 1 < whole_row_partitions && b >= packed_partition_byte_offsets[part + 1]) ++part;
      const int part_byte = part > 0 ? b - packed_partition_byte_offsets[part] : b;
      const int partition_column_end = feature_partition_column_index_offsets[part + 1];
      const int lo_col = feature_partition_column_index_offsets[part] + 2 * part_byte;
      const bool has_hi = lo_col + 1 < partition_column_end;
      const uint32_t lo_span = spans[lo_col];
      const uint32_t hi_span = has_hi ? spans[lo_col + 1] : 1u;
      total_marginal += lo_span + (has_hi ? hi_span : 0u);
    }
    // pass 2: one thread per flat item; resolve the winning slot j and its joint-table base by the same
    // unrolled-comparison selection (j itself stays a small runtime int, never used to index a register array),
    // then recompute that slot's lo_col/spans/hist-start rule exactly as the non-flat path does, for one slot
    // per item instead of once per slot.
    for (uint32_t item = threadIdx.y; item < total_marginal; item += blockDim.y) {
      int j = 0;
      uint32_t sel_offset = slot_offset[0];
      uint32_t sel_joint_base = joint_base[0];
#pragma unroll
      for (int jj = 1; jj < kCodesPerWord; ++jj) {
        if (item >= slot_offset[jj]) {
          j = jj;
          sel_offset = slot_offset[jj];
          sel_joint_base = joint_base[jj];
        }
      }
      const uint32_t m = item - sel_offset;
      const int b = kCodesPerWord * k + j;
      int part = 0;
      while (part + 1 < whole_row_partitions && b >= packed_partition_byte_offsets[part + 1]) ++part;
      const int part_byte = part > 0 ? b - packed_partition_byte_offsets[part] : b;
      const int partition_column_end = feature_partition_column_index_offsets[part + 1];
      const int lo_col = feature_partition_column_index_offsets[part] + 2 * part_byte;
      const bool has_hi = lo_col + 1 < partition_column_end;
      const uint32_t lo_span = spans[lo_col];
      const uint32_t hi_span = has_hi ? spans[lo_col + 1] : 1u;
      const uint32_t hist_start = column_hist_offsets_full[part];
      const bool is_lo = m < lo_span;
      const uint32_t bin = is_lo ? m : m - lo_span;
      const uint32_t count = is_lo ? hi_span : lo_span;
      const uint32_t stride = is_lo ? 1u : hi_span;
      // (FLAT: the table's cell offset read again, so the adds keep only the byte offsets)
      const int32_t* joint = shared_joint + (FLAT ? joint_offsets[lo_col] : sel_joint_base);
      const int32_t* cell = joint + (is_lo ? bin * hi_span : bin);
      uint32_t sum = 0;
      for (uint32_t c = 0; c < count; ++c) {
        sum += static_cast<uint32_t>(cell[c * stride]);
      }
      const int32_t packed_grad_hess = static_cast<int32_t>(sum);
      if (packed_grad_hess == 0) continue;
      const uint32_t hist_pos = hist_start + column_meta[is_lo ? lo_col : lo_col + 1] + bin;
      if (USE_16BIT_HIST) {
        atomicAdd(reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + hist_pos, packed_grad_hess);
      } else {
        const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
        atomicAdd(reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + hist_pos, (atomic_add_long_t)(packed_grad_hess_int64));
      }
    }
  } else {
    for (int j = 0; j < kCodesPerWord; ++j) {
      if (((valid_mask >> j) & 1u) == 0) continue;
      const int b = kCodesPerWord * k + j;
      int part = 0;
      while (part + 1 < whole_row_partitions && b >= packed_partition_byte_offsets[part + 1]) ++part;
      const int part_byte = part > 0 ? b - packed_partition_byte_offsets[part] : b;
      const int partition_column_end = feature_partition_column_index_offsets[part + 1];
      const int lo_col = feature_partition_column_index_offsets[part] + 2 * part_byte;
      const bool has_hi = lo_col + 1 < partition_column_end;
      const uint32_t lo_span = spans[lo_col];
      const uint32_t hi_span = has_hi ? spans[lo_col + 1] : 1u;
      const uint32_t partition_hist_start = column_hist_offsets_full[part];
      // (FLAT: the table's cell offset read again, so the adds keep only the byte offsets)
      const int32_t* joint = shared_joint + (FLAT ? joint_offsets[lo_col] : joint_base[j]);
      const uint32_t num_marginal = lo_span + (has_hi ? hi_span : 0u);
      for (uint32_t m = threadIdx.y; m < num_marginal; m += blockDim.y) {
        const bool is_lo = m < lo_span;
        const uint32_t bin = is_lo ? m : m - lo_span;
        const uint32_t count = is_lo ? hi_span : lo_span;
        const uint32_t stride = is_lo ? 1u : hi_span;
        const int32_t* cell = joint + (is_lo ? bin * hi_span : bin);
        uint32_t sum = 0;
        for (uint32_t c = 0; c < count; ++c) {
          sum += static_cast<uint32_t>(cell[c * stride]);
        }
        const int32_t packed_grad_hess = static_cast<int32_t>(sum);
        if (packed_grad_hess == 0) continue;
        const uint32_t hist_pos = partition_hist_start + column_meta[is_lo ? lo_col : lo_col + 1] + bin;
        if (USE_16BIT_HIST) {
          atomicAdd(reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + hist_pos, packed_grad_hess);
        } else {
          const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
          atomicAdd(reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + hist_pos, (atomic_add_long_t)(packed_grad_hess_int64));
        }
      }
    }
  }
}

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
  const PairJointBlockPrefix block_prefix, \
  const CUDADeviceLevelState* dls

#define FALCATA_PAIR_JOINT_BATCHED_ARGS \
  pair_descs, cuda_gradients_and_hessians, data, column_meta, num_compact_columns, column_hist_offsets_full, \
  feature_partition_column_index_offsets, packed_partition_byte_offsets, num_data, min_data_in_leaf, \
  min_sum_hessian_in_leaf, per_pair_min_grid_dim_y, min_rows_per_thread, saturation_floor_total, \
  num_grad_quant_bins, whole_row_partitions, whole_row_joint, level_min_rows_per_thread, block_prefix, dls

template <bool ALL_ROWS, int BATCH, bool CODE5 = false, bool WORDS = false, bool FLAT = false, bool EPILOGUE_FLAT = false,
          int CODE_BITS = 5>  // NOLINT(whitespace/indent_namespace)
__device__ __forceinline__ void PairJointBatchedBlock(FALCATA_PAIR_JOINT_BATCHED_PARAMS, int32_t* shared_joint,
                                                      const unsigned int pair_index, const int block_y,
                                                      const int level_grid_y, const int level_num_pairs) {
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
    if (WORDS) { \
      ConstructDiscretizedHistogramPairJointCode5WordInner<B16, BATCH, FLAT, EPILOGUE_FLAT, CODE_BITS>( \
        smaller_struct, shared_joint, cuda_gradients_and_hessians, data, column_meta, num_compact_columns, \
        column_hist_offsets_full, feature_partition_column_index_offsets, packed_partition_byte_offsets, \
        dim_y, whole_row_partitions, whole_row_joint, level_min_rows_per_thread, block_y); \
    } else { \
      ConstructDiscretizedHistogramPairJointInner<B16, DIRECT, BATCH, CODE5>( \
        smaller_struct, shared_joint, cuda_gradients_and_hessians, data, column_meta, num_compact_columns, \
        column_hist_offsets_full, feature_partition_column_index_offsets, packed_partition_byte_offsets, \
        num_data, dim_y, whole_row_partitions, whole_row_joint, level_min_rows_per_thread, block_y); \
    }
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

template <bool ALL_ROWS, int BATCH, bool CODE5 = false, bool WORDS = false, bool FLAT = false, bool EPILOGUE_FLAT = false,
          int CODE_BITS = 5, bool DL = false>  // NOLINT(whitespace/indent_namespace)
__device__ __forceinline__ void PairJointBatchedBody(FALCATA_PAIR_JOINT_BATCHED_PARAMS) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next level-chain kernel may get resident
  extern __shared__ int32_t shared_joint[];
  if constexpr (DL) {
    // cuda_plan key device_levels: the level's block map, grid y and rows-per-thread floor from the device
    // (DeviceLevelPairsKernel: the host's pair_block_map / pair_code5_slots / level_row_blocks math on the host's
    // descriptors); the launch's y extent is a fixed count of block rows that strides over the map's
    const int total = dls->construct_total_blocks;
    if (total == 0) {
      return;
    }
    const data_size_t level_min_rows = dls->construct_min_rows;
    const bool direct = total <= static_cast<int>(gridDim.y);
    __shared__ int s_block_row;
    // One processing callsite for both modes. The direct mode never claims or waits on a reuse barrier;
    // overflow retains the original initial/next claim and both CTA barriers in their original order.
    int* claim = gridDim.x <= kDeviceLevelClaimColumns ?
      const_cast<CUDADeviceLevelState*>(dls)->construct_next_block + blockIdx.x : nullptr;
    for (int k = 0;; ++k) {
      int vb = static_cast<int>(blockIdx.y);
      if (!direct) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          s_block_row = claim != nullptr ? atomicAdd(claim, 1) :
            static_cast<int>(blockIdx.y) + k * static_cast<int>(gridDim.y);
        }
        __syncthreads();
        vb = s_block_row;
      }
      if (vb >= total) {
        break;
      }
      const int pair = DeviceLevelBlockPair(dls, vb);
      PairJointBatchedBlock<ALL_ROWS, BATCH, CODE5, WORDS, FLAT, EPILOGUE_FLAT, CODE_BITS>(
        pair_descs, cuda_gradients_and_hessians, data, column_meta, num_compact_columns, column_hist_offsets_full,
        feature_partition_column_index_offsets, packed_partition_byte_offsets, num_data, min_data_in_leaf,
        min_sum_hessian_in_leaf, per_pair_min_grid_dim_y, min_rows_per_thread, saturation_floor_total,
        num_grad_quant_bins, whole_row_partitions, whole_row_joint, level_min_rows, block_prefix, dls, shared_joint,
        static_cast<unsigned int>(pair), vb - dls->construct_block_start[pair], dls->construct_grid_y,
        dls->live_pairs);
      if (direct) {
        break;
      }
      __syncthreads();  // the next mapped block row reuses the joint tables (and s_block_row)
    }
    return;
  }
  // cuda_plan key pair_block_map: the pair and block row this block stands for (see BlockPrefix)
  unsigned int pair_index = blockIdx.z;
  int block_y = static_cast<int>(blockIdx.y);
  int level_grid_y = static_cast<int>(gridDim.y);
  int level_num_pairs = static_cast<int>(gridDim.z);
  if (block_prefix.num_pairs > 0) {
    BlockPrefixLookup(block_prefix, &pair_index, &block_y);
    level_grid_y = block_prefix.grid_y;
    level_num_pairs = block_prefix.num_pairs;
  }
  PairJointBatchedBlock<ALL_ROWS, BATCH, CODE5, WORDS, FLAT, EPILOGUE_FLAT, CODE_BITS>(
    FALCATA_PAIR_JOINT_BATCHED_ARGS, shared_joint, pair_index, block_y, level_grid_y, level_num_pairs);
}

template <bool ALL_ROWS, bool CODE5 = false, bool DL = false>
__global__ void CUDAConstructDiscretizedHistogramPairJointBatchedKernel(FALCATA_PAIR_JOINT_BATCHED_PARAMS) {
  PairJointBatchedBody<ALL_ROWS, 16, CODE5, false, false, false, 5, DL>(FALCATA_PAIR_JOINT_BATCHED_ARGS);
}

#if CUDART_VERSION >= 12040
#define FALCATA_PAIR_JOINT_CAPPED_BUILD 1

#define FALCATA_PAIR_JOINT_MAXNREG(n) __maxnreg__(n)

#else
#define FALCATA_PAIR_JOINT_CAPPED_BUILD 0

#define FALCATA_PAIR_JOINT_MAXNREG(n)

#endif
template <bool ALL_ROWS, int BATCH, int MAXREG, bool CODE5 = false, bool DL = false>
__global__ void FALCATA_PAIR_JOINT_MAXNREG(MAXREG)
CUDAConstructDiscretizedHistogramPairJointCappedKernel(FALCATA_PAIR_JOINT_BATCHED_PARAMS) {
  PairJointBatchedBody<ALL_ROWS, BATCH, CODE5, false, false, false, 5, DL>(FALCATA_PAIR_JOINT_BATCHED_ARGS);
}

#undef FALCATA_PAIR_JOINT_MAXNREG

template <int BATCH, bool FLAT, bool EPILOGUE_FLAT, int CODE_BITS, bool DL = false>
__global__ void CUDAConstructDiscretizedHistogramPairJointCode5WordKernel(FALCATA_PAIR_JOINT_BATCHED_PARAMS) {
  PairJointBatchedBody<false, BATCH, true, true, FLAT, EPILOGUE_FLAT, CODE_BITS, DL>(FALCATA_PAIR_JOINT_BATCHED_ARGS);
}

#undef FALCATA_PAIR_JOINT_BATCHED_ARGS

#undef FALCATA_PAIR_JOINT_BATCHED_PARAMS

struct PairJointLimits {
  bool ready = false;
  int max_threads = 0;
  int warps_by_regs = 0;
  int warps_per_sm = 0;
};

inline const PairJointLimits& PairJointDeviceLimits() {
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

inline int PairJointMaxThreadsPerBlock() {
  return PairJointDeviceLimits().max_threads;
}

inline int PairJointResidentWarps(const int threads) {
  const PairJointLimits& limits = PairJointDeviceLimits();
  const int block_warps = std::max(1, (threads + 31) / 32);
  return std::min(limits.warps_by_regs / block_warps, limits.warps_per_sm / block_warps) * block_warps;
}

using PairJointKernelFn = void (*)(const CUDAHybridPairDescriptor*, const int32_t*, const uint8_t*, const uint32_t*,
                                   int, const uint32_t*, const int*, const int*, data_size_t, data_size_t, double, int,
                                   int, int, int, int, uint32_t, data_size_t, PairJointBlockPrefix,
                                   const CUDADeviceLevelState*);

struct PairJointCappedBuild {
  PairJointKernelFn gather;
  PairJointKernelFn direct;
  PairJointKernelFn gather_code5;
};

inline PairJointCappedBuild PairJointCapped() {
  return {CUDAConstructDiscretizedHistogramPairJointCappedKernel<false, 15, 48>,
          CUDAConstructDiscretizedHistogramPairJointCappedKernel<true, 12, 48>,
          CUDAConstructDiscretizedHistogramPairJointCappedKernel<false, 12, 48, true>};
}

inline PairJointCappedBuild PairJointCappedDL() {
  return {CUDAConstructDiscretizedHistogramPairJointCappedKernel<false, 15, 48, false, true>,
          CUDAConstructDiscretizedHistogramPairJointCappedKernel<true, 12, 48, false, true>,
          CUDAConstructDiscretizedHistogramPairJointCappedKernel<false, 12, 48, true, true>};
}

inline int PairJointCode5MaxThreads(const PairJointKernelFn kernel) {
  struct Entry {
    int device;
    PairJointKernelFn kernel;
    int max_threads;
  };
  static thread_local std::vector<Entry> entries;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  for (const Entry& e : entries) {
    if (e.device == device && e.kernel == kernel) return e.max_threads;
  }
  cudaFuncAttributes attr;
  CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attr, kernel));
  entries.push_back({device, kernel, attr.maxThreadsPerBlock});
  return attr.maxThreadsPerBlock;
}

inline PairJointKernelFn PairJointCode5WordKernel(const bool flat, const bool epilogue_flat, const int code_bits) {
  if (code_bits == kCodeRadix) {
    return flat ? (epilogue_flat ?
                   CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, true, kCodeRadix> :
                   CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, false, kCodeRadix>) :
                  CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<4, false, false, kCodeRadix>;
  }
  if (code_bits == 6) {
    return flat ? (epilogue_flat ? CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, true, 6> :
                                    CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, false, 6>) :
                  CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<4, false, false, 6>;
  }
  return flat ? (epilogue_flat ? CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, true, 5> :
                                  CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, false, 5>) :
                CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<4, false, false, 5>;
}

inline PairJointKernelFn PairJointCode5WordKernelDL(const bool flat, const bool epilogue_flat, const int code_bits) {
  if (code_bits == kCodeRadix) {
    return flat ? (epilogue_flat ?
                   CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, true, kCodeRadix, true> :
                   CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, false, kCodeRadix, true>) :
                  CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<4, false, false, kCodeRadix, true>;
  }
  if (code_bits == 6) {
    return flat ? (epilogue_flat ? CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, true, 6, true> :
                                    CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, false, 6, true>) :
                  CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<4, false, false, 6, true>;
  }
  return flat ? (epilogue_flat ? CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, true, 5, true> :
                                  CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<5, true, false, 5, true>) :
                CUDAConstructDiscretizedHistogramPairJointCode5WordKernel<4, false, false, 5, true>;
}

struct PairJointCode5WordShape {
  int rows;
  int device_slots;
};

inline PairJointCode5WordShape PairJointCode5WordShapeFor(const int row_words, const size_t smem_bytes,
                                                          const int num_grad_quant_bins, const bool flat,
                                                          const bool tallest, const bool epilogue_flat,
                                                          const int code_bits) {
  struct Entry {
    int device;
    int code_bits;
    bool flat;
    int row_words;
    size_t smem_bytes;
    int bins;
    bool tallest;
    bool epilogue_flat;
    PairJointCode5WordShape shape;
  };
  static thread_local std::vector<Entry> entries;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  for (const Entry& e : entries) {
    if (e.device == device && e.code_bits == code_bits && e.flat == flat && e.row_words == row_words &&
        e.smem_bytes == smem_bytes &&
        e.bins == num_grad_quant_bins && e.tallest == tallest && e.epilogue_flat == epilogue_flat) {
      return e.shape;
    }
  }
  const PairJointKernelFn kernel = PairJointCode5WordKernel(flat, epilogue_flat, code_bits);
  cudaFuncAttributes attr;
  CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attr, kernel));
  // the kernel's largest block on this device (registers included), in warps of the device's width
  const int max_threads = attr.maxThreadsPerBlock;
  const int warp_size = CurrentDeviceLaunchLimits().warp_size;
  int best_rows = 0;
  int best_warps = 0;
  int best_blocks = 0;
  for (int c = 1; row_words > 0 && row_words * c <= max_threads &&
       HybridQuantConstructBlockDimY(c, num_grad_quant_bins) == c; ++c) {
    int blocks = 0;
    CUDASUCCESS_OR_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, row_words * c, smem_bytes));
    const int warps = blocks * ((row_words * c + warp_size - 1) / warp_size);
    if (blocks > 0 && (tallest || warps >= best_warps)) {
      best_warps = warps;
      best_rows = c;
      best_blocks = blocks;
    }
  }
  int num_sms = 0;
  CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  const PairJointCode5WordShape shape{best_rows, best_rows > 0 ? best_blocks * std::max(1, num_sms) : 0};
  entries.push_back({device, code_bits, flat, row_words, smem_bytes, num_grad_quant_bins, tallest, epilogue_flat, shape});
  return shape;
}

inline data_size_t PairJointSlotRowsPerThread(const std::vector<data_size_t>& counts, const int block_rows,
                                              const int slots, const data_size_t min_rows, const data_size_t max_rows) {
  const auto blocks_at = [&](const data_size_t r) {
    const int64_t block_span = static_cast<int64_t>(block_rows) * r;
    int64_t total = 0;
    for (const data_size_t n : counts) total += (static_cast<int64_t>(n) + block_span - 1) / block_span;
    return total;
  };
  data_size_t lo = std::max<data_size_t>(1, std::min(min_rows, max_rows));
  data_size_t hi = std::max<data_size_t>(lo, max_rows);
  if (blocks_at(hi) > slots) return hi;
  while (lo < hi) {
    const data_size_t mid = lo + (hi - lo) / 2;
    if (blocks_at(mid) <= slots) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

inline int PairJointOccupancyWarps(PairJointKernelFn gather, PairJointKernelFn direct, const int threads,
                                   const size_t smem_bytes) {
  int blocks_gather = 0;
  int blocks_direct = 0;
  CUDASUCCESS_OR_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_gather, gather, threads, smem_bytes));
  CUDASUCCESS_OR_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_direct, direct, threads, smem_bytes));
  return std::min(blocks_gather, blocks_direct) * ((threads + 31) / 32);
}

inline int PairJointCappedRows(const int row_bytes, const size_t smem_bytes, const int default_y,
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
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next level-chain kernel may get resident
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
    // block-strided over the row's elements: blockDim.x is halved for partitions wider than
    // NUM_THREADS_PER_BLOCK (CalcConstructHistogramKernelDim), so a sparse row can hold more elements
    // than threads; one element per thread silently dropped the rest from the histogram
    if (threadIdx.x < row_size) {
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      for (DATA_PTR_TYPE e = threadIdx.x; e < row_size; e += blockDim.x) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + e]);
        int32_t* pos_ptr = shared_hist_packed + bin;
        atomicAdd_block(pos_ptr, grad_and_hess);
      }
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
    // block-strided over the row's elements: blockDim.x is halved for partitions wider than
    // NUM_THREADS_PER_BLOCK (CalcConstructHistogramKernelDim), so a sparse row can hold more elements
    // than threads; one element per thread silently dropped the rest from the histogram
    if (threadIdx.x < row_size) {
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      for (DATA_PTR_TYPE e = threadIdx.x; e < row_size; e += blockDim.x) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + e]);
        int32_t* pos_ptr = shared_hist_packed + bin;
        atomicAdd_block(pos_ptr, grad_and_hess);
      }
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
    // cuda_plan key dense_wide_hist: see the batched launch
    auto launch_leaf_quant = [&](auto qhist_tag) {
    constexpr size_t QHIST = decltype(qhist_tag)::value;
    if (USE_GLOBAL_MEM_BUFFER) {
      if (cuda_row_data_->is_sparse()) {
        if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory<BIN_TYPE, PTR_TYPE, QHIST, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        } else {
          CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory<BIN_TYPE, PTR_TYPE, QHIST, false><<<grid_dim, block_dim, 0, current_stream()>>>(
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
          CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory<BIN_TYPE, QHIST, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        } else {
          CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory<BIN_TYPE, QHIST, false><<<grid_dim, block_dim, 0, current_stream()>>>(
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
          CUDAConstructDiscretizedHistogramSparseKernel<BIN_TYPE, PTR_TYPE, QHIST, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else {
          CUDAConstructDiscretizedHistogramSparseKernel<BIN_TYPE, PTR_TYPE, QHIST, false><<<grid_dim, block_dim, 0, current_stream()>>>(
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
            CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, QHIST, true, PackNibble4><<<grid_dim, block_dim, 0, current_stream()>>>(
              cuda_smaller_leaf_splits,
              reinterpret_cast<const int32_t*>(cuda_gradients_),
              RowMajorBin<BIN_TYPE>(),
              cuda_row_data_->cuda_column_hist_offsets(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              cuda_row_data_->cuda_feature_partition_column_index_offsets(),
              cuda_row_data_->cuda_packed_partition_byte_offsets(),
              num_data_);
          } else {
            CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, QHIST, false, PackNibble4><<<grid_dim, block_dim, 0, current_stream()>>>(
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
          CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, QHIST, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            nullptr,
            num_data_);
        } else {
          CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, QHIST, false><<<grid_dim, block_dim, 0, current_stream()>>>(
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
    };
    if (cuda_row_data_->dense_wide_hist()) {
      CHECK_EQ(cuda_row_data_->quant_hist_slots(), static_cast<int>(SHARED_HIST_SIZE) * 2);
      launch_leaf_quant(std::integral_constant<size_t, SHARED_HIST_SIZE * 2>{});
    } else {
      CHECK_EQ(cuda_row_data_->quant_hist_slots(), static_cast<int>(SHARED_HIST_SIZE));
      launch_leaf_quant(std::integral_constant<size_t, SHARED_HIST_SIZE>{});
    }
  } else {
    // the non-quantized kernels' tables are not widened (dense_wide_hist divides only under quantized training)
    CHECK(!cuda_row_data_->dense_wide_hist());
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
          count_scratch_dirty_ = true;  // cuda_plan key forest_count_hist: the det slab is also the count scratch
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
          uint8_t* active_data = NibbleViewData();
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
            if (count_codes_ != nullptr) {
              // cuda_plan key forest_count_hist: the count construct, same source and offsets
              LaunchConstructHistogramDenseCount<BIN_TYPE>(
                cuda_smaller_leaf_splits, num_data_in_smaller_leaf,
                grid_dim_x, compact_block_x, &compact_source);
            } else {
              LaunchConstructHistogramDenseDeterministic<BIN_TYPE>(
                cuda_smaller_leaf_splits, num_data_in_smaller_leaf,
                grid_dim_x, compact_block_x, &compact_source);
            }
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
          if (count_codes_ != nullptr) {
            // cuda_plan key forest_count_hist: the count construct
            LaunchConstructHistogramDenseCount<BIN_TYPE>(
              cuda_smaller_leaf_splits, num_data_in_smaller_leaf, grid_dim_x, block_dim_x);
          } else {
            LaunchConstructHistogramDenseDeterministic<BIN_TYPE>(
              cuda_smaller_leaf_splits, num_data_in_smaller_leaf, grid_dim_x, block_dim_x);
          }
        } else if (cuda_row_data_->is_4bit_packed()) {
          CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, current_stream()>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            RowMajorBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            cuda_row_data_->cuda_packed_partition_byte_offsets(),
            column_used_bytree_device(),
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
            column_used_bytree_device(),
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
          count_scratch_dirty_ = true;  // cuda_plan key forest_count_hist: the det slab is also the count scratch
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
        if (count_codes_ != nullptr) {
          // cuda_plan key forest_count_hist: the count construct (count_supported_: every partition fits its table)
          LaunchConstructHistogramDenseCount<BIN_TYPE>(
            cuda_smaller_leaf_splits, num_data_in_smaller_leaf, grid_dim_x, block_dim_x);
        } else {
          LaunchConstructHistogramDenseDeterministic<BIN_TYPE>(
            cuda_smaller_leaf_splits, num_data_in_smaller_leaf, grid_dim_x, block_dim_x);
        }
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
  const bool hist_fp32);

__global__ void SubtractHistogramBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  const uint8_t* bin_used,
  const bool hist_fp32,
  const CUDAHybridGraphLoopStateOpt gstate);

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
  // (coalesced) so the order-exact folds read shared, not a dependent chain of
  // global loads; chunking keeps them exact for any bin count. Lane 0 folds
  // the gradient and lane 1 the hessian, so each fold step is one warp
  // instruction for both sums (the staging interleaves them, so the two
  // lanes' reads fall in distinct banks). (The discretized fix keeps its
  // parallel reduction: integer sums are order-invariant. shared_mem_buffer
  // stays a parameter for that variant's shared call signature.)
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
    if (threadIdx_x < 2) {
      hist_t acc = threadIdx_x == 0 ? sum_gradient : sum_hessian;
      for (uint32_t b = 0; b < chunk; ++b) {
        if (chunk_start + b != most_freq_bin) {
          acc += fix_stage[(b << 1) + threadIdx_x];
        }
      }
      if (threadIdx_x == 0) {
        sum_gradient = acc;
      } else {
        sum_hessian = acc;
      }
    }
  }
  if (threadIdx_x < 2) {
    sum_hessian = __shfl_sync(0x3u, sum_hessian, 1);
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
  const bool hist_fp32);

__global__ void FixHistogramBatchedKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDAHybridPairDescriptor* pair_descs,
  const int8_t* feature_used,
  const bool hist_fp32,
  const CUDAHybridGraphLoopStateOpt gstate);

__device__ __forceinline__ void FixHistogramGroupInner(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const int num_need_fix,
  const int first,
  const int8_t* feature_used,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* larger_for_subtract,
  const bool hist_fp32) {
  __shared__ double fix_group_stage[kFixGroupFeatures][2 * kFixGroupChunk];
  __shared__ uint32_t fix_group_bins;
  constexpr uint32_t kStageWidth = 2 * kFixGroupChunk;
  // the feature this thread stages (every thread stages one element a round)
  const uint32_t stage_k = threadIdx.x / kStageWidth;
  const uint32_t stage_e = threadIdx.x % kStageWidth;
  bool stage_active = false;
  uint32_t stage_offset = 0;
  uint32_t stage_num_bin = 0;
  if (first + static_cast<int>(stage_k) < num_need_fix) {
    const int feature_index = cuda_need_fix_histogram_features[first + stage_k];
    stage_active = feature_used == nullptr || feature_used[feature_index];
    stage_offset = cuda_feature_hist_offsets[feature_index];
    stage_num_bin = cuda_feature_num_bins[feature_index];
  }
  // the feature this lane folds (lanes 0 .. 2 * kFixGroupFeatures - 1)
  const uint32_t fold_k = threadIdx.x >> 1;
  const uint32_t fold_c = threadIdx.x & 1;
  bool fold_active = false;
  uint32_t fold_offset = 0;
  uint32_t fold_num_bin = 0;
  uint32_t fold_mfb = 0;
  if (threadIdx.x < 2 * kFixGroupFeatures && first + static_cast<int>(fold_k) < num_need_fix) {
    const int feature_index = cuda_need_fix_histogram_features[first + fold_k];
    fold_active = feature_used == nullptr || feature_used[feature_index];
    fold_offset = cuda_feature_hist_offsets[feature_index];
    fold_num_bin = cuda_feature_num_bins[feature_index];
    fold_mfb = cuda_feature_most_freq_bins[feature_index];
  }
  if (threadIdx.x == 0) {
    fix_group_bins = 0;
  }
  __syncthreads();
  if (fold_active && fold_c == 0) {
    atomicMax(&fix_group_bins, fold_num_bin);
  }
  __syncthreads();
  const uint32_t group_bins = fix_group_bins;
  const hist_t* smaller_hist = cuda_smaller_leaf_splits->hist_in_leaf;
  const float* smaller_hist32 = reinterpret_cast<const float*>(smaller_hist);
  hist_t acc = 0.0f;
  for (uint32_t chunk_start = 0; chunk_start < group_bins; chunk_start += kFixGroupChunk) {
    const uint32_t stage_bin = chunk_start + (stage_e >> 1);
    if (stage_active && stage_bin < stage_num_bin) {
      const uint32_t pos = ((stage_offset + stage_bin) << 1) + (stage_e & 1);
      fix_group_stage[stage_k][stage_e] = hist_fp32 ? static_cast<hist_t>(smaller_hist32[pos]) : smaller_hist[pos];
    }
    __syncthreads();
    if (fold_active) {
      for (uint32_t b = 0; b < kFixGroupChunk; ++b) {
        const uint32_t bin = chunk_start + b;
        if (bin < fold_num_bin && bin != fold_mfb) {
          acc += fix_group_stage[fold_k][(b << 1) + fold_c];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < 2 * kFixGroupFeatures) {
    const hist_t sum_hessian = __shfl_down_sync((1u << (2 * kFixGroupFeatures)) - 1, acc, 1);
    if (fold_active && fold_c == 0) {
      const hist_t sum_gradient = acc;
      const double leaf_sum_gradients = cuda_smaller_leaf_splits->sum_of_gradients;
      const double leaf_sum_hessians = cuda_smaller_leaf_splits->sum_of_hessians;
      const hist_t fixed_gradient = leaf_sum_gradients - sum_gradient;
      const hist_t fixed_hessian = leaf_sum_hessians - sum_hessian;
      if (hist_fp32) {
        float* feature_hist32 = reinterpret_cast<float*>(cuda_smaller_leaf_splits->hist_in_leaf) + fold_offset * 2;
        feature_hist32[fold_mfb << 1] = static_cast<float>(fixed_gradient);
        feature_hist32[(fold_mfb << 1) + 1] = static_cast<float>(fixed_hessian);
        if (larger_for_subtract != nullptr && larger_for_subtract->leaf_index >= 0) {
          float* larger_feature_hist = reinterpret_cast<float*>(larger_for_subtract->hist_in_leaf) + fold_offset * 2;
          larger_feature_hist[fold_mfb << 1] -= static_cast<float>(fixed_gradient);
          larger_feature_hist[(fold_mfb << 1) + 1] -= static_cast<float>(fixed_hessian);
        }
      } else {
        hist_t* feature_hist = cuda_smaller_leaf_splits->hist_in_leaf + fold_offset * 2;
        feature_hist[fold_mfb << 1] = fixed_gradient;
        feature_hist[(fold_mfb << 1) + 1] = fixed_hessian;
        if (larger_for_subtract != nullptr && larger_for_subtract->leaf_index >= 0) {
          hist_t* larger_feature_hist = larger_for_subtract->hist_in_leaf + fold_offset * 2;
          larger_feature_hist[fold_mfb << 1] -= fixed_gradient;
          larger_feature_hist[(fold_mfb << 1) + 1] -= fixed_hessian;
        }
      }
    }
  }
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
  const CUDAHybridGraphLoopStateOpt gstate);

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

__global__ void SubtractHistogramDiscretizedBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const CUDAHybridGraphLoopStateOpt gstate,
  const CUDADeviceLevelState* dls);

__global__ void CopyChangedNumBitHistogram(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  hist_t* num_bit_change_buffer);

__global__ void CopyChangedNumBitHistogramBatchedKernel(
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const uint8_t* bin_used,
  const CUDAHybridGraphLoopStateOpt gstate,
  const CUDADeviceLevelState* dls);

template <bool USE_16BIT_HIST>
__device__ __forceinline__ void FixHistogramDiscretizedInner(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  int64_t* shared_mem_buffer,
  const unsigned int blockIdx_x) {
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
    cuda_smaller_leaf_splits, shared_mem_buffer, blockIdx.x);
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
  const CUDADeviceLevelState* dls);

__device__ __forceinline__ void SubtractDiscretizedBin(
  const int sub_case, const uint32_t i, const int32_t s32, const int64_t s64,
  hist_t* larger_hist_in_leaf, int32_t* buffer) {
  if (sub_case == 0) {
    reinterpret_cast<int32_t*>(larger_hist_in_leaf)[i] -= s32;
  } else if (sub_case == 3) {
    reinterpret_cast<int64_t*>(larger_hist_in_leaf)[i] -= s64;
  } else {
    int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(larger_hist_in_leaf);
    const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(s32 >> 16)) << 32) |
      static_cast<int64_t>(s32 & 0x0000ffff);
    const int64_t larger_hist_item = larger_leaf_hist[i] - smaller_hist_item_int64;
    if (sub_case == 1) {
      buffer[i] = static_cast<int32_t>(static_cast<int16_t>(larger_hist_item >> 32) << 16) |
        static_cast<int32_t>(larger_hist_item & 0x000000000000ffff);
    } else {
      larger_leaf_hist[i] = larger_hist_item;
    }
  }
}

__device__ __forceinline__ void FixSubtractDiscretizedMaskedPair(
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
  const bool larger_exists,
  const uint8_t parent_num_bits,
  const uint8_t smaller_num_bits,
  const uint8_t larger_num_bits,
  int32_t* buffer,
  const uint8_t* bin_used,
  const int8_t* feature_used) {
  const int sub_case = parent_num_bits <= 16 ? 0 : (larger_num_bits <= 16 ? 1 : (smaller_num_bits <= 16 ? 2 : 3));
  const hist_t* smaller_hist = smaller_struct->hist_in_leaf;
  if (static_cast<int>(blockIdx.x) < num_subtract_blocks) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!larger_exists || i >= static_cast<uint32_t>(num_total_bin) || fix_mfb_mask[2 * static_cast<size_t>(i)] ||
        (bin_used != nullptr && !bin_used[i])) {
      return;
    }
    const int32_t s32 = sub_case == 3 ? 0 : reinterpret_cast<const int32_t*>(smaller_hist)[i];
    const int64_t s64 = sub_case == 3 ? reinterpret_cast<const int64_t*>(smaller_hist)[i] : 0;
    SubtractDiscretizedBin(sub_case, i, s32, s64, larger_struct->hist_in_leaf, buffer);
    return;
  }
  const unsigned int fix_index = blockIdx.x - static_cast<unsigned int>(num_subtract_blocks);
  const int feature_index = cuda_need_fix_histogram_features[fix_index];
  if (feature_used != nullptr && !feature_used[feature_index]) {
    return;  // most-frequent bin of an unused feature: dead storage this tree (its subtract is masked too)
  }
  __shared__ int64_t shared_mem_buffer[WARPSIZE];
  if (smaller_num_bits <= 16) {
    FixHistogramDiscretizedInner<true>(
      cuda_feature_num_bins, cuda_feature_hist_offsets, cuda_feature_most_freq_bins,
      cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned,
      smaller_struct, shared_mem_buffer, fix_index);
  } else {
    FixHistogramDiscretizedInner<false>(
      cuda_feature_num_bins, cuda_feature_hist_offsets, cuda_feature_most_freq_bins,
      cuda_need_fix_histogram_features, cuda_need_fix_histogram_features_num_bin_aligned,
      smaller_struct, shared_mem_buffer, fix_index);
  }
  if (threadIdx.x == 0 && larger_exists) {
    // thread 0 wrote the fixed value; the subtract of that bin reads it back (own write, visible)
    const uint32_t i = cuda_feature_hist_offsets[feature_index] + cuda_feature_most_freq_bins[feature_index];
    const int32_t s32 = sub_case == 3 ? 0 : reinterpret_cast<const int32_t*>(smaller_hist)[i];
    const int64_t s64 = sub_case == 3 ? reinterpret_cast<const int64_t*>(smaller_hist)[i] : 0;
    SubtractDiscretizedBin(sub_case, i, s32, s64, larger_struct->hist_in_leaf, buffer);
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
  const CUDADeviceLevelState* dls);

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
  hist_t* num_bit_change_buffer);

constexpr uint32_t kFixSubtractMaxSpan = 8;

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

__global__ void __launch_bounds__(512) FixSubtractUsedFeaturesBatchedKernel(
  const uint32_t* used_feature_info,
  const int num_used_features,
  const int num_total_bin,
  const CUDAHybridPairDescriptor* pair_descs,
  hist_t* num_bit_change_buffer,
  const CUDADeviceLevelState* dls);

__global__ void ConvertFixedHistBatchedKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_items,
  const double* inv_scale,
  const uint8_t* bin_used,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const CUDAHybridGraphLoopStateOpt gstate);

__global__ void NonquantFixedAbsMaxKernel(const score_t* gradients, const score_t* hessians, const data_size_t n,
                                          uint32_t* absmax_bits);

__global__ void NonquantFixedScaleKernel(const uint32_t* absmax_bits, const int lg_n, int* scale,
                                         double* inv_scale);

void LaunchNonquantFixedScale(const score_t* gradients, const score_t* hessians, const data_size_t n,
                              const bool absmax_ready, uint32_t* absmax_bits, int* scale, double* inv_scale);

__global__ void ApplyFusedRootHistogramKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const hist_t* root_hist_scratch,
  const int num_total_bin,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf);

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
    block_dim_y = std::max(1, BatchedConstructThreads() / block_dim_x);
    block_dim_y = HybridQuantConstructBlockDimY(
      block_dim_y, use_quantized_grad_ ? num_grad_quant_bins_ : 0);
    grid_dim_y = HybridBatchedConstructGridDimYQuant(
      max_num_data_in_smaller_leaf, num_pairs, block_dim_y, min_grid_dim_y_,
      BatchConstructMinRowsPerThread(), BatchConstructSaturationFloor(),
      use_quantized_grad_ ? num_grad_quant_bins_ : 0);
  }
  dim3 grid_dim(grid_dim_x, grid_dim_y, num_pairs);
  dim3 block_dim(block_dim_x, block_dim_y);
  // cuda_plan key device_levels: a device-driven level's dense construct takes its block map, grid y and rows floor
  // from the device (DeviceLevelPairsKernel), striding a fixed launch over the map's block rows
  const CUDADeviceLevelState* dense_dls = device_level_state_;
  if (dense_dls != nullptr) {
    CHECK(use_quantized_grad_);
    // (four times the pair-joint launch's block rows: the dense kernel's mapped block rows differ more in their row
    // counts, and a launch close to their number leaves the hardware scheduler to balance them)
    const int target_blocks = 4 * DeviceLevelConstructTargetBlocks();
    grid_dim = dim3(grid_dim_x, static_cast<unsigned int>(std::max(1, std::min(65535,
      (target_blocks + std::max(1, grid_dim_x) - 1) / std::max(1, grid_dim_x)))), 1);
  }
  // cuda_plan key dense_level_row_blocks (classic flow; the speculative flow derives it on the device): the
  // quantized dense batched construct runs every leaf at no fewer rows per thread than the largest leaf's of this
  // grid (which the grid's overflow guard bounds) or the grid formula's own target, whichever is smaller
  const int64_t dense_dim_y = static_cast<int64_t>(grid_dim_y) * block_dim_y;
  const data_size_t dense_min_rows_per_thread =
    FalcataPlan::Get().dense_level_row_blocks && num_pairs > 1 && level_smaller_num_data == nullptr && dense_dim_y > 0 &&
    dense_dls == nullptr ?
    DenseLevelRowsPerThreadFloor(max_num_data_in_smaller_leaf, dense_dim_y, BatchConstructMinRowsPerThread()) : 0;
  // cuda_plan key dense_block_map (classic flow, as pair_block_map): launch only the block rows that have rows, each
  // pair's count by the kernel's own row grouping (dense_level_row_blocks' floor) from the host copy of the leaf
  // counts the kernel reads; blocks map back to their pair and block row through the prefix. Same blocks with the
  // same rows: bit-identical.
  // Levels with more pairs than a prefix holds launch one block-mapped grid per DenseBlockPrefix::kMaxPairs pairs,
  // each reading its pairs' descriptors from pair_descs + pair_base (the kernel indexes its descriptors by the pair
  // it maps to); levels where a chunk's blocks exceed a launch's y limit take the plain grid.
  struct DenseLaunchChunk {
    dim3 grid;
    int pair_base;
    DenseBlockPrefix prefix;  // num_pairs 0: the plain grid
  };
  std::vector<DenseLaunchChunk> dense_chunks;
  if (FalcataPlan::Get().dense_block_map && num_pairs > 1 && dense_dls == nullptr &&
      level_smaller_num_data == nullptr && hybrid_graph_capture_gstate_ == nullptr && dense_dim_y > 0 &&
      level_host_pair_descs_ != nullptr && level_host_num_pairs_ == num_pairs) {
    const int64_t max_blocks = std::min(65535, CurrentDeviceLaunchLimits().max_grid_dim_y);
    bool fits = true;
    for (int pair_base = 0; pair_base < num_pairs; pair_base += DenseBlockPrefix::kMaxPairs) {
      const int chunk_pairs = std::min(DenseBlockPrefix::kMaxPairs, num_pairs - pair_base);
      DenseLaunchChunk chunk{};
      chunk.pair_base = pair_base;
      int64_t total = 0;
      for (int p = 0; p < chunk_pairs; ++p) {
        chunk.prefix.start[p] = static_cast<uint16_t>(std::min<int64_t>(total, 65535));
        const CUDAHybridPairDescriptor& d = level_host_pair_descs_[pair_base + p];
        const data_size_t n = d.num_data_in_smaller_leaf;
        if (!d.construct_valid || n <= 0) {
          continue;
        }
        const int64_t rows_per_thread = std::max<int64_t>((static_cast<int64_t>(n) + dense_dim_y - 1) / dense_dim_y,
                                                          dense_min_rows_per_thread);
        const int64_t block_rows = static_cast<int64_t>(block_dim_y) * rows_per_thread;
        total += std::min<int64_t>(grid_dim_y, (static_cast<int64_t>(n) + block_rows - 1) / block_rows);
      }
      chunk.prefix.start[chunk_pairs] = static_cast<uint16_t>(std::min<int64_t>(total, 65535));
      if (total > max_blocks) {
        fits = false;
        break;
      }
      if (total == 0) {
        continue;  // no pair of this chunk has rows: nothing to launch
      }
      chunk.prefix.num_pairs = chunk_pairs;
      chunk.prefix.grid_y = grid_dim_y;
      chunk.grid = dim3(grid_dim.x, static_cast<unsigned int>(total), 1);
      dense_chunks.push_back(chunk);
    }
    if (!fits) {
      dense_chunks.clear();
    }
  }
  if (dense_chunks.empty()) {
    dense_chunks.push_back(DenseLaunchChunk{grid_dim, 0, DenseBlockPrefix{}});
  }
  // cuda_plan key forest_count_hist: the count construct takes every level the deterministic one would (and the
  // levels too wide for its slot slab)
  const bool count_batched = CountBatchedEligible();
  const bool det_batched = !count_batched && DetDenseBatchedEligible(num_pairs);
  // nonquant_deterministic: every non-quantized arm below the count and deterministic ones launches the batched
  // kernel with this launch's accumulator, so under a fixed-point launch they all leave int64 words for the convert
  // (set once for the whole region, so no arm can miss it)
  nq_fixed_kernel_launched_ = nq_fixed_launch_ && !use_quantized_grad_ && !count_batched && !det_batched;
  const int* level_dim_y = nullptr;
  const data_size_t* level_sizes_for_kernel = nullptr;
  if (level_smaller_num_data != nullptr && !det_batched && !count_batched) {
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
  // the non-quantized kernels' tables are not widened (dense_wide_hist divides only under quantized training)
  if (!use_quantized_grad_) {
    CHECK(!cuda_row_data_->dense_wide_hist());
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
    // (not while this tree's compact buffer holds the pair_code5 code view: the small-leaf kernels read the nibble
    // view, which would cost the tree a second fill; the code-view pair-joint kernel below takes those levels)
    if (SmallLeafConstructEnabled() && small_leaf_rows > 0 &&
        hybrid_graph_capture_gstate_ == nullptr && dense_dls == nullptr &&
        max_num_data_in_smaller_leaf <= small_leaf_rows && !(use_compact_view_ && code5_valid_)) {
      const int8_t* feat_used =
        any_feature_unused_bytree_ ? column_used_bytree_device() : nullptr;
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
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackBit3x32, reinterpret_cast<const BIN_TYPE*>(NibbleViewData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
            case PackCodecId::kRadix5x32:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRadix5x32, reinterpret_cast<const BIN_TYPE*>(NibbleViewData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
            case PackCodecId::kRadix6x32:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRadix6x32, reinterpret_cast<const BIN_TYPE*>(NibbleViewData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
            case PackCodecId::kRadix7x32:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRadix7x32, reinterpret_cast<const BIN_TYPE*>(NibbleViewData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
            default:
              FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackNibble4, reinterpret_cast<const BIN_TYPE*>(NibbleViewData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), compact_packed_partition_byte_offsets_.RawData(), nullptr);
              break;
          }
        } else {
          FALCATA_LAUNCH_SMALL_LEAF_QUANT(PackRaw8, reinterpret_cast<const BIN_TYPE*>(NibbleViewData()), compact_column_hist_offsets_.RawData(), compact_feature_partition_column_index_offsets_.RawData(), nullptr, nullptr);
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
    // cuda_plan key dense_wide_hist: the dense kernels' packed shared table is instantiated for the slot count
    // the feature partitions were divided against (quant_hist_slots(): SHARED_HIST_SIZE, or 2x it); one body,
    // two instantiations
    auto launch_dense_quant = [&](auto qhist_tag) {
    constexpr size_t QHIST = decltype(qhist_tag)::value;
    if (use_compact_view_) {
      // Device-map geometry is chosen before capture using the exact DL kernel's resources. A rejected pair
      // shape must take the existing dense DL branch, rather than retrying the host variant's pair decision.
      CUDADeviceLevelConstructShape device_shape{};
      if (dense_dls != nullptr) DeviceLevelConstructShape(&device_shape);
      if (compact_is_4bit_ && compact_codec_ == PackCodecId::kNibble4 && FalcataPlan::Get().pair_hist &&
                 compact_pair_joint_max_ > 0 &&
                 compact_pair_joint_max_ * sizeof(int32_t) <= 48 * 1024 &&
                 level_smaller_num_data == nullptr && hybrid_graph_capture_gstate_ == nullptr &&
                 (dense_dls == nullptr || device_shape.pair_mode == 1) &&
                 (std::max(1, max_num_compact_cols_per_partition_) + 1) / 2 * static_cast<int>(block_dim.y) <=
                   PairJointMaxThreadsPerBlock()) {
        if (dense_dls != nullptr) {
          LaunchDeviceLevelPairJoint(pair_descs);  // cuda_plan key device_levels
          return;
        }
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
        // cuda_plan key pair_code5_words: gathering whole-row levels over this tree's code view take the code-word
        // kernel at its own block height (occupancy API); the grid y is re-derived for that height by the same
        // formula and packed-cell guard
        const bool level_all_rows = FalcataPlan::Get().all_rows_direct && num_pairs == 1 &&
          max_num_data_in_smaller_leaf == num_data_;
        int code5_word_rows = 0;
        // cuda_plan key pair_code5_flat: the branch-free adds' dummy cells past the joint tables, where the larger
        // block still fits the device's default dynamic shared memory
        const size_t code5_flat_smem_bytes = pair_smem_bytes + Code5DummyCells(code5_bits_) * sizeof(int32_t);
        const bool code5_flat = FalcataPlan::Get().pair_code5_flat &&
                                code5_flat_smem_bytes <= CurrentDeviceLaunchLimits().shared_bytes_per_block;
        const size_t code5_word_smem_bytes = code5_flat ? code5_flat_smem_bytes : pair_smem_bytes;
        const bool code5_tall = FalcataPlan::Get().pair_code5_slots && FalcataPlan::Get().pair_code5_tall;
        // cuda_plan key pair_code5_epilogue: the flat (slot, marginal) flush, composed only with the branch-free
        // adds (code5_flat) this was measured on
        const bool code5_epilogue_flat = code5_flat && FalcataPlan::Get().pair_code5_epilogue;
        // cuda_plan key pair_code5_slots: the code-word level's one rows-per-thread (0: the generic sizing)
        data_size_t slot_rows_per_thread = 0;
        if (whole_rows && !level_all_rows && code5_valid_ && FalcataPlan::Get().pair_code5 &&
            FalcataPlan::Get().pair_code5_words &&
            Code5RowWords(compact_row_bytes_, code5_bits_) == code5_row_words_) {
          // cuda_plan key pair_code5_tall: the tallest block where the grid is sized to the slots
          const PairJointCode5WordShape word_shape = PairJointCode5WordShapeFor(
            code5_row_words_, code5_word_smem_bytes, use_quantized_grad_ ? num_grad_quant_bins_ : 0, code5_flat,
            code5_tall, code5_epilogue_flat, code5_bits_);
          code5_word_rows = word_shape.rows;
          if (code5_word_rows > 0) {
            int word_grid_y = HybridBatchedConstructGridDimYQuant(
              max_num_data_in_smaller_leaf, num_pairs, code5_word_rows, min_grid_dim_y_,
              BatchConstructMinRowsPerThread(), BatchConstructSaturationFloor(),
              use_quantized_grad_ ? num_grad_quant_bins_ : 0);
            // cuda_plan key pair_code5_slots: the level's smaller leaves (the pairs the kernel constructs, from the host
            // copy of the counts it reads) take one rows-per-thread R, the smallest whose blocks fill no more than the
            // kernel's resident slots on this device, at least the min-rows-per-thread knob and at most the packed-cell
            // cap; the grid is the largest leaf's blocks at R, so every leaf's blocks run R rows per thread
            if (FalcataPlan::Get().pair_code5_slots && word_shape.device_slots > 0 && max_num_data_in_smaller_leaf > 0 &&
                (num_pairs == 1 || (level_host_pair_descs_ != nullptr && level_host_num_pairs_ == num_pairs))) {
              std::vector<data_size_t> counts;
              if (num_pairs == 1) {
                counts.push_back(max_num_data_in_smaller_leaf);
              } else {
                for (int p = 0; p < num_pairs; ++p) {
                  const CUDAHybridPairDescriptor& d = level_host_pair_descs_[p];
                  if (d.construct_valid && d.num_data_in_smaller_leaf > 0) {
                    counts.push_back(d.num_data_in_smaller_leaf);
                  }
                }
              }
              const int bins = use_quantized_grad_ ? num_grad_quant_bins_ : 0;
              const data_size_t max_rows = bins > 0 ?
                static_cast<data_size_t>(HybridQuantConstructMaxRowsPerThread(bins, code5_word_rows)) :
                max_num_data_in_smaller_leaf;
              const data_size_t r = PairJointSlotRowsPerThread(
                counts, code5_word_rows, word_shape.device_slots,
                static_cast<data_size_t>(std::max(1, BatchConstructMinRowsPerThread())), max_rows);
              const int64_t r_grid_y = (static_cast<int64_t>(max_num_data_in_smaller_leaf) +
                                        static_cast<int64_t>(code5_word_rows) * r - 1) /
                                       (static_cast<int64_t>(code5_word_rows) * r);
              if (!counts.empty() && r_grid_y <= CurrentDeviceLaunchLimits().max_grid_dim_y) {
                slot_rows_per_thread = r;
                word_grid_y = static_cast<int>(r_grid_y);
              }
            }
            if (word_grid_y <= CurrentDeviceLaunchLimits().max_grid_dim_y) {
              pair_y = code5_word_rows;
              pair_grid_y = word_grid_y;
            } else {
              code5_word_rows = 0;
              slot_rows_per_thread = 0;
            }
          }
        }
        const dim3 pair_grid_dim(whole_rows ? 1 : grid_dim.x, pair_grid_y, grid_dim.z);
        const dim3 pair_block_dim(code5_word_rows > 0 ? code5_row_words_ :
                                  whole_rows ? compact_row_bytes_ : (cc + 1) / 2, pair_y);
        // level_row_blocks: every leaf at the largest leaf's rows per thread (within the grid's overflow guard)
        const int64_t pair_dim_y = static_cast<int64_t>(pair_grid_y) * pair_y;
        // (pair_code5_slots: every leaf at the level's slot-fitted R, no per-pair grid re-sizing in the kernel)
        const data_size_t pair_min_rows_per_thread = slot_rows_per_thread > 0 ? slot_rows_per_thread :
          FalcataPlan::Get().level_row_blocks && num_pairs > 1 ?
          static_cast<data_size_t>((static_cast<int64_t>(max_num_data_in_smaller_leaf) + pair_dim_y - 1) / pair_dim_y) : 0;
        const int pair_per_pair_min_grid = slot_rows_per_thread > 0 ? 0 :
          FalcataPlan::Get().per_pair_rows ? min_grid_dim_y_ : 0;
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
          const int per_pair_min_grid = pair_per_pair_min_grid;
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
#define FALCATA_PAIR_JOINT_LAUNCH_ARGS(DATA) \
          pair_descs, \
          reinterpret_cast<const int32_t*>(cuda_gradients_), \
          DATA, \
          compact_column_hist_offsets_.RawData(), \
          num_compact_columns_, \
          cuda_row_data_->cuda_partition_hist_offsets(), \
          compact_feature_partition_column_index_offsets_.RawData(), \
          compact_packed_partition_byte_offsets_.RawData(), \
          num_data_, \
          static_cast<data_size_t>(min_data_in_leaf_), \
          min_sum_hessian_in_leaf_, \
          pair_per_pair_min_grid, \
          BatchConstructMinRowsPerThread() * (whole_rows ? static_cast<int>(grid_dim.x) : 1), \
          BatchConstructSaturationFloor(), \
          use_quantized_grad_ ? num_grad_quant_bins_ : 0, \
          whole_rows ? static_cast<int>(grid_dim.x) : 0, \
          whole_rows ? static_cast<uint32_t>(compact_pair_joint_total_) : 0u, \
          pair_min_rows_per_thread, \
          block_prefix, \
          static_cast<const CUDADeviceLevelState*>(nullptr)
#define FALCATA_LAUNCH_PAIR_JOINT_DATA(KERNEL, DATA) \
        KERNEL<<<launch_grid_dim, pair_block_dim, pair_smem_bytes, cuda_stream_>>>(FALCATA_PAIR_JOINT_LAUNCH_ARGS(DATA))
        // a whole-row block does the work of grid_dim.x partition blocks and zeroes / flushes all their joint
        // tables, so its per_pair_rows rows-per-thread floor scales by the partitions it covers to keep the same
        // share of fixed per-block cost (a larger floor only lowers the formula, so the launched grid still bounds it)
#define FALCATA_LAUNCH_PAIR_JOINT(KERNEL) FALCATA_LAUNCH_PAIR_JOINT_DATA(KERNEL, NibbleViewData())
        // cuda_plan key pair_code5: gathering whole-row blocks read this tree's code view (written by the fused fill
        // into the live compact buffer instead of the nibble view) when its instantiation can launch the same block;
        // every other launch reads the nibble view through NibbleViewData(), which writes it over the code view first
        const PairJointKernelFn code5_kernel = capped_y > 0 ? PairJointCapped().gather_code5 :
          CUDAConstructDiscretizedHistogramPairJointBatchedKernel<false, true>;
        const bool code5 = whole_rows && !all_rows && code5_valid_ && FalcataPlan::Get().pair_code5 &&
            code5_bits_ == 5 && Code5RowWords(compact_row_bytes_) == code5_row_words_ &&
            static_cast<int>(pair_block_dim.x * pair_block_dim.y) <= PairJointCode5MaxThreads(code5_kernel);
        if (code5_word_rows > 0) {
          // (cuda_plan key pair_code_radix: the mixed-radix code view)
          DiagConstructKernel(code5_bits_ == kCodeRadix ? "pair_code5_words (mixed radix)" : "pair_code5_words");
          PairJointCode5WordKernel(code5_flat, code5_epilogue_flat, code5_bits_)<<<launch_grid_dim, pair_block_dim, code5_word_smem_bytes, cuda_stream_>>>(
            FALCATA_PAIR_JOINT_LAUNCH_ARGS(compact_data_uint8_t_.RawDataReadOnly()));
          // a new launch shape: a rejected launch must not leave the histogram unbuilt silently
          CUDASUCCESS_OR_FATAL(cudaPeekAtLastError());
        } else if (code5) {
          DiagConstructKernel("pair_code5");
          FALCATA_LAUNCH_PAIR_JOINT_DATA(code5_kernel, compact_data_uint8_t_.RawDataReadOnly());
          // a new variant: a rejected launch must not leave the histogram unbuilt silently
          CUDASUCCESS_OR_FATAL(cudaPeekAtLastError());
        } else if (capped_y > 0) {
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
#undef FALCATA_LAUNCH_PAIR_JOINT_DATA
#undef FALCATA_PAIR_JOINT_LAUNCH_ARGS
      } else if (compact_is_4bit_) {
        DiagConstructKernel(FalcataPlan::Get().row_batch ? "row_batch" : "unbatched");
#define FALCATA_LAUNCH_BATCHED_COMPACT_QUANT(PACKT) \
        for (const DenseLaunchChunk& dense_chunk : dense_chunks) \
        (dense_dls != nullptr ? CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, QHIST, PACKT, true> : \
                                CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, QHIST, PACKT, false>) \
          <<<dense_chunk.grid, block_dim, 0, cuda_stream_>>>( \
          pair_descs + dense_chunk.pair_base, \
          reinterpret_cast<const int32_t*>(cuda_gradients_), \
          reinterpret_cast<const BIN_TYPE*>(NibbleViewData()), \
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
          hybrid_graph_capture_gstate_, FalcataPlan::Get().row_batch, \
          dense_min_rows_per_thread, FalcataPlan::Get().dense_level_row_blocks, dense_chunk.prefix, dense_dls)
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
        for (const DenseLaunchChunk& dense_chunk : dense_chunks) {
        (dense_dls != nullptr ? CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, QHIST, PackRaw8, true> :
                                 CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, QHIST, PackRaw8, false>)
          <<<dense_chunk.grid, block_dim, 0, cuda_stream_>>>(
          pair_descs + dense_chunk.pair_base,
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          reinterpret_cast<const BIN_TYPE*>(NibbleViewData()),
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
          hybrid_graph_capture_gstate_, FalcataPlan::Get().row_batch,
          dense_min_rows_per_thread, FalcataPlan::Get().dense_level_row_blocks, dense_chunk.prefix, dense_dls);
        }
      }
    } else if (cuda_row_data_->is_4bit_packed()) {
      DiagConstructKernel(FalcataPlan::Get().row_batch ? "row_batch" : "unbatched");
      for (const DenseLaunchChunk& dense_chunk : dense_chunks) {
      (dense_dls != nullptr ? CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, QHIST, PackNibble4, true> :
                               CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, QHIST, PackNibble4, false>)
        <<<dense_chunk.grid, block_dim, 0, cuda_stream_>>>(
        pair_descs + dense_chunk.pair_base,
        reinterpret_cast<const int32_t*>(cuda_gradients_),
        RowMajorBin<BIN_TYPE>(),
        cuda_row_data_->cuda_column_hist_offsets(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        cuda_row_data_->cuda_feature_partition_column_index_offsets(),
        cuda_row_data_->cuda_packed_partition_byte_offsets(),
        num_data_,
        any_feature_unused_bytree_ ? column_used_bytree_device() : nullptr,
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
        static_cast<data_size_t>(min_data_in_leaf_),
        min_sum_hessian_in_leaf_,
        level_smaller_num_data,
        hybrid_graph_capture_gstate_, FalcataPlan::Get().row_batch,
        dense_min_rows_per_thread, FalcataPlan::Get().dense_level_row_blocks, dense_chunk.prefix, dense_dls);
      }
    } else {
      DiagConstructKernel(FalcataPlan::Get().row_batch ? "row_batch" : "unbatched");
      for (const DenseLaunchChunk& dense_chunk : dense_chunks) {
      (dense_dls != nullptr ? CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, QHIST, PackRaw8, true> :
                               CUDAConstructDiscretizedHistogramDenseBatchedKernel<BIN_TYPE, QHIST, PackRaw8, false>)
        <<<dense_chunk.grid, block_dim, 0, cuda_stream_>>>(
        pair_descs + dense_chunk.pair_base,
        reinterpret_cast<const int32_t*>(cuda_gradients_),
        RowMajorBin<BIN_TYPE>(),
        cuda_row_data_->cuda_column_hist_offsets(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        cuda_row_data_->cuda_feature_partition_column_index_offsets(),
        nullptr,
        num_data_,
        any_feature_unused_bytree_ ? column_used_bytree_device() : nullptr,
        any_feature_unused_bytree_ ? cuda_bin_used_bytree_.RawDataReadOnly() : nullptr,
        static_cast<data_size_t>(min_data_in_leaf_),
        min_sum_hessian_in_leaf_,
        level_smaller_num_data,
        hybrid_graph_capture_gstate_, FalcataPlan::Get().row_batch,
        dense_min_rows_per_thread, FalcataPlan::Get().dense_level_row_blocks, dense_chunk.prefix, dense_dls);
      }
    }
    };
    if (cuda_row_data_->dense_wide_hist()) {
      CHECK_EQ(cuda_row_data_->quant_hist_slots(), static_cast<int>(SHARED_HIST_SIZE) * 2);
      launch_dense_quant(std::integral_constant<size_t, SHARED_HIST_SIZE * 2>{});
    } else {
      CHECK_EQ(cuda_row_data_->quant_hist_slots(), static_cast<int>(SHARED_HIST_SIZE));
      launch_dense_quant(std::integral_constant<size_t, SHARED_HIST_SIZE>{});
    }
  } else if (count_batched) {
    LaunchConstructHistogramDenseBatchedCount<BIN_TYPE>(
      pair_descs, num_pairs, max_num_data_in_smaller_leaf, grid_dim_x, block_dim_x);
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
        CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true, true, USE_GH2><<<grid_dim, block_dim,
          NqFixedSmem(reinterpret_cast<const void*>(&CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true, true, USE_GH2>)), cuda_stream_>>>(
          pair_descs,
          cuda_gradients_, cuda_hessians_,
          USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
          reinterpret_cast<const BIN_TYPE*>(NibbleViewData()),
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
          hybrid_graph_capture_gstate_,
          nq_fixed_launch_ ? nq_scale_.RawDataReadOnly() : nullptr);
      } else {
        CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true, false, USE_GH2><<<grid_dim, block_dim,
          NqFixedSmem(reinterpret_cast<const void*>(&CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, true, false, USE_GH2>)), cuda_stream_>>>(
          pair_descs,
          cuda_gradients_, cuda_hessians_,
          USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
          reinterpret_cast<const BIN_TYPE*>(NibbleViewData()),
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
          hybrid_graph_capture_gstate_,
          nq_fixed_launch_ ? nq_scale_.RawDataReadOnly() : nullptr);
      }
    } else {
      if (compact_is_4bit_) {
        CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, true, USE_GH2><<<grid_dim, block_dim,
          NqFixedSmem(reinterpret_cast<const void*>(&CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, true, USE_GH2>)), cuda_stream_>>>(
          pair_descs,
          cuda_gradients_, cuda_hessians_,
          USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
          reinterpret_cast<const BIN_TYPE*>(NibbleViewData()),
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
          hybrid_graph_capture_gstate_,
          nq_fixed_launch_ ? nq_scale_.RawDataReadOnly() : nullptr);
      } else {
        CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, false, USE_GH2><<<grid_dim, block_dim,
          NqFixedSmem(reinterpret_cast<const void*>(&CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, false, USE_GH2>)), cuda_stream_>>>(
          pair_descs,
          cuda_gradients_, cuda_hessians_,
          USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
          reinterpret_cast<const BIN_TYPE*>(NibbleViewData()),
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
          hybrid_graph_capture_gstate_,
          nq_fixed_launch_ ? nq_scale_.RawDataReadOnly() : nullptr);
      }
    }
  } else if (cuda_row_data_->is_4bit_packed()) {
    CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, true, USE_GH2><<<grid_dim, block_dim,
          NqFixedSmem(reinterpret_cast<const void*>(&CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, true, USE_GH2>)), cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
      RowMajorBin<BIN_TYPE>(),
      cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      cuda_row_data_->cuda_packed_partition_byte_offsets(),
      column_used_bytree_device(),
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
      hybrid_graph_capture_gstate_,
      nq_fixed_launch_ ? nq_scale_.RawDataReadOnly() : nullptr);
  } else {
    CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, false, USE_GH2><<<grid_dim, block_dim,
          NqFixedSmem(reinterpret_cast<const void*>(&CUDAConstructHistogramDenseBatchedKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE, false, false, USE_GH2>)), cuda_stream_>>>(
      pair_descs,
      cuda_gradients_, cuda_hessians_,
      USE_GH2 ? cuda_gradients_hessians_.RawDataReadOnly() : nullptr,
      RowMajorBin<BIN_TYPE>(),
      cuda_row_data_->cuda_column_hist_offsets(),
      cuda_row_data_->cuda_partition_hist_offsets(),
      cuda_row_data_->cuda_feature_partition_column_index_offsets(),
      nullptr,
      column_used_bytree_device(),
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
      hybrid_graph_capture_gstate_,
      nq_fixed_launch_ ? nq_scale_.RawDataReadOnly() : nullptr);
  }
}


}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_KERNELS_CUH_
