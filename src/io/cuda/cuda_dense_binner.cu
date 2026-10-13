/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>

#include <cstdint>

#include "cuda_dense_binner.hpp"

namespace Falcata {

// defined in cuda_algorithms.cu
void CheckCUDADeviceSupportsThisBuild(int device_id);

namespace {

constexpr uint32_t kDeviceSkipBin = 0xFFFFFFFFu;  // == Bin::kSkipBin

// exact device replica of BinMapper::ValueToBin for numerical bins
// (categorical features are excluded by the host-side eligibility check)
__device__ __forceinline__ uint32_t DeviceValueToBin(
    double value, const double* __restrict__ bounds,
    const CUDADenseBinnerColMeta& meta) {
  if (isnan(value)) {
    if (meta.missing_is_nan) {
      return static_cast<uint32_t>(meta.num_bin - 1);
    }
    value = 0.0f;
  }
  int l = 0;
  int r = meta.num_bin - 1;
  if (meta.missing_is_nan) {
    r -= 1;
  }
  const double* ub = bounds + meta.bounds_offset;
  while (l < r) {
    int m = (r + l - 1) / 2;
    if (value <= ub[m]) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  return static_cast<uint32_t>(l);
}

// exact device replica of BinMapper::ValueToBin for categorical bins. The
// host truncates the value to int: NaN, values at or below -1 (negative after
// truncation) and categories it never saw (at or above the table end, which
// includes values beyond the int range) go to bin 0; the table holds the host
// result for every category below cat_size.
__device__ __forceinline__ uint32_t DeviceCategoryToBin(
    double value, const uint32_t* __restrict__ cat_bins,
    const CUDADenseBinnerColMeta& meta) {
  if (!(value > -1.0) || value >= static_cast<double>(meta.cat_size)) {
    return 0;
  }
  return cat_bins[meta.cat_offset + static_cast<int>(value)];
}

// exact device replica of FeatureGroup::EncodeBinForPush (non multi-val)
__device__ __forceinline__ uint32_t DeviceEncodeBin(
    uint32_t bin, const CUDADenseBinnerColMeta& meta) {
  if (bin == meta.most_freq_bin) {
    return kDeviceSkipBin;
  }
  if (meta.most_freq_bin == 0) {
    bin -= 1;
  }
  return bin + meta.sub_offset;
}

template <typename T>
__device__ __forceinline__ uint32_t DeviceLutIndex(T v);
template <>
__device__ __forceinline__ uint32_t DeviceLutIndex(int8_t v) {
  return static_cast<uint32_t>(static_cast<int>(v) + 128);
}
template <>
__device__ __forceinline__ uint32_t DeviceLutIndex(int16_t v) {
  return static_cast<uint32_t>(static_cast<int>(v) + 32768);
}
template <>
__device__ __forceinline__ uint32_t DeviceLutIndex(uint8_t v) {
  return static_cast<uint32_t>(v);
}
template <>
__device__ __forceinline__ uint32_t DeviceLutIndex(uint16_t v) {
  return static_cast<uint32_t>(v);
}

// one thread handles one row pair of one feature group: it walks the group's
// columns in ascending column order with last-non-skip-wins semantics
// (replicating the host push order) and writes the final packed value once.
// `parity` shifts the pairing by one row so a chunk that starts at an odd
// dataset row keeps its rows in the dataset-global nibble positions: with
// parity=1 the first pair holds only the chunk's first row in its high slot
// (the low slot stays zero and is OR-merged into the byte the previous chunk
// already wrote). parity=0 reproduces the original behavior exactly.
template <typename T, bool ROW_MAJOR, bool USE_LUT>
__global__ void CUDADenseBinChunkKernel(const T* __restrict__ in,
                                        int64_t in_stride, data_size_t rows,
                                        data_size_t pairs, int parity,
                                        CUDADenseBinnerTables tables,
                                        uint8_t* __restrict__ out) {
  const int group = static_cast<int>(blockIdx.y);
  const data_size_t pair = static_cast<data_size_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (pair >= pairs) {
    return;
  }
  const data_size_t row0 = pair * 2 - static_cast<data_size_t>(parity);
  uint32_t vals[2] = {0, 0};
  const int k_end = tables.group_col_ptr[group + 1];
  for (int k = tables.group_col_ptr[group]; k < k_end; ++k) {
    const int col = tables.group_cols[k];
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
      const data_size_t row = row0 + t;
      if (row < 0 || row >= rows) {
        continue;
      }
      const T raw = ROW_MAJOR
          ? in[static_cast<int64_t>(row) * in_stride + col]
          : in[static_cast<int64_t>(col) * in_stride + row];
      uint32_t enc;
      if constexpr (USE_LUT) {
        enc = tables.lut[tables.col_lut_off[col] + DeviceLutIndex<T>(raw)];
      } else {
        const CUDADenseBinnerColMeta meta = tables.col_meta[col];
        const double value = static_cast<double>(raw);
        enc = DeviceEncodeBin(
            meta.cat_size > 0
                ? DeviceCategoryToBin(value, tables.cat_bins, meta)
                : DeviceValueToBin(value, tables.bounds, meta),
            meta);
      }
      if (enc != kDeviceSkipBin) {
        vals[t] = enc;
      }
    }
  }
  const uint8_t width = tables.group_width[group];
  uint8_t* group_out = out + tables.group_pair_off[group] * pairs;
  if (width == 0) {
    group_out[pair] = static_cast<uint8_t>(vals[0] | (vals[1] << 4));
    return;
  }
  #pragma unroll
  for (int t = 0; t < 2; ++t) {
    const data_size_t row = row0 + t;
    if (row < 0 || row >= rows) {
      continue;
    }
    // element index in chunk-aligned output space (row + parity == 2*pair + t)
    const data_size_t elem = pair * 2 + t;
    if (width == 1) {
      group_out[elem] = static_cast<uint8_t>(vals[t]);
    } else if (width == 2) {
      reinterpret_cast<uint16_t*>(group_out)[elem] =
          static_cast<uint16_t>(vals[t]);
    } else {
      reinterpret_cast<uint32_t*>(group_out)[elem] = vals[t];
    }
  }
}

template <typename T, bool USE_LUT>
void LaunchTyped(const void* in, bool is_row_major, int64_t in_stride,
                 data_size_t rows, data_size_t pairs, int parity,
                 int num_groups, const CUDADenseBinnerTables& tables,
                 uint8_t* out, cudaStream_t stream) {
  const int block_dim = 256;
  dim3 grid((pairs + block_dim - 1) / block_dim,
            static_cast<unsigned int>(num_groups));
  const T* typed_in = reinterpret_cast<const T*>(in);
  if (is_row_major) {
    CUDADenseBinChunkKernel<T, true, USE_LUT><<<grid, block_dim, 0, stream>>>(
        typed_in, in_stride, rows, pairs, parity, tables, out);
  } else {
    CUDADenseBinChunkKernel<T, false, USE_LUT><<<grid, block_dim, 0, stream>>>(
        typed_in, in_stride, rows, pairs, parity, tables, out);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

// row-major int8/uint8 input with one column per group (LUT encode): the tile
// kernel below stages tile_rows rows x kTileCols columns in shared memory with
// coalesced reads (a warp walks consecutive columns of one row), then encodes
// from shared memory. Output is written exactly as CUDADenseBinChunkKernel
// writes it, so the bytes are identical. The rows per tile and the block's
// threads are cuda_plan knobs (construct_tile_rows, default 64, and
// construct_tile_threads, default 256); the tile's width is not a knob:
// kTileCols int8 values are one 128-byte line of an input row, the unit the
// coalesced reads fetch, so it is fixed by the memory system, not tuned (the
// int8 row's 128-byte line; FalcataPlan::kConstructTileCols, shared with the
// knob's validation).
constexpr int kTileCols = FalcataPlan::kConstructTileCols;

// kFixedRows > 0: the rows per tile as a compile-time constant (the default
// construct_tile_rows); 0: tile_rows at run time (any other knob value)
template <typename T, int kThreads, int kFixedRows>
__global__ void __launch_bounds__(kThreads)
CUDADenseBinTileKernel(const T* __restrict__ in, int64_t in_stride, int ncol,
                       data_size_t rows, data_size_t pairs, int parity,
                       int runtime_tile_rows, CUDADenseBinnerTables tables,
                       uint8_t* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char tile_bytes[];
  T* tile = reinterpret_cast<T*>(tile_bytes);  // [column][row in block]
  const int tile_rows = kFixedRows > 0 ? kFixedRows : runtime_tile_rows;
  const int tile_pairs = tile_rows / 2;  // row pairs per block
  const data_size_t pair_base = static_cast<data_size_t>(blockIdx.x) * tile_pairs;
  const int col_base = static_cast<int>(blockIdx.y) * kTileCols;
  const data_size_t row_base = pair_base * 2 - static_cast<data_size_t>(parity);
  for (int i = threadIdx.x; i < kTileCols * tile_rows; i += kThreads) {
    const int r = i / kTileCols;  // consecutive threads: consecutive columns
    const int j = i - r * kTileCols;
    const data_size_t row = row_base + r;
    const int col = col_base + j;
    T v = 0;
    if (row >= 0 && row < rows && col < ncol) {
      v = in[static_cast<int64_t>(row) * in_stride + col];
    }
    tile[j * tile_rows + r] = v;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kTileCols * tile_pairs; i += kThreads) {
    const int p = i % tile_pairs;  // consecutive threads: consecutive pairs
    const int j = i / tile_pairs;
    const int col = col_base + j;
    const data_size_t pair = pair_base + p;
    if (col >= ncol || pair >= pairs) {
      continue;
    }
    const int group = tables.col_group[col];
    if (group < 0) {
      continue;
    }
    uint32_t vals[2] = {0, 0};
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
      const data_size_t row = pair * 2 - static_cast<data_size_t>(parity) + t;
      if (row < 0 || row >= rows) {
        continue;
      }
      const uint32_t enc = tables.lut[tables.col_lut_off[col] +
                                      DeviceLutIndex<T>(tile[j * tile_rows + 2 * p + t])];
      if (enc != kDeviceSkipBin) {
        vals[t] = enc;
      }
    }
    const uint8_t width = tables.group_width[group];
    uint8_t* group_out = out + tables.group_pair_off[group] * pairs;
    if (width == 0) {
      group_out[pair] = static_cast<uint8_t>(vals[0] | (vals[1] << 4));
      continue;
    }
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
      const data_size_t row = pair * 2 - static_cast<data_size_t>(parity) + t;
      if (row < 0 || row >= rows) {
        continue;
      }
      const data_size_t elem = pair * 2 + t;
      if (width == 1) {
        group_out[elem] = static_cast<uint8_t>(vals[t]);
      } else if (width == 2) {
        reinterpret_cast<uint16_t*>(group_out)[elem] =
            static_cast<uint16_t>(vals[t]);
      } else {
        reinterpret_cast<uint32_t*>(group_out)[elem] = vals[t];
      }
    }
  }
}

constexpr int kTileDefaultRows = 64;  // construct_tile_rows' default, compiled as a constant

template <typename T, int kThreads>
void LaunchTileThreads(const void* in, int64_t in_stride, int ncol,
                       data_size_t rows, data_size_t pairs, int parity,
                       int tile_rows, const CUDADenseBinnerTables& tables,
                       uint8_t* out, cudaStream_t stream) {
  const int tile_pairs = tile_rows / 2;
  const dim3 grid(static_cast<unsigned int>((pairs + tile_pairs - 1) / tile_pairs),
                  static_cast<unsigned int>((ncol + kTileCols - 1) / kTileCols));
  const size_t smem_bytes = static_cast<size_t>(kTileCols) * static_cast<size_t>(tile_rows) * sizeof(T);
  if (tile_rows == kTileDefaultRows) {
    CUDADenseBinTileKernel<T, kThreads, kTileDefaultRows><<<grid, kThreads, smem_bytes, stream>>>(
        reinterpret_cast<const T*>(in), in_stride, ncol, rows, pairs, parity,
        tile_rows, tables, out);
  } else {
    CUDADenseBinTileKernel<T, kThreads, 0><<<grid, kThreads, smem_bytes, stream>>>(
        reinterpret_cast<const T*>(in), in_stride, ncol, rows, pairs, parity,
        tile_rows, tables, out);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

template <typename T>
void LaunchTileTyped(const void* in, int64_t in_stride, int ncol,
                     data_size_t rows, data_size_t pairs, int parity,
                     int tile_rows, int tile_threads,
                     const CUDADenseBinnerTables& tables, uint8_t* out,
                     cudaStream_t stream) {
  // the knob values the plan accepts (FalcataPlan::ResolveFromConfig): an even
  // tile_rows within FalcataPlan::kConstructTileSmemBytes (48 KB), the dynamic
  // shared memory every CUDA device grants a block without opt-in (so the
  // launch needs no device query or attribute call), a power-of-two block of
  // 32-1024 threads
  CHECK(tile_rows >= 2 && tile_rows % 2 == 0 &&
        static_cast<size_t>(kTileCols) * static_cast<size_t>(tile_rows) * sizeof(T) <=
          static_cast<size_t>(FalcataPlan::kConstructTileSmemBytes));
  switch (tile_threads) {
    case 32:
      LaunchTileThreads<T, 32>(in, in_stride, ncol, rows, pairs, parity, tile_rows, tables, out, stream);
      break;
    case 64:
      LaunchTileThreads<T, 64>(in, in_stride, ncol, rows, pairs, parity, tile_rows, tables, out, stream);
      break;
    case 128:
      LaunchTileThreads<T, 128>(in, in_stride, ncol, rows, pairs, parity, tile_rows, tables, out, stream);
      break;
    case 256:
      LaunchTileThreads<T, 256>(in, in_stride, ncol, rows, pairs, parity, tile_rows, tables, out, stream);
      break;
    case 512:
      LaunchTileThreads<T, 512>(in, in_stride, ncol, rows, pairs, parity, tile_rows, tables, out, stream);
      break;
    case 1024:
      LaunchTileThreads<T, 1024>(in, in_stride, ncol, rows, pairs, parity, tile_rows, tables, out, stream);
      break;
    default:
      Log::Fatal("CUDADenseBinner: tile kernel block of %d threads (expected a power of two in [32, 1024])",
                 tile_threads);
  }
}

template <typename T>
__global__ void CUDAGatherRowsKernel(const T* __restrict__ in,
                                     bool is_row_major, data_size_t nrow,
                                     int ncol,
                                     const int32_t* __restrict__ row_ids,
                                     int num_samples, T* __restrict__ out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (idx >= static_cast<int64_t>(num_samples) * ncol) {
    return;
  }
  const int64_t sample = idx / ncol;
  const int col = static_cast<int>(idx - sample * ncol);
  const int64_t row = row_ids[sample];
  out[sample * ncol + col] = is_row_major
      ? in[row * ncol + col]
      : in[static_cast<int64_t>(col) * nrow + row];
}

template <typename T>
void GatherSampleRowsToHostTyped(const void* device_data, bool is_row_major,
                                 data_size_t nrow, int ncol,
                                 const int32_t* sample_row_ids,
                                 int num_samples, void* host_out) {
  int32_t* d_row_ids = nullptr;
  T* d_out = nullptr;
  const size_t out_elems = static_cast<size_t>(num_samples) * ncol;
  InitCUDAMemoryFromHostMemory<int32_t>(&d_row_ids, sample_row_ids,
                                        static_cast<size_t>(num_samples),
                                        __FILE__, __LINE__);
  AllocateCUDAMemory<T>(&d_out, out_elems, __FILE__, __LINE__);
  const int block_dim = 256;
  const int64_t total = static_cast<int64_t>(num_samples) * ncol;
  const int64_t num_blocks = (total + block_dim - 1) / block_dim;
  CUDAGatherRowsKernel<T><<<static_cast<unsigned int>(num_blocks), block_dim>>>(
      reinterpret_cast<const T*>(device_data), is_row_major, nrow, ncol,
      d_row_ids, num_samples, d_out);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
  CopyFromCUDADeviceToHost<T>(reinterpret_cast<T*>(host_out), d_out,
                              out_elems, __FILE__, __LINE__);
  DeallocateCUDAMemory<T>(&d_out, __FILE__, __LINE__);
  DeallocateCUDAMemory<int32_t>(&d_row_ids, __FILE__, __LINE__);
}

}  // anonymous namespace

void CUDAGatherSampleRowsToHost(const void* device_data, int64_t elem_size,
                                bool is_row_major, data_size_t nrow, int ncol,
                                const int32_t* sample_row_ids, int num_samples,
                                void* host_out) {
  if (num_samples <= 0) {
    return;
  }
  switch (elem_size) {
    case 1:
      GatherSampleRowsToHostTyped<uint8_t>(device_data, is_row_major, nrow,
                                           ncol, sample_row_ids, num_samples,
                                           host_out);
      break;
    case 2:
      GatherSampleRowsToHostTyped<uint16_t>(device_data, is_row_major, nrow,
                                            ncol, sample_row_ids, num_samples,
                                            host_out);
      break;
    case 4:
      GatherSampleRowsToHostTyped<uint32_t>(device_data, is_row_major, nrow,
                                            ncol, sample_row_ids, num_samples,
                                            host_out);
      break;
    case 8:
      GatherSampleRowsToHostTyped<uint64_t>(device_data, is_row_major, nrow,
                                            ncol, sample_row_ids, num_samples,
                                            host_out);
      break;
    default:
      Log::Fatal("CUDADenseBinner: unsupported element size %d",
                 static_cast<int>(elem_size));
  }
}

void LaunchCUDADenseBinChunkKernel(const void* in, int dtype_code,
                                   bool is_row_major, int64_t in_stride,
                                   data_size_t rows, data_size_t pairs,
                                   int parity, int num_groups,
                                   const CUDADenseBinnerTables& tables,
                                   uint8_t* out, cudaStream_t stream) {
  // this entry runs at Dataset construction, before any tree learner exists
  CheckCUDADeviceSupportsThisBuild(-1);
  switch (dtype_code) {
    case 0:
      LaunchTyped<float, false>(in, is_row_major, in_stride, rows, pairs,
                                parity, num_groups, tables, out, stream);
      break;
    case 1:
      LaunchTyped<double, false>(in, is_row_major, in_stride, rows, pairs,
                                parity, num_groups, tables, out, stream);
      break;
    case 2:
      LaunchTyped<int8_t, true>(in, is_row_major, in_stride, rows, pairs,
                                parity, num_groups, tables, out, stream);
      break;
    case 3:
      LaunchTyped<int16_t, true>(in, is_row_major, in_stride, rows, pairs,
                                parity, num_groups, tables, out, stream);
      break;
    case 4:
      LaunchTyped<uint8_t, true>(in, is_row_major, in_stride, rows, pairs,
                                parity, num_groups, tables, out, stream);
      break;
    case 5:
      LaunchTyped<uint16_t, true>(in, is_row_major, in_stride, rows, pairs,
                                parity, num_groups, tables, out, stream);
      break;
    default:
      Log::Fatal("CUDADenseBinner: unsupported dtype code %d", dtype_code);
  }
}

void LaunchCUDADenseBinTileKernel(const void* in, int dtype_code,
                                  int64_t in_stride, int ncol,
                                  data_size_t rows, data_size_t pairs,
                                  int parity, int tile_rows,
                                  int tile_threads,
                                  const CUDADenseBinnerTables& tables,
                                  uint8_t* out, cudaStream_t stream) {
  if (pairs <= 0) {
    return;
  }
  switch (dtype_code) {
    case 2:
      LaunchTileTyped<int8_t>(in, in_stride, ncol, rows, pairs, parity,
                              tile_rows, tile_threads, tables, out, stream);
      break;
    case 4:
      LaunchTileTyped<uint8_t>(in, in_stride, ncol, rows, pairs, parity,
                               tile_rows, tile_threads, tables, out, stream);
      break;
    default:
      Log::Fatal("CUDADenseBinner: no tile kernel for dtype code %d", dtype_code);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
