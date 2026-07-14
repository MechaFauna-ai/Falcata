/*!
 * Copyright (c) 2026 The ExaBoost developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.hu>

#include <cstdint>

#include "cuda_dense_binner.hpp"

namespace LightGBM {

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
// (replicating the host push order) and writes the final packed value once
template <typename T, bool ROW_MAJOR, bool USE_LUT>
__global__ void CUDADenseBinChunkKernel(const T* __restrict__ in,
                                        int64_t in_stride, data_size_t rows,
                                        data_size_t pairs,
                                        CUDADenseBinnerTables tables,
                                        uint8_t* __restrict__ out) {
  const int group = static_cast<int>(blockIdx.y);
  const data_size_t pair = static_cast<data_size_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (pair >= pairs) {
    return;
  }
  const data_size_t row0 = pair * 2;
  const int num_rows = (row0 + 1 < rows) ? 2 : 1;
  uint32_t vals[2] = {0, 0};
  const int k_end = tables.group_col_ptr[group + 1];
  for (int k = tables.group_col_ptr[group]; k < k_end; ++k) {
    const int col = tables.group_cols[k];
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
      if (t >= num_rows) {
        break;
      }
      const data_size_t row = row0 + t;
      const T raw = ROW_MAJOR
          ? in[static_cast<int64_t>(row) * in_stride + col]
          : in[static_cast<int64_t>(col) * in_stride + row];
      uint32_t enc;
      if constexpr (USE_LUT) {
        enc = tables.lut[tables.col_lut_off[col] + DeviceLutIndex<T>(raw)];
      } else {
        const CUDADenseBinnerColMeta meta = tables.col_meta[col];
        enc = DeviceEncodeBin(
            DeviceValueToBin(static_cast<double>(raw), tables.bounds, meta),
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
  } else if (width == 1) {
    group_out[row0] = static_cast<uint8_t>(vals[0]);
    if (num_rows == 2) {
      group_out[row0 + 1] = static_cast<uint8_t>(vals[1]);
    }
  } else if (width == 2) {
    uint16_t* out16 = reinterpret_cast<uint16_t*>(group_out);
    out16[row0] = static_cast<uint16_t>(vals[0]);
    if (num_rows == 2) {
      out16[row0 + 1] = static_cast<uint16_t>(vals[1]);
    }
  } else {
    uint32_t* out32 = reinterpret_cast<uint32_t*>(group_out);
    out32[row0] = vals[0];
    if (num_rows == 2) {
      out32[row0 + 1] = vals[1];
    }
  }
}

template <typename T, bool USE_LUT>
void LaunchTyped(const void* in, bool is_row_major, int64_t in_stride,
                 data_size_t rows, data_size_t pairs, int num_groups,
                 const CUDADenseBinnerTables& tables, uint8_t* out,
                 cudaStream_t stream) {
  const int block_dim = 256;
  dim3 grid((pairs + block_dim - 1) / block_dim,
            static_cast<unsigned int>(num_groups));
  const T* typed_in = reinterpret_cast<const T*>(in);
  if (is_row_major) {
    CUDADenseBinChunkKernel<T, true, USE_LUT><<<grid, block_dim, 0, stream>>>(
        typed_in, in_stride, rows, pairs, tables, out);
  } else {
    CUDADenseBinChunkKernel<T, false, USE_LUT><<<grid, block_dim, 0, stream>>>(
        typed_in, in_stride, rows, pairs, tables, out);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

}  // anonymous namespace

void LaunchCUDADenseBinChunkKernel(const void* in, int dtype_code,
                                   bool is_row_major, int64_t in_stride,
                                   data_size_t rows, data_size_t pairs,
                                   int num_groups,
                                   const CUDADenseBinnerTables& tables,
                                   uint8_t* out, cudaStream_t stream) {
  switch (dtype_code) {
    case 0:
      LaunchTyped<float, false>(in, is_row_major, in_stride, rows, pairs,
                                num_groups, tables, out, stream);
      break;
    case 1:
      LaunchTyped<double, false>(in, is_row_major, in_stride, rows, pairs,
                                num_groups, tables, out, stream);
      break;
    case 2:
      LaunchTyped<int8_t, true>(in, is_row_major, in_stride, rows, pairs,
                                num_groups, tables, out, stream);
      break;
    case 3:
      LaunchTyped<int16_t, true>(in, is_row_major, in_stride, rows, pairs,
                                num_groups, tables, out, stream);
      break;
    case 4:
      LaunchTyped<uint8_t, true>(in, is_row_major, in_stride, rows, pairs,
                                num_groups, tables, out, stream);
      break;
    case 5:
      LaunchTyped<uint16_t, true>(in, is_row_major, in_stride, rows, pairs,
                                num_groups, tables, out, stream);
      break;
    default:
      Log::Fatal("CUDADenseBinner: unsupported dtype code %d", dtype_code);
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
