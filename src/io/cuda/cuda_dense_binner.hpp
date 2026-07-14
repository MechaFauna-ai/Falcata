/*!
 * Copyright (c) 2026 The ExaBoost developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_CUDA_CUDA_DENSE_BINNER_HPP_
#define LIGHTGBM_IO_CUDA_CUDA_DENSE_BINNER_HPP_

#ifdef USE_CUDA

#include <LightGBM/meta.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace LightGBM {

// per-column metadata for the on-device float/double binning
// (replicates BinMapper::ValueToBin + FeatureGroup::EncodeBinForPush)
struct CUDADenseBinnerColMeta {
  int64_t bounds_offset;   // into the flattened bin_upper_bound_ array
  int num_bin;
  uint32_t most_freq_bin;
  uint32_t sub_offset;     // FeatureGroup bin offset of the subfeature (0 pre-decrement)
  uint8_t missing_is_nan;  // missing_type == MissingType::NaN
};

// device tables shared by every chunk of one matrix
struct CUDADenseBinnerTables {
  const int* group_col_ptr;      // [num_groups + 1]
  const int* group_cols;         // ascending column ids per group
  const uint8_t* group_width;    // bytes per element; 0 means 4-bit packed
  const int64_t* group_pair_off;  // bytes-per-row-pair prefix offset per group
  const uint32_t* lut;           // LUT mode: encoded bins (kSkipBin sentinel)
  const int64_t* col_lut_off;    // LUT mode: per-column offset into lut
  const double* bounds;          // float mode: flattened bin upper bounds
  const CUDADenseBinnerColMeta* col_meta;  // float mode
};

// bins one chunk of a dense matrix into per-group packed column bins.
// dtype_code: 0=float32 1=float64 2=int8 3=int16 4=uint8 5=uint16(+fp16 bits)
// in_stride: elements per row (row-major) or per column (col-major)
void LaunchCUDADenseBinChunkKernel(const void* in, int dtype_code,
                                   bool is_row_major, int64_t in_stride,
                                   data_size_t rows, data_size_t pairs,
                                   int num_groups,
                                   const CUDADenseBinnerTables& tables,
                                   uint8_t* out, cudaStream_t stream);

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_IO_CUDA_CUDA_DENSE_BINNER_HPP_
