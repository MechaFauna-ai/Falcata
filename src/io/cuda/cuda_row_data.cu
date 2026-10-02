/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_row_data.hpp>

#include <algorithm>

namespace Falcata {

// One thread per row writes the row's packed bytes of one partition. The partition's columns sit in `staging`
// at col_offsets[c], as the Dataset stores them: 4-bit columns hold two rows per byte (low nibble = even row),
// 8-bit columns one row per byte.
__global__ void PackDenseNibblesPartitionKernel(const uint8_t* staging, const size_t* col_offsets,
                                                const uint8_t* col_bits, const int num_columns,
                                                const int packed_width, const data_size_t num_data,
                                                uint8_t* out) {
  auto fetch = [&](int c, data_size_t row) -> uint32_t {
    const uint8_t* col = staging + col_offsets[c];
    return col_bits[c] == 4 ? (col[row >> 1] >> ((row & 1) << 2)) & 0xf : col[row];
  };
  for (data_size_t row = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x); row < num_data;
       row += static_cast<data_size_t>(gridDim.x * blockDim.x)) {
    uint8_t* out_row = out + static_cast<size_t>(row) * packed_width;
    for (int j = 0; j < packed_width; ++j) {
      const uint32_t lo = fetch(2 * j, row);
      const uint32_t hi = 2 * j + 1 < num_columns ? fetch(2 * j + 1, row) : 0;
      out_row[j] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
}

void CUDARowData::LaunchPackDenseNibblesPartition(const uint8_t* staging, const size_t* col_offsets,
                                                  const uint8_t* col_bits, const int num_columns,
                                                  const int packed_width, uint8_t* out) const {
  const int block = 256;
  const int grid = std::max(1, std::min<int>((num_data_ + block - 1) / block, 65535));
  PackDenseNibblesPartitionKernel<<<grid, block>>>(staging, col_offsets, col_bits, num_columns, packed_width,
                                                  num_data_, out);
}

}  // namespace Falcata

#endif  // USE_CUDA
