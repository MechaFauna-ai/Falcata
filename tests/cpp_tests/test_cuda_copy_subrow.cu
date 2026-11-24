/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include <Falcata/cuda/cuda_column_data.hpp>

// CUDAColumnData::CopySubrow of an empty subset launched its copy kernel with an empty grid, an invalid
// configuration whose error stayed pending until an unrelated check read it. It must copy nothing and leave no
// CUDA error behind.
TEST(CUDAColumnData, CopySubrowOfEmptySubsetLeavesNoError) {
  const std::array<uint8_t, 4> bins = {{0, 1, 2, 3}};
  Falcata::CUDAColumnData full(4, 0);
  full.Init(3, {bins.data(), bins.data(), bins.data()}, {nullptr, nullptr, nullptr}, {8, 8, 8}, {3, 3, 3}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 1, 2});
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  Falcata::CUDAColumnData subset(0, 0);
  const Falcata::data_size_t no_rows[1] = {0};
  EXPECT_NO_THROW(subset.CopySubrow(&full, no_rows, 0));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(CUDAColumnData, DenseBatchesPreserveColumnBytesAndAlignment) {
  // Odd row count exercises the last packed nibble. The columns exceed one
  // batch and mix every supported dense encoding.
  constexpr Falcata::data_size_t rows = 65537;
  constexpr int columns = 160;
  std::vector<std::vector<uint8_t>> bins(columns);
  std::vector<const void*> column_data(columns);
  std::vector<uint8_t> bit_types(columns);
  std::vector<int> feature_to_column(columns);
  constexpr std::array<uint8_t, 4> bits = {{4, 8, 16, 32}};
  for (int c = 0; c < columns; ++c) {
    bit_types[c] = bits[c % bits.size()];
    auto& bytes = bins[c];
    bytes.resize((static_cast<size_t>(rows) * bit_types[c] + 7) / 8);
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<uint8_t>(i + c);
    }
    column_data[c] = bytes.data();
    feature_to_column[c] = c;
  }
  const std::vector<uint32_t> feature_bins(columns, 0);
  const std::vector<uint8_t> missing(columns, 0);
  Falcata::CUDAColumnData full(rows, 0);
  full.Init(columns, column_data, std::vector<Falcata::BinIterator*>(columns, nullptr),
            bit_types, feature_bins, feature_bins, feature_bins, feature_bins, feature_bins,
            missing, missing, missing, missing, feature_to_column);
  for (int c = 0; c < columns; ++c) {
    const uint8_t* device = full.GetColumnData(c);
    ASSERT_NE(device, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(device) % 256, 0U);
    std::vector<uint8_t> actual(bins[c].size());
    ASSERT_EQ(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(actual == bins[c]) << "column " << c;
  }
  // Successive column views share storage. The first two columns fit in the
  // same batch, and the first column occupies ceil(rows/2) bytes rounded up.
  const auto first = reinterpret_cast<uintptr_t>(full.GetColumnData(0));
  const auto second = reinterpret_cast<uintptr_t>(full.GetColumnData(1));
  EXPECT_EQ(second - first, (bins[0].size() + 255) / 256 * 256);
}
