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
