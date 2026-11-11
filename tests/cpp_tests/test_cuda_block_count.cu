/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <gtest/gtest.h>
#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/meta.h>

#include <cstdint>
#include <limits>

using Falcata::CUDABlockCount;
using Falcata::data_size_t;

// Launch block counts over row counts: (n + block - 1) / block evaluated in a
// 32-bit data_size_t overflows for n within block - 1 of INT32_MAX (signed
// overflow; in practice a negative block count and an invalid launch).
TEST(CUDABlockCount, CountsNearInt32Max) {
  constexpr data_size_t kMax = std::numeric_limits<data_size_t>::max();
  static_assert(CUDABlockCount(0, 1024) == 0, "an empty range needs no block");
  static_assert(CUDABlockCount(1, 1024) == 1, "one item needs one block");
  static_assert(CUDABlockCount(1024, 1024) == 1, "an exact multiple needs no extra block");
  static_assert(CUDABlockCount(1025, 1024) == 2, "a remainder needs one more block");
  // every block size the CUDA objective, metric, subset-copy and discretizer
  // launchers pass
  for (const int block : {256, 512, 1024}) {
    for (data_size_t n : {kMax, kMax - 1, kMax - block + 2, kMax - block + 1, kMax - block}) {
      const int64_t expected = (static_cast<int64_t>(n) + block - 1) / block;
      EXPECT_EQ(CUDABlockCount(n, block), expected) << "n=" << n << " block=" << block;
      EXPECT_GT(CUDABlockCount(n, block), 0);
    }
  }
  EXPECT_EQ(CUDABlockCount(kMax, 1024), 2097152);
  EXPECT_EQ(CUDABlockCount(kMax - 1023, 1024), 2097151);
  // the second-level reduction of the pointwise metrics divides a block count again
  EXPECT_EQ(CUDABlockCount(CUDABlockCount(kMax, 1024), 1024), 2048);
}
