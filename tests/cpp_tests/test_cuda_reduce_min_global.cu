/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <Falcata/cuda/cuda_algorithms.hpp>

// ShuffleReduceMinGlobal<label_t, double> reduces each block to a minimum, then reduces the block minima. The block
// reduce used to start every thread at 0 and the first kernel padded the threads past the last value with 0, so any
// non-negative input reduced to 0 instead of its minimum. Sizes cover one partial block, an exact block, several
// blocks with a partial last one, and more blocks than threads in the second kernel.
TEST(CUDAAlgorithms, ShuffleReduceMinGlobalReturnsTheMinimumOfNonNegativeValues) {
  const std::vector<size_t> sizes = {1, 7, 1024, 1025, 5000, 2000003};
  for (const size_t n : sizes) {
    for (const float base : {3.5f, -2.0f}) {
      std::vector<Falcata::label_t> values(n);
      for (size_t i = 0; i < n; ++i) {
        values[i] = base + static_cast<float>((i * 7919) % 1000);
      }
      // the minimum somewhere past the first block, so the block-minimum reduce has to find it
      const size_t min_at = n / 2 + n / 3;
      values[min_at] = base - 1.0f;
      const double expected = static_cast<double>(*std::min_element(values.begin(), values.end()));

      Falcata::label_t* d_values = nullptr;
      double* d_blocks = nullptr;
      const size_t num_blocks = (n + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
      ASSERT_EQ(cudaMalloc(&d_values, n * sizeof(Falcata::label_t)), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_blocks, num_blocks * sizeof(double)), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_values, values.data(), n * sizeof(Falcata::label_t), cudaMemcpyHostToDevice),
                cudaSuccess);
      // stale contents must not reach the result
      ASSERT_EQ(cudaMemset(d_blocks, 0, num_blocks * sizeof(double)), cudaSuccess);
      Falcata::ShuffleReduceMinGlobal<Falcata::label_t, double>(d_values, n, d_blocks);
      ASSERT_EQ(cudaGetLastError(), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      double result = 0.0;
      ASSERT_EQ(cudaMemcpy(&result, d_blocks, sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(result, expected) << "n " << n << " base " << base;
      EXPECT_EQ(cudaFree(d_values), cudaSuccess);
      EXPECT_EQ(cudaFree(d_blocks), cudaSuccess);
    }
  }
}
