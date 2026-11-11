/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <vector>

#include <Falcata/cuda/cuda_algorithms.hpp>

namespace {

__global__ void WarpMinKernel(const float* values, const int len, float* out) {
  const float min_value = Falcata::ShuffleReduceMinWarp<float>(values[threadIdx.x], len);
  if (threadIdx.x == 0) {
    *out = min_value;
  }
}

}  // namespace

// ShuffleReduceMinWarp(value, len): lane 0 returns the minimum over lanes [0, len) while every lane of the warp
// calls it. With a mask of only the first len lanes, the lanes inside it read the padding lanes' values (and the
// lanes outside it executed an undefined shuffle), so a smaller padding value leaked into the result.
TEST(CUDAAlgorithms, ShuffleReduceMinWarpIgnoresLanesPastLen) {
  float* d_values = nullptr;
  float* d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_values, 32 * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, sizeof(float)), cudaSuccess);
  for (const int len : {1, 2, 5, 16, 17, 31, 32}) {
    std::vector<float> values(32, -1000.0f);  // padding lanes hold smaller values than any real one
    for (int lane = 0; lane < len; ++lane) {
      values[lane] = 100.0f + static_cast<float>(len - 1 - lane);  // minimum (100) in the last real lane
    }
    ASSERT_EQ(cudaMemcpy(d_values, values.data(), 32 * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    WarpMinKernel<<<1, 32>>>(d_values, len, d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    float result = 0.0f;
    ASSERT_EQ(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result, 100.0f) << "len " << len;
  }
  EXPECT_EQ(cudaFree(d_values), cudaSuccess);
  EXPECT_EQ(cudaFree(d_out), cudaSuccess);
}
