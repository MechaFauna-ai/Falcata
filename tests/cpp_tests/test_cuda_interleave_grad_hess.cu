/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <Falcata/meta.h>

#include <cstring>
#include <vector>

namespace Falcata {
void LaunchInterleaveGradHessKernel(const score_t* gradients, const score_t* hessians, float2* gradients_hessians,
                                    data_size_t num_data, uint32_t* absmax_bits);
}  // namespace Falcata

// LaunchInterleaveGradHessKernel launched without checking the launch: a rejected launch (an empty grid for zero
// rows) left its error pending for whichever unrelated check read it next, and the caller marked the interleaved
// buffer valid. Zero rows must be a no-op that leaves no CUDA error behind, and a real launch must interleave every
// row and collect the max |gradient| and |hessian|.
TEST(CUDAHistogramConstructor, InterleaveGradHessLaunch) {
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_NO_THROW(Falcata::LaunchInterleaveGradHessKernel(nullptr, nullptr, nullptr, 0, nullptr));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  const int n = 3000;
  std::vector<Falcata::score_t> g(n), h(n);
  for (int i = 0; i < n; ++i) {
    g[i] = static_cast<Falcata::score_t>(i % 17) - 8.5f;
    h[i] = 0.25f + static_cast<Falcata::score_t>(i % 5);
  }
  Falcata::score_t *d_g = nullptr, *d_h = nullptr;
  float2* d_gh = nullptr;
  uint32_t* d_absmax = nullptr;
  ASSERT_EQ(cudaMalloc(&d_g, n * sizeof(Falcata::score_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_h, n * sizeof(Falcata::score_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_gh, n * sizeof(float2)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_absmax, 2 * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_g, g.data(), n * sizeof(Falcata::score_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_h, h.data(), n * sizeof(Falcata::score_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_absmax, 0, 2 * sizeof(uint32_t)), cudaSuccess);
  EXPECT_NO_THROW(Falcata::LaunchInterleaveGradHessKernel(d_g, d_h, d_gh, n, d_absmax));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float2> gh(n);
  uint32_t absmax[2] = {0, 0};
  ASSERT_EQ(cudaMemcpy(gh.data(), d_gh, n * sizeof(float2), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(absmax, d_absmax, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(gh[i].x, g[i]) << i;
    ASSERT_EQ(gh[i].y, h[i]) << i;
  }
  float max_g = 0.0f, max_h = 0.0f;
  memcpy(&max_g, &absmax[0], sizeof(float));
  memcpy(&max_h, &absmax[1], sizeof(float));
  EXPECT_EQ(max_g, 8.5f);
  EXPECT_EQ(max_h, 4.25f);
  EXPECT_EQ(cudaFree(d_g), cudaSuccess);
  EXPECT_EQ(cudaFree(d_h), cudaSuccess);
  EXPECT_EQ(cudaFree(d_gh), cudaSuccess);
  EXPECT_EQ(cudaFree(d_absmax), cudaSuccess);
}
