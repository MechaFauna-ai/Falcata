/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include <Falcata/cuda/cuda_algorithms.hpp>

// ShuffleReduceMax/Min called back to back on one shared scratch buffer, with
// no barrier between the calls. Warp 0 reads every warp's slot after the
// first call's barrier; a warp > 0 that has finished the first call writes
// its slot for the second call at once. Without a leading barrier in the
// second call the write races with warp 0's read (a WAR hazard that
// compute-sanitizer --tool racecheck reports), and warp 0 can fold a
// second-call partial into the first result.

namespace {

constexpr int kThreads = 1024;
constexpr int kBlocks = 64;

__global__ void BackToBackReduceKernel(const float* a, const float* b, float* out) {
  __shared__ float scratch[32];
  const float va = a[threadIdx.x];
  const float vb = b[threadIdx.x];
  const float max_a = Falcata::ShuffleReduceMax<float>(va, scratch, blockDim.x);
  const float max_b = Falcata::ShuffleReduceMax<float>(vb, scratch, blockDim.x);
  const float min_a = Falcata::ShuffleReduceMin<float>(va, scratch, blockDim.x);
  const float min_b = Falcata::ShuffleReduceMin<float>(vb, scratch, blockDim.x);
  if (threadIdx.x == 0) {
    float* block_out = out + 4 * blockIdx.x;
    block_out[0] = max_a;
    block_out[1] = max_b;
    block_out[2] = min_a;
    block_out[3] = min_b;
  }
}

}  // namespace

TEST(CUDAShuffleReduce, BackToBackMaxMinOnOneScratchBuffer) {
  // a: values in [0, 1024) with the max in warp 0 and the min in the last
  // warp; b: every value larger than any of a (and its min above a's max), so
  // a leaked b partial changes the first max and a leaked a partial the
  // second min.
  std::vector<float> a(kThreads), b(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    a[i] = static_cast<float>(kThreads - 1 - i);
    b[i] = 10000.0f + static_cast<float>(i);
  }
  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_a, kThreads * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_b, kThreads * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, 4 * kBlocks * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_a, a.data(), kThreads * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_b, b.data(), kThreads * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  for (int rep = 0; rep < 50; ++rep) {
    BackToBackReduceKernel<<<kBlocks, kThreads>>>(d_a, d_b, d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> out(4 * kBlocks);
    ASSERT_EQ(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int block = 0; block < kBlocks; ++block) {
      EXPECT_EQ(out[4 * block + 0], static_cast<float>(kThreads - 1)) << "block " << block;
      EXPECT_EQ(out[4 * block + 1], 10000.0f + static_cast<float>(kThreads - 1)) << "block " << block;
      EXPECT_EQ(out[4 * block + 2], 0.0f) << "block " << block;
      EXPECT_EQ(out[4 * block + 3], 10000.0f) << "block " << block;
    }
  }
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}
