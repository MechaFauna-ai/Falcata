/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include <Falcata/cuda/cuda_algorithms.hpp>

// ShufflePrefixSumExclusive: after the warp-total scan, warp w + 1 reads its base from scratch slot w while warp w
// overwrote that slot with its hand-off value, with no barrier in between (a race compute-sanitizer racecheck
// reports; for floating types the two values can differ in the last bits). Two calls back to back on one scratch
// buffer also need the leading barrier ShufflePrefixSum has.

namespace {

constexpr int kThreads = 1024;
constexpr int kBlocks = 32;

__global__ void BackToBackExclusiveKernel(const double* in, double* out) {
  __shared__ double scratch[32];
  const double v = in[blockIdx.x * blockDim.x + threadIdx.x];
  const double first = Falcata::ShufflePrefixSumExclusive<double>(v, scratch);
  const double second = Falcata::ShufflePrefixSumExclusive<double>(2.0 * v, scratch);
  out[2 * (blockIdx.x * blockDim.x + threadIdx.x)] = first;
  out[2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1] = second;
}

}  // namespace

TEST(CUDAPrefixSumExclusive, BackToBackCallsOnOneScratchBuffer) {
  const int n = kThreads * kBlocks;
  // integer-valued doubles: every partial sum is exact, so the expected exclusive scan is unambiguous
  std::vector<double> in(n);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<double>((i * 37) % 101) - 50.0;
  }
  double *d_in = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_in, n * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, 2 * n * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_in, in.data(), n * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
  std::vector<double> out(2 * n);
  for (int rep = 0; rep < 20; ++rep) {
    BackToBackExclusiveKernel<<<kBlocks, kThreads>>>(d_in, d_out);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(out.data(), d_out, out.size() * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int b = 0; b < kBlocks; ++b) {
      double prefix = 0.0;
      for (int t = 0; t < kThreads; ++t) {
        const int i = b * kThreads + t;
        ASSERT_EQ(out[2 * i], prefix) << "block " << b << " thread " << t;
        ASSERT_EQ(out[2 * i + 1], 2.0 * prefix) << "block " << b << " thread " << t;
        prefix += in[i];
      }
    }
  }
  cudaFree(d_in);
  cudaFree(d_out);
}
