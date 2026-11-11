/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "../../src/treelearner/cuda/cuda_best_split_finder_kernels.cuh"

// The CPU's Common::RoundInt as fixed in https://github.com/MechaFauna-ai/Falcata/pull/322: halves away from
// zero, saturation beyond the int range, NaN to 0. The device count estimate round(hessian * cnt_factor) must agree
// with it, also for negative products (custom objectives can return negative hessians).
namespace {

int ReferenceRoundInt(double x) {
  const double r = x >= 0.0 ? x + 0.5f : x - 0.5f;
  if (r >= 2147483648.0) return std::numeric_limits<int>::max();
  if (r <= -2147483649.0) return std::numeric_limits<int>::min();
  if (r != r) return 0;
  return static_cast<int>(r);
}

// the float overload rounds in float arithmetic (as before for non-negative values), with the same tie rule,
// saturation and NaN handling
int ReferenceRoundIntFloat(float x) {
  const float r = x >= 0.0f ? x + 0.5f : x - 0.5f;
  if (r >= 2147483648.0f) return std::numeric_limits<int>::max();
  if (r <= -2147483648.0f) return std::numeric_limits<int>::min();
  if (r != r) return 0;
  return static_cast<int>(r);
}

__global__ void RoundIntKernel(const double* in, int n, int* out_double, int* out_float) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out_double[i] = Falcata::CUDARoundInt(in[i]);
    out_float[i] = Falcata::CUDARoundInt(static_cast<float>(in[i]));
  }
}

}  // namespace

TEST(CUDARoundInt, MatchesTheCPURoundIntIncludingNegativeAndOutOfRange) {
  std::vector<double> values = {0.0, -0.0, 0.25, 0.5, 1.5, 2.5, 3.49, -0.25, -0.5, -1.5, -2.5, -3.49, -3.5, -7.0,
                                -1e6 - 0.5, 1e6 + 0.5, 2147483646.4, -2147483647.4, 3e9, -3e9, 1e300, -1e300,
                                std::numeric_limits<double>::quiet_NaN()};
  for (int k = -2000; k <= 2000; ++k) {
    values.push_back(k * 0.25);
  }
  const int n = static_cast<int>(values.size());
  double* d_in = nullptr;
  int *d_double = nullptr, *d_float = nullptr;
  ASSERT_EQ(cudaMalloc(&d_in, n * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_double, n * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_float, n * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_in, values.data(), n * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
  RoundIntKernel<<<(n + 255) / 256, 256>>>(d_in, n, d_double, d_float);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<int> out_double(n), out_float(n);
  ASSERT_EQ(cudaMemcpy(out_double.data(), d_double, n * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out_float.data(), d_float, n * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(out_double[i], ReferenceRoundInt(values[i])) << "double " << values[i];
    const float f = static_cast<float>(values[i]);
    EXPECT_EQ(out_float[i], ReferenceRoundIntFloat(f)) << "float " << f;
  }
  cudaFree(d_in);
  cudaFree(d_double);
  cudaFree(d_float);
}
