/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <vector>

#include "../../src/treelearner/cuda/cuda_gradient_discretizer.hpp"

namespace {

// The dequantization scale the discretizer derives for the hessians of \p hessians (gradients all 0.5).
float HessScale(const std::vector<float>& hessians) {
  const Falcata::data_size_t num_data = static_cast<Falcata::data_size_t>(hessians.size());
  Falcata::CUDAGradientDiscretizer discretizer(4, 1, 7, false, false);
  discretizer.Init(num_data, 4, 0, nullptr);
  std::vector<float> gradients(hessians.size(), 0.5f);
  float* d_grad = nullptr;
  float* d_hess = nullptr;
  EXPECT_EQ(cudaMalloc(&d_grad, gradients.size() * sizeof(float)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&d_hess, hessians.size() * sizeof(float)), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(d_grad, gradients.data(), gradients.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(d_hess, hessians.data(), hessians.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  discretizer.DiscretizeGradients(num_data, d_grad, d_hess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  float scale = 0.0f;
  EXPECT_EQ(cudaMemcpy(&scale, discretizer.hess_scale_ptr(), sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaFree(d_grad), cudaSuccess);
  EXPECT_EQ(cudaFree(d_hess), cudaSuccess);
  return scale;
}

}  // namespace

// ReduceBlockMinMaxKernel block-reduced the hessian maximum twice and never the minimum, so the hessian scale saw
// only thread 0's partial minimum (the first reduce block's). A hessian of magnitude 10 must give the same scale
// whether it is +10 or -10, wherever it sits.
TEST(CUDAGradientDiscretizer, HessianScaleSeesTheMinimumOfEveryBlock) {
  const size_t num_data = 4 * 1024;  // four reduce blocks
  std::vector<float> positive(num_data, 1.0f);
  positive[2500] = 10.0f;  // in the third reduce block
  std::vector<float> negative(num_data, 1.0f);
  negative[2500] = -10.0f;
  const float positive_scale = HessScale(positive);
  const float negative_scale = HessScale(negative);
  EXPECT_GT(positive_scale, 0.0f);
  EXPECT_EQ(negative_scale, positive_scale);
}
