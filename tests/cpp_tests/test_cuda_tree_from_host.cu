/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/tree.h>

#include <vector>

#include <Falcata/cuda/cuda_tree.hpp>

// CUDATree(const Tree* host_tree) uploads the host tree through InitCUDA, which skipped the vector-leaf outputs, so
// a vector-leaf tree built this way had no device leaf outputs for the vector score update to read.
TEST(CUDATree, FromHostTreeUploadsVectorLeafOutputs) {
  const int dim = 3;
  Falcata::Tree host(4, false, false, dim);
  for (int t = 0; t < dim; ++t) host.SetLeafOutput(0, t, 0.25 * (t + 1));
  ASSERT_EQ(host.leaf_value_dim(), dim);

  Falcata::CUDATree cuda_tree(&host);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_NE(cuda_tree.cuda_leaf_values_vec(), nullptr) << "vector-leaf outputs were not uploaded";
  std::vector<double> device(dim, 0.0);
  ASSERT_EQ(cudaMemcpy(device.data(), cuda_tree.cuda_leaf_values_vec(), dim * sizeof(double), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int t = 0; t < dim; ++t) EXPECT_EQ(device[t], host.LeafOutput(0, t)) << "target " << t;
}
