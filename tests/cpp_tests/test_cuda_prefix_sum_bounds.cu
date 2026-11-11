/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_bagging.hpp>

// ShufflePrefixSumGlobal(values, len, ...) bounded its reads by len but wrote values[index] for every thread of
// the last block, overwriting up to a block's worth of memory past len unless the caller padded the buffer.
TEST(CUDAAlgorithms, ShufflePrefixSumGlobalWritesOnlyLen) {
  const size_t len = 1000;  // not a multiple of the 1024-thread block
  const size_t guard = 2048;
  std::vector<uint32_t> host(len + guard, 7u);
  for (size_t i = 0; i < len; ++i) host[i] = 1u;
  uint32_t* d_values = nullptr;
  uint32_t* d_blocks = nullptr;
  ASSERT_EQ(cudaMalloc(&d_values, host.size() * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_blocks, 4 * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_values, host.data(), host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), cudaSuccess);
  Falcata::ShufflePrefixSumGlobal<uint32_t>(d_values, len, d_blocks);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint32_t> out(host.size());
  ASSERT_EQ(cudaMemcpy(out.data(), d_values, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(out[0], 1u);
  EXPECT_EQ(out[len - 1], static_cast<uint32_t>(len));  // inclusive scan of ones
  size_t overwritten = 0;
  for (size_t i = len; i < out.size(); ++i) overwritten += (out[i] != 7u);
  EXPECT_EQ(overwritten, 0u) << "entries past len were written";
  EXPECT_EQ(cudaFree(d_values), cudaSuccess);
  EXPECT_EQ(cudaFree(d_blocks), cudaSuccess);
}

// With no rows, CUDABaggingSample launched empty grids (an invalid configuration) and read the bag count from
// before its buffer. It must return 0 without a CUDA error.
TEST(CUDABagging, EmptyDataReturnsZero) {
  Falcata::CUDAVector<Falcata::data_size_t> scratch;
  Falcata::CUDAVector<Falcata::data_size_t> block_buffer;
  Falcata::CUDAVector<Falcata::data_size_t> out_indices(1);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  Falcata::data_size_t bag_cnt = -1;
  EXPECT_NO_THROW(bag_cnt = Falcata::CUDABaggingSample(0, 0.5, 0, 1, &scratch, &block_buffer, out_indices.RawData()));
  EXPECT_EQ(bag_cnt, 0);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
