/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_bagging.hpp>
#include <Falcata/cuda/cuda_utils.hu>

namespace Falcata {

// Philox4x32-10, identical to the stochastic-rounding generator in
// cuda_gradient_discretizer.cu. Counter-based, so the bag for row r of
// iteration t is a pure function of (seed, t, r): no host RNG state, no
// dependence on thread or block count, and reproducible on any GPU. That is
// what lets the sampling move to the device at all -- a stateful host RNG
// walked in row order cannot be parallelized without changing its output.
__device__ __forceinline__ uint4 BagPhiloxRound(const uint4 ctr, const uint2 key) {
  const uint32_t hi0 = __umulhi(0xD2511F53u, ctr.x);
  const uint32_t lo0 = 0xD2511F53u * ctr.x;
  const uint32_t hi1 = __umulhi(0xCD9E8D57u, ctr.z);
  const uint32_t lo1 = 0xCD9E8D57u * ctr.z;
  return make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

__device__ __forceinline__ uint4 BagPhilox10(uint4 ctr, uint2 key) {
  #pragma unroll
  for (int r = 0; r < 10; ++r) {
    ctr = BagPhiloxRound(ctr, key);
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
  }
  return ctr;
}

// uniform in [0, 1): top 24 bits -> exact float
__device__ __forceinline__ float BagPhiloxUniform(const uint32_t x) {
  return static_cast<float>(x >> 8) * 5.9604644775390625e-08f;  // 2^-24
}

__global__ void BaggingFlagKernel(const data_size_t num_data,
                                  const float bagging_fraction,
                                  const int iter,
                                  const uint32_t seed,
                                  data_size_t* flags) {
  const data_size_t i = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i >= num_data) {
    return;
  }
  const uint4 rand = BagPhilox10(make_uint4(static_cast<uint32_t>(i), static_cast<uint32_t>(iter), 0u, 0u),
                                 make_uint2(seed, 0x9E3779B9u));
  flags[i] = BagPhiloxUniform(rand.x) < bagging_fraction ? 1 : 0;
}

// Scatter into the [in-bag | out-of-bag] layout GBDT::UpdateScore depends on:
// it scores the first bag_cnt entries through the tree learner's partition and
// the remainder by explicit traversal.
//
// prefix holds the INCLUSIVE count of kept rows up to and including i, so a
// kept row lands at prefix[i]-1, and the d-th dropped row (d = i+1-prefix[i])
// lands at num_data-d. Deriving the dropped slot from the tail means neither
// kernel needs the final bag count, so no device->host round trip sits between
// the scan and the scatter.
__global__ void BaggingScatterKernel(const data_size_t num_data,
                                     const data_size_t* prefix,
                                     data_size_t* out_indices) {
  const data_size_t i = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i >= num_data) {
    return;
  }
  const data_size_t kept_through_i = prefix[i];
  const data_size_t kept_before_i = (i == 0) ? 0 : prefix[i - 1];
  if (kept_through_i > kept_before_i) {
    out_indices[kept_through_i - 1] = i;
  } else {
    const data_size_t dropped_through_i = (i + 1) - kept_through_i;
    out_indices[num_data - dropped_through_i] = i;
  }
}

data_size_t CUDABaggingSample(const data_size_t num_data,
                              const double bagging_fraction,
                              const int iter,
                              const int seed,
                              CUDAVector<data_size_t>* scratch,
                              CUDAVector<data_size_t>* block_buffer,
                              data_size_t* out_indices) {
  constexpr int kBlock = GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  const int num_blocks = (num_data + kBlock - 1) / kBlock;
  // ShufflePrefixSumGlobalKernel writes values[index] without a bounds check,
  // so the scan buffer is padded to a whole number of blocks.
  const size_t padded = static_cast<size_t>(num_blocks) * kBlock;
  if (scratch->Size() < padded) {
    scratch->Resize(padded);
  }
  if (block_buffer->Size() < static_cast<size_t>(num_blocks) + 1) {
    block_buffer->Resize(static_cast<size_t>(num_blocks) + 1);
  }
  data_size_t* flags = scratch->RawData();
  SetCUDAMemory<data_size_t>(flags + num_data, 0, padded - num_data, __FILE__, __LINE__);

  BaggingFlagKernel<<<num_blocks, kBlock>>>(num_data, static_cast<float>(bagging_fraction),
                                            iter, static_cast<uint32_t>(seed), flags);
  ShufflePrefixSumGlobal<data_size_t>(flags, padded, block_buffer->RawData());
  BaggingScatterKernel<<<num_blocks, kBlock>>>(num_data, flags, out_indices);

  // The only host round trip left is these 4 bytes: GBDT needs the count to
  // split the score update into its in-bag and out-of-bag halves.
  data_size_t bag_cnt = 0;
  CopyFromCUDADeviceToHost<data_size_t>(&bag_cnt, flags + (num_data - 1), 1, __FILE__, __LINE__);
  return bag_cnt;
}

}  // namespace Falcata

#endif  // USE_CUDA
