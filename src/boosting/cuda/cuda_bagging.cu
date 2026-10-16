/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_bagging.hpp>
#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>

#include "../../treelearner/cuda/cuda_pdl_launch.hpp"

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

// cuda_plan key bagging_fused_scan: the same flags and the same [in-bag | out-of-bag] layout in three kernels without
// the flag array: each block counts its kept rows, one block turns the counts into block bases (and the total), and
// each block recomputes its flags (the Philox draw is a pure function of seed, iteration and row) and scatters every
// row from its block base plus its rank in the block.
// 32-lane warp ballots and scans: CUDA only (ROCm builds keep the flag pass and the global prefix sum)
#if !defined(__HIP_PLATFORM_AMD__)
constexpr int kBagFusedBlock = GLOBAL_PREFIX_SUM_BLOCK_SIZE;
static_assert(kBagFusedBlock % 32 == 0 && kBagFusedBlock <= 1024, "whole warps, at most 32 of them");

__device__ __forceinline__ bool BagKeep(const data_size_t i, const float bagging_fraction, const int iter,
                                        const uint32_t seed) {
  const uint4 rand = BagPhilox10(make_uint4(static_cast<uint32_t>(i), static_cast<uint32_t>(iter), 0u, 0u),
                                 make_uint2(seed, 0x9E3779B9u));
  return BagPhiloxUniform(rand.x) < bagging_fraction;
}

__global__ void BaggingBlockCountKernel(const data_size_t num_data, const float bagging_fraction, const int iter,
                                        const uint32_t seed, data_size_t* block_counts) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel of the chain may get resident
  const data_size_t i = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const bool keep = i < num_data && BagKeep(i, bagging_fraction, iter, seed);
  const int count = __syncthreads_count(keep);
  // cuda_plan key pdl_level_chain: the draw above reads no memory, so it runs while the kernel before it on the stream
  // drains; the store below waits for that kernel's completion and memory (the same store as an ordinary launch)
  PdlWaitForProducer();
  if (threadIdx.x == 0) {
    block_counts[blockIdx.x] = static_cast<data_size_t>(count);
  }
}

// one block: block_counts[b] -> exclusive base of block b, in place; block_counts[num_blocks] = the bag count
__global__ void BaggingBlockBaseKernel(const int num_blocks, data_size_t* block_counts) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the kernel before it on the stream is complete and visible
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel of the chain may get resident
  __shared__ data_size_t warp_sums[kBagFusedBlock / 32];
  __shared__ data_size_t carry;
  if (threadIdx.x == 0) {
    carry = 0;
  }
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  for (int start = 0; start < num_blocks; start += kBagFusedBlock) {
    const int b = start + static_cast<int>(threadIdx.x);
    const data_size_t v = b < num_blocks ? block_counts[b] : 0;
    data_size_t incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const data_size_t n = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += n;
    }
    if (lane == 31) warp_sums[warp] = incl;
    __syncthreads();
    if (warp == 0) {
      data_size_t w = lane < kBagFusedBlock / 32 ? warp_sums[lane] : 0;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const data_size_t n = __shfl_up_sync(0xffffffffu, w, o);
        if (lane >= o) w += n;
      }
      if (lane < kBagFusedBlock / 32) warp_sums[lane] = w;  // inclusive warp prefix
    }
    __syncthreads();
    const data_size_t warp_base = warp > 0 ? warp_sums[warp - 1] : 0;
    const data_size_t base = carry;
    if (b < num_blocks) {
      block_counts[b] = base + warp_base + incl - v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      carry = base + warp_sums[kBagFusedBlock / 32 - 1];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_counts[num_blocks] = carry;
  }
}

__global__ void BaggingBlockScatterKernel(const data_size_t num_data, const float bagging_fraction, const int iter,
                                          const uint32_t seed, const data_size_t* block_bases,
                                          data_size_t* out_indices) {
  __shared__ data_size_t warp_counts[kBagFusedBlock / 32];
  const data_size_t i = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const bool keep = i < num_data && BagKeep(i, bagging_fraction, iter, seed);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const uint32_t ballot = __ballot_sync(0xffffffffu, keep);
  if (lane == 0) warp_counts[warp] = __popc(ballot);
  __syncthreads();
  data_size_t warp_base = 0;
  for (int w = 0; w < warp; ++w) warp_base += warp_counts[w];
  // cuda_plan key pdl_level_chain: the draw and the block's own counts above touch no global memory, so they run
  // while the block base kernel before it drains; the base read and the scatter's stores wait for its completion and
  // memory (the same reads and stores as an ordinary launch)
  PdlWaitForProducer();
  // the inclusive kept count through row i, as the global scan's prefix[i]
  const data_size_t kept_through_i = block_bases[blockIdx.x] + warp_base +
    static_cast<data_size_t>(__popc(ballot & (0xffffffffu >> (31 - lane))));
  if (i >= num_data) {
    return;
  }
  if (keep) {
    out_indices[kept_through_i - 1] = i;
  } else {
    const data_size_t dropped_through_i = (i + 1) - kept_through_i;
    out_indices[num_data - dropped_through_i] = i;
  }
}
#endif  // !defined(__HIP_PLATFORM_AMD__)

data_size_t CUDABaggingSample(const data_size_t num_data,
                              const double bagging_fraction,
                              const int iter,
                              const int seed,
                              CUDAVector<data_size_t>* scratch,
                              CUDAVector<data_size_t>* block_buffer,
                              data_size_t* out_indices) {
#if !defined(__HIP_PLATFORM_AMD__)
  if (FalcataPlan::Get().bagging_fused_scan) {
    const int fused_blocks = (num_data + kBagFusedBlock - 1) / kBagFusedBlock;
    if (block_buffer->Size() < static_cast<size_t>(fused_blocks) + 1) {
      block_buffer->Resize(static_cast<size_t>(fused_blocks) + 1);
    }
    data_size_t* block_counts = block_buffer->RawData();
    // cuda_plan key pdl_level_chain: the count (after the gradients), the base and the scatter as programmatic
    // dependents of the kernel before each
    const bool pdl = PdlLevelChainEnabled();
    LaunchLevelChainKernel(pdl, BaggingBlockCountKernel, dim3(fused_blocks), dim3(kBagFusedBlock), 0, nullptr, num_data,
                           static_cast<float>(bagging_fraction), iter, static_cast<uint32_t>(seed), block_counts);
    LaunchLevelChainKernel(pdl, BaggingBlockBaseKernel, dim3(1), dim3(kBagFusedBlock), 0, nullptr, fused_blocks,
                           block_counts);
    LaunchLevelChainKernel(pdl, BaggingBlockScatterKernel, dim3(fused_blocks), dim3(kBagFusedBlock), 0, nullptr,
                           num_data, static_cast<float>(bagging_fraction), iter, static_cast<uint32_t>(seed),
                           block_counts, out_indices);
    data_size_t fused_bag_cnt = 0;
    CopyFromCUDADeviceToHost<data_size_t>(&fused_bag_cnt, block_counts + fused_blocks, 1, __FILE__, __LINE__);
    return fused_bag_cnt;
  }
#endif  // !defined(__HIP_PLATFORM_AMD__)
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
