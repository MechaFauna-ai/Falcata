/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief cuda_plan key device_levels: the select and pairs kernels of the device-driven level prefix (see
 *  cuda_device_levels.hpp). Each replays, for one level, exactly the host decisions of the two-sync level loop it
 *  stands in for (CollectSplittableLeaves + ApplyLevelBatched + SplitLevelBatched; EnqueueLevelBestSplitSearch +
 *  LaunchConstructHistogramBatchedKernel's grid math), so the kernels after them see the same descriptors.
 */

#ifdef USE_CUDA

#include "cuda_device_levels.hpp"
#include "cuda_device_levels.cuh"

#include <Falcata/cuda/cuda_utils.hu>

#include <algorithm>

#include <Falcata/cuda/cuda_split_info.hpp>
#include <Falcata/cuda/cuda_tree.hpp>

#include "cuda_data_partition.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_hybrid_graph.hpp"
#include "cuda_leaf_splits.hpp"
#include "cuda_pdl_launch.hpp"

namespace Falcata {

namespace {

__global__ void DeviceLevelResetKernel(const CUDADeviceLevelConfig cfg, const int num_leaves) {
  CUDADeviceLevelState* st = cfg.state;
  if (threadIdx.x == 0) {
    if (cfg.whole_tree) {
      // Root search has consumed the initialized fixed root struct on this stream. Preserve its exact output
      // before the first tree record saves leaf 0 as internal_value[0]. The select event orders the record stream.
      cfg.tree_leaf_value[0] = cfg.root_struct->leaf_value;
    }
    st->live_splits = 0;
    st->total_flat_blocks = 0;
    st->num_gaps = 0;
    st->gap_flat_blocks = 0;
    st->split_info_base = 0;
    st->num_leaves = num_leaves;
    st->root_num_data = cfg.leaf_num_data[0];
    st->levels_applied = 0;
    st->done = 0;
    st->live_pairs = 0;
    st->construct_grid_y = 0;
    st->construct_min_rows = 0;
    st->construct_total_blocks = 0;
    st->max_smaller = 0;
    st->sync_blocks_done = 0;
    st->construct_block_start = cfg.construct_block_start;
  }
  if (threadIdx.x < kDeviceLevelJournalHeader) {
    cfg.journal[threadIdx.x] = 0;
  }
  for (int c = static_cast<int>(threadIdx.x); c < kDeviceLevelClaimColumns; c += blockDim.x) {
    st->construct_next_block[c] = 0;
  }
}

// One level's host decisions between its search and its apply (cuda_device_levels.cuh), as a launch of its own where
// the level's sync does not run them in its last block (the prefix's root level, or every level when the sync kernel
// is not the used-task one). MAX_SPLITS: kDeviceLevelMaxPairs for the prefix (never binding: DeviceLevelSelectBlockFast,
// the body the fused copy runs), kDeviceTreeSharedSplits for a whole tree (DeviceLevelSelectBlock, whose final level
// has up to 2^(max_depth - 1) splits and may bind); GLOBAL: a whole tree's level wider than kDeviceTreeSharedSplits
// (the select's arrays in global scratch)
template <int MAX_SPLITS, bool GLOBAL = false>
__global__ void DeviceLevelSelectKernel(const CUDADeviceLevelConfig cfg, const int level) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the level sync's leaf entries are complete and visible
  if constexpr (GLOBAL || MAX_SPLITS == kDeviceTreeSharedSplits) {
    DeviceLevelSelectBlock<MAX_SPLITS, GLOBAL>(cfg, level);
  } else {
    DeviceLevelSelectBlockFast<MAX_SPLITS>(cfg, level);
  }
}

// The children's search descriptors of the level just applied (DeviceLevelPairsBlock, cuda_device_levels.cuh), as a
// launch of its own where the apply does not run it beside its partition
__global__ void DeviceLevelPairsKernel(const CUDADeviceLevelConfig cfg, const int level) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the apply's counts and child structs are complete and visible
  DeviceLevelPairsBlock(cfg, level);
}

// ---- cuda_plan key device_bynode_masks ----

constexpr int kDeviceBynodeThreads = 128;

// ColSampler::MixKey (splitmix64's finalizer)
__device__ __forceinline__ uint64_t DeviceBynodeMixKey(uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// One node's sample (ColSampler::GetByNodeKeyed with Random::Sample), its mask row and its task list. Thread 0 runs
// the generator's draws in Random::Sample's order (each draw depends on the ones before it); the block clears and
// writes the row and compacts the task list in ascending task index.
__device__ __forceinline__ void DeviceBynodeDrawNode(const CUDADeviceBynodeConfig& cfg, const uint64_t key,
                                                     const int leaf, const int slot) {
  extern __shared__ uint32_t s_bits[];
  __shared__ int s_warp[32];
  __shared__ int s_total;
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  const int* sample = cfg.sample;
  const int N = sample[0];
  const int K = sample[1];
  const int mode = sample[2];
  const int* candidate_inner = sample + 5;
  const int words = (N + 31) >> 5;
  for (int w = tid; w < words; w += nt) {
    uint32_t bits = 0;
    if (mode == 1) {
      // K == N: every candidate, in order
      const int left = N - (w << 5);
      bits = left >= 32 ? 0xFFFFFFFFu : ((1u << left) - 1u);
    }
    s_bits[w] = bits;
  }
  __syncthreads();
  if (tid == 0 && (mode == 2 || mode == 3)) {
    const uint64_t seed_mix = static_cast<uint64_t>(static_cast<uint32_t>(sample[3])) |
      (static_cast<uint64_t>(static_cast<uint32_t>(sample[4])) << 32);
    // Random(static_cast<int>(static_cast<uint32_t>(mixed >> 32))): the generator's state is those 32 bits
    uint32_t x = static_cast<uint32_t>(DeviceBynodeMixKey(seed_mix ^ key) >> 32);
    if (mode == 2) {
      // for i in [0, N): take i when NextFloat() < (K - taken) / double(N - i). NextFloat() is r / 32768 with
      // r = (x >> 16) & 0x7FFF, exact in float and double; the compare equals r * (N - i) < (K - taken) * 32768 in
      // integers: a quotient above r / 32768 exceeds it by at least 2^-15 / (N - i), far more than half an ulp, so
      // its rounding never reaches r / 32768, and one at or below it never rounds above it
      int taken = 0;
      for (int i = 0; i < N && taken < K; ++i) {
        x = 214013u * x + 2531011u;
        const uint64_t r = static_cast<uint64_t>((x >> 16) & 0x7FFFu);
        if (r * static_cast<uint64_t>(N - i) < (static_cast<uint64_t>(K - taken) << 15)) {
          s_bits[i >> 5] |= 1u << (i & 31);
          ++taken;
        }
      }
    } else {
      // Floyd: for r in [N - K, N): v = NextInt(0, r + 1); insert v, or r when v is already in
      for (int r = N - K; r < N; ++r) {
        x = 214013u * x + 2531011u;
        const int v = static_cast<int>(x & 0x7FFFFFFFu) % (r + 1);
        const int pick = ((s_bits[v >> 5] >> (v & 31)) & 1u) ? r : v;
        s_bits[pick >> 5] |= 1u << (pick & 31);
      }
    }
  }
  int8_t* row = cfg.rows + static_cast<size_t>(slot) * static_cast<size_t>(cfg.num_features);
  for (int f = tid; f < cfg.num_features; f += nt) {
    row[f] = 0;
  }
  __syncthreads();
  for (int i = tid; i < N; i += nt) {
    if ((s_bits[i >> 5] >> (i & 31)) & 1u) {
      row[candidate_inner[i]] = 1;
    }
  }
  __syncthreads();
  // the tasks of the sampled features, ascending (CommitLevelNodeMasks sorts each row's tasks)
  int* list = cfg.tasks + 2 * cfg.num_slots + static_cast<size_t>(slot) * static_cast<size_t>(cfg.max_tasks);
  int count = 0;
  for (int base = 0; base < cfg.num_tasks; base += nt) {
    const int t = base + tid;
    const int flag = (t < cfg.num_tasks && row[cfg.task_inner_feature[t]] != 0) ? 1 : 0;
    const int pos = count + DeviceLevelBlockScan(flag, s_warp, &s_total);
    if (flag && pos < cfg.max_tasks) {
      list[pos] = t;
    }
    count += s_total;
  }
  if (tid == 0) {
    // int range: PlanDeviceTreeChunks caps the chunk so 2 * num_slots + row slots * max_tasks fits (64-bit math here)
    cfg.tasks[2 * leaf] = static_cast<int>(2 * static_cast<int64_t>(cfg.num_slots) +
                                           static_cast<int64_t>(slot) * cfg.max_tasks);
    cfg.tasks[2 * leaf + 1] = min(count, cfg.max_tasks);
    if (cfg.table != nullptr) {
      cfg.table[leaf] = row;  // pair chunking: the leaf's row lives in its chunk slot
    }
  }
}

__global__ void DeviceBynodeRootMaskKernel(const CUDADeviceBynodeConfig cfg, const uint64_t root_key) {
  if (threadIdx.x == 0) {
    cfg.keys[0] = root_key;
  }
  DeviceBynodeDrawNode(cfg, root_key, 0, 0);
}

// block (k, side): side 0 the left child of the level's split chunk_base + k (the parent's leaf index), side 1 the
// right child. Pair chunking (cfg.table): the child's row slot is 1 + 2 k + side within the chunk; otherwise its leaf
__global__ void DeviceBynodeLevelMasksKernel(const CUDADeviceBynodeConfig cfg, const CUDADeviceLevelState* st,
                                             const CUDAHybridApplyDescriptor* apply_descs, const int level,
                                             const int chunk_base) {
  const int k = chunk_base + static_cast<int>(blockIdx.x);
  if (k >= st->live_pairs) {
    return;
  }
  const CUDAHybridApplyDescriptor& a = apply_descs[k];
  const int left = a.left_leaf_index;
  const int leaf = blockIdx.y == 0 ? left : a.right_leaf_index;
  // CUDASingleGPUTreeLearner::BynodeKeySplit
  const uint64_t parent = cfg.keys[static_cast<size_t>(level & 1) * cfg.num_slots + left];
  const uint64_t key = DeviceBynodeMixKey(parent ^ (blockIdx.y == 0 ? 0x6A09E667F3BCC908ULL : 0xBB67AE8584CAA73BULL));
  if (threadIdx.x == 0) {
    cfg.keys[static_cast<size_t>((level + 1) & 1) * cfg.num_slots + leaf] = key;
  }
  DeviceBynodeDrawNode(cfg, key, leaf, cfg.table != nullptr ? 1 + 2 * static_cast<int>(blockIdx.x) +
                                                                  static_cast<int>(blockIdx.y) : leaf);
}

}  // anonymous namespace

void LaunchDeviceBynodeRootMask(const CUDADeviceBynodeConfig& config, const uint64_t root_key, cudaStream_t stream) {
  DeviceBynodeRootMaskKernel<<<1, kDeviceBynodeThreads, DeviceBynodeSharedBytes(config.num_features), stream>>>(
    config, root_key);
}

void LaunchDeviceBynodeLevelMasks(const CUDADeviceBynodeConfig& config, const CUDADeviceLevelState* state,
                                  const CUDAHybridApplyDescriptor* apply_descs, const int level, const int max_splits,
                                  const int chunk_base, cudaStream_t stream) {
  DeviceBynodeLevelMasksKernel<<<dim3(max_splits, 2), kDeviceBynodeThreads,
                                 DeviceBynodeSharedBytes(config.num_features), stream>>>(
    config, state, apply_descs, level, chunk_base);
}

void LaunchDeviceLevelReset(const CUDADeviceLevelConfig& config, const int num_leaves, cudaStream_t stream) {
  DeviceLevelResetKernel<<<1, 32, 0, stream>>>(config, num_leaves);
}

void LaunchDeviceLevelSelect(const CUDADeviceLevelConfig& config, const int level, const int max_splits,
                             cudaStream_t stream) {
  // cuda_plan key pdl_level_chain: a programmatic dependent of the level sync launched just before on this stream
  if (config.whole_tree && max_splits > kDeviceTreeSharedSplits) {
    // a level wider than the shared select: the same select over global scratch (DeviceLevelBitonicSortGlobal)
    LaunchLevelChainKernel(PdlLevelChainEnabled(), DeviceLevelSelectKernel<kDeviceTreeSharedSplits, true>, dim3(1),
                           dim3(DeviceLevelThreads(max_splits)), 0, stream, config, level);
  } else if (config.whole_tree) {
    LaunchLevelChainKernel(PdlLevelChainEnabled(), DeviceLevelSelectKernel<kDeviceTreeSharedSplits>, dim3(1),
                           dim3(DeviceLevelThreads(max_splits)), 0, stream, config, level);
  } else {
    LaunchLevelChainKernel(PdlLevelChainEnabled(), DeviceLevelSelectKernel<kDeviceLevelMaxPairs>, dim3(1),
                           dim3(DeviceLevelThreads(max_splits)), 0, stream, config, level);
  }
}

void LaunchDeviceLevelPairs(const CUDADeviceLevelConfig& config, const int level, const int max_splits,
                            cudaStream_t stream) {
  // cuda_plan key pdl_level_chain: a programmatic dependent of the apply's partition kernel launched just before
  LaunchLevelChainKernel(PdlLevelChainEnabled(), DeviceLevelPairsKernel, dim3(1), dim3(DeviceLevelThreads(max_splits)),
                         0, stream, config, level);
}

}  // namespace Falcata

#endif  // USE_CUDA
