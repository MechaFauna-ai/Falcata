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
  }
  if (threadIdx.x < kDeviceLevelJournalHeader) {
    cfg.journal[threadIdx.x] = 0;
  }
  for (int c = static_cast<int>(threadIdx.x); c < kDeviceLevelClaimColumns; c += blockDim.x) {
    st->construct_next_block[c] = 0;
  }
}

// One level's host decisions between its search and its apply (DeviceLevelSelectBlock, cuda_device_levels.cuh), as a
// launch of its own where the level's sync does not run them in its last block (MAX_SPLITS: kDeviceLevelMaxPairs for
// the prefix, kDeviceTreeMaxSplits for a whole tree, whose final level has up to 2^(max_depth - 1) splits)
template <int MAX_SPLITS>
__global__ void DeviceLevelSelectKernel(const CUDADeviceLevelConfig cfg, const int level) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the level sync's leaf entries are complete and visible
  DeviceLevelSelectBlock<MAX_SPLITS>(cfg, level);
}

// The children's search descriptors of the level just applied (DeviceLevelPairsBlock, cuda_device_levels.cuh), as a
// launch of its own where the apply does not run it beside its partition
__global__ void DeviceLevelPairsKernel(const CUDADeviceLevelConfig cfg, const int level) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the apply's counts and child structs are complete and visible
  DeviceLevelPairsBlock(cfg, level);
}

}  // anonymous namespace

void LaunchDeviceLevelReset(const CUDADeviceLevelConfig& config, const int num_leaves, cudaStream_t stream) {
  DeviceLevelResetKernel<<<1, 32, 0, stream>>>(config, num_leaves);
}

void LaunchDeviceLevelSelect(const CUDADeviceLevelConfig& config, const int level, const int max_splits,
                             cudaStream_t stream) {
  // cuda_plan key pdl_level_chain: a programmatic dependent of the level sync launched just before on this stream
  if (config.whole_tree) {
    LaunchLevelChainKernel(PdlLevelChainEnabled(), DeviceLevelSelectKernel<kDeviceTreeMaxSplits>, dim3(1),
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
