/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef FALCATA_SRC_BOOSTING_CUDA_NCCL_ROW_SPLIT_HPP_
#define FALCATA_SRC_BOOSTING_CUDA_NCCL_ROW_SPLIT_HPP_

#include <Falcata/meta.h>

#include <algorithm>
#include <cstdint>

namespace Falcata {

// The data-parallel multi-GPU row split, kept free of CUDA so tests/cpp_tests can pin it on a CPU build.
// The rows of one machine are cut into num_ranks consecutive chunks of ceil(num_data / num_ranks) rows, rank r taking
// [r * chunk, (r + 1) * chunk) clipped to num_data. num_ranks is the number of ranks that TRAIN on this machine (the
// NCCL topology's num_gpu(): num_gpu clamped to the device count, or the size of gpu_device_id_list), never the
// number of devices the machine has: dividing by the device count left the rows past num_ranks * chunk on no rank.
struct NCCLRowRange {
  data_size_t start;
  data_size_t end;
  data_size_t num_data() const { return end - start; }
};

// Rows of the largest rank's share (rank 0's): ceil(num_data / num_ranks).
inline data_size_t NCCLMaxNumDataPerRank(const data_size_t num_data, const int num_ranks) {
  const int64_t ranks = std::max(num_ranks, 1);
  return static_cast<data_size_t>((static_cast<int64_t>(num_data) + ranks - 1) / ranks);
}

// Rank `rank`'s rows. Every row lands on exactly one rank and the counts sum to num_data; a rank past the last row
// (num_data < num_ranks, or a short tail chunk) gets an empty range at num_data, never a negative count.
inline NCCLRowRange NCCLRankRowRange(const data_size_t num_data, const int num_ranks, const int rank) {
  const int64_t chunk = NCCLMaxNumDataPerRank(num_data, num_ranks);
  const int64_t start = std::min<int64_t>(chunk * rank, num_data);
  const int64_t end = std::min<int64_t>(start + chunk, num_data);
  return {static_cast<data_size_t>(start), static_cast<data_size_t>(end)};
}

}  // namespace Falcata

#endif  // FALCATA_SRC_BOOSTING_CUDA_NCCL_ROW_SPLIT_HPP_
