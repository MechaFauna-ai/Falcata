/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Standalone CUDA regression probe, linked to the library under test. Poison the claim counters to make a missing
 * initialization deterministic, independently of allocator contents. Run through the GPU scheduler, also with
 * compute-sanitizer --tool initcheck. See runs/device_loop/recovery for the exact build and execution manifest.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../../src/treelearner/cuda/cuda_device_levels.hpp"

namespace {
void Check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
    std::exit(2);
  }
}

bool Empty(const Falcata::CUDADeviceLevelState& state) {
  if (state.live_pairs != 0 || state.construct_total_blocks != 0) {
    return false;
  }
  for (int value : state.construct_next_block) {
    if (value != 0) {
      return false;
    }
  }
  return true;
}
}  // namespace

int main() {
  using Falcata::CUDADeviceLevelState;
  Falcata::CUDADeviceLevelConfig cfg{};
  Falcata::data_size_t* rows = nullptr;
  Check(cudaMalloc(&cfg.state, sizeof(CUDADeviceLevelState)));
  Check(cudaMalloc(&cfg.journal, Falcata::kDeviceLevelJournalHeader * sizeof(int)));
  Check(cudaMalloc(&rows, sizeof(*rows)));
  Check(cudaMemset(rows, 0, sizeof(*rows)));
  cfg.leaf_num_data = rows;
  Check(cudaMemset(cfg.state, 0xff, sizeof(CUDADeviceLevelState)));
  Falcata::LaunchDeviceLevelReset(cfg, 1, nullptr);
  CUDADeviceLevelState actual{};
  Check(cudaMemcpy(&actual, cfg.state, sizeof(actual), cudaMemcpyDeviceToHost));
  const bool reset_ok = Empty(actual);
  CUDADeviceLevelState stopped{};
  for (int& value : stopped.construct_next_block) {
    value = -1;
  }
  stopped.construct_grid_y = -1;
  stopped.construct_min_rows = -1;
  stopped.max_smaller = -1;
  Check(cudaMemcpy(cfg.state, &stopped, sizeof(stopped), cudaMemcpyHostToDevice));
  Falcata::LaunchDeviceLevelPairs(cfg, 0, 1, nullptr);
  Check(cudaMemcpy(&actual, cfg.state, sizeof(actual), cudaMemcpyDeviceToHost));
  const bool pairs_ok = Empty(actual) && actual.construct_grid_y == 0 && actual.construct_min_rows == 0 &&
    actual.max_smaller == 0;
  Check(cudaFree(rows));
  Check(cudaFree(cfg.journal));
  Check(cudaFree(cfg.state));
  std::printf("poisoned empty state: reset=%s pairs=%s\n", reset_ok ? "PASS" : "FAIL", pairs_ok ? "PASS" : "FAIL");
  return reset_ok && pairs_ok ? 0 : 1;
}
