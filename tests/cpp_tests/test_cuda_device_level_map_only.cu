/*!
 * Copyright (c) 2026 Falcata contributors. MIT license; see LICENSE.
 * UNBUILT/UNRUN standalone probe linked to the proposed library, not a replacement CUDA implementation.
 * Future execution/compilation requires its own authorized gpuq reservation. Not added to the normal CMake suite.
 */
#include <cuda_runtime.h>
#include <Falcata/c_api.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>
#include <Falcata/falcata_plan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/treelearner/cuda/cuda_data_partition.hpp"
#include "../../src/treelearner/cuda/cuda_device_levels.hpp"

namespace {
using Falcata::CUDASplitInfo;
using Falcata::data_size_t;
using Falcata::CUDADataPartition;
using Falcata::CUDAVector;
using Falcata::hist_t;
using Falcata::Dataset;
using Falcata::Config;
using Falcata::FalcataPlan;
using Falcata::CUDALeafSplitsStruct;
using Falcata::CUDAHybridApplyDescriptor;
using Falcata::CUDADeviceLevelState;
using Falcata::CUDADeviceLevelConfig;
void Check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <typename T> void Put(T* dst, const std::vector<T>& src) {
  Check(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T> std::vector<T> Get(const T* src, size_t size) {
  std::vector<T> result(size);
  Check(cudaMemcpy(result.data(), src, size * sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}
CUDASplitInfo MakeSplitInfo(int left_count, int right_count) {
  // User-provided ctor initializes ownership only. Initialize EVERY scalar read by its copy assignment.
  CUDASplitInfo split;
  split.is_valid = true;
  split.leaf_index = 0;
  split.inner_feature_index = 0;
  split.threshold = 0;
  split.default_left = true;
  split.left_count = left_count;
  split.right_count = right_count;
  split.left_sum_gradients = -5.0 * left_count;
  split.right_sum_gradients = 10.0 * right_count;
  split.left_sum_hessians = 4.0 * left_count;
  split.right_sum_hessians = 4.0 * right_count;
  // Packed signed grad32/hess32, without undefined negative signed shifts.
  split.left_sum_of_gradients_hessians = (-5LL * left_count) * 4294967296LL + 4LL * left_count;
  split.right_sum_of_gradients_hessians = (10LL * right_count) * 4294967296LL + 4LL * right_count;
  split.left_value = 1.25;
  split.right_value = -2.5;
  split.left_gain = split.left_sum_gradients * split.left_sum_gradients / split.left_sum_hessians;
  split.right_gain = split.right_sum_gradients * split.right_sum_gradients / split.right_sum_hessians;
  split.gain = split.left_gain + split.right_gain;
  split.num_cat_threshold = 0;
  split.cat_threshold = nullptr;
  split.cat_threshold_real = nullptr;
  split.num_vec_targets = 0;
  split.vec_payload = nullptr;
  return split;
}
struct Snapshot {
  std::vector<data_size_t> indices, counts, starts;
  std::vector<int> map, split_info;
};

Snapshot Run(const Dataset* dataset, bool map_only, const std::string& action) {
  constexpr int kLeaves = 8;
  if (action == "plan_transition") {
    Config creation;
    creation.Set({{"device_type", "cuda"}, {"cuda_plan", "auto,leaf_map_small_blocks:off"}});
    FalcataPlan::ResolveFromConfig(creation);
    Require(CUDADataPartition::DeviceLevelGapBlockDim() == 1024, "creation plan must use1024-thread gaps");
  }
  const int creation_gap_block = CUDADataPartition::DeviceLevelGapBlockDim();
  const int rows = dataset->num_data();
  const int parent_rows = rows / 2 + 3;
  const int gap_rows = rows - parent_rows;
  const int flat = (parent_rows + 1023) / 1024;
  CUDAVector<hist_t> hist;
  hist.Resize(2 * kLeaves);
  CUDADataPartition partition(dataset, 1, kLeaves, 32, true, hist.RawData());
  partition.Init();
  partition.EnsureDeviceLevelCapacity(kLeaves);
  partition.BeforeTrain();
  Check(cudaDeviceSynchronize());
  auto* old_main = partition.hybrid_graph_main_indices();
  auto* old_out = partition.hybrid_graph_out_indices();
  std::vector<data_size_t> original(rows);
  std::iota(original.rbegin(), original.rend(), 0);  // shuffled old parent row IDs; not child order
  Put(old_main, original);
  const std::vector<data_size_t> poison(rows, -777777);
  Put(old_out, poison);
  std::vector<uint8_t> bins(rows);
  for (int i = 0; i < rows; ++i) bins[i] = (i * 17 + 9) % 5 > 1 ? 1 : 0;
  CUDAVector<uint8_t> column;
  column.InitFromHostVector(bins);
  CUDAVector<CUDASplitInfo> best;
  int left_count = 0;
  for (int pos = 0; pos < parent_rows; ++pos) left_count += bins[original[pos]] == 0;
  Require(left_count > 0 && left_count < parent_rows, "split must have two nonempty children");
  const CUDASplitInfo split = MakeSplitInfo(left_count, parent_rows - left_count);
  best.InitFromHostVector(std::vector<CUDASplitInfo>{split});
  CUDAVector<CUDALeafSplitsStruct> slots;
  slots.Resize(2);
  CUDAHybridApplyDescriptor d{};
  d.column_data = column.RawDataReadOnly();
  d.best_split_info = best.RawDataReadOnly();
  d.smaller_leaf_splits = slots.RawData();
  d.larger_leaf_splits = slots.RawData() + 1;
  d.num_data_in_leaf = parent_rows;
  d.num_blocks = flat;
  d.left_leaf_index = 0;
  d.right_leaf_index = 3;
  d.bit_type = 8;
  d.use_min_bin = 1;
  d.max_bin = 1;
  CUDAHybridApplyDescriptor gap{};
  gap.leaf_data_start = parent_rows;
  gap.num_data_in_leaf = gap_rows;
  gap.left_leaf_index = 2;  // nonzero terminal ID; leaf1 is empty and has no gap descriptor
  gap.num_blocks = (gap_rows + creation_gap_block - 1) / creation_gap_block;
  gap.flat_block_start = flat;
  Put(partition.device_level_apply_descs(), std::vector<CUDAHybridApplyDescriptor>{d, gap});
  std::vector<data_size_t> counts(kLeaves, 0), starts(kLeaves, 0);
  counts[0] = parent_rows;
  counts[2] = gap_rows;
  starts[2] = parent_rows;
  Put(const_cast<data_size_t*>(partition.cuda_leaf_num_data()), counts);
  Put(const_cast<data_size_t*>(partition.cuda_leaf_data_start()), starts);
  CUDADeviceLevelState state{};
  state.live_splits = 1;
  state.total_flat_blocks = flat;
  state.num_gaps = 1;
  state.gap_flat_blocks = (gap_rows + 1023) / 1024;
  state.num_leaves = 3;
  state.levels_applied = 1;
  CUDAVector<CUDADeviceLevelState> device_state;
  device_state.InitFromHostVector(std::vector<CUDADeviceLevelState>{state});
  CUDADeviceLevelConfig cfg{};
  cfg.state = device_state.RawData();
  cfg.whole_tree = 1;
  cfg.gap_block_dim = creation_gap_block;
  cfg.max_depth = 1;  // pairs block MUST skip child-search pointers on final depth
  CUDAVector<CUDADeviceLevelConfig> device_config;
  device_config.InitFromHostVector(std::vector<CUDADeviceLevelConfig>{cfg});
  partition.LaunchDeviceLevelApply(2, device_state.RawData(), device_config.RawData(), true, map_only);
  Check(cudaDeviceSynchronize());
  partition.FinishDeviceLevels(1, 1, 3);  // pre-level three leaves were set above; actual leaves=4
  if (map_only) {
    Require(Get(old_out, rows) == poison, "deferred fused apply wrote child or terminal out-index bytes");
    partition.DeferDeviceLevelPartition(1, 1, gap.num_blocks, flat, creation_gap_block);
    Require(partition.cuda_data_indices() == old_main, "positive final correction did not restore parents");
  }
  std::vector<int> expected(rows, 2);
  std::vector<data_size_t> expected_order;
  for (int bin : {0, 1}) {
    for (int pos = 0; pos < parent_rows; ++pos) {
      const int row = original[pos];
      expected[row] = bins[row] == 0 ? 0 : 3;
      if (bins[row] == bin) expected_order.push_back(row);
    }
  }
  expected_order.insert(expected_order.end(), original.begin() + parent_rows, original.end());
  Require(Get(partition.cuda_data_index_to_leaf_index(), rows) == expected,
          "complete map omitted split rows, nonzero-ID terminal rows or the odd tail");
  const auto exact_counts = Get(partition.cuda_leaf_num_data(), 4);
  const auto exact_starts = Get(partition.cuda_leaf_data_start(), 4);
  partition.PrefetchSplitBatchAsync(1, partition.apply_stream());
  Check(cudaStreamSynchronize(partition.apply_stream()));
  std::vector<int> info;
  partition.ReadPrefetchedSplitBatch(1, &info);
  info.resize(16);  // slots16/17 are NCCL-only; do not compare uninitialized irrelevant bytes
  if (action == "plan_transition") {
    // A DIFFERENT Dataset resolves the process-global plan while our DL receipt still owns1024-thread units.
    std::vector<float> other_values{0, 1, 0, 1};
    DatasetHandle other = nullptr;
    Require(FLC_DatasetCreateFromMat(other_values.data(), C_API_DTYPE_FLOAT32, 4, 1, 1,
      "device_type=cuda max_bin=7 min_data_in_leaf=1 feature_pre_filter=false cuda_plan=auto,leaf_map_small_blocks:on",
      nullptr, &other) == 0, "create other plan-resolving Dataset failed");
    FLC_DatasetFree(other);
    Require(CUDADataPartition::DeviceLevelGapBlockDim() == 256, "other Dataset must actually switch gap width");
    Require(creation_gap_block == 1024, "descriptor creation width was not snapshotted");
    if (rows == 4111) Require(gap_rows == 2053 && gap.num_blocks == 3, "2053-row/3-block counterexample changed");
  } else if (action == "reset") {
    Config config;
    config.num_leaves = kLeaves + 3;
    partition.ResetConfig(&config, hist.RawData());
  } else if (action == "leaf_pred") {
    partition.ResetByLeafPred(expected, 4);  // MUST repair before replacing the map (smoothed-refit boundary)
  } else if (action == "scratch") {
    partition.PresizeLevelBuffers();
  } else if (action == "next_root") {
    partition.BeforeTrain();  // old assignment deliberately superseded; no old repair may run afterward
    Require(partition.cuda_data_indices() == old_out,
            "next root did not restore the ordinary materialized target before identity fill");
    expected_order.resize(rows);
    std::iota(expected_order.begin(), expected_order.end(), 0);
  } else {
    Require(action == "ensure", "unknown native probe action");
  }
  partition.EnsureLevelPartitioned();
  Check(cudaDeviceSynchronize());
  const auto* repaired = partition.cuda_data_indices();
  Require(Get(repaired, rows) == expected_order, "lazy repair changed any stable partition row/window order");
  partition.EnsureLevelPartitioned();
  Require(partition.cuda_data_indices() == repaired, "second Ensure swapped the buffers again");
  // Compare metadata at successful final completion, and row-window bytes/map AFTER the boundary action.
  // A next-root assignment intentionally overwrites root counts; do not confuse it with the completed tree.
  return {Get(repaired, rows), exact_counts, exact_starts,
          Get(partition.cuda_data_index_to_leaf_index(), rows), info};
}
}  // namespace

int main(int argc, char** argv) {
  // Each invocation is a future gpuq child, same reservation SID. Never use as an out-of-queue CPU probe.
  DatasetHandle handle = nullptr;
  try {
    Require(argc == 3, "usage: probe rows ensure|reset|leaf_pred|scratch|next_root|plan_transition");
    const int rows = std::stoi(argv[1]);
    Require(rows >= 2051, "fixture needs split and terminal chunks, with tails");
    std::vector<float> input(rows);
    for (int i = 0; i < rows; ++i) input[i] = static_cast<float>(i % 5);
    Require(FLC_DatasetCreateFromMat(input.data(), C_API_DTYPE_FLOAT32, rows, 1, 1,
      "max_bin=7 min_data_in_leaf=1 feature_pre_filter=false num_threads=32", nullptr, &handle) == 0,
      "create dataset failed");
    Falcata::Config config;
    config.Set({{"device_type", "cuda"}, {"num_leaves", "8"}, {"quant_mode", "fixedpoint"},
      {"num_threads", "32"}, {"cuda_plan", "auto"}});
    Falcata::FalcataPlan::ResolveFromConfig(config);
    auto* dataset = reinterpret_cast<Falcata::Dataset*>(handle);
    const auto full = Run(dataset, false, argv[2]);
    const auto deferred = Run(dataset, true, argv[2]);
    Require(full.indices == deferred.indices && full.map == deferred.map && full.counts == deferred.counts &&
      full.starts == deferred.starts && full.split_info == deferred.split_info, "full/deferred outputs differ");
    FLC_DatasetFree(handle);
    std::puts("PASS: map/counts/records equal, untouched deferred out storage, exact lazy repair and idempotence");
    return 0;
  } catch (const std::exception& error) {
    if (handle != nullptr) FLC_DatasetFree(handle);
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
