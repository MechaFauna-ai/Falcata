/*!
 * Copyright (c) 2026 Falcata contributors. MIT license; see LICENSE.
 * Standalone real-kernel row-coverage and captured physical-grid probe. Run only inside gpuq.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
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
void CheckAt(cudaError_t code, const char* call, int line) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "CUDA_CHECK line=%d call=%s code=%d: %s\n",
                 line, call, static_cast<int>(code), cudaGetErrorString(code));
    throw std::runtime_error(cudaGetErrorString(code));
  }
}
#define CUDA_CHECK(call) CheckAt((call), #call, __LINE__)
#define FALCATA_TEST_STRINGIFY_IMPL(x) #x
#define FALCATA_TEST_STRINGIFY(x) FALCATA_TEST_STRINGIFY_IMPL(x)
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <typename T> void Put(T* dst, const std::vector<T>& src) {
  CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T> std::vector<T> Get(const T* src, size_t size) {
  std::vector<T> result(size);
  CUDA_CHECK(cudaMemcpy(result.data(), src, size * sizeof(T), cudaMemcpyDeviceToHost));
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

Snapshot Run(const Dataset* dataset, bool bounded, bool capture, unsigned int* physical_grid) {
  constexpr int kLeaves = 1024;
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
  CUDA_CHECK(cudaDeviceSynchronize());
  auto* old_main = partition.device_level_main_indices();
  auto* old_out = partition.device_level_out_indices();
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
  d.left_leaf_index = 511;
  d.right_leaf_index = 512;
  d.bit_type = 8;
  d.use_min_bin = 1;
  d.max_bin = 1;
  CUDAHybridApplyDescriptor gap{};
  gap.leaf_data_start = parent_rows;
  gap.num_data_in_leaf = gap_rows;
  gap.left_leaf_index = 509;  // nonzero terminal ID; leaf1 is empty and has no gap descriptor
  gap.num_blocks = (gap_rows + creation_gap_block - 1) / creation_gap_block;
  gap.flat_block_start = flat;
  Put(partition.device_level_apply_descs(), std::vector<CUDAHybridApplyDescriptor>{d, gap});
  std::vector<data_size_t> counts(kLeaves, 0), starts(kLeaves, 0);
  counts[511] = parent_rows;
  counts[509] = gap_rows;
  starts[509] = parent_rows;
  Put(const_cast<data_size_t*>(partition.cuda_leaf_num_data()), counts);
  Put(const_cast<data_size_t*>(partition.cuda_leaf_data_start()), starts);
  CUDADeviceLevelState state{};
  state.live_splits = 1;
  state.total_flat_blocks = flat;
  state.num_gaps = 1;
  state.gap_flat_blocks = (gap_rows + 1023) / 1024;
  state.num_leaves = 512;
  state.levels_applied = 10;
  CUDAVector<CUDADeviceLevelState> device_state;
  device_state.InitFromHostVector(std::vector<CUDADeviceLevelState>{state});
  CUDADeviceLevelConfig cfg{};
  cfg.state = device_state.RawData();
  cfg.whole_tree = 1;
  cfg.gap_block_dim = creation_gap_block;
  cfg.max_depth = 10;  // pairs block MUST skip child-search pointers on final depth
  CUDAVector<CUDADeviceLevelConfig> device_config;
  device_config.InitFromHostVector(std::vector<CUDADeviceLevelConfig>{cfg});
  auto apply = [&]() {
    partition.LaunchDeviceLevelApply(512, device_state.RawData(), device_config.RawData(), true, false, bounded);
  };
  *physical_grid = 0;
  if (capture) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(partition.apply_stream(), cudaStreamCaptureModeGlobal));
    apply();
    CUDA_CHECK(cudaStreamEndCapture(partition.apply_stream(), &graph));
    size_t count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
    // Driver query returns the real CUfunction/grid, without translating a library kernel to a runtime host stub.
    static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    Require(driver != nullptr, "load CUDA driver for standalone graph introspection failed");
    using GetKernelParams = decltype(&cuGraphKernelNodeGetParams);
    const auto get_kernel_params = reinterpret_cast<GetKernelParams>(
      dlsym(driver, FALCATA_TEST_STRINGIFY(cuGraphKernelNodeGetParams)));
    Require(get_kernel_params != nullptr, "header-selected CUDA driver graph-query symbol missing");
    std::printf("Captured kernel grids via %s\n", FALCATA_TEST_STRINGIFY(cuGraphKernelNodeGetParams));
    for (auto node : nodes) {
      cudaGraphNodeType type;
      CUDA_CHECK(cudaGraphNodeGetType(node, &type));
      if (type == cudaGraphNodeTypeKernel) {
        CUDA_KERNEL_NODE_PARAMS params{};
        const CUresult status = get_kernel_params(reinterpret_cast<CUgraphNode>(node), &params);
        if (status != CUDA_SUCCESS) {
          std::fprintf(stderr, "DRIVER_CHECK line=%d cuGraphKernelNodeGetParams code=%d\n",
                       __LINE__, static_cast<int>(status));
          throw std::runtime_error("CUDA driver graph parameter query failed");
        }
        *physical_grid = std::max(*physical_grid, params.gridDimX);
      }
    }
    // Fused apply is the unique largest grid (one controller above the capped flat-row workers).
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, partition.apply_stream()));
    CUDA_CHECK(cudaStreamSynchronize(partition.apply_stream()));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
  } else {
    apply();
    CUDA_CHECK(cudaStreamSynchronize(partition.apply_stream()));
  }
  partition.FinishDeviceLevels(1, 1, 512);
  std::vector<int> expected(rows, 509);
  std::vector<data_size_t> expected_order;
  for (int bin : {0, 1}) {
    for (int pos = 0; pos < parent_rows; ++pos) {
      const int row = original[pos];
      expected[row] = bins[row] == 0 ? 511 : 512;
      if (bins[row] == bin) expected_order.push_back(row);
    }
  }
  expected_order.insert(expected_order.end(), original.begin() + parent_rows, original.end());
  Require(Get(partition.cuda_data_index_to_leaf_index(), rows) == expected,
          "complete map omitted split rows, nonzero-ID terminal rows or the odd tail");
  const auto exact_counts = Get(partition.cuda_leaf_num_data(), 513);
  const auto exact_starts = Get(partition.cuda_leaf_data_start(), 513);
  partition.PrefetchSplitBatchAsync(1, partition.apply_stream());
  CUDA_CHECK(cudaStreamSynchronize(partition.apply_stream()));
  std::vector<int> info;
  partition.ReadPrefetchedSplitBatch(1, &info);
  info.resize(16);  // slots16/17 are NCCL-only; do not compare uninitialized irrelevant bytes
  partition.EnsureLevelPartitioned();
  CUDA_CHECK(cudaDeviceSynchronize());
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
  DatasetHandle handle = nullptr;
  try {
    Require(argc == 4, "usage: probe rows graph|plain on|off(row batching)");
    const int rows = std::stoi(argv[1]);
    const bool capture = std::string(argv[2]) == "graph";
    Require(rows >= 2051, "need both odd split/gap tails");
    std::vector<float> input(rows);
    for (int i = 0; i < rows; ++i) input[i] = static_cast<float>(i % 5);
    Require(FLC_DatasetCreateFromMat(input.data(), C_API_DTYPE_FLOAT32, rows, 1, 1,
      "max_bin=7 min_data_in_leaf=1 feature_pre_filter=false num_threads=32", nullptr, &handle) == 0,
      "create dataset failed");
    Config config;
    config.Set({{"device_type", "cuda"}, {"num_leaves", "1024"}, {"quant_mode", "fixedpoint"},
      {"num_threads", "32"}, {"cuda_plan", std::string("auto,apply_row_batch:") + argv[3]}});
    FalcataPlan::ResolveFromConfig(config);
    auto* dataset = reinterpret_cast<Dataset*>(handle);
    unsigned int old_grid = 0, new_grid = 0;
    const auto old = Run(dataset, false, capture, &old_grid);
    const auto bounded = Run(dataset, true, capture, &new_grid);
    Require(old.indices == bounded.indices && old.map == bounded.map && old.counts == bounded.counts &&
      old.starts == bounded.starts && old.split_info == bounded.split_info, "old/new actual row bytes or metadata differ");
    if (capture) {
      Require(new_grid > 1 && new_grid < old_grid, "captured actual physical grid did not shrink");
      if (rows == 5463797) Require(old_grid == 8193 && new_grid == 6360, "actual target grid differs");
    }
    FLC_DatasetFree(handle);
    std::printf("PASS: every row/map/window/count/record equal; captured physical grids old=%u new=%u rows=%d\n",
                old_grid, new_grid, rows);
    return 0;
  } catch (const std::exception& error) {
    if (handle != nullptr) FLC_DatasetFree(handle);
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
