// Copyright (c) 2026 Falcata contributors. MIT license; see LICENSE.
// Manual correctness probe: launch the ACTUAL extracted production cubin, no copied processing kernel or counters.
#include <Falcata/c_api.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/treelearner/cuda/cuda_histogram_constructor.hpp"
#include "../../src/treelearner/cuda/cuda_device_levels.hpp"
#include "../../src/treelearner/cuda/cuda_leaf_splits.hpp"

namespace {
using Falcata::CUDADeviceLevelState;
using Falcata::CUDAHybridPairDescriptor;
using Falcata::CUDALeafSplitsStruct;
using Falcata::data_size_t;

void Need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void Check(cudaError_t code) { Need(code == cudaSuccess, cudaGetErrorString(code)); }
void Driver(CUresult code) {
  if (code == CUDA_SUCCESS) return;
  const char* error = nullptr;
  cuGetErrorString(code, &error);
  throw std::runtime_error(error != nullptr ? error : "driver API failure");
}
template <typename T> struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { Check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
  ~Buffer() { if (p != nullptr) cudaFree(p); }
  void Put(const T* h, size_t n) { Check(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice)); }
};
// Exact by-value production BlockPrefix<1024> layout; the queued runner checks this declaration against the source.
struct Prefix { int num_pairs; int grid_y; uint16_t start[1025]; };
static_assert(sizeof(Prefix) == 2060, "production BlockPrefix<1024> ABI");

void Probe(const char* module_file, const char* symbol, int n, bool overflow, bool words, int by, bool empty) {
  Need(FLC_GetLastError() != nullptr, "private Falcata C API did not load");
  Check(cudaSetDevice(0));
  Check(cudaFree(nullptr));
  Driver(cuInit(0));
  CUmodule module = nullptr;
  Driver(cuModuleLoad(&module, module_file));
  CUfunction kernel = nullptr;
  Driver(cuModuleGetFunction(&kernel, module, symbol));
  const int total = empty ? 0 : (n > 100000 ? 511 : 7);
  const unsigned int workers = overflow ? 2u : static_cast<unsigned int>(total + 3);
  std::vector<int32_t> gradients(n);
  std::vector<data_size_t> indices(n);
  std::vector<uint8_t> nibble(n);
  // Match the production code-view row stride, including its eight-word sector padding.
  const size_t row_words = words ? static_cast<size_t>(Falcata::Code5RowWords(1, 5)) : 1u;
  std::vector<uint32_t> codes(static_cast<size_t>(n) * row_words, 0u);
  std::array<int64_t, 10> gsum{};
  std::array<uint32_t, 10> hsum{};
  for (int i = 0; i < n; ++i) {
    const int lo = i % 5;
    const int hi = (i / 5) % 5;
    const int g = i % 7 - 3;
    gradients[i] = static_cast<int32_t>((static_cast<uint32_t>(g) << 16) | 1u);
    indices[i] = n - 1 - i;
    nibble[i] = static_cast<uint8_t>(lo | (hi << 4));
    codes[static_cast<size_t>(i) * row_words] = static_cast<uint32_t>(lo * 5 + hi);
    if (!empty) {
      gsum[lo] += g;
      gsum[5 + hi] += g;
      ++hsum[lo];
      ++hsum[5 + hi];
    }
  }
  Buffer<int32_t> grad(n); grad.Put(gradients.data(), n);
  Buffer<data_size_t> idx(n); idx.Put(indices.data(), n);
  Buffer<uint8_t> bytes(n); bytes.Put(nibble.data(), n);
  Buffer<uint32_t> code(codes.size()); code.Put(codes.data(), codes.size());
  Buffer<uint64_t> hist(10); Check(cudaMemset(hist.p, 0, 10 * sizeof(uint64_t)));
  std::array<CUDALeafSplitsStruct, 2> leaves;
  std::memset(leaves.data(), 0, sizeof(leaves));
  leaves[0].leaf_index = 7;
  leaves[0].num_data_in_leaf = n;
  leaves[0].sum_of_hessians = n;
  leaves[0].data_indices_in_leaf = idx.p;
  leaves[0].hist_in_leaf = reinterpret_cast<Falcata::hist_t*>(hist.p);
  leaves[1].leaf_index = -1;
  Buffer<CUDALeafSplitsStruct> leaf(2); leaf.Put(leaves.data(), 2);
  CUDAHybridPairDescriptor descriptor;
  std::memset(&descriptor, 0, sizeof(descriptor));
  descriptor.construct_valid = 1;
  descriptor.smaller_struct = leaf.p;
  descriptor.larger_struct = leaf.p + 1;
  descriptor.smaller_num_bits = 32;
  Buffer<CUDAHybridPairDescriptor> pair(1); pair.Put(&descriptor, 1);
  CUDADeviceLevelState state{};
  state.live_pairs = 1;
  state.construct_grid_y = total;
  state.construct_total_blocks = total;
  const std::array<int, 2> block_start{{0, total}};
  Buffer<int> device_block_start(2); device_block_start.Put(block_start.data(), 2);
  state.construct_block_start = device_block_start.p;
  Buffer<CUDADeviceLevelState> device_state(1); device_state.Put(&state, 1);
  // Two5-bin numerical columns, one packed dense partition; 25 joint cells and ten marginal cells.
  const std::array<uint32_t, 8> metadata{{0, 5, 5, 5, 0, 0, 0, 0}};
  const std::array<uint32_t, 2> hist_offsets{{0, 10}};
  const std::array<int, 2> column_offsets{{0, 2}};
  const std::array<int, 2> byte_offsets{{-1, 1}};
  Buffer<uint32_t> meta(metadata.size()); meta.Put(metadata.data(), metadata.size());
  Buffer<uint32_t> ho(2); ho.Put(hist_offsets.data(), 2);
  Buffer<int> co(2); co.Put(column_offsets.data(), 2);
  Buffer<int> bo(2); bo.Put(byte_offsets.data(), 2);
  const uint8_t* data = words ? reinterpret_cast<const uint8_t*>(code.p) : bytes.p;
  int columns = 2;
  data_size_t min_leaf = 0;
  double min_hessian = 0;
  int per_pair = 0, min_rows = 0, saturation = 0, quant_bins = 64, partitions = 1;
  uint32_t joint = 25;
  data_size_t level_min_rows = 0;
  Prefix prefix{};
  void* args[] = {&pair.p, &grad.p, &data, &meta.p, &columns, &ho.p, &co.p, &bo.p, &n, &min_leaf,
    &min_hessian, &per_pair, &min_rows, &saturation, &quant_bins, &partitions, &joint, &level_min_rows,
    &prefix, &device_state.p};
  const std::string kernel_name(symbol);
  const bool word_kernel = kernel_name.find("PairJointCode5WordKernel") != std::string::npos;
  const bool flat_word = word_kernel && kernel_name.find("ILi5ELb1ELb0ELi5ELb1EEEv") != std::string::npos;
  Need(!word_kernel || words, "Word kernel requires the CODE5 row view");
  Need(!word_kernel || flat_word || kernel_name.find("ILi4ELb0ELb0ELi5ELb1EEEv") != std::string::npos,
       "unreviewed Word template in this fixed diagnostic matrix");
  const unsigned int block_x = word_kernel ? static_cast<unsigned int>(row_words) : 1u;
  // Exact production Code5DummyCells(5): 1u << 5; only FLAT adds into these unflushed cells.
  const size_t dynamic_cells = static_cast<size_t>(joint) + (flat_word ? (1u << 5) : 0u);
  std::printf("LAYOUT: code_row_words=%zu block_x=%u block_y=%d dynamic_bytes=%zu word=%d flat=%d\n",
              row_words, block_x, by, dynamic_cells * sizeof(int32_t), word_kernel, flat_word);
  std::fflush(stdout);
  Driver(cuLaunchKernel(kernel, 1, workers, 1, block_x, by, 1,
                        dynamic_cells * sizeof(int32_t), nullptr, args, nullptr));
  Check(cudaDeviceSynchronize());
  std::array<uint64_t, 10> actual{};
  Check(cudaMemcpy(actual.data(), hist.p, sizeof(actual), cudaMemcpyDeviceToHost));
  for (size_t bin = 0; bin < actual.size(); ++bin) {
    const uint64_t expected = (static_cast<uint64_t>(static_cast<uint32_t>(gsum[bin])) << 32) | hsum[bin];
    Need(actual[bin] == expected, "integer histogram differs from CPU oracle (missing/duplicate/reordered rows)");
  }
  Check(cudaMemcpy(&state, device_state.p, sizeof(state), cudaMemcpyDeviceToHost));
  Need(state.construct_total_blocks == total && state.construct_grid_y == total, "kernel changed immutable work map");
  if (!empty && overflow) Need(state.construct_next_block[0] >= total, "actual atomic-claim overflow path did not execute");
  else Need(state.construct_next_block[0] == 0, "direct/empty path unexpectedly performed a claim");
  std::printf("PASS: actual production kernel CPU hist bins=10 rows=%d virtual=%d workers=%u claims=%d mode=%s\n",
    n, total, workers, state.construct_next_block[0], empty ? "empty" : (overflow ? "claim" : "direct"));
  Driver(cuModuleUnload(module));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Need(argc == 8, "module symbol rows direct|claim bytes|words block_y normal|empty");
    Probe(argv[1], argv[2], std::stoi(argv[3]), std::string(argv[4]) == "claim",
      std::string(argv[5]) == "words", std::stoi(argv[6]), std::string(argv[7]) == "empty");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
