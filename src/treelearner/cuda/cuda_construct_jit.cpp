/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_construct_jit.hpp"

#include <LightGBM/utils/log.h>

#include <cuda.h>
#include <nvrtc.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

namespace LightGBM {

std::string ConstructJITShapeKey::Signature() const {
  return "b" + std::to_string(bins) + "_p" + std::to_string(num_partitions) +
         "_c" + std::to_string(cols_per_partition) + "_s" + std::to_string(shared_hist_size) +
         "_h" + std::to_string(use_16bit_hist) + "_q" + std::to_string(is_4bit) +
         "_a" + std::to_string(sm_major) + std::to_string(sm_minor);
}

CUDAConstructJIT::~CUDAConstructJIT() {
  for (auto& kv : cache_) {
    if (kv.second.module != nullptr) {
      cuModuleUnload(reinterpret_cast<CUmodule>(kv.second.module));
      kv.second.module = nullptr;
    }
  }
}

bool CUDAConstructJIT::Enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("EXABOOST_CONSTRUCT_JIT");
    // opt-in for now: default OFF so the AOT compact-quant fast path is the
    // shipped behavior. EXABOOST_CONSTRUCT_JIT=1 requests the JIT.
    return env != nullptr && std::string(env) == std::string("1");
  }();
  return enabled;
}

bool CUDAConstructJIT::Available() {
  static const bool avail = []() {
    // NVRTC present?
    int major = 0, minor = 0;
    if (nvrtcVersion(&major, &minor) != NVRTC_SUCCESS) {
      Log::Debug("CUDAConstructJIT: NVRTC unavailable, JIT disabled");
      return false;
    }
    // Driver API present and a current/primary context obtainable? The runtime
    // API has already created a primary context by the time we get here.
    if (cuInit(0) != CUDA_SUCCESS) {
      Log::Debug("CUDAConstructJIT: cuInit failed, JIT disabled");
      return false;
    }
    Log::Debug("CUDAConstructJIT: NVRTC %d.%d available", major, minor);
    return true;
  }();
  return avail;
}

// The specialized kernel source. Shape constants are baked as `#define`s (the
// OpenCL backend's precedent). The kernel replicates, bit-for-bit, the (bin,
// gradient) accumulation of ConstructDiscretizedHistogramDenseInner for the
// host-launched, non-graph, non-4bit, uniform-low-bin compact path: one block
// per (partition x row-group), block_dim = (COLS_PER_PARTITION, dim_y). Integer
// atomics are order-invariant, so visiting the same pairs => identical sums.
std::string CUDAConstructJIT::BuildKernelSource(const ConstructJITShapeKey& key) const {
  const std::string bins = std::to_string(key.bins);
  const std::string cols = std::to_string(key.cols_per_partition);
  const std::string shist = std::to_string(key.shared_hist_size);
  std::string src;
  src += "typedef int int32_t; typedef long long int64_t; typedef short int16_t;\n";
  src += "typedef unsigned int uint32_t; typedef unsigned char uint8_t;\n";
  src += "typedef int data_size_t;\n";
  src += "#define BINS " + bins + "\n";
  src += "#define COLS_PER_PARTITION " + cols + "\n";
  src += "#define SHARED_HIST_SIZE " + shist + "\n";
  src += "#define USE_16BIT_HIST " + std::to_string(key.use_16bit_hist) + "\n";
  src += "#define IS_4BIT " + std::to_string(key.is_4bit) + "\n";
  src += R"CUDA(
struct LeafSplits {
  int leaf_index;
  double sum_of_gradients;
  double sum_of_hessians;
  int64_t sum_of_gradients_hessians;
  data_size_t num_data_in_leaf;
  double gain;
  double leaf_value;
  const data_size_t* data_indices_in_leaf;
  void* hist_in_leaf;
};

// Specialized construct: one smaller-leaf histogram. Baked COLS_PER_PARTITION,
// BINS (per-column bin count, uniform), SHARED_HIST_SIZE. dim_y == gridDim.y *
// blockDim.y (host-launched exact grid). column_hist_offsets holds each column's
// hist offset relative to the partition; here uniform, offset(col) = col*BINS,
// but we read the array to stay layout-exact.
extern "C" __global__ void construct_jit(
    const LeafSplits* smaller_leaf,
    const int32_t* grad_and_hess,
    const uint8_t* data,
    const uint32_t* column_hist_offsets,
    const uint32_t* partition_hist_offsets,
    const int* feature_partition_column_index_offsets,
    const data_size_t num_data) {
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  int32_t* shared_hist_packed = (int32_t*)shared_hist;

  const data_size_t num_data_in_smaller_leaf = smaller_leaf->num_data_in_leaf;
  const int dim_y = (int)(gridDim.y * blockDim.y);
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = ((size_t)blockIdx_y * blockDim.y) * num_data_per_thread;
  if (block_start >= num_data_in_smaller_leaf) return;

  const data_size_t* data_indices_ref = smaller_leaf->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const int row_stride = num_columns_in_partition;
  const uint8_t* data_ptr = data + (size_t)partition_column_start * num_data;
  const uint32_t partition_hist_start = partition_hist_offsets[blockIdx.x];
  const uint32_t partition_hist_end = partition_hist_offsets[blockIdx.x + 1];
  const uint32_t num_items_in_partition = partition_hist_end - partition_hist_start;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0;
  }
  __syncthreads();

  const unsigned int threadIdx_y = threadIdx.y;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start,
      num_data_per_thread * (data_size_t)blockDim.y));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total :
      num_iteration_total - (data_size_t)(threadIdx_y >= remainder);
  data_size_t inner_data_index = (data_size_t)threadIdx_y;
  const int column_index = (int)threadIdx.x + partition_column_start;
  if (threadIdx.x < (unsigned int)num_columns_in_partition) {
    int32_t* shared_hist_ptr = shared_hist_packed + column_hist_offsets[column_index];
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const int32_t gh = grad_and_hess[data_index];
      const uint32_t bin = (uint32_t)data_ptr[(size_t)data_index * row_stride + threadIdx.x];
      atomicAdd_block(shared_hist_ptr + bin, gh);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();

#if USE_16BIT_HIST
  int32_t* feature_histogram_ptr = (int32_t*)smaller_leaf->hist_in_leaf + partition_hist_start;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd((int*)(feature_histogram_ptr + i), (int)shared_hist_packed[i]);
  }
#else
  long long* feature_histogram_ptr = (long long*)smaller_leaf->hist_in_leaf + partition_hist_start;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    const int32_t p = shared_hist_packed[i];
    const int64_t v = ((int64_t)((int16_t)(p >> 16)) << 32) | (int64_t)(p & 0x0000ffff);
    atomicAdd((unsigned long long*)(feature_histogram_ptr + i), (unsigned long long)v);
  }
#endif
}

// -----------------------------------------------------------------------------
// The LIVE batched construct kernel. This is the one the dispatch launches with
// EXABOOST_CONSTRUCT_JIT=1 on the (non-graph, host-launched, non-4bit) dense
// compact-quant path -- exactly numerai's shape. It replicates, bit-for-bit, the
// (bin, gradient) accumulation of the AOT
// CUDAConstructDiscretizedHistogramDenseBatchedKernel for that case:
//   - one block per (partition = blockIdx.x, row-group = blockIdx.y, pair = blockIdx.z)
//   - the per-pair validity / min_data / min_hessian construct gate from the
//     descriptor (host-launched: level_smaller_num_data == null, gstate inactive,
//     is_feature_used == null, bin_used == null -> the simple offsets-array body)
//   - dim_y = gridDim.y * blockDim.y (the host's exact launch grid)
//   - per-pair 16-/32-bit histogram flush (block-uniform from smaller_num_bits)
// Because the shared-hist SIZE and the hist bit width are the only shape facts the
// body needs (the per-tree column/partition counts are read from the offset arrays
// at runtime), one compile serves every tree of a run -- the compact column set
// resamples per tree but the kernel reads the resampled offsets, unchanged.
// Integer atomics are order-invariant -> identical sums to the AOT kernel.
//
// The CUDAHybridPairDescriptor / CUDALeafSplitsStruct layouts below MUST match
// src/treelearner/cuda/cuda_leaf_splits.hpp exactly (asserted host-side).
struct HybridPairDescriptor {
  const LeafSplits* smaller_struct;
  const LeafSplits* larger_struct;
  int smaller_leaf_index;
  int larger_leaf_index;
  data_size_t num_data_in_smaller_leaf;
  data_size_t num_data_in_larger_leaf;
  uint8_t construct_valid;
  uint8_t smaller_valid;
  uint8_t larger_valid;
  uint8_t parent_num_bits;
  uint8_t smaller_num_bits;
  uint8_t larger_num_bits;
};

// The accumulation body, templated on the flush bit width (matches
// ConstructDiscretizedHistogramDenseInner). Reads the same offset arrays the AOT
// kernel does; dim_y passed in (host-launched grid).
template <bool USE_16BIT>
__device__ __forceinline__ void construct_jit_inner(
    const LeafSplits* smaller_leaf,
    int16_t* shared_hist,
    const int32_t* grad_and_hess,
    const uint8_t* data,
    const uint32_t* column_hist_offsets,
    const uint32_t* partition_hist_offsets,
    const int* feature_partition_column_index_offsets,
    const int* packed_partition_byte_offsets,
    const data_size_t num_data,
    const int dim_y) {
  int32_t* shared_hist_packed = (int32_t*)shared_hist;
  const data_size_t num_data_in_smaller_leaf = smaller_leaf->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = ((size_t)blockIdx_y * blockDim.y) * num_data_per_thread;
  if (block_start >= num_data_in_smaller_leaf) return;

  const data_size_t* data_indices_ref = smaller_leaf->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const int num_columns_in_partition = partition_column_end - partition_column_start;
#if IS_4BIT
  const int row_stride = packed_partition_byte_offsets[blockIdx.x + 1] -
                         packed_partition_byte_offsets[blockIdx.x];
  const uint8_t* data_ptr = data + (size_t)packed_partition_byte_offsets[blockIdx.x] * num_data;
#else
  const int row_stride = num_columns_in_partition;
  const uint8_t* data_ptr = data + (size_t)partition_column_start * num_data;
#endif
  const uint32_t partition_hist_start = partition_hist_offsets[blockIdx.x];
  const uint32_t partition_hist_end = partition_hist_offsets[blockIdx.x + 1];
  const uint32_t num_items_in_partition = partition_hist_end - partition_hist_start;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;

  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0;
  }
  __syncthreads();

  const unsigned int threadIdx_y = threadIdx.y;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start,
      num_data_per_thread * (data_size_t)blockDim.y));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total :
      num_iteration_total - (data_size_t)(threadIdx_y >= remainder);
  data_size_t inner_data_index = (data_size_t)threadIdx_y;
  const int column_index = (int)threadIdx.x + partition_column_start;
  if (threadIdx.x < (unsigned int)num_columns_in_partition) {
    int32_t* shared_hist_ptr = shared_hist_packed + column_hist_offsets[column_index];
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const int32_t gh = grad_and_hess[data_index];
      const uint8_t* row_ptr = data_ptr + (size_t)data_index * row_stride;
#if IS_4BIT
      const uint32_t packed = (uint32_t)row_ptr[threadIdx.x >> 1];
      const uint32_t bin = (packed >> ((threadIdx.x & 1) << 2)) & 0xfu;
#else
      const uint32_t bin = (uint32_t)row_ptr[threadIdx.x];
#endif
      atomicAdd_block(shared_hist_ptr + bin, gh);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();

  if (USE_16BIT) {
    int32_t* feature_histogram_ptr = (int32_t*)smaller_leaf->hist_in_leaf + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      atomicAdd((int*)(feature_histogram_ptr + i), (int)shared_hist_packed[i]);
    }
  } else {
    long long* feature_histogram_ptr = (long long*)smaller_leaf->hist_in_leaf + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t p = shared_hist_packed[i];
      const int64_t v = ((int64_t)((int16_t)(p >> 16)) << 32) | (int64_t)(p & 0x0000ffff);
      atomicAdd((unsigned long long*)(feature_histogram_ptr + i), (unsigned long long)v);
    }
  }
}

extern "C" __global__ void construct_jit_batched(
    const HybridPairDescriptor* pair_descs,
    const int32_t* grad_and_hess,
    const uint8_t* data,
    const uint32_t* column_hist_offsets,
    const uint32_t* partition_hist_offsets,
    const int* feature_partition_column_index_offsets,
    const int* packed_partition_byte_offsets,
    const data_size_t num_data,
    const data_size_t min_data_in_leaf,
    const double min_sum_hessian_in_leaf) {
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  const HybridPairDescriptor* desc = pair_descs + blockIdx.z;
  if (!desc->construct_valid) return;
  const LeafSplits* smaller_struct = desc->smaller_struct;
  const data_size_t num_data_smaller = smaller_struct->num_data_in_leaf;
  const double sum_hessians_smaller = smaller_struct->sum_of_hessians;
  const LeafSplits* larger_struct = desc->larger_struct;
  const bool has_larger = larger_struct->leaf_index >= 0;
  const data_size_t num_data_larger = has_larger ? larger_struct->num_data_in_leaf : 0;
  const double sum_hessians_larger = has_larger ? larger_struct->sum_of_hessians : 0.0;
  if ((num_data_smaller <= min_data_in_leaf || sum_hessians_smaller <= min_sum_hessian_in_leaf) &&
      (num_data_larger <= min_data_in_leaf || sum_hessians_larger <= min_sum_hessian_in_leaf)) {
    return;
  }
  const int dim_y = (int)(gridDim.y * blockDim.y);
  const uint8_t smaller_num_bits = desc->smaller_num_bits;
  if (smaller_num_bits <= 16) {
    construct_jit_inner<true>(smaller_struct, shared_hist, grad_and_hess, data,
        column_hist_offsets, partition_hist_offsets,
        feature_partition_column_index_offsets, packed_partition_byte_offsets,
        num_data, dim_y);
  } else {
    construct_jit_inner<false>(smaller_struct, shared_hist, grad_and_hess, data,
        column_hist_offsets, partition_hist_offsets,
        feature_partition_column_index_offsets, packed_partition_byte_offsets,
        num_data, dim_y);
  }
}
)CUDA";
  return src;
}

void* CUDAConstructJIT::GetOrCompile(const ConstructJITShapeKey& key, double* compile_ms_out) {
  if (compile_ms_out != nullptr) *compile_ms_out = 0.0;
  if (!Enabled() || !Available()) return nullptr;
  // supported-shape gate: uniform low-bin, sane partition layout
  if (key.bins <= 0 || key.bins > 256 || key.cols_per_partition <= 0 ||
      key.num_partitions <= 0 || key.shared_hist_size <= 0) {
    return nullptr;
  }
  const std::string sig = key.Signature();
  auto it = cache_.find(sig);
  if (it != cache_.end()) {
    if (compile_ms_out != nullptr) *compile_ms_out = it->second.compile_ms;
    return it->second.func;  // may be nullptr (permanent fallback)
  }

  ConstructJITEntry entry;
  entry.attempted = true;
  const std::string source = BuildKernelSource(key);
  const auto t0 = std::chrono::steady_clock::now();

  nvrtcProgram prog = nullptr;
  if (nvrtcCreateProgram(&prog, source.c_str(), "construct_jit.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) {
    Log::Warning("CUDAConstructJIT: nvrtcCreateProgram failed for %s, using AOT", sig.c_str());
    cache_[sig] = entry;
    return nullptr;
  }
  const std::string arch = "--gpu-architecture=compute_" + std::to_string(key.sm_major) +
                           std::to_string(key.sm_minor);
  const char* opts[] = {arch.c_str(), "--std=c++14"};
  const nvrtcResult comp = nvrtcCompileProgram(prog, 2, opts);
  if (comp != NVRTC_SUCCESS) {
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::vector<char> nvlog(log_size + 1, 0);
    nvrtcGetProgramLog(prog, nvlog.data());
    Log::Warning("CUDAConstructJIT: NVRTC compile failed for %s: %s -- using AOT", sig.c_str(), nvlog.data());
    nvrtcDestroyProgram(&prog);
    cache_[sig] = entry;
    return nullptr;
  }
  size_t ptx_size = 0;
  nvrtcGetPTXSize(prog, &ptx_size);
  std::vector<char> ptx(ptx_size);
  nvrtcGetPTX(prog, ptx.data());
  nvrtcDestroyProgram(&prog);

  CUmodule module = nullptr;
  if (cuModuleLoadData(&module, ptx.data()) != CUDA_SUCCESS) {
    Log::Warning("CUDAConstructJIT: cuModuleLoadData failed for %s, using AOT", sig.c_str());
    cache_[sig] = entry;
    return nullptr;
  }
  CUfunction func = nullptr;
  if (cuModuleGetFunction(&func, module, "construct_jit") != CUDA_SUCCESS) {
    Log::Warning("CUDAConstructJIT: cuModuleGetFunction failed for %s, using AOT", sig.c_str());
    cuModuleUnload(module);
    cache_[sig] = entry;
    return nullptr;
  }
  CUfunction func_batched = nullptr;
  if (cuModuleGetFunction(&func_batched, module, "construct_jit_batched") != CUDA_SUCCESS) {
    Log::Warning("CUDAConstructJIT: cuModuleGetFunction(batched) failed for %s, using AOT", sig.c_str());
    cuModuleUnload(module);
    cache_[sig] = entry;
    return nullptr;
  }
  const auto t1 = std::chrono::steady_clock::now();
  entry.compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  entry.module = module;
  entry.func = func;
  entry.func_batched = func_batched;
  cache_[sig] = entry;
  Log::Info("CUDAConstructJIT: compiled construct kernel %s in %.1f ms", sig.c_str(), entry.compile_ms);
  if (compile_ms_out != nullptr) *compile_ms_out = entry.compile_ms;
  return func;
}

void* CUDAConstructJIT::GetBatchedFunc(const ConstructJITShapeKey& key) const {
  auto it = cache_.find(key.Signature());
  if (it == cache_.end()) return nullptr;
  return it->second.func_batched;
}

void* CUDAConstructJIT::GetBatchedIfValidated(const ConstructJITShapeKey& key) const {
  auto it = cache_.find(key.Signature());
  if (it == cache_.end() || !it->second.validated) return nullptr;
  return it->second.func_batched;
}

bool CUDAConstructJIT::IsValidated(const ConstructJITShapeKey& key) const {
  auto it = cache_.find(key.Signature());
  return it != cache_.end() && it->second.validated;
}

void CUDAConstructJIT::SetValidated(const ConstructJITShapeKey& key, bool ok) {
  auto it = cache_.find(key.Signature());
  if (it == cache_.end()) return;
  it->second.validated = ok;
  if (!ok) {
    // permanent fallback: drop the function so GetOrCompile returns nullptr
    Log::Warning("CUDAConstructJIT: %s failed bit-identity validation, permanent AOT fallback",
                 key.Signature().c_str());
    it->second.func = nullptr;
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
