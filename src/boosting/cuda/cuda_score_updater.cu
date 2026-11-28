/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_score_updater.hpp"

#ifdef USE_CUDA

#include <cstdint>

namespace Falcata {

__global__ void AddScoreConstantKernel(
  const double val,
  const data_size_t num_data,
  double* score) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    score[data_index] += val;
  }
}

__global__ void AddScoreConstantVec2Kernel(
  const double val,
  const data_size_t num_data,
  double* score) {
  const data_size_t pair_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const data_size_t base = pair_index * 2;
  if (base + 1 < num_data) {
    double2* vec_score = reinterpret_cast<double2*>(score);
    double2 v = vec_score[pair_index];
    v.x += val;
    v.y += val;
    vec_score[pair_index] = v;
  } else if (base < num_data) {
    score[base] += val;
  }
}

void CUDAScoreUpdater::LaunchAddScoreConstantKernel(const double val, const size_t offset) {
  if (num_data_ <= 0) {
    return;
  }
  double* dst = cuda_score_.RawData() + offset;
  Log::Debug("Adding init score = %lf", val);
  if ((reinterpret_cast<uintptr_t>(dst) & 0xfu) == 0) {
    const data_size_t num_pairs = (num_data_ + 1) / 2;
    const int threads = 256;
    const int num_blocks = (num_pairs + threads - 1) / threads;
    AddScoreConstantVec2Kernel<<<num_blocks, threads>>>(val, num_data_, dst);
  } else {
    const int num_blocks = (num_data_ + num_threads_per_block_ - 1) / num_threads_per_block_;
    AddScoreConstantKernel<<<num_blocks, num_threads_per_block_>>>(val, num_data_, dst);
  }
}

__global__ void MultiplyScoreConstantKernel(
  const double val,
  const data_size_t num_data,
  double* score) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    score[data_index] *= val;
  }
}

__global__ void MultiplyScoreConstantVec2Kernel(
  const double val,
  const data_size_t num_data,
  double* score) {
  const data_size_t pair_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const data_size_t base = pair_index * 2;
  if (base + 1 < num_data) {
    double2* vec_score = reinterpret_cast<double2*>(score);
    double2 v = vec_score[pair_index];
    v.x *= val;
    v.y *= val;
    vec_score[pair_index] = v;
  } else if (base < num_data) {
    score[base] *= val;
  }
}

void CUDAScoreUpdater::LaunchMultiplyScoreConstantKernel(const double val, const size_t offset) {
  if (num_data_ <= 0) {
    return;
  }
  double* dst = cuda_score_.RawData() + offset;
  if ((reinterpret_cast<uintptr_t>(dst) & 0xfu) == 0) {
    const data_size_t num_pairs = (num_data_ + 1) / 2;
    const int threads = 256;
    const int num_blocks = (num_pairs + threads - 1) / threads;
    MultiplyScoreConstantVec2Kernel<<<num_blocks, threads>>>(val, num_data_, dst);
  } else {
    const int num_blocks = (num_data_ + num_threads_per_block_ - 1) / num_threads_per_block_;
    MultiplyScoreConstantKernel<<<num_blocks, num_threads_per_block_>>>(val, num_data_, dst);
  }
}

// score += leaf_const + sum(coeff * raw), or the leaf's constant output when a used raw value is NaN: Tree's
// PredictionFunLinear, operation for operation. The explicit _rn intrinsics keep nvcc from contracting the
// multiply-add into an FMA, which the host evaluator does not use, so the device score is the host one bit for bit.
template <bool USE_INDICES>
__global__ void LinearTreeScoreKernel(
  const data_size_t num_rows,
  const data_size_t* __restrict__ data_indices,
  const int* __restrict__ leaf_index,
  const float* __restrict__ raw,
  const data_size_t num_data,
  const int num_leaves,
  const double* __restrict__ model,        // [leaf_const | leaf_output | coeff]
  const int* __restrict__ model_index,     // [coeff_offset (num_leaves + 1) | coeff raw slot]
  double* __restrict__ score) {
  const data_size_t row = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (row >= num_rows) {
    return;
  }
  const data_size_t i = USE_INDICES ? data_indices[row] : row;
  const int leaf = leaf_index[i];
  const double* coeff = model + 2 * num_leaves;
  const int* slot = model_index + num_leaves + 1;
  double add_score = model[leaf];
  const int begin = model_index[leaf];
  const int end = model_index[leaf + 1];
  for (int k = begin; k < end; ++k) {
    const float feat_val = raw[static_cast<size_t>(slot[k]) * num_data + i];
    if (isnan(feat_val)) {
      score[i] += model[num_leaves + leaf];
      return;
    }
    add_score = __dadd_rn(add_score, __dmul_rn(coeff[k], static_cast<double>(feat_val)));
  }
  score[i] += add_score;
}

void CUDAScoreUpdater::LaunchLinearTreeScoreKernel(const data_size_t* data_indices, const data_size_t data_cnt,
                                                   const int num_leaves, const size_t offset) {
  if (data_cnt <= 0) {
    return;
  }
  const int threads = 256;
  const int blocks = (data_cnt + threads - 1) / threads;
  if (data_indices == nullptr) {
    LinearTreeScoreKernel<false><<<blocks, threads>>>(
      data_cnt, nullptr, cuda_linear_leaf_index_.RawData(), cuda_linear_raw_.RawData(), num_data_, num_leaves,
      cuda_linear_model_.RawData(), cuda_linear_model_index_.RawData(), cuda_score_.RawData() + offset);
  } else {
    LinearTreeScoreKernel<true><<<blocks, threads>>>(
      data_cnt, data_indices, cuda_linear_leaf_index_.RawData(), cuda_linear_raw_.RawData(), num_data_, num_leaves,
      cuda_linear_model_.RawData(), cuda_linear_model_index_.RawData(), cuda_score_.RawData() + offset);
  }
}

__global__ void FillDoubleKernel(double* out, const double val, const size_t n) {
  const size_t i = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockIdx.x) * blockDim.x;
  if (i < n) {
    out[i] = val;
  }
}

__global__ void FillDoubleVec2Kernel(double* out, const double val, const size_t n) {
  const size_t pair_idx = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockIdx.x) * blockDim.x;
  const size_t base = pair_idx * 2;
  if (base + 1 < n) {
    reinterpret_cast<double2*>(out)[pair_idx] = make_double2(val, val);
  } else if (base < n) {
    out[base] = val;
  }
}

void LaunchFillDoubleKernel(double* out, const double val, const size_t n) {
  if (n == 0) {
    return;
  }
  const int threads = 256;
  if ((reinterpret_cast<uintptr_t>(out) & 0xfu) == 0) {
    const size_t num_pairs = (n + 1) / 2;
    const size_t blocks = (num_pairs + threads - 1) / threads;
    FillDoubleVec2Kernel<<<static_cast<unsigned int>(blocks), threads>>>(out, val, n);
  } else {
    const size_t blocks = (n + threads - 1) / threads;
    FillDoubleKernel<<<static_cast<unsigned int>(blocks), threads>>>(out, val, n);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
