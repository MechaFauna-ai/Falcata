/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <string>
#include <vector>

#include "cuda_rank_objective.hpp"

namespace Falcata {

CUDALambdarankNDCG::CUDALambdarankNDCG(const Config& config)
    : CUDALambdaRankObjectiveInterface<LambdarankNDCG>(config), deterministic_(config.deterministic) {}

CUDALambdarankNDCG::CUDALambdarankNDCG(const std::vector<std::string>& strs): CUDALambdaRankObjectiveInterface<LambdarankNDCG>(strs) {}

CUDALambdarankNDCG::~CUDALambdarankNDCG() {}

void CUDALambdarankNDCG::Init(const Metadata& metadata, data_size_t num_data) {
  CUDALambdaRankObjectiveInterface<LambdarankNDCG>::Init(metadata, num_data);
  cuda_inverse_max_dcgs_.Resize(this->inverse_max_dcgs_.size());
  CopyFromHostToCUDADevice(cuda_inverse_max_dcgs_.RawData(), this->inverse_max_dcgs_.data(), this->inverse_max_dcgs_.size(), __FILE__, __LINE__);
  cuda_label_gain_.Resize(this->label_gain_.size());
  CopyFromHostToCUDADevice(cuda_label_gain_.RawData(), this->label_gain_.data(), this->label_gain_.size(), __FILE__, __LINE__);
}


CUDARankXENDCG::CUDARankXENDCG(const Config& config): CUDALambdaRankObjectiveInterface<RankXENDCG>(config) {}

CUDARankXENDCG::CUDARankXENDCG(const std::vector<std::string>& strs): CUDALambdaRankObjectiveInterface<RankXENDCG>(strs) {}

CUDARankXENDCG::~CUDARankXENDCG() {}

void CUDARankXENDCG::Init(const Metadata& metadata, data_size_t num_data) {
  CUDALambdaRankObjectiveInterface<RankXENDCG>::Init(metadata, num_data);
  // RankXENDCG::Init seeds rands_[q] with seed_ + q (Random(int) stores the seed as its unsigned state); the device
  // generators start from the same states, so every Init reseeds them
  std::vector<uint32_t> rand_states(static_cast<size_t>(num_queries_));
  for (data_size_t i = 0; i < num_queries_; ++i) {
    rand_states[i] = static_cast<uint32_t>(seed_) + static_cast<uint32_t>(i);
  }
  cuda_query_rand_states_.InitFromHostVector(rand_states);
  cuda_item_rands_.Resize(static_cast<size_t>(num_data));
  if (max_items_in_query_aligned_ >= 2048) {
    cuda_params_buffer_.Resize(static_cast<size_t>(num_data_));
  }
}


}  // namespace Falcata

#endif  // USE_CUDA
