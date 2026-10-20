/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_metadata.hpp>

#include <vector>

namespace Falcata {

CUDAMetadata::CUDAMetadata(const int gpu_device_id)
  : gpu_device_id_(gpu_device_id >= 0 ? gpu_device_id : 0) {
  SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
}

CUDAMetadata::~CUDAMetadata() {}

void CUDAMetadata::Init(const std::vector<label_t>& label,
                        const std::vector<label_t>& weight,
                        const std::vector<data_size_t>& query_boundaries,
                        const std::vector<label_t>& query_weights,
                        const std::vector<double>& /*init_score*/) {
  if (label.size() == 0) {
    cuda_label_.Clear();
  } else {
    cuda_label_.InitFromHostVector(label);
  }
  if (weight.size() == 0) {
    cuda_weights_.Clear();
  } else {
    cuda_weights_.InitFromHostVector(weight);
  }
  if (query_boundaries.size() == 0) {
    cuda_query_boundaries_.Clear();
  } else {
    cuda_query_boundaries_.InitFromHostVector(query_boundaries);
  }
  cuda_query_weights_.Set(query_weights.data(), query_weights.size(), gpu_device_id_);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAMetadata::SetLabel(const label_t* label, data_size_t len) {
  cuda_label_.InitFromHostMemory(label, static_cast<size_t>(len));
}

void CUDAMetadata::SetWeights(const label_t* weights, data_size_t len) {
  cuda_weights_.InitFromHostMemory(weights, static_cast<size_t>(len));
}

void CUDAMetadata::SetQuery(const data_size_t* query_boundaries, const label_t* query_weights, data_size_t num_queries) {
  if (num_queries == 0 && query_boundaries == nullptr) {
    cuda_query_boundaries_.Clear();
  } else {
    cuda_query_boundaries_.InitFromHostMemory(query_boundaries, static_cast<size_t>(num_queries) + 1);
  }
  // A null weight pointer clears a previous weighted ranking snapshot.
  cuda_query_weights_.Set(query_weights, query_weights != nullptr ? static_cast<size_t>(num_queries) : 0,
                          gpu_device_id_);
}

void CUDAMetadata::SetInitScore(const double* /*init_score*/, data_size_t /*len*/) {}

}  // namespace Falcata

#endif  // USE_CUDA
