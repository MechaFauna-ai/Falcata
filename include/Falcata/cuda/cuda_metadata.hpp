/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_METADATA_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_METADATA_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/meta.h>

#include <vector>

namespace Falcata {

class CUDAMetadata {
 public:
  explicit CUDAMetadata(const int gpu_device_id);

  ~CUDAMetadata();

  void Init(const std::vector<label_t>& label,
            const std::vector<label_t>& weight,
            const std::vector<data_size_t>& query_boundaries,
            const std::vector<label_t>& query_weights,
            const std::vector<double>& init_score);

  void SetLabel(const label_t* label, data_size_t len);

  void SetWeights(const label_t* weights, data_size_t len);

  void SetQuery(const data_size_t* query, const label_t* query_weights, data_size_t num_queries);

  void SetInitScore(const double* init_score, data_size_t len);

  const label_t* cuda_label() const { return cuda_label_.RawData(); }

  const label_t* cuda_weights() const { return cuda_weights_.RawData(); }

  const data_size_t* cuda_query_boundaries() const { return cuda_query_boundaries_.RawData(); }

  const label_t* cuda_query_weights() const { return cuda_query_weights_.RawData(); }

 private:
  CUDAVector<label_t> cuda_label_;
  CUDAVector<label_t> cuda_weights_;
  CUDAVector<data_size_t> cuda_query_boundaries_;
  CUDAVector<label_t> cuda_query_weights_;
  CUDAVector<double> cuda_init_score_;
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_METADATA_HPP_
