/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root.
 */
// Standalone OpenCL regression probe (requires a USE_GPU build and an OpenCL device).
// Compile with the library's USE_GPU, MM_MALLOC and MM_PREFETCH definitions,
// -D_GLIBCXX_ASSERTIONS, include paths and link flags, then run through gpuq.
#include <Falcata/c_api.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "../../src/treelearner/gpu_tree_learner.h"

namespace {
void CheckAPI(int result) {
  if (result != 0) {
    throw std::runtime_error(FLC_GetLastError());
  }
}

class BufferProbe : public Falcata::GPUTreeLearner {
 public:
  explicit BufferProbe(const Falcata::Config* config) : GPUTreeLearner(config) {}

  void CheckExtent() {
    // The OpenCL buffers include a full workgroup prefetch margin.
    const size_t extent = static_cast<size_t>(num_data_) + 256 * (1 << 10);
    std::cout << "rows=" << num_data_ << " gradients=" << ordered_gradients_.size()
              << " hessians=" << ordered_hessians_.size() << " expected=" << extent << std::endl;
    if (ordered_gradients_.size() != extent || ordered_hessians_.size() != extent) {
      throw std::runtime_error("Pinned host buffer extent exceeds live vector elements");
    }
    ordered_gradients_[extent - 1] = 0;
    ordered_hessians_[extent - 1] = 0;
  }
};
}  // namespace

int main() {
  Falcata::Config config;
  config.Set(Falcata::Config::Str2Map("device_type=gpu max_bin=63 num_threads=2 num_leaves=7"));
  std::vector<DatasetHandle> datasets;
  {
    BufferProbe learner(&config);
    for (int rows : {2048, 8192, 1024, 16384, 300000, 2048}) {
      std::vector<double> features(rows * 8);
      for (size_t i = 0; i < features.size(); ++i) {
        features[i] = static_cast<double>((i * 17 + i / 8) % 101) / 101;
      }
      DatasetHandle dataset = nullptr;
      CheckAPI(FLC_DatasetCreateFromMat(features.data(), 1, rows, 8, 1, "max_bin=63 num_threads=2",
                                       datasets.empty() ? nullptr : datasets.front(), &dataset));
      datasets.push_back(dataset);
      if (datasets.size() == 1) {
        learner.Init(static_cast<const Falcata::Dataset*>(dataset), false);
      } else {
        learner.ResetTrainingData(static_cast<const Falcata::Dataset*>(dataset), false);
      }
      learner.CheckExtent();
    }
  }
  for (auto dataset : datasets) {
    CheckAPI(FLC_DatasetFree(dataset));
  }
}
