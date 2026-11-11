/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root.
 */
// OpenCL learner only (USE_GPU builds); needs an OpenCL device at run time.
#ifdef USE_GPU

#include <gtest/gtest.h>
#include <Falcata/c_api.h>

#include <vector>

#include "../../src/treelearner/gpu_tree_learner.h"

namespace {

class BufferProbe : public Falcata::GPUTreeLearner {
 public:
  explicit BufferProbe(const Falcata::Config* config) : GPUTreeLearner(config) {}

  // The OpenCL buffers include a full workgroup prefetch margin, so the pinned host vectors must hold that extent.
  void ExpectFullExtent() {
    const size_t extent = static_cast<size_t>(num_data_) + 256 * (1 << 10);
    EXPECT_EQ(ordered_gradients_.size(), extent) << "rows=" << num_data_;
    EXPECT_EQ(ordered_hessians_.size(), extent) << "rows=" << num_data_;
    if (ordered_gradients_.size() == extent && ordered_hessians_.size() == extent) {
      ordered_gradients_[extent - 1] = 0;
      ordered_hessians_[extent - 1] = 0;
    }
  }
};

}  // namespace

// GPUTreeLearner hands OpenCL host buffers spanning num_data + one workgroup's prefetch margin; the ordered
// gradient/hessian vectors behind them must be that large on Init and after every ResetTrainingData, growing or
// shrinking, or the device reads past the vectors' live elements.
TEST(OpenCLHostBuffers, OrderedVectorsCoverThePinnedExtent) {
  Falcata::Config config;
  config.Set(Falcata::Config::Str2Map("device_type=gpu max_bin=63 num_threads=2 num_leaves=7"));
  std::vector<DatasetHandle> datasets;
  {
    BufferProbe learner(&config);
    for (int rows : {2048, 8192, 1024, 16384, 300000, 2048}) {
      std::vector<double> features(static_cast<size_t>(rows) * 8);
      for (size_t i = 0; i < features.size(); ++i) {
        features[i] = static_cast<double>((i * 17 + i / 8) % 101) / 101;
      }
      DatasetHandle dataset = nullptr;
      ASSERT_EQ(FLC_DatasetCreateFromMat(features.data(), 1, rows, 8, 1, "max_bin=63 num_threads=2",
                                         datasets.empty() ? nullptr : datasets.front(), &dataset),
                0)
          << FLC_GetLastError();
      datasets.push_back(dataset);
      if (datasets.size() == 1) {
        learner.Init(static_cast<const Falcata::Dataset*>(dataset), false);
      } else {
        learner.ResetTrainingData(static_cast<const Falcata::Dataset*>(dataset), false);
      }
      learner.ExpectFullExtent();
    }
  }
  for (DatasetHandle dataset : datasets) {
    EXPECT_EQ(FLC_DatasetFree(dataset), 0);
  }
}

#endif  // USE_GPU
