/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/c_api.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../src/treelearner/cuda/cuda_best_split_finder.hpp"
#include "../../src/treelearner/cuda/cuda_data_partition.hpp"
#include "../../src/treelearner/cuda/cuda_histogram_constructor.hpp"

namespace {

// A small host Dataset and the per-feature histogram offsets the learner components are built from.
Falcata::Dataset* MakeDataset(std::vector<uint32_t>* hist_offsets) {
  std::vector<double> X(200 * 3);
  for (size_t i = 0; i < X.size(); ++i) X[i] = static_cast<double>((i * 7919) % 31);
  DatasetHandle handle = nullptr;
  if (FLC_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT64, 200, 3, 1, "verbose=-1", nullptr, &handle) != 0) {
    return nullptr;
  }
  auto* dataset = static_cast<Falcata::Dataset*>(handle);
  hist_offsets->assign(1, 0);
  for (int f = 0; f < dataset->num_features(); ++f) {
    const auto* mapper = dataset->FeatureBinMapper(f);
    hist_offsets->push_back(hist_offsets->back() + mapper->num_bin() - (mapper->GetMostFreqBin() == 0 ? 1 : 0));
  }
  return dataset;
}

}  // namespace

// The CUDA learner components create their streams (and the histogram constructor its events) in Init, but the
// destructors indexed cuda_streams_[0..] (an empty vector before Init) or destroyed the never-set cuda_stream_. A
// component destroyed without Init (e.g. a learner whose Init threw) must release nothing and exit cleanly. Each case
// runs in a fresh process, since the old destructors crash.
TEST(CUDALearnerComponents, DestroyWithoutInit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        std::vector<uint32_t> offsets;
        Falcata::Dataset* dataset = MakeDataset(&offsets);
        Falcata::Config config;
        { Falcata::CUDABestSplitFinder finder(nullptr, dataset, offsets, false, &config); }
        std::exit(dataset != nullptr ? 0 : 3);
      },
      ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT(
      {
        std::vector<uint32_t> offsets;
        Falcata::Dataset* dataset = MakeDataset(&offsets);
        { Falcata::CUDADataPartition partition(dataset, static_cast<int>(offsets.back()), 31, 1, false, nullptr); }
        std::exit(dataset != nullptr ? 0 : 3);
      },
      ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT(
      {
        std::vector<uint32_t> offsets;
        Falcata::Dataset* dataset = MakeDataset(&offsets);
        { Falcata::CUDAHistogramConstructor hist(dataset, 31, 1, offsets, 20, 1e-3, 0, false, false, 4, 1.0); }
        std::exit(dataset != nullptr ? 0 : 3);
      },
      ::testing::ExitedWithCode(0), "");
}
