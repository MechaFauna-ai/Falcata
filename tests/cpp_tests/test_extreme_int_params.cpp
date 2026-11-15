/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/c_api.h>

#include <random>
#include <string>
#include <vector>

namespace {

// Train a few iterations on a small random dataset; the parameters must give no signed overflow
// (undefined behaviour, which the sanitizer build stops on).
void TrainWith(const std::string& params, int nrow = 200) {
  const int ncol = 4;
  std::mt19937 gen(0);
  std::normal_distribution<double> dist;
  std::vector<double> data(nrow * ncol);
  for (auto& v : data) v = dist(gen);
  std::vector<float> label(nrow);
  for (int i = 0; i < nrow; ++i) label[i] = static_cast<float>(data[i * ncol] > 0);
  DatasetHandle dataset = nullptr;
  ASSERT_EQ(FLC_DatasetCreateFromMat(data.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1, "verbose=-1", nullptr, &dataset), 0)
      << FLC_GetLastError();
  ASSERT_EQ(FLC_DatasetSetField(dataset, "label", label.data(), nrow, C_API_DTYPE_FLOAT32), 0);
  BoosterHandle booster = nullptr;
  ASSERT_EQ(FLC_BoosterCreate(dataset, ("objective=binary verbose=-1 " + params).c_str(), &booster), 0)
      << FLC_GetLastError();
  int finished = 0;
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(FLC_BoosterUpdateOneIter(booster, &finished), 0) << FLC_GetLastError();
  }
  ASSERT_EQ(FLC_BoosterFree(booster), 0);
  ASSERT_EQ(FLC_DatasetFree(dataset), 0);
}

}  // namespace

// min_data_in_leaf * 2 overflowed int in SerialTreeLearner::BeforeFindBestSplit
TEST(ExtremeIntParams, MinDataInLeafAtIntMax) {
  TrainWith("min_data_in_leaf=2147483647 min_sum_hessian_in_leaf=0");
}

// extra_seed + feature index overflowed int in HistogramPool::SetFeatureInfo
TEST(ExtremeIntParams, ExtraSeedAtIntMax) {
  TrainWith("extra_trees=true extra_seed=2147483647");
}

// bagging_seed + block index overflowed int in the bagging and GOSS samplers (one generator per
// block of 1024 rows, so more than one block)
TEST(ExtremeIntParams, BaggingSeedAtIntMax) {
  TrainWith("bagging_fraction=0.5 bagging_freq=1 bagging_seed=2147483647", 5000);
  TrainWith("data_sample_strategy=goss bagging_seed=2147483647", 5000);
}
