/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/c_api.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

DatasetHandle MakeDataset(int nrow, int ncol, int seed, const DatasetHandle reference) {
  std::vector<double> X(static_cast<size_t>(nrow) * ncol);
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      X[static_cast<size_t>(i) * ncol + j] = std::sin(0.13 * i * (j + 1) + seed) + 0.01 * ((i * 7 + j) % 11);
    }
  }
  DatasetHandle out = nullptr;
  EXPECT_EQ(0, FLC_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1,
                                        "verbosity=-1 min_data_in_bin=1", reference, &out));
  return out;
}

}  // namespace

// Adding features to the Dataset a Booster trains on made the next
// FLC_BoosterUpdateOneIter read the extended Dataset through the tree learner's
// sizes for the old one (a segfault in BasicLeafConstraints::Update on the
// CPU; use of the freed device columns on CUDA). Training now fails cleanly;
// the Booster still predicts, and a new Booster trains on the extended Dataset.
TEST(AddFeaturesFrom, BoosterOnTheChangedDatasetRefusesToTrain) {
  const int nrow = 2000;
  DatasetHandle train = MakeDataset(nrow, 4, 0, nullptr);
  std::vector<float> y(nrow);
  for (int i = 0; i < nrow; ++i) {
    y[i] = static_cast<float>(std::sin(0.13 * i) + 0.5 * std::sin(0.26 * i + 1));
  }
  ASSERT_EQ(0, FLC_DatasetSetField(train, "label", y.data(), nrow, C_API_DTYPE_FLOAT32));
  BoosterHandle booster = nullptr;
  ASSERT_EQ(0, FLC_BoosterCreate(train, "objective=regression verbosity=-1 num_leaves=15 min_data_in_leaf=5",
                                 &booster));
  int finished = 0;
  for (int it = 0; it < 3; ++it) {
    ASSERT_EQ(0, FLC_BoosterUpdateOneIter(booster, &finished));
  }

  DatasetHandle extra = MakeDataset(nrow, 3, 5, nullptr);
  ASSERT_EQ(0, FLC_DatasetAddFeaturesFrom(train, extra));
  int num_feature = 0;
  ASSERT_EQ(0, FLC_DatasetGetNumFeature(train, &num_feature));
  ASSERT_EQ(num_feature, 7);

  EXPECT_NE(0, FLC_BoosterUpdateOneIter(booster, &finished));
  EXPECT_NE(std::string(FLC_GetLastError()).find("gained features"), std::string::npos) << FLC_GetLastError();

  // the trained trees still predict
  std::vector<double> row(4, 0.25), out(1, 0.0);
  int64_t out_len = 0;
  EXPECT_EQ(0, FLC_BoosterPredictForMat(booster, row.data(), C_API_DTYPE_FLOAT64, 1, 4, 1, C_API_PREDICT_NORMAL, 0,
                                        -1, "", &out_len, out.data()));

  // and a new Booster trains on the extended Dataset
  BoosterHandle booster2 = nullptr;
  ASSERT_EQ(0, FLC_BoosterCreate(train, "objective=regression verbosity=-1 num_leaves=15 min_data_in_leaf=5",
                                 &booster2));
  for (int it = 0; it < 3; ++it) {
    EXPECT_EQ(0, FLC_BoosterUpdateOneIter(booster2, &finished));
  }
  FLC_BoosterFree(booster2);
  FLC_BoosterFree(booster);
  FLC_DatasetFree(extra);
  FLC_DatasetFree(train);
}
