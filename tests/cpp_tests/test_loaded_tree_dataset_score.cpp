/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/c_api.h>
#include <Falcata/tree.h>

#include <stdexcept>

TEST(TreeDatasetScore, LoadedSplitTreeRejectsMissingBins) {
  const double values[] = {0, 1, 2, 3};
  DatasetHandle handle;
  ASSERT_EQ(FLC_DatasetCreateFromMat(values, C_API_DTYPE_FLOAT64, 4, 1, 1,
      "min_data_in_bin=1 min_data_in_leaf=1 feature_pre_filter=false verbosity=-1",
      nullptr, &handle), 0);
  const auto* data = static_cast<const Falcata::Dataset*>(handle);
  Falcata::Tree source(2, false, false);
  source.Split(0, 0, 0, 1, 0.5, -1, 1, 1, 1, 1, 1, 1, Falcata::MissingType::None, true);
  const auto text = source.ToString();
  size_t used = 0;
  Falcata::Tree loaded(text.c_str(), &used);
  double scores[4] = {};
  const Falcata::data_size_t indices[] = {0, 2};
  EXPECT_THROW(loaded.AddPredictionToScore(data, 4, scores), std::runtime_error);
  EXPECT_THROW(loaded.AddPredictionToScore(data, indices, 2, scores), std::runtime_error);
  EXPECT_EQ(loaded.Predict(values), -1);
  EXPECT_NO_THROW(source.AddPredictionToScore(data, 4, scores));
  EXPECT_EQ(FLC_DatasetFree(handle), 0);
}
