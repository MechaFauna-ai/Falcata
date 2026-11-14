/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/c_api.h>

#include <cstdio>
#include <string>
#include <vector>

// A dataset whose only column is constant keeps no features, so the per-feature
// arrays it serializes are empty and their data() is null. LocalFile::Write passed
// that null pointer to fwrite (UBSan: "null pointer passed as argument 1").
TEST(SaveBinary, DatasetWithoutUsableFeaturesRoundTrips) {
  const std::vector<double> data = {5.0, 5.0, 5.0, 5.0};
  const std::vector<float> label = {1.0f, 0.0f, 1.0f, 0.0f};
  DatasetHandle ds = nullptr;
  ASSERT_EQ(0, FLC_DatasetCreateFromMat(data.data(), C_API_DTYPE_FLOAT64, 4, 1, 1, "verbose=-1", nullptr, &ds));
  ASSERT_EQ(0, FLC_DatasetSetField(ds, "label", label.data(), 4, C_API_DTYPE_FLOAT32));

  const std::string path = ::testing::TempDir() + "save_binary_no_features.bin";
  ASSERT_EQ(0, FLC_DatasetSaveBinary(ds, path.c_str()));

  DatasetHandle loaded = nullptr;
  ASSERT_EQ(0, FLC_DatasetCreateFromFile(path.c_str(), "verbose=-1", nullptr, &loaded));
  int num_data = 0;
  int num_feature = 0;
  EXPECT_EQ(0, FLC_DatasetGetNumData(loaded, &num_data));
  EXPECT_EQ(0, FLC_DatasetGetNumFeature(loaded, &num_feature));
  EXPECT_EQ(4, num_data);
  EXPECT_EQ(1, num_feature);

  EXPECT_EQ(0, FLC_DatasetFree(loaded));
  EXPECT_EQ(0, FLC_DatasetFree(ds));
  std::remove(path.c_str());
}
