/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <Falcata/c_api.h>

#include <cstdint>
#include <vector>

using Falcata::TestUtils;

// FLC_DatasetGetSubset must carry over the query boundaries of the queries
// whose rows it keeps (examples/lambdarank/rank.train.query starts 1, 13, 5).
TEST(DatasetSubset, KeepsQueryBoundariesOfSelectedQueries) {
  DatasetHandle full;
  ASSERT_EQ(0, TestUtils::LoadDatasetFromExamples("lambdarank/rank.train", "max_bin=15 verbose=-1", &full));

  // Query 0 is row 0; query 2 is rows 14..18.
  std::vector<int32_t> used = {0, 14, 15, 16, 17, 18};
  DatasetHandle subset;
  ASSERT_EQ(0, FLC_DatasetGetSubset(full, used.data(), static_cast<int32_t>(used.size()), "verbose=-1", &subset));

  int out_len = 0;
  const void* out_ptr = nullptr;
  int out_type = 0;
  ASSERT_EQ(0, FLC_DatasetGetField(subset, "group", &out_len, &out_ptr, &out_type));
  ASSERT_EQ(C_API_DTYPE_INT32, out_type);
  ASSERT_EQ(3, out_len);
  const int32_t* boundaries = static_cast<const int32_t*>(out_ptr);
  EXPECT_EQ(0, boundaries[0]);
  EXPECT_EQ(1, boundaries[1]);
  EXPECT_EQ(6, boundaries[2]);

  EXPECT_EQ(0, FLC_DatasetFree(subset));
  EXPECT_EQ(0, FLC_DatasetFree(full));
}

// A subset that cuts through a query keeps today's behaviour: no queries are
// carried over (callers such as the Python package set the group afterwards).
TEST(DatasetSubset, PartialQuerySubsetCarriesNoQueries) {
  DatasetHandle full;
  ASSERT_EQ(0, TestUtils::LoadDatasetFromExamples("lambdarank/rank.train", "max_bin=15 verbose=-1", &full));

  std::vector<int32_t> used = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  DatasetHandle subset;
  ASSERT_EQ(0, FLC_DatasetGetSubset(full, used.data(), static_cast<int32_t>(used.size()), "verbose=-1", &subset));

  int out_len = 0;
  const void* out_ptr = nullptr;
  int out_type = 0;
  ASSERT_EQ(0, FLC_DatasetGetField(subset, "group", &out_len, &out_ptr, &out_type));
  EXPECT_EQ(1, out_len);

  EXPECT_EQ(0, FLC_DatasetFree(subset));
  EXPECT_EQ(0, FLC_DatasetFree(full));
}
