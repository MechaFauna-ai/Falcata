/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/bin.h>
#include <Falcata/meta.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace Falcata {
// Exclusive feature bundling's group search, defined in src/io/dataset.cpp.
std::vector<std::vector<int>> FindGroups(const std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
                                         const std::vector<int>& find_order, int** sample_indices,
                                         const int* num_per_col, int num_sample_col, data_size_t total_sample_cnt,
                                         data_size_t num_data, bool is_use_gpu, bool is_sparse,
                                         std::vector<int8_t>* multi_val_group);
}  // namespace Falcata

using Falcata::BinMapper;
using Falcata::data_size_t;

namespace {

std::unique_ptr<BinMapper> MakeBinMapper(std::vector<double> values, data_size_t total) {
  std::unique_ptr<BinMapper> mapper(new BinMapper());
  mapper->FindBin(values.data(), static_cast<int>(values.size()), static_cast<size_t>(total), 255, 1, 1, false,
                  Falcata::BinType::NumericalBin, true, false, {});
  return mapper;
}

}  // namespace

// With device bundling (is_use_gpu) FindGroups keeps every group within 256
// bins. FeatureGroup sizes a group as 1 + sum(num_bin - (most_freq_bin == 0)),
// and so do FindGroups' candidate check and new-group count, but its merge
// update used default_bin instead of most_freq_bin. For features whose zero
// bin is 0 while their most frequent bin is not, every merge undercounted the
// group by one bin, so enough of them overfilled a group past 256.
TEST(FindGroups, DeviceBundlesStayWithin256Bins) {
  const data_size_t total = 20000;
  const int num_mostly_five = 80;
  const int rows_per_feature = 20;

  std::vector<std::unique_ptr<BinMapper>> mappers;
  std::vector<std::vector<int>> indices;

  // feature 0: sparse, 15 distinct nonzero values on rows [0, 200); most frequent bin is the zero bin
  {
    std::vector<double> values;
    std::vector<int> rows;
    for (int i = 0; i < 200; ++i) {
      values.push_back(1.0 + i % 15);
      rows.push_back(i);
    }
    mappers.push_back(MakeBinMapper(values, total));
    indices.push_back(rows);
  }
  // features 1..80: 5.0 everywhere except 20 own rows holding 1, 2 or 3, so the
  // zero (default) bin is bin 0 but the most frequent bin is 5.0's; the rows
  // FindGroups sees for them are the ones off the most frequent bin
  for (int k = 0; k < num_mostly_five; ++k) {
    std::vector<double> values(total, 5.0);
    std::vector<int> rows;
    const int first = 200 + k * rows_per_feature;
    for (int r = 0; r < rows_per_feature; ++r) {
      values[first + r] = 1.0 + r % 3;
      rows.push_back(first + r);
    }
    mappers.push_back(MakeBinMapper(values, total));
    indices.push_back(rows);
  }
  ASSERT_EQ(mappers[0]->GetMostFreqBin(), 0u);
  for (int f = 1; f <= num_mostly_five; ++f) {
    ASSERT_EQ(mappers[f]->GetDefaultBin(), 0u) << f;
    ASSERT_NE(mappers[f]->GetMostFreqBin(), 0u) << f;
  }

  const int num_col = static_cast<int>(mappers.size());
  std::vector<int*> index_ptrs;
  std::vector<int> num_per_col;
  std::vector<int> order;
  for (int f = 0; f < num_col; ++f) {
    index_ptrs.push_back(indices[f].data());
    num_per_col.push_back(static_cast<int>(indices[f].size()));
    order.push_back(f);
  }
  std::vector<int8_t> multi_val;
  const auto groups = Falcata::FindGroups(mappers, order, index_ptrs.data(), num_per_col.data(), num_col, total, total,
                                          true, false, &multi_val);

  size_t max_group_size = 0;
  for (const auto& group : groups) {
    int group_bins = 1;
    for (int f : group) {
      group_bins += mappers[f]->num_bin() - (mappers[f]->GetMostFreqBin() == 0 ? 1 : 0);
    }
    EXPECT_LE(group_bins, 256) << "a device bundle of " << group.size() << " features";
    max_group_size = std::max(max_group_size, group.size());
  }
  EXPECT_GT(max_group_size, 1u);  // the features do bundle
}
