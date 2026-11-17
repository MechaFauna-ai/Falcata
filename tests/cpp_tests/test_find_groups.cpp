/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/bin.h>
#include <Falcata/meta.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace Falcata {
// Exclusive feature bundling's group search, defined in src/io/dataset.cpp.
std::vector<std::vector<int>> FindGroups(const std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
                                         const std::vector<int>& find_order, int** sample_indices,
                                         const int* num_per_col, int num_sample_col, data_size_t total_sample_cnt,
                                         data_size_t num_data, bool is_use_gpu, bool is_sparse,
                                         std::vector<int8_t>* multi_val_group);
// The same search on the sets as FastFeatureBundling holds them: bitmaps, and arrays for the columns with fewer than
// min_bitmap_count rows.
std::vector<std::vector<int>> FindGroupsOnRowSetBitmaps(const std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
                                                        const std::vector<int>& find_order, int** sample_indices,
                                                        const int* num_per_col, int num_sample_col,
                                                        data_size_t total_sample_cnt, data_size_t num_data,
                                                        bool is_use_gpu, bool is_sparse,
                                                        std::vector<int8_t>* multi_val_group, size_t min_bitmap_count);
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

// FastFeatureBundling holds a column's row set as a bitmap, or as an array when it has fewer rows than the bitmap has
// words; the group search counts and marks either kind. Bitmaps only, arrays only, the mixed rule and the array
// interface must find the same groups, on columns from a few rows to most rows (some bundle, some conflict, some
// end in multi-value groups), for every device / sparsity setting.
TEST(FindGroups, BitmapAndArrayRowSetsGiveTheSameGroups) {
  const data_size_t total = 6000;  // 94 bitmap words
  const size_t words = (static_cast<size_t>(total) + 63) / 64;
  std::mt19937 gen(7);
  std::vector<std::unique_ptr<BinMapper>> mappers;
  std::vector<std::vector<int>> indices;
  const int counts[] = {3, 20, 60, 93, 94, 95, 150, 400, 1200, 3000, 5500};
  for (int f = 0; f < 120; ++f) {
    int n = counts[f % 11];
    std::vector<int> rows(total);
    for (int r = 0; r < total; ++r) {
      rows[r] = r;
    }
    if (f % 5 == 0) {
      // disjoint-ish blocks: these bundle
      std::rotate(rows.begin(), rows.begin() + (f * 97) % total, rows.end());
    } else if (f % 5 == 1) {
      // rows in the last 100 only (the partial last bitmap word and the one before): their conflicts decide
      n = std::min(n, 12);
      std::shuffle(rows.end() - 100, rows.end(), gen);
      std::rotate(rows.begin(), rows.end() - 100, rows.end());
    } else {
      std::shuffle(rows.begin(), rows.end(), gen);
    }
    rows.resize(n);
    std::sort(rows.begin(), rows.end());
    std::vector<double> values;
    for (int i = 0; i < n; ++i) {
      values.push_back(1.0 + (i + f) % 7);
    }
    mappers.push_back(MakeBinMapper(values, total));
    indices.push_back(rows);
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
  std::shuffle(order.begin(), order.end(), gen);
  bool bundled = false;
  for (int setting = 0; setting < 4; ++setting) {
    const bool is_use_gpu = setting & 1;
    const bool is_sparse = setting & 2;
    std::vector<int8_t> mv_ref;
    const auto ref = Falcata::FindGroups(mappers, order, index_ptrs.data(), num_per_col.data(), num_col, total, total,
                                         is_use_gpu, is_sparse, &mv_ref);
    for (size_t min_bitmap_count : {static_cast<size_t>(0), words, static_cast<size_t>(total) + 1}) {
      std::vector<int8_t> mv;
      const auto groups = Falcata::FindGroupsOnRowSetBitmaps(mappers, order, index_ptrs.data(), num_per_col.data(),
                                                             num_col, total, total, is_use_gpu, is_sparse, &mv,
                                                             min_bitmap_count);
      EXPECT_EQ(groups, ref) << "setting " << setting << ", min_bitmap_count " << min_bitmap_count;
      EXPECT_EQ(mv, mv_ref) << "setting " << setting << ", min_bitmap_count " << min_bitmap_count;
    }
    for (const auto& group : ref) {
      bundled = bundled || group.size() > 1;
    }
  }
  EXPECT_TRUE(bundled);  // the search does merge columns, so the conflict counts matter
}

// Two columns off their most frequent bin on one common sampled row must not share a single-value group: that row
// would keep the bin of only one of them, and the other's splits would route it wrongly. LightGBM tolerated
// total_sample_cnt / 10000 such rows per group (here 2); bundles are now exact. On the CPU second round (is_sparse)
// the pair may still share a group, but only a multi-value one, which keeps both bins.
TEST(FindGroups, ColumnsThatShareARowDoNotShareASingleValueGroup) {
  const data_size_t total = 20000;
  std::vector<std::unique_ptr<BinMapper>> mappers;
  std::vector<std::vector<int>> indices;
  // column 0: rows [0, 400); column 1: rows [399, 799) (row 399 in common); column 2: rows [1000, 1400) (exclusive)
  for (int first : {0, 399, 1000}) {
    std::vector<double> values(400, 1.0);
    std::vector<int> rows;
    for (int r = 0; r < 400; ++r) {
      rows.push_back(first + r);
    }
    mappers.push_back(MakeBinMapper(values, total));
    indices.push_back(rows);
  }
  std::vector<int*> index_ptrs;
  std::vector<int> num_per_col, order;
  for (int f = 0; f < 3; ++f) {
    index_ptrs.push_back(indices[f].data());
    num_per_col.push_back(static_cast<int>(indices[f].size()));
    order.push_back(f);
  }
  for (int setting = 0; setting < 4; ++setting) {
    const bool is_use_gpu = setting & 1;
    const bool is_sparse = setting & 2;
    std::vector<int8_t> multi_val;
    const auto groups = Falcata::FindGroups(mappers, order, index_ptrs.data(), num_per_col.data(), 3, total, total,
                                            is_use_gpu, is_sparse, &multi_val);
    ASSERT_EQ(groups.size(), multi_val.size());
    bool bundled = false;
    for (size_t g = 0; g < groups.size(); ++g) {
      const bool has0 = std::count(groups[g].begin(), groups[g].end(), 0) > 0;
      const bool has1 = std::count(groups[g].begin(), groups[g].end(), 1) > 0;
      EXPECT_FALSE(has0 && has1 && !multi_val[g]) << "setting " << setting;
      bundled = bundled || groups[g].size() > 1;
    }
    EXPECT_TRUE(bundled) << "setting " << setting;  // the exclusive column still bundles
  }
}

// FindBin sorts a copy: bundling pairs the sampled values with their row ids afterwards (the most-frequent-bin masks
// of FixSampleIndices), which an in-place sort scrambles.
TEST(FindBin, LeavesTheSampledValuesInPlace) {
  std::vector<double> values;
  for (int i = 0; i < 3000; ++i) {
    values.push_back(i % 7 == 0 ? std::nan("") : 0.5 + (i * 7919) % 13);
  }
  const std::vector<double> before = values;
  BinMapper mapper;
  mapper.FindBin(values.data(), static_cast<int>(values.size()), values.size() + 500, 255, 1, 1, false,
                 Falcata::BinType::NumericalBin, true, false, {});
  ASSERT_EQ(values.size(), before.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (std::isnan(before[i])) {
      EXPECT_TRUE(std::isnan(values[i])) << i;
    } else {
      EXPECT_EQ(values[i], before[i]) << i;
    }
  }
}
