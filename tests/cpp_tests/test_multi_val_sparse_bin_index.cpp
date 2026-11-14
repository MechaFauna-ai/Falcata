/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/bin.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using Falcata::data_size_t;
using Falcata::MultiValBin;

// CreateMultiValSparseBin picks the row-offset type (uint16/uint32/size_t)
// from the estimated entry count. When the rows hold more entries than that
// type can count, the row offsets wrapped and the merged data was silently
// truncated. Loading now fails with an error instead.
TEST(MultiValSparseBinIndex, MoreEntriesThanTheIndexTypeHoldsIsAnError) {
  // estimate 1000 * 0.01 * 1.1 = 11 entries: a uint16_t row index
  std::unique_ptr<MultiValBin> bin(MultiValBin::CreateMultiValSparseBin(1000, 256, 0.01));
  std::vector<uint32_t> row(100);
  for (int j = 0; j < 100; ++j) {
    row[j] = static_cast<uint32_t>(j + 1);
  }
  for (data_size_t i = 0; i < 1000; ++i) {  // 100,000 entries > 65,535
    bin->PushOneRow(0, i, row);
  }
  // the builder learns the exact count (and rebuilds with a wider index; see Dataset)
  uint64_t num_entries = 0;
  EXPECT_FALSE(bin->PushedEntriesFit(&num_entries));
  EXPECT_EQ(num_entries, 100000u);
  EXPECT_THROW(bin->FinishLoad(), std::runtime_error);
  // a bin created from that count holds the same rows
  std::unique_ptr<MultiValBin> wide(MultiValBin::CreateMultiValSparseBin(
      1000, 256, static_cast<double>(num_entries) / 1000));
  for (data_size_t i = 0; i < 1000; ++i) {
    wide->PushOneRow(0, i, row);
  }
  EXPECT_TRUE(wide->PushedEntriesFit(&num_entries));
  EXPECT_NO_THROW(wide->FinishLoad());
  EXPECT_DOUBLE_EQ(wide->num_element_per_row(), 100.0);
}

// A single row longer than the index type can count is not stored; the bin reports it and
// refuses to finish loading.
TEST(MultiValSparseBinIndex, RowLongerThanTheIndexTypeIsAnError) {
  std::unique_ptr<MultiValBin> bin(MultiValBin::CreateMultiValSparseBin(2, 1 << 17, 1.0));
  std::vector<uint32_t> row(70000);
  for (int j = 0; j < 70000; ++j) {
    row[j] = static_cast<uint32_t>(j + 1);
  }
  EXPECT_NO_THROW(bin->PushOneRow(0, 0, row));
  bin->PushOneRow(0, 1, std::vector<uint32_t>{1});
  uint64_t num_entries = 0;
  EXPECT_FALSE(bin->PushedEntriesFit(&num_entries));
  EXPECT_EQ(num_entries, 70001u);
  EXPECT_THROW(bin->FinishLoad(), std::runtime_error);
}

// Below the limit nothing changes: 60,000 entries fit a uint16_t index.
TEST(MultiValSparseBinIndex, EntriesWithinTheIndexTypeLoad) {
  std::unique_ptr<MultiValBin> bin(MultiValBin::CreateMultiValSparseBin(600, 256, 0.01));
  std::vector<uint32_t> row(100, 7);
  for (data_size_t i = 0; i < 600; ++i) {
    bin->PushOneRow(0, i, row);
  }
  bin->FinishLoad();
  EXPECT_DOUBLE_EQ(bin->num_element_per_row(), 100.0);
}
