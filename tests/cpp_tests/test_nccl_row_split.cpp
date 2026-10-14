/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/meta.h>

#include <vector>

#include "../../src/boosting/cuda/nccl_row_split.hpp"

using Falcata::data_size_t;
using Falcata::NCCLMaxNumDataPerRank;
using Falcata::NCCLRankRowRange;
using Falcata::NCCLRowRange;

namespace {

// The split over num_ranks ranks: every row on exactly one rank (the ranges tile [0, num_data) in rank order),
// no negative count, no rank larger than the advertised maximum (which is what dense_rowdata is sized for).
std::vector<NCCLRowRange> CheckedSplit(const data_size_t num_data, const int num_ranks) {
  std::vector<NCCLRowRange> ranges;
  data_size_t next = 0;
  data_size_t total = 0;
  const data_size_t max_rows = NCCLMaxNumDataPerRank(num_data, num_ranks);
  for (int rank = 0; rank < num_ranks; ++rank) {
    const NCCLRowRange r = NCCLRankRowRange(num_data, num_ranks, rank);
    EXPECT_EQ(r.start, next) << "rank " << rank;
    EXPECT_GE(r.num_data(), 0) << "rank " << rank;
    EXPECT_LE(r.num_data(), max_rows) << "rank " << rank;
    next = r.end;
    total += r.num_data();
    ranges.push_back(r);
  }
  EXPECT_EQ(next, num_data);
  EXPECT_EQ(total, num_data);
  if (num_data > 0) {
    EXPECT_EQ(ranges[0].num_data(), max_rows);  // rank 0 holds the largest share
  }
  return ranges;
}

}  // namespace

TEST(NCCLRowSplit, OneRankTakesEveryRow) {
  for (const data_size_t n : {0, 1, 7, 1000000}) {
    const auto ranges = CheckedSplit(n, 1);
    EXPECT_EQ(ranges[0].start, 0);
    EXPECT_EQ(ranges[0].end, n);
    EXPECT_EQ(NCCLMaxNumDataPerRank(n, 1), n);
  }
}

TEST(NCCLRowSplit, RanksEqualDevicesEvenSplit) {
  // 4 ranks on a 4-device node, 1000 rows: 250 each (the arithmetic master used when num_gpu == device count)
  const auto ranges = CheckedSplit(1000, 4);
  for (int rank = 0; rank < 4; ++rank) {
    EXPECT_EQ(ranges[rank].start, 250 * rank);
    EXPECT_EQ(ranges[rank].num_data(), 250);
  }
}

TEST(NCCLRowSplit, FewerRanksThanDevicesCoverAllRows) {
  // the bug: 2 ranks on a 4-device node split by the device count took rows [0, 500) of 1000 and left [500, 1000)
  // untrained. Split over the 2 ranks that train, they cover every row.
  const auto ranges = CheckedSplit(1000, 2);
  EXPECT_EQ(ranges[0].start, 0);
  EXPECT_EQ(ranges[0].end, 500);
  EXPECT_EQ(ranges[1].start, 500);
  EXPECT_EQ(ranges[1].end, 1000);
  EXPECT_EQ(NCCLMaxNumDataPerRank(1000, 2), 500);
  // 3 ranks (e.g. gpu_device_id_list=0,2,5 on an 8-device node)
  const auto three = CheckedSplit(1000, 3);
  EXPECT_EQ(three[2].end, 1000);
}

TEST(NCCLRowSplit, RemainderRowsGoToLeadingRanks) {
  // 10 rows over 4 ranks: chunk ceil(10/4) = 3, so 3, 3, 3, 1 (the existing ceil chunking, unchanged)
  const auto ranges = CheckedSplit(10, 4);
  EXPECT_EQ(NCCLMaxNumDataPerRank(10, 4), 3);
  EXPECT_EQ(ranges[0].num_data(), 3);
  EXPECT_EQ(ranges[1].num_data(), 3);
  EXPECT_EQ(ranges[2].num_data(), 3);
  EXPECT_EQ(ranges[3].num_data(), 1);
  // 1000001 rows over 3 ranks: 333334, 333334, 333333
  const auto odd = CheckedSplit(1000001, 3);
  EXPECT_EQ(odd[0].num_data(), 333334);
  EXPECT_EQ(odd[2].num_data(), 333333);
}

TEST(NCCLRowSplit, RanksPastTheLastRowAreEmptyNotNegative) {
  // 5 rows over 4 ranks: chunk 2, so 2, 2, 1, 0; rank 3 starts past the data and must get an empty range at 5
  const auto ranges = CheckedSplit(5, 4);
  EXPECT_EQ(ranges[3].start, 5);
  EXPECT_EQ(ranges[3].num_data(), 0);
  CheckedSplit(9, 4);  // chunk 3: 3, 3, 3, 0
  CheckedSplit(2, 8);
}

TEST(NCCLRowSplit, LargeRowCountsDoNotOverflow) {
  // chunk * rank is computed in 64 bits
  const data_size_t n = 2147483000;
  CheckedSplit(n, 8);
  CheckedSplit(n, 7);
}
