/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/bin.h>

#include <memory>
#include <vector>

namespace {

using Falcata::data_size_t;
using Falcata::hist_t;
using Falcata::MultiValBin;
using Falcata::score_t;

// per-bin count of stored values, read back through the histogram (gradient 1)
std::vector<int> BinCounts(const MultiValBin& bin) {
  const data_size_t n = bin.num_data();
  std::vector<score_t> grad(n, 1.0f), hess(n, 1.0f);
  std::vector<hist_t> hist(2 * bin.num_bin(), 0.0);
  bin.ConstructHistogram(0, n, grad.data(), hess.data(), hist.data());
  std::vector<int> counts;
  for (int b = 0; b < bin.num_bin(); ++b) {
    counts.push_back(static_cast<int>(hist[2 * b]));
  }
  return counts;
}

}  // namespace

// The copy constructor behind Clone() copied data_ and row_ptr_ but not the
// loader state (t_data_, t_size_, offsets_), so PushOneRow on a clone taken
// before FinishLoad indexed the empty t_size_.
TEST(MultiValSparseBin, CloneBeforeFinishLoadCanStillLoad) {
  std::unique_ptr<MultiValBin> bin(MultiValBin::CreateMultiValSparseBin(4, 6, 1.0));
  bin->PushOneRow(0, 0, {1, 3});
  bin->PushOneRow(0, 1, {2});
  std::unique_ptr<MultiValBin> clone(bin->Clone());
  clone->PushOneRow(0, 2, {5});
  clone->PushOneRow(0, 3, {1});
  clone->FinishLoad();
  EXPECT_EQ(BinCounts(*clone), std::vector<int>({0, 2, 1, 1, 0, 1}));
  EXPECT_EQ(clone->offsets(), bin->offsets());
}

// CopySubcol advanced k while val >= upper[k] with no bound, so a stored bin at
// or past upper.back() read past the range vectors. Values past the last range
// are dropped like values in an unused range.
TEST(MultiValSparseBin, CopySubcolStopsAtTheLastRange) {
  std::unique_ptr<MultiValBin> full(MultiValBin::CreateMultiValSparseBin(3, 10, 1.0));
  full->PushOneRow(0, 0, {1, 6, 9});
  full->PushOneRow(0, 1, {4});
  full->PushOneRow(0, 2, {2, 8});
  full->FinishLoad();
  // keep bins [1, 5) as [1, 5) (delta 0); 6, 8 and 9 are past every range
  std::unique_ptr<MultiValBin> sub(full->CreateLike(3, 5, 1, 1.0, {}));
  sub->CopySubcol(full.get(), {0}, {1}, {5}, {0});
  EXPECT_EQ(BinCounts(*sub), std::vector<int>({0, 1, 1, 0, 1}));
}
