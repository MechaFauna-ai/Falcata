/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/bin.h>

#include <memory>
#include <vector>

namespace {

using Falcata::Bin;
using Falcata::BinIterator;
using Falcata::data_size_t;

// a 4-bit dense bin (num_bin <= 16) keeps the odd rows' nibbles in a side
// buffer until FinishLoad merges it into the packed data
std::unique_ptr<Bin> Make4BitBin(data_size_t num_data) {
  return std::unique_ptr<Bin>(Bin::CreateDenseBin(num_data, 16));
}

std::vector<uint32_t> RawValues(const Bin& bin, data_size_t num_data) {
  std::unique_ptr<BinIterator> it(bin.GetIterator(0, 15, 0));
  std::vector<uint32_t> out;
  for (data_size_t i = 0; i < num_data; ++i) {
    out.push_back(it->RawGet(i));
  }
  return out;
}

}  // namespace

// Push after FinishLoad dereferenced the released side buffer for an odd row
// (and an even-row push cleared the odd neighbour's nibble).
TEST(DenseBin4Bit, PushAfterFinishLoadUpdatesInPlace) {
  auto bin = Make4BitBin(5);
  for (data_size_t i = 0; i < 5; ++i) {
    bin->Push(0, i, static_cast<uint32_t>(i + 1));
  }
  bin->FinishLoad();
  EXPECT_EQ(RawValues(*bin, 5), std::vector<uint32_t>({1, 2, 3, 4, 5}));
  bin->Push(0, 1, 9);
  bin->Push(0, 2, 15);
  bin->Push(0, 4, 0);
  EXPECT_EQ(RawValues(*bin, 5), std::vector<uint32_t>({1, 9, 15, 4, 0}));
}

// ReSize grew the packed data but not the side buffer, so an odd-row push past
// the original size wrote past the side buffer.
TEST(DenseBin4Bit, ReSizeGrowsTheSideBuffer) {
  auto bin = Make4BitBin(4);
  bin->Push(0, 1, 7);
  bin->ReSize(100);
  for (data_size_t i = 4; i < 100; ++i) {
    bin->Push(0, i, static_cast<uint32_t>(i % 16));
  }
  bin->FinishLoad();
  const auto values = RawValues(*bin, 100);
  EXPECT_EQ(values[1], 7u);
  for (data_size_t i = 4; i < 100; ++i) {
    EXPECT_EQ(values[i], static_cast<uint32_t>(i % 16)) << i;
  }
}

// Clone (the private copy constructor) copied the packed data but not the side
// buffer, so a clone taken before FinishLoad lost every odd row.
TEST(DenseBin4Bit, CloneBeforeFinishLoadKeepsOddRows) {
  auto bin = Make4BitBin(6);
  for (data_size_t i = 0; i < 6; ++i) {
    bin->Push(0, i, static_cast<uint32_t>(10 + i));
  }
  std::unique_ptr<Bin> clone(bin->Clone());
  clone->Push(0, 5, 3);
  clone->FinishLoad();
  bin->FinishLoad();
  EXPECT_EQ(RawValues(*clone, 6), std::vector<uint32_t>({10, 11, 12, 13, 14, 3}));
  EXPECT_EQ(RawValues(*bin, 6), std::vector<uint32_t>({10, 11, 12, 13, 14, 15}));
}

// MaxStoredBin scans every stored value of a loaded binary dataset (FeatureGroup::CheckLoadedBins): it must equal the
// element-wise maximum for every width, with the maximum in either nibble of a 4-bit byte, an odd row count, and
// sizes past the vectorized loop's blocks.
TEST(DenseBin, MaxStoredBinIsTheLargestStoredValue) {
  for (const int num_bin : {16, 255, 60000, 70000}) {
    for (const data_size_t n : {1, 2, 3, 31, 64, 1001}) {
      for (data_size_t at = 0; at < n; at += (n > 64 ? 97 : 1)) {
        std::unique_ptr<Bin> bin(Bin::CreateDenseBin(n, num_bin));
        const uint32_t top = static_cast<uint32_t>(num_bin - 1);
        for (data_size_t i = 0; i < n; ++i) {
          bin->Push(0, i, i == at ? top : static_cast<uint32_t>(i % 3));
        }
        bin->FinishLoad();
        EXPECT_EQ(bin->MaxStoredBin(), top) << "num_bin " << num_bin << " n " << n << " at " << at;
      }
    }
  }
  std::unique_ptr<Bin> zeros(Bin::CreateDenseBin(7, 16));
  zeros->FinishLoad();
  EXPECT_EQ(zeros->MaxStoredBin(), 0u);
}
