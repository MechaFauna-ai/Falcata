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
