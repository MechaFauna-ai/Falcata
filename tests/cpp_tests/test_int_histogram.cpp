/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/bin.h>

#include <cstdint>
#include <memory>
#include <vector>

// Quantized-gradient histograms take one packed int16 per row: the int8
// gradient in the low byte and the int8 hessian in the high byte. These tests
// compare the variants that must agree.

namespace {

using Falcata::Bin;
using Falcata::data_size_t;
using Falcata::hist_t;
using Falcata::MultiValBin;
using Falcata::score_t;

std::vector<int16_t> PackedGradients(data_size_t n) {
  std::vector<int16_t> packed(n);
  for (data_size_t i = 0; i < n; ++i) {
    const int8_t grad = static_cast<int8_t>((i * 37) % 23 - 11);
    const int8_t hess = static_cast<int8_t>(1 + i % 5);
    packed[i] = static_cast<int16_t>((static_cast<uint16_t>(static_cast<uint8_t>(hess)) << 8) |
                                     static_cast<uint8_t>(grad));
  }
  return packed;
}

const score_t* AsScore(const std::vector<int16_t>& v) { return reinterpret_cast<const score_t*>(v.data()); }

}  // namespace

// MultiValDenseBin's integer histograms ignored ORDERED and always read the
// gradient at the row index, so ConstructHistogramOrderedInt* given gradients
// ordered by position (as the float ConstructHistogramOrdered is) read the
// wrong rows.
TEST(IntHistogram, MultiValDenseOrderedMatchesRowIndexed) {
  const data_size_t n = 100;  // more than the 32-row prefetch distance
  const int num_bin = 8;
  std::unique_ptr<MultiValBin> bin(MultiValBin::CreateMultiValDenseBin(n, num_bin, 2, {0, 4, 8}));
  for (data_size_t i = 0; i < n; ++i) {
    bin->PushOneRow(0, i, {static_cast<uint32_t>(i % 4), static_cast<uint32_t>((i / 3) % 4)});
  }
  bin->FinishLoad();
  const auto packed = PackedGradients(n);
  std::vector<data_size_t> indices;
  for (data_size_t i = n - 1; i >= 0; i -= 2) {
    indices.push_back(i);
  }
  std::vector<int16_t> ordered;
  for (data_size_t idx : indices) {
    ordered.push_back(packed[idx]);
  }
  const data_size_t cnt = static_cast<data_size_t>(indices.size());

  std::vector<hist_t> by_row(num_bin, 0.0), by_position(num_bin, 0.0);  // int64 per bin
  bin->ConstructHistogramInt32(indices.data(), 0, cnt, AsScore(packed), nullptr, by_row.data());
  bin->ConstructHistogramOrderedInt32(indices.data(), 0, cnt, AsScore(ordered), nullptr, by_position.data());
  EXPECT_EQ(std::vector<int64_t>(reinterpret_cast<int64_t*>(by_row.data()),
                                 reinterpret_cast<int64_t*>(by_row.data()) + num_bin),
            std::vector<int64_t>(reinterpret_cast<int64_t*>(by_position.data()),
                                 reinterpret_cast<int64_t*>(by_position.data()) + num_bin));

  std::vector<hist_t> by_row16(num_bin, 0.0), by_position16(num_bin, 0.0);
  bin->ConstructHistogramInt16(indices.data(), 0, cnt, AsScore(packed), nullptr, by_row16.data());
  bin->ConstructHistogramOrderedInt16(indices.data(), 0, cnt, AsScore(ordered), nullptr, by_position16.data());
  EXPECT_EQ(std::vector<int32_t>(reinterpret_cast<int32_t*>(by_row16.data()),
                                 reinterpret_cast<int32_t*>(by_row16.data()) + num_bin),
            std::vector<int32_t>(reinterpret_cast<int32_t*>(by_position16.data()),
                                 reinterpret_cast<int32_t*>(by_position16.data()) + num_bin));
}

// SparseBin's no-hessian integer histogram over a contiguous range read the
// int8 gradient at [row] instead of [row << 1] (the pair stride the indexed
// variant uses), so it summed every other row's gradient and hessian bytes.
TEST(IntHistogram, SparseContiguousMatchesIndexedWithoutHessian) {
  const data_size_t n = 200;
  const int num_bin = 6;
  std::unique_ptr<Bin> bin(Bin::CreateSparseBin(n, num_bin));
  for (data_size_t i = 0; i < n; i += 3) {
    bin->Push(0, i, static_cast<uint32_t>(1 + i % (num_bin - 1)));
  }
  bin->FinishLoad();
  const auto packed = PackedGradients(n);
  std::vector<data_size_t> all(n);
  for (data_size_t i = 0; i < n; ++i) {
    all[i] = i;
  }
  // int32 gradient sum and uint32 count per bin
  std::vector<hist_t> contiguous(num_bin, 0.0), indexed(num_bin, 0.0);
  bin->ConstructHistogramInt32(0, n, AsScore(packed), contiguous.data());
  bin->ConstructHistogramInt32(all.data(), 0, n, AsScore(packed), indexed.data());
  const int32_t* c = reinterpret_cast<const int32_t*>(contiguous.data());
  const int32_t* x = reinterpret_cast<const int32_t*>(indexed.data());
  EXPECT_EQ(std::vector<int32_t>(c, c + 2 * num_bin), std::vector<int32_t>(x, x + 2 * num_bin));
  // and the indexed variant sums the gradient bytes
  int32_t expected_bin1 = 0;
  for (data_size_t i = 0; i < n; i += 3) {
    if (1 + i % (num_bin - 1) == 1) {
      expected_bin1 += static_cast<int8_t>(packed[i] & 0xff);
    }
  }
  EXPECT_EQ(x[2 * 1], expected_bin1);
}
