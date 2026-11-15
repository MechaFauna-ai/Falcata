/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/bin.h>

#include <cstring>
#include <new>
#include <vector>

namespace {

using Falcata::BinMapper;
using Falcata::BinType;

// Build a BinMapper in storage filled with a non-zero byte, so a member that is never set
// shows up as garbage instead of passing as the zero a fresh page happens to hold.
struct DirtyBinMapper {
  alignas(BinMapper) unsigned char storage[sizeof(BinMapper)];
  BinMapper* mapper;
  DirtyBinMapper() {
    std::memset(storage, 0xbe, sizeof(storage));
    mapper = new (storage) BinMapper();
  }
  ~DirtyBinMapper() { mapper->~BinMapper(); }
};

void ExpectRoundTrip(const BinMapper& mapper) {
  std::vector<char> buffer(mapper.SizesInByte());
  mapper.CopyTo(buffer.data());
  BinMapper copy;
  // the sized reader rejects a default or most frequent bin outside [0, num_bin)
  ASSERT_NO_THROW(copy.CopyFrom(buffer.data(), buffer.size()));
  EXPECT_EQ(copy.num_bin(), mapper.num_bin());
  EXPECT_EQ(copy.GetMostFreqBin(), mapper.GetMostFreqBin());
  EXPECT_EQ(copy.GetDefaultBin(), mapper.GetDefaultBin());
}

}  // namespace

// Distributed bin finding serializes every machine's bin mappers and gathers them, trivial
// features included. A feature that FindBin found trivial (constant, or pre-filtered by
// min_data_in_leaf) never set most_freq_bin_, so data-parallel training failed with
// "Serialized bin mapper is corrupt: ... most_freq_bin=<garbage>" on every machine.
TEST(BinMapperSerialize, TrivialFeatureRoundTrips) {
  std::vector<double> values(50, 3.0);  // a constant feature
  DirtyBinMapper dirty;
  dirty.mapper->FindBin(values.data(), static_cast<int>(values.size()), values.size(), 255, 3, 20, true,
                        BinType::NumericalBin, true, false, {});
  ASSERT_TRUE(dirty.mapper->is_trivial());
  ExpectRoundTrip(*dirty.mapper);
}

TEST(BinMapperSerialize, PreFilteredFeatureRoundTrips) {
  // two values, too few rows on one side for min_data_in_leaf=20: pre-filtered to trivial
  std::vector<double> values(30, 1.0);
  values[0] = 2.0;
  DirtyBinMapper dirty;
  dirty.mapper->FindBin(values.data(), static_cast<int>(values.size()), values.size(), 255, 1, 20, true,
                        BinType::NumericalBin, true, false, {});
  ASSERT_TRUE(dirty.mapper->is_trivial());
  ExpectRoundTrip(*dirty.mapper);
}

TEST(BinMapperSerialize, DefaultConstructedRoundTrips) {
  // the distributed loader also serializes a default BinMapper for a feature past the sampled columns
  DirtyBinMapper dirty;
  ExpectRoundTrip(*dirty.mapper);
}
