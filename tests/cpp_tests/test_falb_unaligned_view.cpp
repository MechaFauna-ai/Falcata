/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "../../src/boosting/falb.h"

// A binary model's sections and mixed-width stats arrays need not be aligned for their element type; reading one
// through ByteReader::View must not perform a misaligned typed load.
TEST(FALBByteReader, ViewReadsAndCopiesAnUnalignedArray) {
  alignas(8) char data[16] = {};
  const uint32_t value = 123;
  std::memcpy(data + 1, &value, sizeof(value));
  Falcata::FALB::ByteReader reader(data, sizeof(data));
  const auto view = reader.View<uint32_t>(1, 1, "test");
  EXPECT_EQ(view[0], value);
  uint32_t copied = 0;
  view.CopyTo(&copied);
  EXPECT_EQ(copied, value);
}
