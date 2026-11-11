/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <stdexcept>

#include "../../src/boosting/falb.h"

TEST(FalbWriter, RejectsZeroAlignment) {
  Falcata::FALB::ByteWriter writer;
  EXPECT_THROW(writer.Align(0), std::runtime_error);
  EXPECT_EQ(writer.size(), 0u);
  writer.Write<char>('x');
  EXPECT_THROW(writer.Align(0), std::runtime_error);
  EXPECT_EQ(writer.size(), 1u);
}

TEST(FalbWriter, PadsToPositiveAlignment) {
  Falcata::FALB::ByteWriter writer;
  writer.Align(3);
  EXPECT_EQ(writer.size(), 0u);
  writer.Write<char>('x');
  writer.Align(1);
  EXPECT_EQ(writer.size(), 1u);
  writer.Align(3);
  ASSERT_EQ(writer.size(), 3u);
  EXPECT_EQ(writer.buffer()[0], 'x');
  EXPECT_EQ(writer.buffer()[1], 0);
  EXPECT_EQ(writer.buffer()[2], 0);
  writer.Align(3);
  EXPECT_EQ(writer.size(), 3u);
}
