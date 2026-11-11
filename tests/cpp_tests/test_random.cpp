/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/utils/random.h>

#include <limits>
#include <stdexcept>
#include <vector>

// NextShort / NextInt took the generator output modulo (upper - lower) with no
// check: equal bounds divided by zero (SIGFPE), inverted bounds returned values
// outside the range, and a range wider than int overflowed the subtraction.
TEST(Random, EmptyOrInvalidRangeThrows) {
  Falcata::Random r(42);
  EXPECT_THROW(r.NextInt(5, 5), std::runtime_error);
  EXPECT_THROW(r.NextShort(5, 5), std::runtime_error);
  EXPECT_THROW(r.NextInt(7, 3), std::runtime_error);
  EXPECT_THROW(r.NextShort(7, 3), std::runtime_error);
  EXPECT_THROW(r.NextInt(std::numeric_limits<int>::min(), std::numeric_limits<int>::max()), std::runtime_error);
  for (int i = 0; i < 1000; ++i) {
    const int v = r.NextInt(-3, 4);
    EXPECT_GE(v, -3);
    EXPECT_LT(v, 4);
    EXPECT_EQ(r.NextShort(9, 10), 9);
  }
}

// The draws for a seed are unchanged (bagging, sampling and every seeded
// result depend on them).
TEST(Random, SeededSequenceIsUnchanged) {
  Falcata::Random r(42);
  std::vector<int> ints, shorts;
  for (int i = 0; i < 6; ++i) {
    ints.push_back(r.NextInt(0, 1000));
  }
  for (int i = 0; i < 6; ++i) {
    shorts.push_back(r.NextShort(10, 20));
  }
  EXPECT_EQ(ints, std::vector<int>({557, 348, 367, 998, 289, 904}));
  EXPECT_EQ(shorts, std::vector<int>({16, 14, 19, 19, 19, 10}));
  EXPECT_EQ(r.Sample(100, 5), std::vector<int>({24, 37, 52, 65, 69}));
}
