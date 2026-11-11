/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include "../../swig/StringArray.hpp"

// Each slot of a StringArray holds string_size characters plus the terminator, so a string that exactly fills the
// slot must be accepted (#309 fixed setitem rejecting it) and a longer one rejected.
TEST(SWIGStringArray, SetItemUsesTheFullSlotCapacity) {
  StringArray strings(1, 4);
  ASSERT_EQ(strings.setitem(0, "abcd"), 0);
  EXPECT_STREQ(strings.getitem(0), "abcd");
  EXPECT_EQ(strings.setitem(0, "abcde"), -1);
  EXPECT_EQ(strings.setitem(1, ""), -1);
  StringArray empty(1, 0);
  EXPECT_EQ(empty.setitem(0, ""), 0);
}
