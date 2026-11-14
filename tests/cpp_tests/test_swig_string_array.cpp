/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <new>

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

// A slot read before it is set holds the empty string: the slots were not initialized, so getitem() returned
// uninitialized bytes (and strlen-based readers in the JNI layer read past them).
TEST(SWIGStringArray, UnsetSlotsAreEmptyStrings) {
  StringArray strings(3, 0);
  EXPECT_STREQ(strings.getitem(0), "");
  StringArray wide(2, 16);
  EXPECT_STREQ(wide.getitem(1), "");
}

// A huge element count (a negative Java long arrives as SIZE_MAX) used to wrap num_elements + 1 to 0 and then
// write the string pointers past the empty vector; it must fail cleanly instead.
TEST(SWIGStringArray, HugeElementCountIsRejected) {
  EXPECT_THROW(StringArray(static_cast<size_t>(-1), 5), std::bad_alloc);
  // likewise a string size of SIZE_MAX wrapped string_size + 1 to a 0-byte slot that accepted any string
  EXPECT_THROW(StringArray(2, static_cast<size_t>(-1)), std::bad_alloc);
}
