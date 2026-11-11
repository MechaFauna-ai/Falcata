/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <new>

#include "../../src/network/linkers.h"

// The default constructor (Network::Dispose's placeholder) set only is_init_:
// rank(), num_machines() and the accumulated network time were indeterminate.
// Build it over memory filled with garbage to show what it leaves behind.
TEST(Linkers, DefaultConstructedIsASingleMachine) {
  alignas(Falcata::Linkers) unsigned char storage[sizeof(Falcata::Linkers)];
  std::memset(storage, 0xAB, sizeof(storage));
  Falcata::Linkers* linkers = new (storage) Falcata::Linkers();
  EXPECT_EQ(linkers->rank(), 0);
  EXPECT_EQ(linkers->num_machines(), 1);
  linkers->~Linkers();
}
