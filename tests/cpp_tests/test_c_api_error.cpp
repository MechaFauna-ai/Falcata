/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/c_api.h>

#include <string>
#include <thread>

// FLC_SetLastError used to be an inline function in c_api.h writing a buffer
// that the header declared static, so every translation unit (and every
// library such as the SWIG wrapper) that called it wrote its own copy, which
// FLC_GetLastError (in c_api.cpp) never read. This file is such a separate
// translation unit.
TEST(CApiLastError, SetLastErrorIsWhatGetLastErrorReturns) {
  FLC_SetLastError("set from another translation unit");
  EXPECT_STREQ(FLC_GetLastError(), "set from another translation unit");

  // and an error the library sets is still read back
  FLC_SetLastError("first");
  int out_num_iterations = 0;
  BoosterHandle booster = nullptr;
  EXPECT_NE(FLC_BoosterCreateFromModelfile("/nonexistent/falcata/model.txt", &out_num_iterations, &booster), 0);
  EXPECT_STRNE(FLC_GetLastError(), "first");
}

TEST(CApiLastError, MessageIsPerThreadAndTruncated) {
  FLC_SetLastError("main thread");
  std::thread([] {
    EXPECT_STRNE(FLC_GetLastError(), "main thread");
    FLC_SetLastError("worker thread");
    EXPECT_STREQ(FLC_GetLastError(), "worker thread");
  }).join();
  EXPECT_STREQ(FLC_GetLastError(), "main thread");

  const std::string long_msg(2000, 'x');
  FLC_SetLastError(long_msg.c_str());
  EXPECT_EQ(std::string(FLC_GetLastError()), long_msg.substr(0, 511));
}
