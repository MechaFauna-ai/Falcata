/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "../../src/cuda/device_check_once.hpp"

using Falcata::DeviceCheckOnce;

// CheckCUDADeviceSupportsThisBuild's once-per-device guard. It used one bit for every device id from 63 up, so once
// one of those devices had been checked, every other one skipped the architecture check.
TEST(DeviceCheckOnce, EveryDeviceIsClaimedExactlyOnce) {
  DeviceCheckOnce once;
  for (int device = 0; device < 200; ++device) {
    EXPECT_TRUE(once.Claim(device)) << "device " << device;
  }
  for (int device = 0; device < 200; ++device) {
    EXPECT_FALSE(once.Claim(device)) << "device " << device;
  }
}

TEST(DeviceCheckOnce, DevicesPast62DoNotShareAClaim) {
  DeviceCheckOnce once;
  EXPECT_TRUE(once.Claim(63));
  EXPECT_TRUE(once.Claim(64));
  EXPECT_TRUE(once.Claim(100));
  EXPECT_FALSE(once.Claim(64));
}

TEST(DeviceCheckOnce, ConcurrentClaimsHaveOneWinnerPerDevice) {
  DeviceCheckOnce once;
  constexpr int kThreads = 8;
  constexpr int kDevices = 130;
  std::vector<std::atomic<int>> winners(kDevices);
  for (auto& w : winners) w = 0;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      for (int device = 0; device < kDevices; ++device) {
        if (once.Claim(device)) ++winners[device];
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (int device = 0; device < kDevices; ++device) {
    EXPECT_EQ(winners[device].load(), 1) << "device " << device;
  }
}
