/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_CUDA_DEVICE_CHECK_ONCE_HPP_
#define FALCATA_SRC_CUDA_DEVICE_CHECK_ONCE_HPP_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_set>

namespace Falcata {

// Per-device "run this once" bookkeeping for CheckCUDADeviceSupportsThisBuild, which SetCUDADevice calls on every
// CUDA entry path. Claim(device) is true the first time a device is claimed and false afterwards, for every device
// id: ids 0-63 use a lock-free bit each, larger ids a set under a mutex. (One bit used to stand for every id from 63
// up, so after one of those devices was checked the others skipped the check.) Kept free of CUDA so tests/cpp_tests
// can pin it on a CPU build.
class DeviceCheckOnce {
 public:
  bool Claim(const int device) {
    if (device >= 0 && device < 64) {
      const uint64_t bit = uint64_t{1} << device;
      return (low_.fetch_or(bit) & bit) == 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return high_.insert(device).second;
  }

 private:
  std::atomic<uint64_t> low_{0};
  std::mutex mutex_;
  std::unordered_set<int> high_;
};

}  // namespace Falcata

#endif  // FALCATA_SRC_CUDA_DEVICE_CHECK_ONCE_HPP_
