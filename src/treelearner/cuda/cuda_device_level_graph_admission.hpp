/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_DEVICE_LEVEL_GRAPH_ADMISSION_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_DEVICE_LEVEL_GRAPH_ADMISSION_HPP_

#include <cstddef>
#include <utility>
#include <vector>

namespace Falcata {

// Per-learner history of executable-cache MISSES, using the complete capture key. Hits bypass this policy.
// Promote on the third observation while resident in a bounded LRU. Promotion removes the history entry, so an
// executable evicted later starts cold instead of being recaptured immediately. No CUDA state is owned here.
class CUDADeviceLevelGraphAdmission {
 public:
  bool Observe(const std::vector<const void*>& key, const size_t capacity) {
    while (history_.size() > capacity) history_.erase(history_.begin());
    if (capacity == 0) return false;
    for (size_t i = 0; i < history_.size(); ++i) {
      if (history_[i].key == key) {
        Entry entry = std::move(history_[i]);
        history_.erase(history_.begin() + i);
        if (++entry.observations == 3) return true;
        history_.push_back(std::move(entry));
        return false;
      }
    }
    if (history_.size() == capacity) history_.erase(history_.begin());
    history_.push_back({key, 1});
    return false;
  }

  void Clear() { history_.clear(); }
  size_t size() const { return history_.size(); }

 private:
  struct Entry {
    std::vector<const void*> key;
    unsigned observations;
  };
  std::vector<Entry> history_;  // oldest at the front; all entries have one or two observations
};

}  // namespace Falcata
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_DEVICE_LEVEL_GRAPH_ADMISSION_HPP_
