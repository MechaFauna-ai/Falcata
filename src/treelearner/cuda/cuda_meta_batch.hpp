/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_META_BATCH_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_META_BATCH_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>

#include <cstdint>
#include <cstring>
#include <vector>

namespace Falcata {

// cuda_plan key tree_meta_batch: the tree start's KB-scale host -> device metadata uploads, staged into one host
// buffer and moved by one H2D copy into a device arena, from which one kernel scatters every segment to its
// destination. Both run on the legacy default stream the uploads used, in the same place of its order relative to
// every other GPU operation: the batch is flushed before any GPU operation issued while it collects (see
// TreeStartMetaBatchScope and its Flush calls). The bytes that land in each destination are the same.
constexpr int kMetaBatchMaxSegments = 32;
constexpr size_t kMetaBatchDirectBytes = 64 << 10;
struct MetaBatchSegments {
  uint8_t* dst[kMetaBatchMaxSegments];
  uint32_t src_offset[kMetaBatchMaxSegments];
  uint32_t bytes[kMetaBatchMaxSegments];
  int count;
};

// defined in cuda_histogram_constructor.cu; launches on the legacy default stream
void LaunchMetaBatchScatter(const uint8_t* staged, const MetaBatchSegments& segments);

class TreeStartMetaBatch {
 public:
  TreeStartMetaBatch() { segments_.count = 0; }
  void Add(void* device_dst, const void* host_src, const size_t bytes) {
    if (bytes == 0) {
      return;
    }
    if (bytes > kMetaBatchDirectBytes) {
      // large enough that a copy of its own costs no more than the scatter: as without the batch (it lands before
      // the batch's readers too, which all follow the next flush)
      CopyFromHostToCUDADeviceAsync<uint8_t>(static_cast<uint8_t*>(device_dst), static_cast<const uint8_t*>(host_src),
                                             bytes, 0, __FILE__, __LINE__);
      return;
    }
    if (segments_.count == kMetaBatchMaxSegments) {
      Flush();
    }
    const size_t offset = (host_.size() + 15) & ~static_cast<size_t>(15);
    host_.resize(offset + bytes);
    std::memcpy(host_.data() + offset, host_src, bytes);
    segments_.dst[segments_.count] = static_cast<uint8_t*>(device_dst);
    segments_.src_offset[segments_.count] = static_cast<uint32_t>(offset);
    segments_.bytes[segments_.count] = static_cast<uint32_t>(bytes);
    ++segments_.count;
  }
  // one H2D copy of the staged bytes (a pageable source is staged before cudaMemcpyAsync returns, so the host buffer
  // is reusable at once) and one scatter kernel, both on the legacy default stream; the arena is only rewritten by
  // the next flush's copy, which the stream orders behind this scatter
  void Flush() {
    if (segments_.count == 0) {
      return;
    }
    if (device_.Size() < host_.size()) {
      device_.Resize(host_.size() * 2);
    }
    CopyFromHostToCUDADeviceAsync<uint8_t>(device_.RawData(), host_.data(), host_.size(), 0, __FILE__, __LINE__);
    LaunchMetaBatchScatter(device_.RawDataReadOnly(), segments_);
    Discard();
  }
  // drops what has not been flushed (a tree start abandoned by an exception)
  void Discard() {
    host_.clear();
    segments_.count = 0;
  }

 private:
  std::vector<uint8_t> host_;
  CUDAVector<uint8_t> device_;
  MetaBatchSegments segments_;
};

// the batch collecting this thread's uploads (nullptr: uploads go out one by one)
inline TreeStartMetaBatch*& CurrentTreeStartMetaBatch() {
  static thread_local TreeStartMetaBatch* current = nullptr;
  return current;
}

// makes `batch` (the tree learner's own; nullptr: off) collect this thread's tree start uploads until End(), which
// flushes them; destruction without End() (an exception unwinding the tree start) drops them instead
class TreeStartMetaBatchScope {
 public:
  explicit TreeStartMetaBatchScope(TreeStartMetaBatch* batch)
    : batch_(CurrentTreeStartMetaBatch() == nullptr ? batch : nullptr) {
    if (batch_ != nullptr) {
      CurrentTreeStartMetaBatch() = batch_;
    }
  }
  ~TreeStartMetaBatchScope() {
    if (batch_ != nullptr) {
      batch_->Discard();
      CurrentTreeStartMetaBatch() = nullptr;
    }
  }
  void End() {
    if (batch_ != nullptr) {
      batch_->Flush();
      CurrentTreeStartMetaBatch() = nullptr;
      batch_ = nullptr;
    }
  }

 private:
  TreeStartMetaBatch* batch_;
};

// flush point: call before a GPU operation that may read a batched destination
inline void FlushTreeStartMetaBatch() {
  if (CurrentTreeStartMetaBatch() != nullptr) {
    CurrentTreeStartMetaBatch()->Flush();
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_META_BATCH_HPP_
