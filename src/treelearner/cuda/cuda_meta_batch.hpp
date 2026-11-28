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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
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
      // the batch's readers too, which all follow the next flush), from a kept copy of the bytes (the source may be
      // local to the caller; see Flush)
      const uint8_t* src = static_cast<const uint8_t*>(host_src);
      std::vector<uint8_t>& kept = NextKept();
      kept.assign(src, src + bytes);
      CopyFromHostToCUDADeviceAsync<uint8_t>(static_cast<uint8_t*>(device_dst), kept.data(), bytes, 0,
                                             __FILE__, __LINE__);
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
  // one H2D copy of the staged bytes and one scatter kernel, both on the legacy default stream; the arena is only
  // rewritten by the next flush's copy, which the stream orders behind this scatter. For a pageable source the CUDA
  // documentation promises staging before return only for the synchronous cudaMemcpy, so the staged bytes are kept
  // unmodified until Release() (as TreeStartUploads keeps its copies) and the next flush stages into another buffer.
  void Flush() {
    if (segments_.count == 0) {
      return;
    }
    if (device_.Size() < host_.size()) {
      device_.Resize(host_.size() * 2);
    }
    CopyFromHostToCUDADeviceAsync<uint8_t>(device_.RawData(), host_.data(), host_.size(), 0, __FILE__, __LINE__);
    LaunchMetaBatchScatter(device_.RawDataReadOnly(), segments_);
    std::swap(host_, NextKept());  // a swap keeps both heap buffers: the copy's source stays where it is
    Discard();
  }
  // only after a host synchronization that follows every flush so far (Train()'s tree-end device sync): the kept
  // buffers are reused from then on
  void Release() { num_kept_ = 0; }
  // drops what has not been flushed (a tree start abandoned by an exception)
  void Discard() {
    host_.clear();
    segments_.count = 0;
  }

 private:
  // the next buffer to keep until Release() (a moved std::vector keeps its buffer: earlier ones stay valid)
  std::vector<uint8_t>& NextKept() {
    if (num_kept_ == kept_.size()) {
      kept_.emplace_back();
    }
    return kept_[num_kept_++];
  }
  std::vector<uint8_t> host_;
  std::vector<std::vector<uint8_t>> kept_;
  size_t num_kept_ = 0;
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

// cuda_plan key compact_prefill_dt: cross-tree overlapped metadata staging.
// While tree N executes on the GPU (device_tree), the host pre-computes tree N+1's metadata
// (BestSplitFinder masks/task lists, HistogramConstructor bin mask / used_feature_info /
// compact layout & fill tables) into a single pinned host buffer and asynchronously DMAs it
// into an inactive device staging arena on low-priority prefetch_stream_. At tree N+1 start,
// stream 0 waits on the H2D event (already complete) and runs ONE vectorized MetaBatchScatter
// into the fixed destination device pointers — zero host prep, zero PCIe wait, zero graph misses!
class PrefillMetaBatch {
 public:
  PrefillMetaBatch() { segments_.count = 0; }
  ~PrefillMetaBatch() {
    Drain();
    if (h2d_done_event_ != nullptr) {
      gpuAssert(cudaEventDestroy(h2d_done_event_), __FILE__, __LINE__);
    }
    for (int i = 0; i < 2; ++i) {
      if (scatter_done_event_[i] != nullptr) {
        gpuAssert(cudaEventDestroy(scatter_done_event_[i]), __FILE__, __LINE__);
      }
    }
    if (pinned_host_ != nullptr) {
      gpuAssert(cudaFreeHost(pinned_host_), __FILE__, __LINE__);
    }
  }

  void BeginPrepare() {
    if (h2d_pending_) {
      CUDASUCCESS_OR_FATAL(cudaEventSynchronize(h2d_done_event_));
      h2d_pending_ = false;
    }
    host_size_ = 0;
    segments_.count = 0;
  }

  void AbortPrepare() {
    host_size_ = 0;
    segments_.count = 0;
  }

  bool Add(void* device_dst, const void* host_src, const size_t bytes) {
    if (bytes == 0) {
      return true;
    }
    uint8_t* dst = ReserveDirect(device_dst, bytes);
    if (dst == nullptr) {
      return false;
    }
    std::memcpy(dst, host_src, bytes);
    return true;
  }

  template <typename T>
  bool AddTyped(T* device_dst, const T* host_src, const size_t count) {
    return Add(static_cast<void*>(device_dst), static_cast<const void*>(host_src), count * sizeof(T));
  }

  // Reserve `bytes` directly in the pinned staging buffer so large arrays (e.g., 512 KB bin_used_bytree)
  // can be constructed in place with zero extra host memcpy.
  uint8_t* ReserveDirect(void* device_dst, const size_t bytes) {
    if (bytes == 0 || segments_.count >= kMetaBatchMaxSegments) {
      return nullptr;
    }
    const size_t offset = (host_size_ + 15) & ~static_cast<size_t>(15);
    const size_t needed = offset + bytes;
    if (needed > pinned_cap_) {
      const size_t fresh_cap = std::max<size_t>(size_t{64} << 10, (needed * 2 + 4095) & ~static_cast<size_t>(4095));
      void* fresh = nullptr;
      CUDASUCCESS_OR_FATAL(cudaHostAlloc(&fresh, fresh_cap, cudaHostAllocDefault));
      if (pinned_host_ != nullptr) {
        if (host_size_ > 0) {
          std::memcpy(fresh, pinned_host_, host_size_);
        }
        CUDASUCCESS_OR_FATAL(cudaFreeHost(pinned_host_));
      }
      pinned_host_ = fresh;
      pinned_cap_ = fresh_cap;
    }
    segments_.dst[segments_.count] = static_cast<uint8_t*>(device_dst);
    segments_.src_offset[segments_.count] = static_cast<uint32_t>(offset);
    segments_.bytes[segments_.count] = static_cast<uint32_t>(bytes);
    ++segments_.count;
    host_size_ = needed;
    return static_cast<uint8_t*>(pinned_host_) + offset;
  }

  bool SubmitAsync(cudaStream_t prefetch_stream) {
    if (segments_.count == 0 || host_size_ == 0) {
      return false;
    }
    if (h2d_done_event_ == nullptr) {
      CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&h2d_done_event_, cudaEventDisableTiming));
      CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&scatter_done_event_[0], cudaEventDisableTiming));
      CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&scatter_done_event_[1], cudaEventDisableTiming));
    }
    const int slot = write_slot_;
    if (device_arena_[slot].Size() < host_size_) {
      if (scatter_pending_[slot]) {
        CUDASUCCESS_OR_FATAL(cudaEventSynchronize(scatter_done_event_[slot]));
        scatter_pending_[slot] = false;
      }
      const size_t cap = std::max<size_t>(size_t{64} << 10, host_size_ * 2);
      if (!device_arena_[slot].TryResizeDiscard(cap)) {
        AbortPrepare();
        return false;
      }
    }
    if (scatter_pending_[slot]) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(prefetch_stream, scatter_done_event_[slot], 0));
    }
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_arena_[slot].RawData(), pinned_host_, host_size_,
                                         cudaMemcpyHostToDevice, prefetch_stream));
    CUDASUCCESS_OR_FATAL(cudaEventRecord(h2d_done_event_, prefetch_stream));
    h2d_pending_ = true;
    return true;
  }

  void ConsumeOnDefaultStream() {
    if (segments_.count == 0) {
      return;
    }
    const int slot = write_slot_;
    if (h2d_pending_) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(0, h2d_done_event_, 0));
    }
    LaunchMetaBatchScatter(device_arena_[slot].RawDataReadOnly(), segments_);
    CUDASUCCESS_OR_FATAL(cudaEventRecord(scatter_done_event_[slot], 0));
    scatter_pending_[slot] = true;
    write_slot_ ^= 1;
    segments_.count = 0;
    host_size_ = 0;
  }

  void Drain() {
    if (h2d_pending_ && h2d_done_event_ != nullptr) {
      CUDASUCCESS_OR_FATAL(cudaEventSynchronize(h2d_done_event_));
      h2d_pending_ = false;
    }
    for (int i = 0; i < 2; ++i) {
      if (scatter_pending_[i] && scatter_done_event_[i] != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaEventSynchronize(scatter_done_event_[i]));
        scatter_pending_[i] = false;
      }
    }
    host_size_ = 0;
    segments_.count = 0;
  }

 private:
  void* pinned_host_ = nullptr;
  size_t pinned_cap_ = 0;
  size_t host_size_ = 0;
  CUDAVector<uint8_t> device_arena_[2];
  cudaEvent_t h2d_done_event_ = nullptr;
  cudaEvent_t scatter_done_event_[2] = {nullptr, nullptr};
  bool h2d_pending_ = false;
  bool scatter_pending_[2] = {false, false};
  int write_slot_ = 0;
  MetaBatchSegments segments_;
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_META_BATCH_HPP_
