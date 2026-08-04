/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * Feature-parallel multi-GPU (tree_learner=feature, num_gpu>1): every rank
 * holds the FULL dataset and runs the ordinary single-GPU hybrid flow, but
 * constructs histograms and searches splits only over its feature stripe
 * (inner feature index % num_ranks == rank). Once per level, the ranks merge
 * their per-leaf best candidates on the HOST -- the NCCLGBDT ranks are
 * threads of one process, so the merge is a shared buffer and a barrier, no
 * NCCL involved -- and every rank applies the identical winning splits.
 * Rows never move between GPUs and every statistic is computed on the full
 * data, so the rank-local-vs-global count hazards of the data-parallel path
 * cannot exist here by construction.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_FEATURE_PARALLEL_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_FEATURE_PARALLEL_HPP_

#ifdef USE_CUDA

#include <condition_variable>
#include <mutex>
#include <vector>

#include <Falcata/cuda/cuda_split_info.hpp>

namespace Falcata {

/*! \brief shared merge state of one feature-parallel training run: reusable
 *  N-thread barrier plus per-rank pointers to the host leaf-cache buffers */
class FeatureParallelMergeState {
 public:
  explicit FeatureParallelMergeState(int num_ranks)
      : num_ranks_(num_ranks), rank_bufs_(num_ranks, nullptr) {}

  int num_ranks() const { return num_ranks_; }

  void Publish(int rank, const std::vector<CUDASplitInfo>* buf) {
    rank_bufs_[rank] = buf;
  }

  const std::vector<CUDASplitInfo>* rank_buf(int rank) const {
    return rank_bufs_[rank];
  }

  /*! \brief classic sense-reversing barrier (std::barrier is C++20; the rest
   *  of the codebase targets C++14) */
  void Arrive() {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool sense = sense_;
    if (++arrived_ == num_ranks_) {
      arrived_ = 0;
      sense_ = !sense_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, sense] { return sense_ != sense; });
    }
  }

  /*! \brief deterministic per-leaf argmax across ranks. Every rank computes
   *  the same winner: strictly greater gain wins; exact ties break to the
   *  smaller inner feature index, then the smaller threshold, so the result
   *  cannot depend on rank arrival order. */
  static bool FirstBeatsSecond(const CUDASplitInfo& a, const CUDASplitInfo& b) {
    if (!a.is_valid) return false;
    if (!b.is_valid) return true;
    if (a.gain != b.gain) return a.gain > b.gain;
    if (a.inner_feature_index != b.inner_feature_index) {
      return a.inner_feature_index < b.inner_feature_index;
    }
    return a.threshold < b.threshold;
  }

 private:
  const int num_ranks_;
  std::vector<const std::vector<CUDASplitInfo>*> rank_bufs_;
  std::mutex mutex_;
  std::condition_variable cv_;
  int arrived_ = 0;
  bool sense_ = false;
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_FEATURE_PARALLEL_HPP_
