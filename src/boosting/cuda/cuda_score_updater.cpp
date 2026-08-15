/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_score_updater.hpp"

#ifdef USE_CUDA

#include <algorithm>
#include <vector>

namespace Falcata {

CUDAScoreUpdater::CUDAScoreUpdater(const Dataset* data, int num_tree_per_iteration, const bool boosting_on_cuda):
  ScoreUpdater(data, num_tree_per_iteration), num_threads_per_block_(1024), boosting_on_cuda_(boosting_on_cuda) {
  num_data_ = data->num_data();
  int64_t total_size = static_cast<int64_t>(num_data_) * num_tree_per_iteration;
  InitCUDA(total_size);
  has_init_score_ = false;
  const double* init_score = data->metadata().init_score();
  // if exists initial score, will start from it
  if (init_score != nullptr) {
    if ((data->metadata().num_init_score() % num_data_) != 0
        || (data->metadata().num_init_score() / num_data_) != num_tree_per_iteration) {
      Log::Fatal("Number of class for initial score error");
    }
    has_init_score_ = true;
    CopyFromHostToCUDADevice<double>(cuda_score_.RawData(), init_score, total_size, __FILE__, __LINE__);
  } else {
    SetCUDAMemory<double>(cuda_score_.RawData(), 0, static_cast<size_t>(total_size), __FILE__, __LINE__);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (boosting_on_cuda_) {
    // clear host score buffer
    score_.clear();
    score_.shrink_to_fit();
  }
}

void CUDAScoreUpdater::InitCUDA(const size_t total_size) {
  cuda_score_.Resize(total_size);
}

CUDAScoreUpdater::~CUDAScoreUpdater() {}

inline void CUDAScoreUpdater::AddScore(double val, int cur_tree_id) {
  Common::FunctionTimer fun_timer("CUDAScoreUpdater::AddScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  LaunchAddScoreConstantKernel(val, offset);
  if (!boosting_on_cuda_) {
    CopyFromCUDADeviceToHost<double>(score_.data() + offset, cuda_score_.RawData() + offset, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
}

inline void CUDAScoreUpdater::AddScore(const Tree* tree, int cur_tree_id) {
  Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  if (tree->HasLinearMetadata()) {
    ScoreLinearTreeOnHost(tree, offset);
    return;
  }
  tree->AddPredictionToScore(data_, num_data_, cuda_score_.RawData() + offset);
  if (!boosting_on_cuda_) {
    CopyFromCUDADeviceToHost<double>(score_.data() + offset, cuda_score_.RawData() + offset, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
}





void CUDAScoreUpdater::ScoreLinearTreeOnHost(const Tree* tree, size_t offset) {
  // The device traversal kernel knows only leaf constants, so a linear tree
  // scored there lands on the wrong value: training reported a metric that
  // disagreed with what the same model predicts (l2=1118.85 against a real
  // 431.77), and early stopping acted on it. Tree::AddPredictionToScore
  // evaluates the leaf regressions, so round-trip through the host.
  //
  // Only this overload. The tree-learner overload is also used by
  // GBDT::RefitTree, where the tree's inner feature indices belong to the
  // dataset it was GROWN on while data_ is the narrower refit one -- the host
  // evaluator dereferences those and segfaults. Bagged training scores
  // therefore stay on the device path; task #88 tracks that.
  std::vector<double> host_score(static_cast<size_t>(num_data_));
  CopyFromCUDADeviceToHost<double>(host_score.data(), cuda_score_.RawData() + offset,
                                   static_cast<size_t>(num_data_), __FILE__, __LINE__);
  tree->Tree::AddPredictionToScore(data_, num_data_, host_score.data());
  CopyFromHostToCUDADevice<double>(cuda_score_.RawData() + offset, host_score.data(),
                                   static_cast<size_t>(num_data_), __FILE__, __LINE__);
  if (!boosting_on_cuda_) {
    std::copy(host_score.begin(), host_score.end(), score_.data() + offset);
  }
}

inline void CUDAScoreUpdater::AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) {
  Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  tree_learner->AddPredictionToScore(tree, cuda_score_.RawData() + offset);
  if (!boosting_on_cuda_) {
    CopyFromCUDADeviceToHost<double>(score_.data() + offset, cuda_score_.RawData() + offset, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
}

inline void CUDAScoreUpdater::AddScore(const Tree* tree, const data_size_t* data_indices,
                      data_size_t data_cnt, int cur_tree_id) {
  Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  tree->AddPredictionToScore(data_, data_indices, data_cnt, cuda_score_.RawData() + offset);
  if (!boosting_on_cuda_) {
    CopyFromCUDADeviceToHost<double>(score_.data() + offset, cuda_score_.RawData() + offset, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
}

inline void CUDAScoreUpdater::MultiplyScore(double val, int cur_tree_id) {
  Common::FunctionTimer fun_timer("CUDAScoreUpdater::MultiplyScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  LaunchMultiplyScoreConstantKernel(val, offset);
  if (!boosting_on_cuda_) {
    CopyFromCUDADeviceToHost<double>(score_.data() + offset, cuda_score_.RawData() + offset, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
