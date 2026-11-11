/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_score_updater.hpp"

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_tree.hpp>

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

void CUDAScoreUpdater::SetBoostingOnCUDA(bool boosting_on_cuda) {
  if (boosting_on_cuda == boosting_on_cuda_) {
    return;
  }
  boosting_on_cuda_ = boosting_on_cuda;
  if (boosting_on_cuda_) {
    score_.clear();
    score_.shrink_to_fit();
  } else {
    score_.resize(cuda_score_.Size());
    CopyFromCUDADeviceToHost<double>(score_.data(), cuda_score_.RawData(), cuda_score_.Size(), __FILE__, __LINE__);
  }
}

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
    if (!ScoreLinearTreeOnDevice(tree, offset, nullptr, 0)) {
      ScoreLinearTreeOnHost(tree, offset, nullptr, 0);
    }
    return;
  }
  tree->AddPredictionToScore(data_, num_data_, cuda_score_.RawData() + offset);
  if (!boosting_on_cuda_) {
    CopyFromCUDADeviceToHost<double>(score_.data() + offset, cuda_score_.RawData() + offset, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
}





bool CUDAScoreUpdater::ScoreLinearTreeOnDevice(const Tree* tree, size_t offset,
                                               const data_size_t* data_indices,
                                               data_size_t data_cnt) {
  // The validation sets (and GBDT::UpdateScore's out-of-bag rows) used to take ScoreLinearTreeOnHost: a D2H copy
  // of the score, the host evaluator and an H2D copy per tree. Same model, same rows, on the device: the tree's
  // own traversal gives each row's leaf (the routing every non-linear tree is scored with), and the kernel
  // evaluates leaf_const + sum(coeff * raw) exactly as Tree's PredictionFunLinear does. data_ is the dataset this
  // updater scores, so the tree's inner feature indices are the ones its raw columns are keyed by (the refit
  // caveat on ScoreLinearTreeOnHost concerns the tree-learner overload only).
  const CUDATree* cuda_tree = dynamic_cast<const CUDATree*>(tree);
  if (cuda_tree == nullptr) {
    return false;
  }
  const int num_leaves = tree->num_leaves();
  // flatten the leaf models; a feature's raw column is uploaded the first time a leaf regression uses it
  if (linear_raw_slot_.empty()) {
    linear_raw_slot_.assign(static_cast<size_t>(data_->num_features()), -1);
  }
  std::vector<double> model(static_cast<size_t>(2 * num_leaves));
  std::vector<int> model_index(static_cast<size_t>(num_leaves) + 1, 0);
  for (int leaf = 0; leaf < num_leaves; ++leaf) {
    model[leaf] = tree->LeafConst(leaf);
    model[num_leaves + leaf] = tree->LeafOutput(leaf);
    model_index[leaf + 1] = model_index[leaf] + static_cast<int>(tree->LeafCoeffs(leaf).size());
  }
  for (int leaf = 0; leaf < num_leaves; ++leaf) {
    const std::vector<double>& coeffs = tree->LeafCoeffs(leaf);
    const std::vector<int>& features = tree->LeafFeaturesInner(leaf);
    for (size_t k = 0; k < coeffs.size(); ++k) {
      const int feature = features[k];
      if (linear_raw_slot_[feature] < 0) {
        const float* raw = data_->raw_index(feature);
        if (raw == nullptr) {
          return false;
        }
        // The columns are a second copy when data_ is the training set (bagging's out-of-bag pass, DART): the
        // tree learner holds its own. Keep the host path rather than spend more than half of what is free.
        size_t free_bytes = 0, total_bytes = 0;
        CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_bytes, &total_bytes));
        // Resize holds the old and the grown block at once
        const size_t grow_bytes =
          static_cast<size_t>(num_data_) * sizeof(float) * (2 * static_cast<size_t>(num_linear_raw_slots_) + 1);
        if (grow_bytes > free_bytes / 2) {
          return false;
        }
        const int slot = num_linear_raw_slots_++;
        cuda_linear_raw_.Resize(static_cast<size_t>(num_linear_raw_slots_) * static_cast<size_t>(num_data_));
        CopyFromHostToCUDADevice<float>(cuda_linear_raw_.RawData() + static_cast<size_t>(slot) * num_data_, raw,
                                        static_cast<size_t>(num_data_), __FILE__, __LINE__);
        linear_raw_slot_[feature] = slot;
      }
      model.push_back(coeffs[k]);
      model_index.push_back(linear_raw_slot_[feature]);
    }
  }
  cuda_linear_model_.InitFromHostVector(model);
  cuda_linear_model_index_.InitFromHostVector(model_index);
  if (cuda_linear_leaf_index_.Size() != static_cast<size_t>(num_data_)) {
    cuda_linear_leaf_index_.Resize(static_cast<size_t>(num_data_));
  }
  const data_size_t num_rows = data_indices == nullptr ? num_data_ : data_cnt;
  cuda_tree->LeafIndexOnDevice(data_, data_indices, num_rows, cuda_linear_leaf_index_.RawData());
  LaunchLinearTreeScoreKernel(data_indices, num_rows, num_leaves, offset);
  if (!boosting_on_cuda_) {
    CopyFromCUDADeviceToHost<double>(score_.data() + offset, cuda_score_.RawData() + offset,
                                     static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
  return true;
}

void CUDAScoreUpdater::ScoreLinearTreeOnHost(const Tree* tree, size_t offset,
                                             const data_size_t* data_indices,
                                             data_size_t data_cnt) {
  // The device traversal kernel knows only leaf constants, so a linear tree
  // scored there lands on the wrong value: training reported a metric that
  // disagreed with what the same model predicts (l2=1118.85 against a real
  // 431.77), and early stopping acted on it. Tree::AddPredictionToScore
  // evaluates the leaf regressions, so round-trip through the host.
  //
  // Not for the tree-learner overload, which GBDT::RefitTree also uses: there
  // the tree's inner feature indices belong to the dataset it was GROWN on
  // while data_ is the narrower refit one, and the host evaluator dereferences
  // those and segfaults. That overload has its own linear path on the device
  // (CUDASingleGPUTreeLearner::AddPredictionToScore).
  std::vector<double> host_score(static_cast<size_t>(num_data_));
  CopyFromCUDADeviceToHost<double>(host_score.data(), cuda_score_.RawData() + offset,
                                   static_cast<size_t>(num_data_), __FILE__, __LINE__);
  if (data_indices == nullptr) {
    tree->Tree::AddPredictionToScore(data_, num_data_, host_score.data());
  } else {
    // data_indices is a DEVICE pointer here: GBDT::UpdateScore hands the CUDA
    // score updater cuda_bag_data_indices() because every other overload feeds
    // it straight to a kernel. The host evaluator needs them on the host.
    std::vector<data_size_t> host_indices(static_cast<size_t>(data_cnt));
    CopyFromCUDADeviceToHost<data_size_t>(host_indices.data(), data_indices,
                                          host_indices.size(), __FILE__, __LINE__);
    tree->Tree::AddPredictionToScore(data_, host_indices.data(), data_cnt, host_score.data());
  }
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
  if (tree->HasLinearMetadata()) {
    // GBDT::UpdateScore scores the out-of-bag rows here. The device traversal
    // knows only leaf constants, so under bagging those rows got the constant
    // while the in-bag rows (scored by the tree learner) got the leaf
    // regression -- one model, two scorings, and a training metric its own
    // predictions could not reproduce (l2 377 against a real 535).
    if (!ScoreLinearTreeOnDevice(tree, offset, data_indices, data_cnt)) {
      ScoreLinearTreeOnHost(tree, offset, data_indices, data_cnt);
    }
    return;
  }
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
