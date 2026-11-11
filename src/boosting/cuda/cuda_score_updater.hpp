/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef FALCATA_SRC_BOOSTING_CUDA_CUDA_SCORE_UPDATER_HPP_
#define FALCATA_SRC_BOOSTING_CUDA_CUDA_SCORE_UPDATER_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>

#include <vector>

#include "../score_updater.hpp"

namespace Falcata {

/*! \brief out[0, n) = val on the device (legacy default stream): a forest's init score column (RF::Boosting) */
void LaunchFillDoubleKernel(double* out, double val, size_t n);

class CUDAScoreUpdater: public ScoreUpdater {
 public:
  CUDAScoreUpdater(const Dataset* data, int num_tree_per_iteration, const bool boosting_on_cuda);

  ~CUDAScoreUpdater();

  void AddScore(double val, int cur_tree_id) override;

  inline void AddScore(const Tree* tree, int cur_tree_id) override;

  void AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) override;

  inline void AddScore(const Tree* tree, const data_size_t* data_indices,
                       data_size_t data_cnt, int cur_tree_id) override;

  inline void MultiplyScore(double val, int cur_tree_id) override;

  /*!
  * \brief Switch where score() points (the objective moved between the device and the host). cuda_score_ is always
  * current (every update runs on the device and mirrors to the host only when not boosting on CUDA), so the host
  * mirror is filled from it, or dropped.
  */
  void SetBoostingOnCUDA(bool boosting_on_cuda);

  inline const double* score() const override {
    if (boosting_on_cuda_) {
      return cuda_score_.RawData();
    } else {
      return score_.data();
    }
  }

  /*! \brief Disable copy */
  CUDAScoreUpdater& operator=(const CUDAScoreUpdater&) = delete;

  CUDAScoreUpdater(const CUDAScoreUpdater&) = delete;

 private:
  void InitCUDA(const size_t total_size);

  /*! \brief Add a linear tree's predictions using the host evaluator, the only
   *  one that applies the leaf regressions. data_indices == nullptr scores every
   *  row; otherwise only the data_cnt rows it lists. */
  void ScoreLinearTreeOnHost(const Tree* tree, size_t offset,
                             const data_size_t* data_indices, data_size_t data_cnt);

  /*! \brief Add a linear tree's predictions on the device: the tree routes each row to its leaf (CUDATree's
   *  traversal) and a kernel evaluates the leaf regression from the dataset's raw feature values, in the host
   *  evaluator's order and rounding. Returns false (nothing done) for a tree that is not a CUDATree. */
  bool ScoreLinearTreeOnDevice(const Tree* tree, size_t offset,
                               const data_size_t* data_indices, data_size_t data_cnt);

  void LaunchLinearTreeScoreKernel(const data_size_t* data_indices, data_size_t data_cnt, int num_leaves,
                                   size_t offset);

  void LaunchAddScoreConstantKernel(const double val, const size_t offset);

  void LaunchMultiplyScoreConstantKernel(const double val, const size_t offset);

  CUDAVector<double> cuda_score_;

  /*! \brief linear trees: each row's leaf, from CUDATree::LeafIndexOnDevice */
  CUDAVector<int> cuda_linear_leaf_index_;
  /*! \brief linear trees: data_'s raw values of the features the leaf regressions used so far, one num_data_ slot
   *  per feature, uploaded on first use (they never change) */
  CUDAVector<float> cuda_linear_raw_;
  /*! \brief inner feature -> slot in cuda_linear_raw_, -1 until uploaded */
  std::vector<int> linear_raw_slot_;
  int num_linear_raw_slots_ = 0;
  /*! \brief the tree's flattened linear model: [leaf_const | leaf_output | coeff] and [coeff_offset | coeff slot] */
  CUDAVector<double> cuda_linear_model_;
  CUDAVector<int> cuda_linear_model_index_;

  const int num_threads_per_block_;

  bool boosting_on_cuda_;
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_SRC_BOOSTING_CUDA_CUDA_SCORE_UPDATER_HPP_
