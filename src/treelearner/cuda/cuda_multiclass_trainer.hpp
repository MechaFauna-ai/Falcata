/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_MULTICLASS_TRAINER_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_MULTICLASS_TRAINER_HPP_

#ifdef USE_CUDA

#include <Falcata/config.h>
#include <Falcata/dataset.h>
#include <Falcata/tree.h>
#include <Falcata/tree_learner.h>

#include <memory>
#include <vector>

#include "cuda_single_gpu_tree_learner.hpp"

namespace Falcata {

class ScoreUpdater;

/*!
 * \brief cuda_plan key multiclass_concurrent: single-host-thread multi-stream concurrent multiclass GBDT round
 *        execution.
 *
 * Slot 0 is the booster's own CUDASingleGPUTreeLearner; slots 1..S-1 are additional learners sharing immutable
 * CUDARowData and using private CUDAColumnData::InitViewOf views, each bound to its own non-blocking execution
 * and tree CUDA streams while keeping device_tree / device_level_graph enabled.
 *
 * Before launching a round's waves, per-tree RNG counters (bynode, discretizer) and feature_fraction samples are
 * assigned in strict serial class order 0..C-1 from slot 0 so every tree is bit-identical to serial execution.
 * Class c maps deterministically to slot (trained_idx % S) across rounds so fixedpoint quant_ef residuals remain
 * isolated per class.
 */
class CUDAMulticlassTrainer {
 public:
  static constexpr int kMaxConcurrentStreams = 8;

  CUDAMulticlassTrainer(const Config* config, const Dataset* train_data, bool is_constant_hessian,
                        bool boosting_on_gpu, TreeLearner* main_learner);
  ~CUDAMulticlassTrainer();

  static bool Supports(TreeLearner* learner);

  /*! \brief Create up to min(wanted, kMaxConcurrentStreams) slots within VRAM reserve. Returns num_slots(). */
  int EnsureSlots(int wanted);

  int num_slots() const { return static_cast<int>(slots_.size()); }

  /*! \brief Propagate config changes from GBDT::ResetConfig to secondary slots 1..S-1. */
  void ResetConfig(const Config* config);

  /*! \brief Train all active classes for one multiclass GBDT iteration concurrently across slots' streams,
   *  apply shrinkage and train_score_updater updates on each slot's stream (plus bias if apply_bias_on_stream is true),
   *  and order stream 0 behind all score updates via cudaStreamWaitEvent (zero host stall between rounds). */
  void TrainRound(const score_t* gradients, const score_t* hessians,
                  const std::vector<bool>& class_need_train, bool is_first_iter,
                  double shrinkage_rate, const std::vector<double>& init_scores,
                  bool apply_bias_on_stream, ScoreUpdater* train_score_updater,
                  std::vector<std::unique_ptr<Tree>>* out_trees);

 private:
  const Config* config_;
  const Dataset* train_data_;
  bool is_constant_hessian_;
  bool boosting_on_gpu_;
  std::vector<CUDASingleGPUTreeLearner*> slots_;
  std::vector<std::unique_ptr<TreeLearner>> owned_;
  std::vector<bool> slot_trained_;
  size_t slot_init_bytes_ = 0;
  size_t slot_tree_bytes_ = 0;
  bool memory_bound_ = false;
  cudaEvent_t grad_ready_event_ = nullptr;
  std::vector<cudaEvent_t> score_done_events_;

  size_t Reserve(size_t total) const;
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_MULTICLASS_TRAINER_HPP_
