/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_FOREST_TRAINER_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_FOREST_TRAINER_HPP_

#ifdef USE_CUDA

#include <Falcata/config.h>
#include <Falcata/dataset.h>
#include <Falcata/tree.h>
#include <Falcata/tree_learner.h>

#include <memory>
#include <vector>

#include "../col_sampler.hpp"
#include "cuda_single_gpu_tree_learner.hpp"

namespace Falcata {

/*!
 * \brief cuda_plan key forest_batched: grows the trees of a random forest group concurrently, one tree per learner
 *        ("slot") and per host thread, on one GPU (docs/design/batched-forest.md).
 *
 * Slot 0 is the booster's own learner; the others are learners of the same config and data, each with a private
 * view of the dataset's columns (CUDAColumnData::InitViewOf) and its own partition, histograms, split finder and
 * discretizer. A group's trees are given what sequential training would give them, in tree order: the counters of
 * the booster learner plus the tree's position (bynode node keys, stochastic rounding noise) and the next
 * feature_fraction sample of the booster learner's sampler. With the bag and gradients the booster hands each tree,
 * a tree is therefore the tree sequential training grows, whatever the group size (the model does not depend on it).
 */
class CUDAForestTrainer {
 public:
  /*! \brief One tree of a group: its slot, its bag (device indices, in-bag first; nullptr = every row) and its
   *  gradients and hessians (device) */
  struct Job {
    int slot;
    const data_size_t* bag;
    data_size_t bag_cnt;
    const score_t* gradients;
    const score_t* hessians;
  };

  /*! \brief main_learner: the booster's learner (slot 0); nullptr unless it is the single-GPU CUDA learner */
  CUDAForestTrainer(const Config* config, const Dataset* train_data, bool is_constant_hessian,
                    bool boosting_on_gpu, TreeLearner* main_learner);
  ~CUDAForestTrainer();

  /*! \brief Whether learner is the single-GPU CUDA learner the trainer drives */
  static bool Supports(TreeLearner* learner);

  /*! \brief Create slots up to wanted (at least the one there is) while device memory allows: a new slot needs its
   *  footprint (what an extra slot's Init took plus what its first tree allocated, that again until measured) free
   *  beyond the reserve (Reserve). Returns the number of slots. */
  int EnsureSlots(int wanted);

  int num_slots() const { return static_cast<int>(slots_.size()); }

  /*! \brief How many trees of the last Train were the first tree of their slot (they allocate the slot's per-tree
   *  buffers: the group's time is not the steady state) */
  int last_first_trees() const { return last_first_trees_; }

  /*! \brief The learner of a slot (renewal, train score update and bag order of its last tree) */
  TreeLearner* learner(int slot) const;

  /*! \brief Grow the jobs' trees, one thread per job (distinct slots), and hand them back in job order. The jobs are
   *  in tree order: the order sequential training grows them in. Rethrows the first error of a thread. */
  void Train(const std::vector<Job>& jobs, std::vector<std::unique_ptr<Tree>>* trees);

  /*! \brief Undo the last Train from job job_index on (the booster drops the trees it did not use): the booster
   *  learner's tree counters and feature sampler go back to what they were before that job's tree, so the next tree
   *  (in a group or in sequential training) gets what sequential training would have given it */
  void RewindTo(int job_index);

  /*! \brief The booster's new config (GBDT::ResetConfig gave it to the booster learner): every other slot's learner
   *  takes it too */
  void ResetConfig(const Config* config);

  /*! \brief Whether the device had less than the reserve free after the last Train (the next group should shrink) */
  bool memory_low() const { return memory_low_; }

  /*! \brief Free the slots beyond keep (at least slot 0 stays) and create no more: after a group left the device
   *  below the reserve or failed for device memory. Only between groups (no round of the last group pending). */
  void DropSlots(int keep);

 private:
  const Config* config_;
  const Dataset* train_data_;
  bool is_constant_hessian_;
  bool boosting_on_gpu_;
  /*! \brief slot learners, slot 0 the booster's */
  std::vector<CUDASingleGPUTreeLearner*> slots_;
  /*! \brief the learners of slots 1.. */
  std::vector<std::unique_ptr<TreeLearner>> owned_;
  /*! \brief largest device memory an extra slot's Init took */
  size_t slot_init_bytes_ = 0;
  /*! \brief largest device memory an extra slot's first tree allocated (0: no first tree grown yet) */
  size_t slot_tree_bytes_ = 0;
  /*! \brief per slot: whether it grew a tree (slot 0 counts as grown) */
  std::vector<bool> slot_trained_;
  int last_first_trees_ = 0;
  /*! \brief per job of the last Train: the booster learner's counters and (with a per-tree sample) feature sampler
   *  before the job's tree (RewindTo) */
  std::vector<CUDASingleGPUTreeLearner::TreeCounters> job_counters_;
  std::vector<std::unique_ptr<ColSampler>> job_samplers_;
  /*! \brief whether a slot could not be created for lack of memory (no further attempts) */
  bool memory_bound_ = false;
  /*! \brief whether the device had less than the reserve free after the last Train */
  bool memory_low_ = false;
  /*! \brief free device memory a new slot must leave: max(1 GiB, an eighth of the device, 2 x slot_tree_bytes_) */
  size_t Reserve(size_t total) const;
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_FOREST_TRAINER_HPP_
