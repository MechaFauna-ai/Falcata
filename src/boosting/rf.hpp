/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_BOOSTING_RF_HPP_
#define FALCATA_SRC_BOOSTING_RF_HPP_

#include <Falcata/boosting.h>
#include <Falcata/falcata_plan.h>
#include <Falcata/metric.h>

#include <algorithm>
#include <string>
#include <chrono>
#include <cstdio>
#include <deque>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "gbdt.h"
#include "score_updater.hpp"
#ifdef USE_CUDA
#include "../treelearner/cuda/cuda_forest_trainer.hpp"
#endif  // USE_CUDA

namespace Falcata {
/*!
* \brief Random Forest implementation
*/
class RF : public GBDT {
 public:
  RF() : GBDT() {
    average_output_ = true;
  }

  ~RF() {}

  void Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
    const std::vector<const Metric*>& training_metrics) override {
    if (config->data_sample_strategy == std::string("bagging")) {
      CHECK((config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f) ||
            (config->feature_fraction < 1.0f && config->feature_fraction > 0.0f));
    } else {
      CHECK_EQ(config->data_sample_strategy, std::string("goss"));
    }
    GBDT::Init(config, train_data, objective_function, training_metrics);

    if (num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        MultiplyScore(cur_tree_id, 1.0f / num_init_iteration_);
      }
    } else {
      CHECK_EQ(train_data->metadata().init_score(), nullptr);
    }
    CHECK_EQ(num_tree_per_iteration_, num_class_);
    // not shrinkage rate for the RF
    shrinkage_rate_ = 1.0f;
    // only boosting one time
    Boosting();
    if (data_sample_strategy_->is_use_subset() && data_sample_strategy_->bag_data_cnt() < num_data_) {
      tmp_grad_.resize(num_data_);
      tmp_hess_.resize(num_data_);
    }
  }

  void ResetConfig(const Config* config) override {
    if (config->data_sample_strategy == std::string("bagging")) {
      CHECK((config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f) ||
            (config->feature_fraction < 1.0f && config->feature_fraction > 0.0f));
    } else {
      CHECK_EQ(config->data_sample_strategy, std::string("goss"));
    }
    // the rounds grown ahead were grown under the old config: dropped (and the slots take the new one below)
    ResetForest(true);
    GBDT::ResetConfig(config);
    #ifdef USE_CUDA
    if (forest_trainer_ != nullptr) {
      if (ForestBatchedUsable()) {
        forest_trainer_->ResetConfig(config_.get());
      } else {
        forest_trainer_.reset();
      }
    }
    #endif  // USE_CUDA
    // not shrinkage rate for the RF
    shrinkage_rate_ = 1.0f;
  }

  void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
    const std::vector<const Metric*>& training_metrics) override {
    SyncTrainScore();
    ResetForest();
    GBDT::ResetTrainingData(train_data, objective_function, training_metrics);
    if (iter_ + num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        train_score_updater_->MultiplyScore(1.0f / (iter_ + num_init_iteration_), cur_tree_id);
      }
    }
    CHECK_EQ(num_tree_per_iteration_, num_class_);
    // only boosting one time
    Boosting();
    if (data_sample_strategy_->is_use_subset() && data_sample_strategy_->bag_data_cnt() < num_data_) {
      tmp_grad_.resize(num_data_);
      tmp_hess_.resize(num_data_);
    }
  }

  void Boosting() override {
    if (objective_function_ == nullptr) {
      Log::Fatal("RF mode do not support custom objective function, please use built-in objectives.");
    }
    init_scores_.resize(num_tree_per_iteration_, 0.0);
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      init_scores_[cur_tree_id] = BoostFromAverage(cur_tree_id, false);
    }
    size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
    #ifdef USE_CUDA
    // cuda_plan key rf_device_objective: the objective's kernel computes the gradients into device memory, where
    // they stay for every tree; the init score column is kept on the device only for the objectives that renew leaf
    // outputs (their residual base, see TrainOneIter). The columns are filled on the device, one constant per class
    // (no host column, no upload; the same doubles)
    cuda_init_score_.Clear();
    if (boosting_on_gpu_) {
      CUDAVector<double> gradient_scores;
      CUDAVector<double>& cuda_scores = objective_function_->IsRenewTreeOutput() ? cuda_init_score_ : gradient_scores;
      cuda_scores.Resize(total_size);
      for (int j = 0; j < num_tree_per_iteration_; ++j) {
        LaunchFillDoubleKernel(cuda_scores.RawData() + static_cast<size_t>(j) * num_data_, init_scores_[j],
                               static_cast<size_t>(num_data_));
      }
      SynchronizeCUDADevice(__FILE__, __LINE__);  // as the synchronous upload did: the columns before any stream reads
      objective_function_->GetGradients(cuda_scores.RawData(), gradients_pointer_, hessians_pointer_);
      SynchronizeCUDADevice(__FILE__, __LINE__);
      return;
    }
    #endif  // USE_CUDA
    std::vector<double> tmp_scores(total_size, 0.0f);
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int j = 0; j < num_tree_per_iteration_; ++j) {
      size_t offset = static_cast<size_t>(j)* num_data_;
      for (data_size_t i = 0; i < num_data_; ++i) {
        tmp_scores[offset + i] = init_scores_[j];
      }
    }
    objective_function_->
      GetGradients(tmp_scores.data(), gradients_.data(), hessians_.data());
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override {
    CHECK_EQ(gradients, nullptr);
    CHECK_EQ(hessians, nullptr);
    #ifdef USE_CUDA
    if (!forest_rounds_.empty() || ForestBatchedUsable()) {
      if (TrainOneIterForest()) {
        return false;
      }
    }
    #endif  // USE_CUDA
    // bagging logic
    data_sample_strategy_ ->Bagging(iter_, tree_learner_.get(), gradients_.data(), hessians_.data());
    const bool is_use_subset = data_sample_strategy_->is_use_subset();
    const data_size_t bag_data_cnt = data_sample_strategy_->bag_data_cnt();
    const std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>>& bag_data_indices = data_sample_strategy_->bag_data_indices();

    // GOSSStrategy->Bagging may modify value of bag_data_cnt_
    if (is_use_subset && bag_data_cnt < num_data_) {
      tmp_grad_.resize(num_data_);
      tmp_hess_.resize(num_data_);
    }

    CHECK_EQ(gradients, nullptr);
    CHECK_EQ(hessians, nullptr);

    // the device buffers under the device objective (rf_device_objective), else gradients_ and hessians_
    gradients = gradients_pointer_;
    hessians = hessians_pointer_;
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      std::unique_ptr<Tree> new_tree(new Tree(2, false, false));
      size_t offset = static_cast<size_t>(cur_tree_id)* num_data_;
      if (class_need_train_[cur_tree_id]) {
        auto grad = gradients + offset;
        auto hess = hessians + offset;

        if (is_use_subset && bag_data_cnt < num_data_ && !boosting_on_gpu_) {
          for (int i = 0; i < bag_data_cnt; ++i) {
            tmp_grad_[i] = grad[bag_data_indices[i]];
            tmp_hess_[i] = hess[bag_data_indices[i]];
          }
          grad = tmp_grad_.data();
          hess = tmp_hess_.data();
        }

        new_tree.reset(tree_learner_->Train(grad, hess, false));
      }

      if (new_tree->num_leaves() > 1) {
        double pred = init_scores_[cur_tree_id];
        auto residual_getter = [pred](const label_t* label, int i) {return static_cast<double>(label[i]) - pred; };
        // the residual base is the init score (residual_getter on the host; on the device, the objective's kernel
        // reads it from the score column it is handed: the init score column, not the train score)
        const double* renew_score = train_score_updater_->score();
        #ifdef USE_CUDA
        if (boosting_on_gpu_ && objective_function_->IsRenewTreeOutput()) {
          CHECK_GE(cuda_init_score_.Size(), offset + static_cast<size_t>(num_data_));
          renew_score = cuda_init_score_.RawData() + offset;
        }
        #endif  // USE_CUDA
        tree_learner_->RenewTreeOutput(new_tree.get(), objective_function_, residual_getter,
          num_data_, bag_data_indices.data(), bag_data_cnt, renew_score);
        if (std::fabs(init_scores_[cur_tree_id]) > kEpsilon) {
          new_tree->AddBias(init_scores_[cur_tree_id]);
        }
        // update score
        const int num_averaged = iter_ + num_init_iteration_;
        if (DeferTrainScore()) {
          // cuda_plan key rf_lazy_train_score: the validation scores now, the train score when it is read; the bag
          // order the in-bag score update restores is restored all the same (the next tree of the bag starts from it)
          deferred_train_trees_.push_back({new_tree.get(), cur_tree_id, num_averaged});
          tree_learner_->RestoreBagOrder();
          for (auto& score_updater : valid_score_updater_) {
            score_updater->MultiplyScore(num_averaged, cur_tree_id);
            score_updater->AddScore(new_tree.get(), cur_tree_id);
            score_updater->MultiplyScore(1.0 / (num_averaged + 1), cur_tree_id);
          }
        } else {
          FlushDeferredTrainTrees();
          MultiplyScore(cur_tree_id, num_averaged);
          UpdateScore(new_tree.get(), cur_tree_id);
          MultiplyScore(cur_tree_id, 1.0 / (num_averaged + 1));
        }
      } else {
        // only add default score one-time
        if (models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
          double output = 0.0;
          if (!class_need_train_[cur_tree_id]) {
            if (objective_function_ != nullptr) {
              output = objective_function_->BoostFromScore(cur_tree_id);
            } else {
              output = init_scores_[cur_tree_id];
            }
          }
          new_tree->AsConstantTree(output, num_data_);
          FlushDeferredTrainTrees();
          MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
          UpdateScore(new_tree.get(), cur_tree_id);
          MultiplyScore(cur_tree_id, 1.0 / (iter_ + num_init_iteration_ + 1));
        }
      }
      // add model
      models_.push_back(std::move(new_tree));
    }
    ++iter_;
    return false;
  }

  void RollbackOneIter() override {
    if (iter_ <= 0) {
      return;
    }
    SyncTrainScore();
    ResetForest(true);  // the trees grown ahead belong to rounds after the one rolled back
    int cur_iter = iter_ + num_init_iteration_ - 1;
    // reset score
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      auto curr_tree = cur_iter * num_tree_per_iteration_ + cur_tree_id;
      models_[curr_tree]->Shrinkage(-1.0);
      MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
      train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      for (auto& score_updater : valid_score_updater_) {
        score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
      MultiplyScore(cur_tree_id, 1.0f / (iter_ + num_init_iteration_ - 1));
    }
    // remove model
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      models_.pop_back();
    }
    --iter_;
  }

  void MultiplyScore(const int cur_tree_id, double val) {
    train_score_updater_->MultiplyScore(val, cur_tree_id);
    for (auto& score_updater : valid_score_updater_) {
      score_updater->MultiplyScore(val, cur_tree_id);
    }
  }

  void AddValidDataset(const Dataset* valid_data,
    const std::vector<const Metric*>& valid_metrics) override {
    GBDT::AddValidDataset(valid_data, valid_metrics);
    if (iter_ + num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        valid_score_updater_.back()->MultiplyScore(1.0f / (iter_ + num_init_iteration_), cur_tree_id);
      }
    }
  }

  bool NeedAccuratePrediction() const override {
    // No early stopping for prediction
    return true;
  };

  // the readers of the train score (cuda_plan key rf_lazy_train_score): each brings it up to date first

  std::vector<double> GetEvalAt(int data_idx) const override {
    if (data_idx == 0 && !training_metrics_.empty()) {
      SyncTrainScore();
    }
    return GBDT::GetEvalAt(data_idx);
  }

  const double* GetTrainingScore(int64_t* out_len) override {  // GetPredictAt(0) reads through it
    SyncTrainScore();
    return GBDT::GetTrainingScore(out_len);
  }

  void RefitTree(const int* tree_leaf_prediction, const size_t nrow, const size_t ncol) override {
    SyncTrainScore();
    GBDT::RefitTree(tree_leaf_prediction, nrow, ncol);
  }

  // these replace the trees a deferred addition points to

  void MergeFrom(const Falcata::Boosting* other) override {
    FlushDeferredTrainTrees();
    ResetForest(true);
    GBDT::MergeFrom(other);
  }

  void ShuffleModels(int start_iter, int end_iter) override {
    FlushDeferredTrainTrees();
    ResetForest(true);
    GBDT::ShuffleModels(start_iter, end_iter);
  }

  bool LoadModelFromString(const char* buffer, size_t len) override {
    FlushDeferredTrainTrees();
    ResetForest(true);
    return GBDT::LoadModelFromString(buffer, len);
  }

  bool LoadModelFromBinary(const char* buffer, size_t len) override {
    FlushDeferredTrainTrees();
    ResetForest(true);
    return GBDT::LoadModelFromBinary(buffer, len);
  }

 protected:
  bool EvalAndCheckEarlyStopping() override {
    if (!training_metrics_.empty()) {
      SyncTrainScore();
    } else if (early_stopping_round_ > 0) {
      FlushDeferredTrainTrees();  // early stopping pops trees the queue may point to
    }
    const bool met = GBDT::EvalAndCheckEarlyStopping();
    if (met) {
      ResetForest(true);  // training ends: the rounds grown ahead are dropped
      ReleaseForestSlots();
    }
    return met;
  }

 private:
  /*! \brief A tree whose addition to the train score is deferred (rf_lazy_train_score) */
  struct DeferredTrainTree {
    const Tree* tree;
    int cur_tree_id;
    /*! \brief Trees averaged in the score before this one (iter_ + num_init_iteration_ when it was trained) */
    int num_averaged;
  };

  /*!
   * \brief Whether the tree just trained is added to the train score only when the score is read (cuda_plan key
   * rf_lazy_train_score): device objective, no read of the score so far, one GPU, constant leaves, and the training
   * data's per-column view kept (the replay walks the tree on the training data after later trees replaced the
   * learner's view)
   */
  bool DeferTrainScore() const {
    #ifdef USE_CUDA
    return boosting_on_gpu_ && !train_score_read_ && FalcataPlan::Get().rf_lazy_train_score &&
           config_->num_gpu <= 1 && !linear_tree_ &&
           train_data_->has_cuda_column_data() && train_data_->cuda_column_data()->has_original_column_view();
    #else
    return false;
    #endif  // USE_CUDA
  }

  /*! \brief Bring the train score up to date for a reader; from then on it is updated per tree */
  void SyncTrainScore() const {
    train_score_read_ = true;
    FlushDeferredTrainTrees();
  }

  /*!
   * \brief Add the deferred trees to the train score, in training order, by the operations TrainOneIter applies
   * per tree (rescale to the sum, add, rescale to the average). The rows of a tree's bag take, by the walk over
   * every row, the leaf the learner's partition put them in, so the score is the one the per-tree updates give.
   */
  void FlushDeferredTrainTrees() const {
    for (const DeferredTrainTree& deferred : deferred_train_trees_) {
      train_score_updater_->MultiplyScore(deferred.num_averaged, deferred.cur_tree_id);
      train_score_updater_->AddScore(deferred.tree, deferred.cur_tree_id);
      train_score_updater_->MultiplyScore(1.0 / (deferred.num_averaged + 1), deferred.cur_tree_id);
    }
    deferred_train_trees_.clear();
  }

  /*!
   * \brief Drop the trees grown ahead (cuda_plan key forest_batched) and, unless keep_trainer, the forest trainer: the
   *        booster learner's tree counters and feature sampler go back to the first dropped tree's (the next tree,
   *        in a group or sequential, gets what sequential training gives it); a no-op without them. The trainer
   *        (its slots' learners) is kept wherever sequential training keeps its learner's state: under fixedpoint
   *        error feedback a class's residual lives in the slot that grows the class's trees.
   */
  void ResetForest(const bool keep_trainer = false) {
    (void)keep_trainer;
    #ifdef USE_CUDA
    if (!forest_rounds_.empty() && forest_rounds_.front().first_job >= 0) {
      forest_trainer_->RewindTo(forest_rounds_.front().first_job);
    }
    forest_rounds_.clear();
    if (!keep_trainer) {
      forest_trainer_.reset();
    }
    forest_auto_rounds_ = 0;
    forest_auto_groups_ = 0;
    forest_auto_last_rounds_ = 0;
    forest_auto_last_ms_ = 0.0;
    forest_auto_settled_ = false;
    forest_auto_size_groups_ = 0;
    forest_auto_size_ms_ = 0.0;
    #endif  // USE_CUDA
  }

  /*! \brief The device memory of the forest slots back once the training's rounds are trained (the booster lives on
   *  for predictions): the trainer goes, unless a slot carries a class's fixedpoint error-feedback residual (a
   *  continued training needs it); another update() makes the slots again */
  void ReleaseForestSlots() {
    #ifdef USE_CUDA
    if (forest_trainer_ != nullptr && forest_rounds_.empty() &&
        !(config_->ResolvedQuantMode() == QuantMode::kFixedPoint && FalcataPlan::Get().quant_ef)) {
      forest_trainer_.reset();
    }
    if (forest_rounds_.empty()) {
      forest_bags_.clear();  // the per-round bag copies (num_data indices each)
    }
    #endif  // USE_CUDA
  }

  #ifdef USE_CUDA
  /*!
   * \brief cuda_plan key forest_batched: whether this training grows its rounds in groups (CUDAForestTrainer): one
   *        CUDA device and the single-GPU learner, the device objective, bagging (or no row sampling), and nothing
   *        that couples a tree to the trees grown before it on the same learner or to a sampler's shared stream
   *        (CEGB's coupled feature penalty, per-node draws from the shared stream, compact prefill), no linear trees,
   *        forced splits or vector leaves
   */
  bool ForestBatchedUsable() const {
    const FalcataPlan& plan = FalcataPlan::Get();
    const bool cegb = !(config_->cegb_tradeoff >= 1.0f && config_->cegb_penalty_split <= 0.0f &&
                        config_->cegb_penalty_feature_coupled.empty() && config_->cegb_penalty_feature_lazy.empty());
    const bool shared_stream_bynode = config_->feature_fraction_bynode < 1.0 &&
      (!plan.level_bynode_masks || !config_->interaction_constraints_vector.empty());
    // the device sampler keyed by the round (no state carried from one round's bag to the next): every round
    // re-bags, or no round does; not by query, not balanced (the host sampler's stream)
    const bool bagging = config_->bagging_freq > 0 && config_->bagging_fraction < 1.0;
    const bool keyed_bags = !config_->bagging_by_query && config_->pos_bagging_fraction >= 1.0 &&
      config_->neg_bagging_fraction >= 1.0 && (!bagging || config_->bagging_freq == 1);
    return plan.forest_batched && config_->device_type == std::string("cuda") && config_->num_gpu <= 1 &&
           boosting_on_gpu_ && config_->data_sample_strategy == std::string("bagging") && !config_->linear_tree &&
           config_->forcedsplits_filename.empty() && config_->tree_mode != std::string("vector_leaf") && !cegb &&
           !shared_stream_bynode && keyed_bags && !plan.compact_prefill &&
           CUDAForestTrainer::Supports(tree_learner_.get());
  }

  /*!
   * \brief Train round iter_ from the trees grown ahead, growing the next group first when none are left (cuda_plan
   *        key forest_batched). Returns false when the group could not be grown in parallel (too little memory for a
   *        round's class trees): the caller trains the round sequentially.
   */
  bool TrainOneIterForest() {
    if (forest_rounds_.empty() && !GrowForestGroup()) {
      return false;
    }
    ForestRound round = std::move(forest_rounds_.front());
    forest_rounds_.pop_front();
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      std::unique_ptr<Tree> new_tree(round.trees[cur_tree_id] ? round.trees[cur_tree_id].release() :
                                     new Tree(2, false, false));
      TreeLearner* const learner = round.slots[cur_tree_id] >= 0 ?
        forest_trainer_->learner(round.slots[cur_tree_id]) : tree_learner_.get();
      if (new_tree->num_leaves() > 1) {
        double pred = init_scores_[cur_tree_id];
        auto residual_getter = [pred](const label_t* label, int i) {return static_cast<double>(label[i]) - pred; };
        const size_t offset = static_cast<size_t>(cur_tree_id) * num_data_;
        const double* renew_score = train_score_updater_->score();
        if (objective_function_->IsRenewTreeOutput()) {
          CHECK_GE(cuda_init_score_.Size(), offset + static_cast<size_t>(num_data_));
          renew_score = cuda_init_score_.RawData() + offset;
        }
        // the slot's partition still holds this tree's leaves: no slot grows a second tree before its group is used up
        learner->RenewTreeOutput(new_tree.get(), objective_function_, residual_getter,
          num_data_, nullptr, round.bag_cnt, renew_score);
        if (std::fabs(init_scores_[cur_tree_id]) > kEpsilon) {
          new_tree->AddBias(init_scores_[cur_tree_id]);
        }
        const int num_averaged = iter_ + num_init_iteration_;
        if (DeferTrainScore()) {
          // as TrainOneIter, without the bag order restore: a slot's next tree starts from a fresh copy of its bag
          deferred_train_trees_.push_back({new_tree.get(), cur_tree_id, num_averaged});
          for (auto& score_updater : valid_score_updater_) {
            score_updater->MultiplyScore(num_averaged, cur_tree_id);
            score_updater->AddScore(new_tree.get(), cur_tree_id);
            score_updater->MultiplyScore(1.0 / (num_averaged + 1), cur_tree_id);
          }
        } else {
          FlushDeferredTrainTrees();
          MultiplyScore(cur_tree_id, num_averaged);
          // GBDT::UpdateScore with the tree's own learner and bag
          train_score_updater_->AddScore(learner, new_tree.get(), cur_tree_id);
          if (round.bag != nullptr && num_data_ - round.bag_cnt > 0) {
            train_score_updater_->AddScore(new_tree.get(), round.bag + round.bag_cnt, num_data_ - round.bag_cnt,
                                           cur_tree_id);
          }
          for (auto& score_updater : valid_score_updater_) {
            score_updater->AddScore(new_tree.get(), cur_tree_id);
          }
          MultiplyScore(cur_tree_id, 1.0 / (num_averaged + 1));
        }
      } else if (models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
        // only add default score one-time (as TrainOneIter)
        double output = 0.0;
        if (!class_need_train_[cur_tree_id]) {
          output = objective_function_->BoostFromScore(cur_tree_id);
        }
        new_tree->AsConstantTree(output, num_data_);
        FlushDeferredTrainTrees();
        MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
        UpdateScore(new_tree.get(), cur_tree_id);
        MultiplyScore(cur_tree_id, 1.0 / (iter_ + num_init_iteration_ + 1));
      }
      models_.push_back(std::move(new_tree));
    }
    ++iter_;
    if (iter_ >= config_->num_iterations && forest_rounds_.empty()) {
      ReleaseForestSlots();  // the planned rounds are trained
    }
    return true;
  }

  /*!
   * \brief Grow the trees of rounds iter_ .. iter_ + R - 1 together (cuda_plan key forest_batched): the rounds' bags
   *        drawn in round order, one job per trained class tree in the order sequential training grows them. R whole
   *        rounds: forest_group trees, or under forest_group auto the size ForestAutoSize settles on; at most the
   *        rounds left of num_iterations (at least one), one round under fixedpoint error feedback (cross-tree state
   *        per class), and fewer when device memory holds fewer slots. Returns false when not even one round's class
   *        trees fit.
   */
  bool GrowForestGroup() {
    std::vector<int> trained;
    for (int c = 0; c < num_tree_per_iteration_; ++c) {
      if (class_need_train_[c]) {
        trained.push_back(c);
      }
    }
    const int per_round = static_cast<int>(trained.size());
    const FalcataPlan& plan = FalcataPlan::Get();
    const bool error_feedback = config_->ResolvedQuantMode() == QuantMode::kFixedPoint && plan.quant_ef;
    const bool auto_size = plan.forest_group == 0 && !error_feedback && per_round > 0;
    int rounds = 1;
    if (error_feedback) {
      rounds = 1;
    } else if (plan.forest_group > 0) {
      rounds = std::max(1, plan.forest_group / std::max(1, per_round));
    } else {
      rounds = std::max(1, forest_auto_rounds_);
    }
    rounds = std::max(1, std::min(rounds, config_->num_iterations - iter_));
    if (forest_trainer_ == nullptr) {
      forest_trainer_.reset(new CUDAForestTrainer(config_.get(), train_data_, is_constant_hessian_, boosting_on_gpu_,
                                                  tree_learner_.get()));
    }
    if (forest_trainer_->memory_low() && per_round > 0) {
      // the last group left the device below the trainer's reserve: half the rounds per group, and the slots beyond
      // them freed (the model does not depend on the group size)
      const int fewer = std::max(1, std::min(rounds, forest_trainer_->num_slots() / per_round) / 2);
      forest_trainer_->DropSlots(fewer * per_round);
      forest_auto_rounds_ = fewer;
      forest_auto_settled_ = true;
      rounds = std::min(rounds, fewer);
    }
    const int slots = per_round == 0 ? 1 : forest_trainer_->EnsureSlots(rounds * per_round);
    if (slots < per_round) {
      return false;
    }
    if (per_round > 0) {
      if (auto_size && slots / per_round < rounds) {
        forest_auto_rounds_ = slots / per_round;  // the memory bound: settle on what fits
        forest_auto_settled_ = true;
      }
      rounds = std::min(rounds, slots / per_round);
    }
    while (forest_bags_.size() < static_cast<size_t>(rounds)) {
      forest_bags_.emplace_back(new CUDAVector<data_size_t>());
    }
    std::vector<CUDAForestTrainer::Job> jobs;
    for (int r = 0; r < rounds; ++r) {
      ForestRound round;
      // the bag of round iter_ + r, drawn exactly as sequential training draws it (the strategy's buffer keeps the
      // last draw; bagging_freq > 1 rounds reuse it), copied: every row, in-bag first
      data_sample_strategy_->Bagging(iter_ + r, tree_learner_.get(), gradients_.data(), hessians_.data());
      round.bag_cnt = data_sample_strategy_->bag_data_cnt();
      if (round.bag_cnt < num_data_) {
        CUDAVector<data_size_t>& copy = *forest_bags_[r];
        if (copy.Size() < static_cast<size_t>(num_data_)) {
          copy.Resize(static_cast<size_t>(num_data_));
        }
        CopyFromCUDADeviceToCUDADevice<data_size_t>(copy.RawData(),
          data_sample_strategy_->cuda_bag_data_indices().RawData(), static_cast<size_t>(num_data_),
          __FILE__, __LINE__);
        round.bag = copy.RawData();
      }
      round.trees.resize(num_tree_per_iteration_);
      round.slots.assign(num_tree_per_iteration_, -1);
      round.first_job = per_round > 0 ? static_cast<int>(jobs.size()) : -1;
      for (int k = 0; k < per_round; ++k) {
        const int c = trained[k];
        const size_t offset = static_cast<size_t>(c) * num_data_;
        round.slots[c] = r * per_round + k;
        jobs.push_back({round.slots[c], round.bag, round.bag_cnt, gradients_pointer_ + offset,
                        hessians_pointer_ + offset});
      }
      forest_rounds_.push_back(std::move(round));
    }
    std::vector<std::unique_ptr<Tree>> trees;
    const auto group_start = std::chrono::steady_clock::now();
    try {
      forest_trainer_->Train(jobs, &trees);
    } catch (const std::exception& error) {
      // no round of a failed group may be popped (its trees are missing): drop them and rewind the counters and the
      // sampler, so a caller that catches the error and trains on gets the sequential trees
      ResetForest(true);
      // a group that ran out of device memory with more slots than one round needs is grown again once, on one
      // round's slots (the others freed): the trees do not depend on the group size
      if (!forest_oom_retried_ && per_round > 0 && forest_trainer_->num_slots() > per_round &&
          std::string(error.what()).find("out of memory") != std::string::npos) {
        forest_oom_retried_ = true;
        Log::Warning("rf: a group of %d trees ran out of device memory; growing %d tree(s) per group from here on",
                     static_cast<int>(jobs.size()), per_round);
        SynchronizeCUDADevice(__FILE__, __LINE__);
        forest_trainer_->DropSlots(per_round);
        forest_auto_rounds_ = 1;
        forest_auto_settled_ = true;
        return GrowForestGroup();
      }
      throw;
    } catch (...) {
      ResetForest(true);  // as above: no round of the failed group is popped
      throw;
    }
    const double group_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - group_start).count();
    if (auto_size) {
      ForestAutoSize(rounds, per_round, static_cast<int>(jobs.size()), group_ms);
    }
    if (FalcataDebug().diag) {
      size_t free_bytes = 0, total_bytes = 0;
      cudaMemGetInfo(&free_bytes, &total_bytes);
      fprintf(stderr, "[forest-diag] rounds %d..%d: %d trees on %d slots in %.1f ms (%d first trees), %.0f MiB free%s, "
              "mode %s\n", iter_, iter_ + rounds - 1, static_cast<int>(jobs.size()), forest_trainer_->num_slots(),
              group_ms, forest_trainer_->last_first_trees(), free_bytes / 1048576.0,
              auto_size ? (forest_auto_settled_ ? ", auto size settled" : ", auto size growing") : "",
              forest_trainer_->last_solo() ? "single job (non-slot)" : "slots");
    }
    size_t next = 0;
    for (ForestRound& round : forest_rounds_) {
      for (const int c : trained) {
        round.trees[c] = std::move(trees[next++]);
      }
    }
    return true;
  }

  /*!
   * \brief forest_group auto: the group size from measured time per tree. The first group holds one round (its class
   *        trees in parallel); a size's time per tree is the mean of its first two groups whose time is the steady
   *        state (no tree was its slot's first, not the training's first group). The size doubles while that time
   *        drops below 0.85 of the last size's and enough rounds are left (four groups of the doubled size), up to
   *        kForestGroupMaxTrees trees; a doubling that gains less is kept if it gained at all, else the size goes
   *        back. The model does not depend on the size.
   */
  void ForestAutoSize(const int rounds, const int per_round, const int num_trees, const double group_ms) {
    const bool first_group = forest_auto_groups_++ == 0;
    if (forest_auto_settled_ || first_group || forest_trainer_->last_first_trees() > 0 || num_trees == 0 ||
        rounds != std::max(1, forest_auto_rounds_)) {
      return;
    }
    // two steady-state groups per size: one group's time moves by several percent from group to group
    forest_auto_size_ms_ += group_ms / num_trees;
    if (++forest_auto_size_groups_ < 2) {
      return;
    }
    const double ms_per_tree = forest_auto_size_ms_ / forest_auto_size_groups_;
    forest_auto_size_ms_ = 0.0;
    forest_auto_size_groups_ = 0;
    if (forest_auto_last_rounds_ > 0 && ms_per_tree > 0.85 * forest_auto_last_ms_) {
      // the doubling gained less than 15 %: the growth ends, at the doubled size if it gained at all
      forest_auto_rounds_ = ms_per_tree < forest_auto_last_ms_ ? rounds : forest_auto_last_rounds_;
      forest_auto_settled_ = true;
      return;
    }
    forest_auto_last_rounds_ = rounds;
    forest_auto_last_ms_ = ms_per_tree;
    const int next = 2 * rounds;
    const int rounds_left = config_->num_iterations - (iter_ + rounds);
    if (next * per_round > kForestGroupMaxTrees || rounds_left < 4 * next) {
      forest_auto_rounds_ = rounds;
      forest_auto_settled_ = true;
      return;
    }
    forest_auto_rounds_ = next;
  }

  /*! \brief One round of a forest group (cuda_plan key forest_batched): its trees by class (null: class not trained),
   *  the slot that grew each (-1: none) and its bag (nullptr: every row) */
  struct ForestRound {
    std::vector<std::unique_ptr<Tree>> trees;
    std::vector<int> slots;
    const data_size_t* bag = nullptr;
    data_size_t bag_cnt = 0;
    /*! \brief the group's job of its first tree (-1: none), for CUDAForestTrainer::RewindTo */
    int first_job = -1;
  };
  /*! \brief forest_group auto: the largest group ForestAutoSize grows to */
  static constexpr int kForestGroupMaxTrees = 32;
  /*! \brief forest_group auto (ForestAutoSize): rounds per group now (0: one), groups grown, the last steady-state
   *  size and its time per tree, and whether the size settled */
  int forest_auto_rounds_ = 0;
  int forest_auto_groups_ = 0;
  int forest_auto_last_rounds_ = 0;
  double forest_auto_last_ms_ = 0.0;
  bool forest_auto_settled_ = false;
  /*! \brief whether a group was already grown again after running out of device memory (once per training) */
  bool forest_oom_retried_ = false;
  /*! \brief the current size's steady-state groups so far and their summed time per tree */
  int forest_auto_size_groups_ = 0;
  double forest_auto_size_ms_ = 0.0;
  std::unique_ptr<CUDAForestTrainer> forest_trainer_;
  /*! \brief rounds grown ahead, oldest first */
  std::deque<ForestRound> forest_rounds_;
  /*! \brief per round of a group: the copy of its bag */
  std::vector<std::unique_ptr<CUDAVector<data_size_t>>> forest_bags_;
  #endif  // USE_CUDA

  std::vector<score_t> tmp_grad_;
  std::vector<score_t> tmp_hess_;
  std::vector<double> init_scores_;
  /*! \brief Trees not yet added to the train score, oldest first (rf_lazy_train_score) */
  mutable std::vector<DeferredTrainTree> deferred_train_trees_;
  /*! \brief Whether the train score was read during training (rf_lazy_train_score: then no more deferral) */
  mutable bool train_score_read_ = false;
  #ifdef USE_CUDA
  /*! \brief The init score of every row and class on the device: the residual base of leaf renewal with the
   * device objective (rf_device_objective), only for the objectives that renew */
  CUDAVector<double> cuda_init_score_;
  #endif  // USE_CUDA
};

}  // namespace Falcata
#endif  // FALCATA_SRC_BOOSTING_RF_HPP_
