/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_CUDA

#include "cuda_multiclass_trainer.hpp"

#include <Falcata/cuda/cuda_tree.hpp>
#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>
#include <Falcata/utils/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "../../boosting/score_updater.hpp"
#include "../col_sampler.hpp"

namespace Falcata {

CUDAMulticlassTrainer::CUDAMulticlassTrainer(const Config* config, const Dataset* train_data,
                                             const bool is_constant_hessian, const bool boosting_on_gpu,
                                             TreeLearner* main_learner)
    : config_(config),
      train_data_(train_data),
      is_constant_hessian_(is_constant_hessian),
      boosting_on_gpu_(boosting_on_gpu) {
  CHECK(Supports(main_learner));
  CUDASingleGPUTreeLearner* main = static_cast<CUDASingleGPUTreeLearner*>(main_learner);
  SetCUDADevice(main->gpu_device_id(), __FILE__, __LINE__);
  main->PresizeLevelBuffers();
  main->EnableMulticlassStream();
  slots_.push_back(main);
  slot_trained_.push_back(true);
  CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&grad_ready_event_, cudaEventDisableTiming));
  cudaEvent_t ev = nullptr;
  CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  score_done_events_.push_back(ev);
}

CUDAMulticlassTrainer::~CUDAMulticlassTrainer() {
  if (!slots_.empty() && slots_[0] != nullptr) {
    SetCUDADevice(slots_[0]->gpu_device_id(), __FILE__, __LINE__);
    for (CUDASingleGPUTreeLearner* slot : slots_) {
      slot->DisableMulticlassStream();
    }
  }
  for (cudaEvent_t ev : score_done_events_) {
    if (ev != nullptr) {
      cudaEventDestroy(ev);
    }
  }
  if (grad_ready_event_ != nullptr) {
    cudaEventDestroy(grad_ready_event_);
  }
}

bool CUDAMulticlassTrainer::Supports(TreeLearner* learner) {
  auto* single = dynamic_cast<CUDASingleGPUTreeLearner*>(learner);
  return single != nullptr && single->MulticlassConcurrentSupported();
}

size_t CUDAMulticlassTrainer::Reserve(const size_t total) const {
  return std::max({size_t{1} << 30, total / 8, 2 * slot_tree_bytes_});
}

int CUDAMulticlassTrainer::EnsureSlots(const int wanted) {
  const int capped = std::min(wanted, kMaxConcurrentStreams);
  SetCUDADevice(slots_[0]->gpu_device_id(), __FILE__, __LINE__);
  while (num_slots() < capped && !memory_bound_) {
    size_t free_before = 0, total = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_before, &total));
    const size_t reserve = Reserve(total);
    const size_t tree_bytes = slot_tree_bytes_ > 0 ? slot_tree_bytes_ : slot_init_bytes_;
    size_t pending = 0;
    for (const bool trained : slot_trained_) {
      pending += trained ? 0 : tree_bytes;
    }
    if (slot_init_bytes_ > 0 && free_before < reserve + pending + slot_init_bytes_ + tree_bytes) {
      memory_bound_ = true;
      break;
    }
    std::unique_ptr<TreeLearner> created(
        TreeLearner::CreateTreeLearner(config_->tree_learner, config_->device_type, config_, boosting_on_gpu_));
    CUDASingleGPUTreeLearner* slot = dynamic_cast<CUDASingleGPUTreeLearner*>(created.get());
    CHECK(slot != nullptr);
    slot->EnableMulticlassSlot(true, slots_[0]);
    const auto init_start = std::chrono::steady_clock::now();
    slot->Init(train_data_, is_constant_hessian_);
    slot->PresizeLevelBuffers();
    slot->EnableMulticlassStream();
    size_t free_after = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_after, &total));
    const size_t init_bytes = free_before > free_after ? free_before - free_after : size_t{0};
    slot_init_bytes_ = std::max(slot_init_bytes_, init_bytes);
    if (FalcataDebug().diag) {
      fprintf(stderr, "[multiclass-diag] slot %d init %.1f ms, %.0f MiB, %.0f MiB free after\n", num_slots(),
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - init_start).count(),
              init_bytes / 1048576.0, free_after / 1048576.0);
    }
    cudaEvent_t ev = nullptr;
    CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    score_done_events_.push_back(ev);
    slots_.push_back(slot);
    slot_trained_.push_back(false);
    owned_.push_back(std::move(created));
  }
  return num_slots();
}

void CUDAMulticlassTrainer::ResetConfig(const Config* config) {
  config_ = config;
  for (int slot = 1; slot < num_slots(); ++slot) {
    slots_[slot]->ResetConfig(config);
  }
}

void CUDAMulticlassTrainer::TrainRound(const score_t* gradients, const score_t* hessians,
                                       const std::vector<bool>& class_need_train, const bool is_first_iter,
                                       const double shrinkage_rate, const std::vector<double>& init_scores,
                                       const bool apply_bias_on_stream, ScoreUpdater* train_score_updater,
                                       std::vector<std::unique_ptr<Tree>>* out_trees) {
  const int num_classes = static_cast<int>(class_need_train.size());
  out_trees->clear();
  out_trees->resize(num_classes);

  std::vector<int> active_classes;
  active_classes.reserve(num_classes);
  for (int c = 0; c < num_classes; ++c) {
    if (class_need_train[c]) {
      active_classes.push_back(c);
    }
  }
  const int num_active = static_cast<int>(active_classes.size());
  if (num_active == 0) {
    return;
  }

  CUDASingleGPUTreeLearner* const main = slots_[0];
  SetCUDADevice(main->gpu_device_id(), __FILE__, __LINE__);
  const int S = EnsureSlots(num_active);
  if (FalcataDebug().diag && !diag_logged_) {
    diag_logged_ = true;
    fprintf(stderr, "[multiclass-diag] multiclass_concurrent: active=%d slots=%d\n", num_active, S);
  }
  const data_size_t num_data = train_data_->num_data();

  // Record completion of GBDT::Boosting() (GetGradients on stream 0) so every slot stream waits on it.
  CUDASUCCESS_OR_FATAL(cudaEventRecord(grad_ready_event_, nullptr));

  // Strict serial pre-pass in class order 0..C-1: preserve exact RNG draw sequence & counters.
  const CUDASingleGPUTreeLearner::TreeCounters base = main->tree_counters();
  ColSampler* const sampler = main->mutable_col_sampler();
  const bool need_bytree = sampler->need_reset_bytree();
  std::vector<CUDASingleGPUTreeLearner::TreeCounters> class_counters(num_active);
  std::vector<std::vector<int>> class_samples(need_bytree ? num_active : 0);
  for (int idx = 0; idx < num_active; ++idx) {
    class_counters[idx].bynode = base.bynode + static_cast<uint64_t>(idx);
    class_counters[idx].discretizer = base.discretizer + idx;
    if (need_bytree) {
      sampler->ResetByTree();
      class_samples[idx] = sampler->used_feature_indices();
    }
  }

  std::vector<bool> slot_updated_score(S, false);
  const size_t retained_size = static_cast<size_t>(std::max(1, config_->num_leaves));

  // Process active classes in waves of S slots.
  // Class at index `idx` maps deterministically to `slot = idx % S` across rounds,
  // preserving per-class error-feedback residuals (quant_ef) even when C > S.
  for (int wave_start = 0; wave_start < num_active; wave_start += S) {
    const int wave_len = std::min(S, num_active - wave_start);

    // Measure first-tree memory footprint for newly initialized slots in this wave.
    int first_trees = 0;
    for (int w = 0; w < wave_len; ++w) {
      if (w > 0 && !slot_trained_[w]) {
        ++first_trees;
      }
    }
    size_t free_before = 0, total = 0;
    if (first_trees > 0) {
      CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_before, &total));
    }

    // Pre-allocate tiny retained leaf-value buffers BEFORE launching any stream in the wave
    // so FinalizeTrainAsync performs zero cudaMalloc calls during concurrent wave execution.
    std::vector<double*> retained_bufs(wave_len, nullptr);
    for (int w = 0; w < wave_len; ++w) {
      AllocateCUDAMemory<double>(&retained_bufs[w], retained_size, __FILE__, __LINE__);
    }

    std::vector<CUDASingleGPUTreeLearner::AsyncTrainState> states(wave_len);
    std::vector<bool> active_step(wave_len, false);

    // 1. Launch all W classes in this wave back-to-back on their private non-blocking streams.
    for (int w = 0; w < wave_len; ++w) {
      const int idx = wave_start + w;
      const int class_id = active_classes[idx];
      CUDASingleGPUTreeLearner* const slot = slots_[w];
      slot->SetTreeCounters(class_counters[idx]);
      if (need_bytree) {
        slot->SetNextTreeSample(class_samples[idx]);
      }
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(slot->multiclass_exec_stream(), grad_ready_event_, 0));
      const size_t offset = static_cast<size_t>(class_id) * static_cast<size_t>(num_data);
      double* buf = retained_bufs[w];
      retained_bufs[w] = nullptr;  // ownership transferred to StartTrainAsync / CUDATree
      slot->StartTrainAsync(gradients + offset, hessians + offset, is_first_iter, buf, retained_size, &states[w]);
      active_step[w] = (states[w].phase != CUDASingleGPUTreeLearner::AsyncTrainState::Phase::kWholeDeviceTree &&
                        states[w].phase != CUDASingleGPUTreeLearner::AsyncTrainState::Phase::kDoneSync &&
                        states[w].phase != CUDASingleGPUTreeLearner::AsyncTrainState::Phase::kSelectiveReadyFinalize);
    }

    // 2. Step any multi-wave (selective) trees in lockstep across the wave until all are ready to finalize.
    bool any_stepping = true;
    while (any_stepping) {
      any_stepping = false;
      for (int w = 0; w < wave_len; ++w) {
        if (!active_step[w]) {
          continue;
        }
        active_step[w] = slots_[w]->StepTrainAsync(&states[w]);
        if (active_step[w]) {
          any_stepping = true;
        }
      }
    }

    // 3. Finalize trees as their streams complete (overlapping host tree extraction + score update launch
    //    of finished classes with GPU execution of remaining classes).
    std::vector<bool> finalized(wave_len, false);
    int remaining = wave_len;
    const auto finalize_one = [&](const int w) {
      const int idx = wave_start + w;
      const int class_id = active_classes[idx];
      CUDASingleGPUTreeLearner* const slot = slots_[w];
      cudaStream_t exec_stream = slot->multiclass_exec_stream();
      std::unique_ptr<Tree> tree(slot->FinalizeTrainAsync(&states[w]));
      if (tree->num_leaves() > 1 && tree->is_cuda_tree()) {
        CUDATree* cuda_tree = static_cast<CUDATree*>(tree.get());
        cuda_tree->ShrinkageOnStream(shrinkage_rate, exec_stream);
        train_score_updater->AddScore(slot, cuda_tree, class_id);
        if (apply_bias_on_stream && std::fabs(init_scores[class_id]) > kEpsilon) {
          cuda_tree->AddBiasOnStream(init_scores[class_id], exec_stream);
        }
        CUDASUCCESS_OR_FATAL(cudaEventRecord(score_done_events_[w], exec_stream));
        slot_updated_score[w] = true;
      }
      (*out_trees)[class_id] = std::move(tree);
      finalized[w] = true;
      --remaining;
    };

    while (remaining > 0) {
      bool progressed = false;
      for (int w = 0; w < wave_len; ++w) {
        if (finalized[w]) {
          continue;
        }
        const auto phase = states[w].phase;
        if (phase == CUDASingleGPUTreeLearner::AsyncTrainState::Phase::kDoneSync ||
            phase == CUDASingleGPUTreeLearner::AsyncTrainState::Phase::kSelectiveReadyFinalize) {
          finalize_one(w);
          progressed = true;
          continue;
        }
        const cudaError_t q = cudaEventQuery(slots_[w]->multiclass_done_event());
        if (q == cudaSuccess) {
          finalize_one(w);
          progressed = true;
        } else if (q != cudaErrorNotReady) {
          CUDASUCCESS_OR_FATAL(q);
        }
      }
      if (!progressed) {
        // Wait synchronously on the first unfinished slot in the wave
        for (int w = 0; w < wave_len; ++w) {
          if (!finalized[w]) {
            finalize_one(w);
            break;
          }
        }
      }
    }

    if (first_trees > 0) {
      size_t free_after = 0;
      CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_after, &total));
      const size_t per_tree = free_before > free_after ? (free_before - free_after) / first_trees : size_t{0};
      slot_tree_bytes_ = std::max(slot_tree_bytes_, per_tree);
      for (int w = 0; w < wave_len; ++w) {
        slot_trained_[w] = true;
      }
    }
  }

  // Advance slot 0's tree counters to match serial execution (base + num_active).
  CUDASingleGPUTreeLearner::TreeCounters next;
  next.bynode = base.bynode + static_cast<uint64_t>(num_active);
  next.discretizer = base.discretizer + num_active;
  main->SetTreeCounters(next);

  // Order stream 0 (next round's GetGradients / validation scoring / metric eval) behind every slot's score update.
  for (int w = 0; w < S; ++w) {
    if (slot_updated_score[w]) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(nullptr, score_done_events_[w], 0));
    }
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
