/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_CUDA

#include "cuda_forest_trainer.hpp"

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>
#include <Falcata/utils/log.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "cuda_single_gpu_tree_learner.hpp"

namespace Falcata {

CUDAForestTrainer::CUDAForestTrainer(const Config* config, const Dataset* train_data, const bool is_constant_hessian,
                                     const bool boosting_on_gpu, TreeLearner* main_learner)
  : config_(config), train_data_(train_data), is_constant_hessian_(is_constant_hessian),
    boosting_on_gpu_(boosting_on_gpu) {
  CHECK(Supports(main_learner));
  CUDASingleGPUTreeLearner* main = static_cast<CUDASingleGPUTreeLearner*>(main_learner);
  // slot 0 keeps the dataset's column view: nothing else installs one while the group trains
  main->EnableForestSlot(false);
  main->PresizeLevelBuffers();
  slots_.push_back(main);
  slot_trained_.push_back(true);
}

CUDAForestTrainer::~CUDAForestTrainer() {
  slots_[0]->LeaveForest();
}

bool CUDAForestTrainer::Supports(TreeLearner* learner) {
  return learner != nullptr && dynamic_cast<CUDASingleGPUTreeLearner*>(learner) != nullptr;
}

TreeLearner* CUDAForestTrainer::learner(const int slot) const {
  CHECK(slot >= 0 && slot < num_slots());
  return slots_[slot];
}

size_t CUDAForestTrainer::Reserve(const size_t total) const {
  // the device keeps an eighth free (at least 1 GiB), and at least twice what a slot's first tree allocated (the
  // transient allocations of a group's trees growing together come out of it)
  return std::max({size_t{1} << 30, total / 8, 2 * slot_tree_bytes_});
}

int CUDAForestTrainer::EnsureSlots(const int wanted) {
  SetCUDADevice(slots_[0]->gpu_device_id(), __FILE__, __LINE__);
  while (num_slots() < wanted && !memory_bound_) {
    size_t free_before = 0, total = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_before, &total));
    // a slot's footprint: what its Init took (its presized level buffers included) plus what its first tree
    // allocates (measured once a group grew a slot's first tree; until then assumed as large again)
    const size_t reserve = Reserve(total);
    const size_t tree_bytes = slot_tree_bytes_ > 0 ? slot_tree_bytes_ : slot_init_bytes_;
    // the slots created but not grown yet still have their first tree's allocations to make
    size_t pending = 0;
    for (const bool trained : slot_trained_) {
      pending += trained ? 0 : tree_bytes;
    }
    if (slot_init_bytes_ > 0 && free_before < reserve + pending + slot_init_bytes_ + tree_bytes) {
      memory_bound_ = true;
      break;
    }
    std::unique_ptr<TreeLearner> created(TreeLearner::CreateTreeLearner(config_->tree_learner, config_->device_type,
                                                                        config_, boosting_on_gpu_));
    CUDASingleGPUTreeLearner* slot = dynamic_cast<CUDASingleGPUTreeLearner*>(created.get());
    CHECK(slot != nullptr);
    slot->EnableForestSlot(true, slots_[0]);
    const auto init_start = std::chrono::steady_clock::now();
    slot->Init(train_data_, is_constant_hessian_);
    slot->PresizeLevelBuffers();
    size_t free_after = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_after, &total));
    const size_t init_bytes = free_before > free_after ? free_before - free_after : size_t{0};
    slot_init_bytes_ = std::max(slot_init_bytes_, init_bytes);
    if (FalcataDebug().diag) {
      fprintf(stderr, "[forest-diag] slot %d init %.1f ms, %.0f MiB, %.0f MiB free after\n", num_slots(),
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - init_start).count(),
              init_bytes / 1048576.0, free_after / 1048576.0);
    }
    slots_.push_back(slot);
    slot_trained_.push_back(false);
    owned_.push_back(std::move(created));
  }
  return num_slots();
}

void CUDAForestTrainer::Train(const std::vector<Job>& jobs, std::vector<std::unique_ptr<Tree>>* trees) {
  const int num_jobs = static_cast<int>(jobs.size());
  trees->clear();
  trees->resize(num_jobs);
  if (num_jobs == 0) {
    return;
  }
  // what sequential training gives each tree, in tree order: the booster learner's counters plus the tree's position
  // and its sampler's next per-tree sample (drawn here, on this thread, before any slot trains)
  CUDASingleGPUTreeLearner* const main = slots_[0];
  const CUDASingleGPUTreeLearner::TreeCounters base = main->tree_counters();
  ColSampler* const sampler = main->mutable_col_sampler();
  std::vector<bool> slot_taken(slots_.size(), false);
  job_counters_.assign(num_jobs, base);
  job_samplers_.clear();
  job_samplers_.resize(num_jobs);
  for (int k = 0; k < num_jobs; ++k) {
    const Job& job = jobs[k];
    CHECK(job.slot >= 0 && job.slot < num_slots());
    CHECK(!slot_taken[job.slot]);
    slot_taken[job.slot] = true;
    CUDASingleGPUTreeLearner* const slot = slots_[job.slot];
    CUDASingleGPUTreeLearner::TreeCounters counters;
    counters.bynode = base.bynode + static_cast<uint64_t>(k);
    counters.discretizer = base.discretizer + k;
    job_counters_[k] = counters;
    slot->SetTreeCounters(counters);
    if (sampler->need_reset_bytree()) {
      job_samplers_[k].reset(new ColSampler(*sampler));
      sampler->ResetByTree();
      slot->SetNextTreeSample(sampler->used_feature_indices());
    }
  }
  // device memory the slots' first trees allocate (EnsureSlots' footprint estimate)
  int first_trees = 0;
  for (const Job& job : jobs) {
    first_trees += (job.slot > 0 && !slot_trained_[job.slot]) ? 1 : 0;
  }
  size_t free_before = 0, total = 0;
  if (first_trees > 0) {
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_before, &total));
  }
  std::vector<std::exception_ptr> errors(num_jobs);
  // test hook: FALCATA_TEST_FOREST_OOM_GROUP=n makes the n-th group (0-based, of a group with more than one slot) of
  // the process fail in its last slot as a device-memory failure would
  static std::atomic<int> group_counter{0};
  static const int inject_group = [] {
    const char* v = std::getenv("FALCATA_TEST_FOREST_OOM_GROUP");
    return v != nullptr ? std::atoi(v) : -1;
  }();
  const bool inject_oom = inject_group >= 0 && num_slots() > 1 && group_counter.fetch_add(1) == inject_group;
  // the log level and callback are per thread: the slot threads log as this one does (verbosity, the Python or R
  // logger), not at a new thread's default Info level to stdout
  const LogLevel log_level = Log::CurrentLevel();
  const Log::Callback log_callback = Log::CurrentCallBack();
  const auto grow = [&](const int k) {
    Log::ResetLogLevel(log_level);
    Log::ResetCallBack(log_callback);
    try {
      const Job& job = jobs[k];
      CUDASingleGPUTreeLearner* const slot = slots_[job.slot];
      SetCUDADevice(slot->gpu_device_id(), __FILE__, __LINE__);
      if (job.bag != nullptr) {
        slot->SetBaggingData(nullptr, job.bag, job.bag_cnt);
      }
      if (inject_oom && job.slot == num_slots() - 1) {
        // test hook (FALCATA_TEST_FOREST_OOM_GROUP): the device-memory failure of a group, as AllocateCUDAMemory
        // reports it
        throw std::runtime_error("[CUDA] out of memory allocating 0.00 GiB (injected for a test)");
      }
      (*trees)[k].reset(slot->Train(job.gradients, job.hessians, false));
    } catch (...) {
      errors[k] = std::current_exception();
    }
  };
  last_solo_ = num_jobs == 1 && FalcataPlan::Get().forest_single_job_device;
  if (num_jobs == 1) {
    // cuda_plan key forest_single_job_device: no other thread runs while the group's only tree trains (on this
    // thread), so it trains as the sequential learner does (graph capture, the device level flows)
    CUDASingleGPUTreeLearner* const solo = slots_[jobs[0].slot];
    solo->SetForestSoloTree(last_solo_);
    grow(0);
    solo->SetForestSoloTree(false);
  } else {
    std::vector<std::thread> threads;
    threads.reserve(num_jobs - 1);
    for (int k = 1; k < num_jobs; ++k) {
      threads.emplace_back(grow, k);
    }
    grow(0);
    for (std::thread& thread : threads) {
      thread.join();
    }
  }
  // the booster learner continues the count where the group ended (a later sequential tree gets what it would)
  CUDASingleGPUTreeLearner::TreeCounters next;
  next.bynode = base.bynode + static_cast<uint64_t>(num_jobs);
  next.discretizer = base.discretizer + num_jobs;
  main->SetTreeCounters(next);
  SetCUDADevice(main->gpu_device_id(), __FILE__, __LINE__);
  last_first_trees_ = first_trees;
  {
    // the device must keep the reserve after a group: below it, the next group shrinks (RF::GrowForestGroup)
    size_t free_now = 0, total_now = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_now, &total_now));
    memory_low_ = free_now < Reserve(total_now);
  }
  if (first_trees > 0) {
    size_t free_after = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_after, &total));
    // the booster learner's own first tree may be among them: an overestimate, on the safe side
    const size_t per_tree = free_before > free_after ? (free_before - free_after) / first_trees : size_t{0};
    slot_tree_bytes_ = std::max(slot_tree_bytes_, per_tree);
    for (const Job& job : jobs) {
      slot_trained_[job.slot] = true;
    }
  }
  for (const std::exception_ptr& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

void CUDAForestTrainer::DropSlots(const int keep) {
  SetCUDADevice(slots_[0]->gpu_device_id(), __FILE__, __LINE__);
  const int target = std::max(1, keep);
  while (num_slots() > target) {
    owned_.pop_back();  // the learner of the last slot, with its device memory
    slots_.pop_back();
    slot_trained_.pop_back();
  }
  memory_bound_ = true;
  memory_low_ = false;
  if (FalcataDebug().diag) {
    size_t free_bytes = 0, total = 0;
    cudaMemGetInfo(&free_bytes, &total);
    fprintf(stderr, "[forest-diag] slots dropped to %d, %.0f MiB free\n", num_slots(), free_bytes / 1048576.0);
  }
}

void CUDAForestTrainer::RewindTo(const int job_index) {
  CHECK(job_index >= 0 && job_index < static_cast<int>(job_counters_.size()));
  CUDASingleGPUTreeLearner* const main = slots_[0];
  main->SetTreeCounters(job_counters_[job_index]);
  if (job_samplers_[job_index] != nullptr) {
    *main->mutable_col_sampler() = *job_samplers_[job_index];
  }
  job_counters_.clear();
  job_samplers_.clear();
}

void CUDAForestTrainer::ResetConfig(const Config* config) {
  config_ = config;
  for (int slot = 1; slot < num_slots(); ++slot) {
    slots_[slot]->ResetConfig(config);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
