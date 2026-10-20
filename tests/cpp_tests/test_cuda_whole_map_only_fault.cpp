// Copyright (c) 2026 Falcata contributors. MIT license; see LICENSE.
/* UNAPPLIED/UNBUILT/UNRUN diagnostic integration projection. MIT license; see LICENSE.
 * Requires FALCATA_WHOLE_MAP_ONLY_FAULT_TEST in BOTH library C++/CUDA TUs and this standalone executable.
 * No forest hook, FalcataDebug growth hook, normal-build feature, per-level readback or performance claim.
 */
#include <Falcata/c_api.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/treelearner/cuda/cuda_single_gpu_tree_learner.hpp"
#include "../../src/treelearner/cuda/cuda_device_levels.hpp"
#include "whole_map_only_fault_test.hpp"

namespace Falcata {
namespace {
void Need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void Check(cudaError_t code) { Need(code == cudaSuccess, cudaGetErrorString(code)); }
template <typename T> void Read(const T* device, std::vector<T>* host) {
  Check(cudaMemcpy(host->data(), device, host->size() * sizeof(T), cudaMemcpyDeviceToHost));
}
struct InjectedFault { WholeMapOnlyFaultPoint point; };
struct State {
  bool progress, pending, role_undo, final_map, early_map, early_pending, count_ready, complete, prefix, stale, scratch_dirty;
  int ready_leaves, dirty, iter, graph_hits, graph_misses;
  std::array<int, 5> owned;
  const data_size_t* main;
  const data_size_t* out;
  const CUDAHybridApplyDescriptor* desc;
  const uint16_t* bits;
};
struct Context {
  CUDASingleGPUTreeLearner* target = nullptr;
  bool defer = false, armed = false, retry = false, injected = false;
  WholeMapOnlyFaultPoint wanted = WholeMapOnlyFaultPoint::BeforePublish;
  std::exception_ptr prebuilt_fault;
  int hits = 0, before = 0, after = 0, recovery_before = 0, recovery_after = 0, cleared = 0, root_checks = 0, flag_checks = 0;
  State failed{}, repaired{};
  std::vector<data_size_t> rows, counts, starts, saved_rows;
  std::vector<int> map;
  const std::vector<data_size_t>* eager_windows = nullptr;
};
thread_local Context* context = nullptr;
}  // namespace

// Friend only in the explicit diagnostic build; no private-access macro and no class layout/member addition.
struct WholeMapOnlyFaultPeer {
  static State Observe(CUDASingleGPUTreeLearner* l) {
    auto* p = l->cuda_data_partition_.get();
    auto* h = l->cuda_histogram_constructor_.get();
    return {l->tree_in_progress_, p->level_partition_pending_, p->pending_device_level_role_undo_, l->hybrid_map_final_written_,
      l->hybrid_map_early_written_, l->hybrid_map_early_pending_, l->prefetched_tree_counts_,
      l->device_tree_completed_, l->prefix_completes_tree_, l->whole_host_leaf_cache_stale_,
      h->count_scratch_dirty_, l->prefetched_tree_num_leaves_, h->num_dirty_leaves_,
      l->cuda_gradient_discretizer_->iteration(), l->device_level_graph_hits_, l->device_level_graph_misses_,
      {p->pending_num_splits_, p->pending_num_gaps_, p->pending_max_gap_blocks_,
       p->pending_total_flat_blocks_, p->pending_gap_block_dim_},
      p->cuda_data_indices_.RawDataReadOnly(), p->cuda_out_data_indices_in_leaf_.RawDataReadOnly(),
      p->cuda_apply_descs_.RawDataReadOnly(), p->cuda_block_to_left_offset_.RawDataReadOnly()};
  }
  static void CaptureWindows(CUDASingleGPUTreeLearner* l, Context* c) {
    auto* p = l->cuda_data_partition_.get();
    // Observation-only barriers in the DIAGNOSTIC test, never a normal-tree fence change.
    Check(cudaStreamSynchronize(p->apply_stream()));
    Read(p->cuda_data_indices_.RawDataReadOnly(), &c->rows);
    Read(p->cuda_leaf_num_data_.RawDataReadOnly(), &c->counts);
    Read(p->cuda_leaf_data_start_.RawDataReadOnly(), &c->starts);
    Read(p->cuda_data_index_to_leaf_index_.RawDataReadOnly(), &c->map);
  }
  static void Checkpoint(CUDASingleGPUTreeLearner* l, WholeMapOnlyFaultPoint point) {
    auto* c = context;
    if (c == nullptr || c->target != l) return;
    const State s = Observe(l);
    Need(s.progress, "checkpoint must be inside the actual Train/recovery lifetime");
    Need(!l->forest_slot_ && l->config_->boosting == "gbdt" && l->config_->objective == "regression",
         "not a canonical GBDT learner");
    if (point == WholeMapOnlyFaultPoint::BeforePublish || point == WholeMapOnlyFaultPoint::AfterPublish) {
      const int* j = l->pinned_device_level_journal_;
      Need(l->host_device_level_config_.whole_tree == 1 && j != nullptr && j[2] == 0 && j[0] == 3 && j[1] == 5,
           "fault must reach actual whole static-final six-leaf completion, never fallback");
      // The selected/gap descriptors are real captured-kernel output. A different learner may have resolved
      // another gap width between trees; stale graph replay must never disagree with THIS host receipt unit.
      std::array<CUDAHybridApplyDescriptor, 4> descriptors{};
      Check(cudaMemcpy(descriptors.data(), l->cuda_data_partition_->cuda_apply_descs_.RawDataReadOnly(),
        descriptors.size() * sizeof(CUDAHybridApplyDescriptor), cudaMemcpyDeviceToHost));
      const int unit = l->host_device_level_config_.gap_block_dim;
      for (int gap = 2; gap < 4; ++gap) Need(descriptors[gap].num_data_in_leaf > 0 &&
        descriptors[gap].num_blocks == (descriptors[gap].num_data_in_leaf + unit - 1) / unit,
        "captured descriptor width disagrees with the current owned receipt unit");
      if (point == WholeMapOnlyFaultPoint::BeforePublish) {
        ++c->before;
        Need(!s.pending && !s.role_undo && !s.final_map && !s.early_map && !s.early_pending && !s.complete && !s.prefix &&
          !s.count_ready && s.ready_leaves == 0, "pre-publication fault exposed a partial successful receipt");
      } else {
        ++c->after;
        Need(s.pending == c->defer && s.role_undo == c->defer && s.final_map && s.early_map && !s.early_pending && s.complete && s.prefix &&
          s.count_ready && s.ready_leaves == 6 && s.stale, "post-publication receipt/flags are incomplete");
        if (c->defer) Need(s.owned[0] == 2 && s.owned[1] == 2 && s.owned[2] > 0 && s.owned[3] >= 2 &&
          (s.owned[4] == 256 || s.owned[4] == 1024), "pending receipt does not own exact counts and width");
      }
      if (c->armed && point == c->wanted) {
        c->failed = s;
        c->injected = true;
        ++c->hits;
        c->armed = false;  // disarm BEFORE unwinding; destructor/retry must never inject twice
        std::rethrow_exception(c->prebuilt_fault);  // payload allocated BEFORE arming, not a runtime_error string
      }
    } else if (c->retry && point == WholeMapOnlyFaultPoint::BeforeRecoveryRepair) {
      ++c->recovery_before;
      Need(s.pending == c->failed.pending && s.role_undo == c->failed.role_undo && s.main == c->failed.main && s.out == c->failed.out,
           "recovery lost the receipt/parent roles before repair");
      Need(s.desc == c->failed.desc && s.bits == c->failed.bits, "required scratch changed before recovery");
      if (c->failed.pending) Need(s.owned == c->failed.owned, "owned receipt changed during unwinding");
    } else if (c->retry && point == WholeMapOnlyFaultPoint::AfterRecoveryRepair) {
      ++c->recovery_after;
      Need(!s.pending && !s.role_undo, "published pending assignment/role survived AbortedTreeReset repair");
      Need(s.main == (c->failed.pending ? c->failed.out : c->failed.main), "repair did not swap exactly once");
      Need(s.out == (c->failed.pending ? c->failed.main : c->failed.out), "repair changed the other role twice");
      c->repaired = s;
      if (c->wanted == WholeMapOnlyFaultPoint::AfterPublish) {
        CaptureWindows(l, c);
        if (c->eager_windows != nullptr) Need(c->rows == *c->eager_windows,
          "lazy recovery did not reproduce EVERY eager child/terminal window row in order");
      }
    } else if (c->retry && point == WholeMapOnlyFaultPoint::RecoveryCleared) {
      ++c->cleared;
      Need(!s.pending && !s.role_undo && !s.final_map && !s.early_map && !s.early_pending && !s.count_ready &&
        s.ready_leaves == 0 && s.dirty == -1 && s.scratch_dirty,
        "recovery left stale map/readback/dirty-hist/count-scratch state");
    } else if (c->retry && point == WholeMapOnlyFaultPoint::BeforeRootReuse) {
      Need(c->recovery_before == 1 && c->recovery_after == 1 && c->cleared == 1,
           "root/scratch reuse preceded real same-learner recovery");
      Need(!s.pending && !s.role_undo && !s.final_map && !s.early_map && !s.early_pending && !s.count_ready && s.ready_leaves == 0 &&
        s.dirty == -1 && s.scratch_dirty && s.main == c->repaired.main, "root entry inherited stale readiness");
    } else if (c->retry && point == WholeMapOnlyFaultPoint::AfterTrainFlagsReset) {
      ++c->flag_checks;
      Need(!s.pending && !s.role_undo && !s.final_map && !s.early_map && !s.early_pending && !s.count_ready &&
        s.ready_leaves == 0 && !s.complete && !s.prefix && !s.stale,
        "current-tree reset did not invalidate every stale completion/cache/readback receipt before growth");
    } else if (c->retry && point == WholeMapOnlyFaultPoint::AfterRootInit) {
      ++c->root_checks;
      CaptureWindows(l, c);
      Need(!s.pending && !s.role_undo && c->counts[0] == static_cast<data_size_t>(c->rows.size()) && c->starts[0] == 0,
           "retry root count/start not rebuilt");
      for (size_t i = 0; i < c->rows.size(); ++i) Need(c->rows[i] == static_cast<data_size_t>(i),
        "root identity did not overwrite every aborted/current main row");
      for (size_t i = 1; i < c->counts.size(); ++i) Need(c->counts[i] == 0 && c->starts[i] == 0,
        "old child count/start leaked into the retry root");
    }
  }
  static void PoisonDisposableOutput(CUDASingleGPUTreeLearner* l, bool pending) {
    auto* p = l->cuda_data_partition_.get();
    Check(cudaStreamSynchronize(p->apply_stream()));
    // Published case: ONLY the unwritten repair destination, NEVER saved parent/descriptor/bit/prefix input.
    // Unpublished case: next root deliberately replaces current main; force all-row identity reconstruction.
    auto* dst = pending ? p->cuda_out_data_indices_in_leaf_.RawData() : p->cuda_data_indices_.RawData();
    Check(cudaMemset(dst, 0xa5, static_cast<size_t>(p->num_data_) * sizeof(data_size_t)));
  }
};
}  // namespace Falcata

namespace {
using Falcata::Dataset;
using Falcata::WholeMapOnlyFaultPoint;
using Falcata::Config;
using Falcata::CUDASingleGPUTreeLearner;
using Falcata::CUDAVector;
using Falcata::score_t;
using Falcata::Context;
using Falcata::InjectedFault;
using Falcata::context;
using Falcata::SetWholeMapOnlyFaultCallback;
using Falcata::WholeMapOnlyFaultPeer;
using Falcata::Need;
using Falcata::Check;
using Falcata::Read;
using Falcata::data_size_t;
using Falcata::CUDADataPartition;
using Falcata::Tree;
struct Outcome { std::vector<std::string> models; std::vector<double> scores; std::vector<data_size_t> failed_windows; };
Outcome Run(const Dataset* data, const std::vector<float>& x, bool defer, bool inject,
            WholeMapOnlyFaultPoint fault, bool stump_retry, const std::string& graph,
            const std::vector<data_size_t>* eager_windows) {
  Config config;
  config.Set({{"device_type", "cuda"}, {"objective", "regression"}, {"boosting", "gbdt"}, {"metric", "None"},
    {"quant_mode", "fixedpoint"}, {"quant_bins", "64"}, {"seed", "42"}, {"num_threads", "32"},
    {"num_leaves", "16"}, {"max_depth", "3"}, {"min_data_in_leaf", "2"}, {"min_gain_to_split", "100"},
    {"feature_fraction", "1"}, {"quant_train_renew_leaf", "false"}, {"boost_from_average", "false"},
    {"feature_pre_filter", "false"}, {"verbosity", "-1"},
    {"cuda_plan", "auto,device_levels:off,device_tree:on,quant_ef:off,device_level_graph:" + graph +
      ",leaf_map_small_blocks:off,final_map_only:" + (defer ? "on" : "off")}});
  CUDASingleGPUTreeLearner learner(&config, true);
  learner.Init(data, true);
  CUDAVector<score_t> g, h;
  CUDAVector<double> scores;
  std::vector<score_t> warm(data->num_data()), mixed(warm.size()), constant(warm.size(), -1.0f);
  for (size_t row = 0; row < warm.size(); ++row) {
    warm[row] = -(1 + 7 * x[3 * row]);
    mixed[row] = -(1 + 7 * x[3 * row] + 3 * x[3 * row + 1] + 2 * x[3 * row] * x[3 * row + 2]);
  }
  h.InitFromHostVector(std::vector<score_t>(warm.size(), 1.0f));
  scores.InitFromHostVector(std::vector<double>(warm.size(), 0.0));
  Context c;
  c.target = &learner; c.defer = defer; c.wanted = fault; c.eager_windows = eager_windows;
  c.rows.resize(warm.size()); c.counts.resize(16); c.starts.resize(16); c.map.resize(warm.size());
  c.prebuilt_fault = std::make_exception_ptr(InjectedFault{fault});
  context = &c;
  SetWholeMapOnlyFaultCallback(&WholeMapOnlyFaultPeer::Checkpoint);
  struct ResetCallback { ~ResetCallback() { SetWholeMapOnlyFaultCallback(nullptr); context = nullptr; } } reset;
  Outcome result;
  std::vector<double> expected(warm.size(), 0.0);
  auto accept = [&](std::unique_ptr<Tree> tree, int leaves) {
    Need(WholeMapOnlyFaultPeer::Observe(&learner).complete,
         "just-returned Train did not complete the whole controller before score publication");
    Need(tree->num_leaves() == leaves, "fixture lost actual warm/final/stump tree shape");
    tree->Shrinkage(0.001);
    learner.AddPredictionToScore(tree.get(), scores.RawData());
    Check(cudaDeviceSynchronize());
    std::vector<double> live(warm.size()); Read(scores.RawDataReadOnly(), &live);
    for (size_t row = 0; row < warm.size(); ++row) {
      const double features[] = {x[3 * row], x[3 * row + 1], x[3 * row + 2]};
      expected[row] += tree->Predict(features);
      Need(std::isfinite(live[row]) && std::abs(live[row] - expected[row]) <= 1e-12, "live/CPU score mismatch");
    }
    result.models.push_back(tree->ToString()); result.scores = live;
  };
  g.InitFromHostVector(warm);
  accept(std::unique_ptr<Tree>(learner.Train(g.RawData(), h.RawData(), true)), 2);
  const int warm_iter = WholeMapOnlyFaultPeer::Observe(&learner).iter;
  const auto warm_score = result.scores;
  Need(WholeMapOnlyFaultPeer::Observe(&learner).dirty < 4, "warm fixture must leave fewer dirty slots than failed tree");
  g.InitFromHostVector(mixed);
  if (inject) {
    c.armed = true;
    std::unique_ptr<Tree> failed;
    try {
      failed.reset(learner.Train(g.RawData(), h.RawData(), false));
      Need(false, "fault callback did not throw from actual Train");
    } catch (const InjectedFault& tag) {
      Need(tag.point == fault && c.injected && c.hits == 1 && !c.armed && failed == nullptr,
           "wrong/missing fault or a returned partial model");
    }
    Need(c.failed.progress && c.failed.iter == warm_iter + 1, "failed attempt did not advance the real quantizer counter once");
    std::vector<double> unchanged(warm.size()); Read(scores.RawDataReadOnly(), &unchanged);
    Need(unchanged == warm_score, "failed Train changed score without a returned model");
    if (fault == WholeMapOnlyFaultPoint::AfterPublish && !defer) {
      WholeMapOnlyFaultPeer::CaptureWindows(&learner, &c); result.failed_windows = c.rows;
    }
    // The diagnostic observer never corrupts still-owned INPUT scratch to manufacture a failure.
    WholeMapOnlyFaultPeer::PoisonDisposableOutput(&learner, c.failed.pending);
    c.retry = true;
  }
  // REAL other-object plan resolution, no ResetConfig on our learner. For injected post-publication cases,
  // its pending gaps still own1024-thread units while the global plan now says256. New captures/replays must
  // use the new current cfg unit too. The independent clean arm makes the same sidecar transition.
  {
    Config other_config = config;
    const auto width_pos = other_config.cuda_plan.find("leaf_map_small_blocks:off");
    Need(width_pos != std::string::npos, "creation plan did not bind gap width explicitly");
    other_config.cuda_plan.replace(width_pos, std::string("leaf_map_small_blocks:off").size(), "leaf_map_small_blocks:on");
    CUDASingleGPUTreeLearner other(&other_config, true);
    other.Init(data, true);
  }
  Need(CUDADataPartition::DeviceLevelGapBlockDim() == 256, "other learner failed to switch the global gap unit");
  g.InitFromHostVector(stump_retry ? constant : mixed);
  accept(std::unique_ptr<Tree>(learner.Train(g.RawData(), h.RawData(), false)), stump_retry ? 1 : 6);
  if (inject) {
    Need(c.recovery_before == 1 && c.recovery_after == 1 && c.cleared == 1 && c.root_checks == 1 && c.flag_checks == 1,
         "same-learner next-Train recovery callbacks missing/duplicated");
    Need(WholeMapOnlyFaultPeer::Observe(&learner).iter == warm_iter + 2, "retry counter not advanced exactly once");
  }
  c.retry = false;
  g.InitFromHostVector(mixed);
  for (int repeat = 0; repeat < 3; ++repeat) {
    accept(std::unique_ptr<Tree>(learner.Train(g.RawData(), h.RawData(), false)), 6);
  }
  const auto final = WholeMapOnlyFaultPeer::Observe(&learner);
  if (graph == "on") Need(final.graph_hits > 0 && final.graph_misses > 0, "graph path must really miss AND reuse");
  else Need(final.graph_hits == 0 && final.graph_misses == 0, "plain mode unexpectedly used capture");
  Need(!WholeMapOnlyFaultPeer::Observe(&learner).progress, "successful Train did not clear in-progress state");
  Need(c.before > 0 && c.after > 0, "actual positive publication checkpoints were never observed");
  std::fprintf(stderr, "[whole-map-fault] defer=%d inject=%d hits=%d recovery=%d/%d cleared=%d root=%d\n",
    defer, inject, c.hits, c.recovery_before, c.recovery_after, c.cleared, c.root_checks);
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  DatasetHandle handle = nullptr;
  try {
    Need(std::getenv("GPUQ_JOB_ID") != nullptr, "actual CUDA execution requires a future authorized gpuq job");
    Need(argc == 4, "usage: diagnostic-probe before|after positive|stump graph:on|off");
    const auto fault = std::string(argv[1]) == "before" ? WholeMapOnlyFaultPoint::BeforePublish : WholeMapOnlyFaultPoint::AfterPublish;
    Need(std::string(argv[1]) == "before" || std::string(argv[1]) == "after", "unknown phase");
    const bool stump = std::string(argv[2]) == "stump";
    Need(stump || std::string(argv[2]) == "positive", "unknown retry");
    const std::string graph = argv[3]; Need(graph == "on" || graph == "off", "unknown graph mode");
    constexpr int rows = 32771;
    std::vector<float> x(3 * rows), y(rows);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < 3; ++col) x[3 * row + col] = (row >> col) & 1;
      y[row] = 1 + 7 * x[3 * row] + 3 * x[3 * row + 1] + 2 * x[3 * row] * x[3 * row + 2];
    }
    Need(FLC_DatasetCreateFromMat(x.data(), C_API_DTYPE_FLOAT32, rows, 3, 1,
      "device_type=cuda max_bin=7 min_data_in_leaf=2 feature_pre_filter=false num_threads=32", nullptr, &handle) == 0,
      "create dataset failed");
    Need(FLC_DatasetSetField(handle, "label", y.data(), rows, C_API_DTYPE_FLOAT32) == 0, "set labels failed");
    const auto* data = reinterpret_cast<const Dataset*>(handle);
    const auto eager = Run(data, x, false, true, fault, stump, graph, nullptr);
    const auto deferred = Run(data, x, true, true, fault, stump, graph,
      fault == WholeMapOnlyFaultPoint::AfterPublish ? &eager.failed_windows : nullptr);
    const auto clean = Run(data, x, false, false, fault, stump, graph, nullptr);
    Need(eager.models == deferred.models && eager.models == clean.models && eager.scores == deferred.scores &&
      eager.scores == clean.scores, "exact retry models/live scores differ from independent clean eager control");
    FLC_DatasetFree(handle);
    std::puts("PASS: actual guarded phases, same-learner repair/reset/retry, clean full-model/live/CPU oracle");
    return 0;
  } catch (const std::exception& error) {
    if (handle != nullptr) FLC_DatasetFree(handle);
    std::fprintf(stderr, "%s\n", error.what()); return 1;
  } catch (...) {
    if (handle != nullptr) FLC_DatasetFree(handle);
    std::fprintf(stderr, "unexpected fault tag outside intended Train catch\n"); return 1;
  }
}
