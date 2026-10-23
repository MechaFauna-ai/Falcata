/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cuda_runtime.h>
#include <Falcata/cuda/cuda_utils.hu>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <Falcata/cuda/cuda_tree.hpp>

#include "../../src/treelearner/cuda/cuda_data_partition.hpp"
#include "../../src/treelearner/cuda/cuda_device_levels.hpp"
#include "../../src/treelearner/cuda/cuda_leaf_splits.hpp"

namespace {
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t size) {
    EXPECT_EQ(cudaMalloc(&data_, size * sizeof(T)), cudaSuccess);
  }
  ~DeviceBuffer() { EXPECT_EQ(cudaFree(data_), cudaSuccess); }
  T* data() { return data_; }
 private:
  T* data_ = nullptr;
};

// Exercise the library's actual selector, descriptors and journal. The oracle is the host's stable_sort contract,
// not another CUDA implementation. The final pairs launch deliberately has no child-search descriptor storage.
// selective: a selective tree's level (no whole tree, dynamic R1, no tree record, the split-info journal, the level
// from the state): it applies every valid leaf in ascending order while leaves + valid <= budget, else stops at its
// first binding level (journal[3]) without applying it
void CheckSelection(const std::vector<double>& gains, const std::vector<bool>& valid, const int budget,
                    const bool final_depth = true, const bool selective = false) {
  using Falcata::kDeviceLevelJournalRecord;
  using Falcata::kDeviceTreeSharedSplits;
  using Falcata::CUDADeviceLevelConfig;
  using Falcata::CUDADeviceLevelState;
  using Falcata::CUDAHybridApplyDescriptor;
  using Falcata::CUDAHybridGraphFeatureMeta;
  using Falcata::CUDAHybridGraphFeatureSource;
  using Falcata::CUDALeafSplitsStruct;
  using Falcata::CUDASplitInfo;
  using Falcata::CUDATreeBatchSplit;
  using Falcata::data_size_t;
  using Falcata::kDeviceLevelJournalHeader;
  using Falcata::LaunchDeviceLevelPairs;
  using Falcata::LaunchDeviceLevelSelect;
  const int leaves = static_cast<int>(gains.size());
  ASSERT_EQ(valid.size(), gains.size());
  int level = 0;
  while ((1 << level) < leaves) ++level;
  const int cursor = leaves - 1;
  DeviceBuffer<CUDASplitInfo> best(leaves);
  DeviceBuffer<data_size_t> counts(leaves), starts(leaves);
  DeviceBuffer<CUDADeviceLevelState> device_state(1);
  DeviceBuffer<CUDATreeBatchSplit> batch(leaves);
  DeviceBuffer<CUDAHybridApplyDescriptor> descs(2 * leaves);
  DeviceBuffer<CUDALeafSplitsStruct> slots(2 * leaves);
  DeviceBuffer<CUDAHybridGraphFeatureMeta> device_meta(1);
  DeviceBuffer<CUDAHybridGraphFeatureSource> device_source(1);
  DeviceBuffer<double> thresholds(1);
  DeviceBuffer<int> device_journal(kDeviceLevelJournalHeader + 2 * kDeviceLevelJournalRecord * leaves);
  // a level wider than the shared select: the select's arrays in global scratch (DeviceLevelBitonicSortGlobal)
  const bool global_select = leaves > kDeviceTreeSharedSplits;
  DeviceBuffer<int> select_leaves(global_select ? leaves : 1);
  DeviceBuffer<double> select_gains(global_select ? leaves : 1);
  DeviceBuffer<unsigned int> select_bits(global_select ? 2 * leaves / 32 : 1);
  DeviceBuffer<int2> select_gaps(global_select ? 2 * leaves : 1);
  DeviceBuffer<CUDASplitInfo> split_info_journal(2 * leaves);
  std::vector<CUDASplitInfo> splits(leaves);
  std::vector<data_size_t> host_counts(leaves), host_starts(leaves);
  std::vector<int> expected;
  int next_start = 0;
  for (int leaf = 0; leaf < leaves; ++leaf) {
    auto& s = splits[leaf];
    s.is_valid = valid[leaf];
    s.leaf_index = leaf;
    s.gain = gains[leaf];
    s.inner_feature_index = 0;
    s.threshold = 0;
    s.default_left = false;
    s.left_sum_gradients = s.right_sum_gradients = 0.0;
    s.left_sum_hessians = s.right_sum_hessians = 1.0;
    s.left_sum_of_gradients_hessians = s.right_sum_of_gradients_hessians = 1;
    s.left_count = s.right_count = 1;
    s.left_gain = s.right_gain = 0.0;
    s.left_value = s.right_value = 0.0;
    host_counts[leaf] = 2 + leaf % 3;
    host_starts[leaf] = next_start;
    next_start += host_counts[leaf];
    if (valid[leaf]) expected.push_back(leaf);
  }
  const bool binding = static_cast<int>(expected.size()) > budget - leaves;
  const bool rejected = binding && !final_depth && !selective;
  if (binding && selective) {
    expected.clear();  // the first binding level: the host's selective loop takes it
  } else if (binding) {
    std::stable_sort(expected.begin(), expected.end(), [&](int a, int b) { return gains[a] > gains[b]; });
    expected.resize(rejected ? 0 : static_cast<size_t>(budget - leaves));
  }
  ASSERT_EQ(cudaMemcpy(best.data(), splits.data(), leaves * sizeof(CUDASplitInfo), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(counts.data(), host_counts.data(), leaves * sizeof(data_size_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(starts.data(), host_starts.data(), leaves * sizeof(data_size_t), cudaMemcpyHostToDevice), cudaSuccess);
  DeviceBuffer<int> block_start(leaves + 1);  // the block map (the reset sets this pointer in production)
  CUDADeviceLevelState state{};
  state.construct_block_start = block_start.data();
  state.num_leaves = leaves;
  state.split_info_base = cursor;
  state.levels_applied = level;
  ASSERT_EQ(cudaMemcpy(device_state.data(), &state, sizeof(state), cudaMemcpyHostToDevice), cudaSuccess);
  CUDAHybridGraphFeatureMeta meta{};
  meta.real_feature_index = 3;
  meta.max_bin = 1;
  meta.most_freq_bin = 1;
  CUDAHybridGraphFeatureSource source{};
  source.bit_type = 8;
  const double threshold = 0.5;
  ASSERT_EQ(cudaMemcpy(device_meta.data(), &meta, sizeof(meta), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_source.data(), &source, sizeof(source), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(thresholds.data(), &threshold, sizeof(threshold), cudaMemcpyHostToDevice), cudaSuccess);
  std::vector<int> journal(kDeviceLevelJournalHeader + 2 * kDeviceLevelJournalRecord * leaves, 0);
  journal[0] = level;
  journal[1] = cursor;
  ASSERT_EQ(cudaMemcpy(device_journal.data(), journal.data(), journal.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
  CUDADeviceLevelConfig cfg{};
  cfg.state = device_state.data();
  cfg.leaf_best_split_info = best.data();
  cfg.leaf_num_data = counts.data();
  cfg.leaf_data_start = starts.data();
  cfg.tree_batch_splits = batch.data();
  cfg.apply_descs = descs.data();
  cfg.pair_slots = slots.data();
  cfg.feature_meta = device_meta.data();
  cfg.feature_source = device_source.data();
  cfg.real_thresholds = thresholds.data();
  cfg.journal = device_journal.data();
  cfg.construct_block_start = block_start.data();
  cfg.max_depth = level + (final_depth ? 1 : 2);
  cfg.num_leaves_budget = budget;
  cfg.whole_tree = selective ? 0 : 1;
  cfg.r1_binding = selective ? 1 : 0;
  cfg.skip_tree_record = selective ? 1 : 0;
  cfg.split_info_journal = selective ? split_info_journal.data() : nullptr;
  cfg.gap_block_dim = 1024;
  if (global_select) {
    cfg.select_leaves = select_leaves.data();
    cfg.select_gains = select_gains.data();
    cfg.select_split_bits = select_bits.data();
    cfg.select_gaps = select_gaps.data();
    cfg.select_capacity = leaves;
  }
  if (selective) {
    // the learner's launch: the level's most splits (R1: at most budget / 2) and leaves pick the select's capacity
    // (register-batched up to 1024 leaves, shared up to 4096, global beyond); the level from the state
    LaunchDeviceLevelSelect(cfg, -1, std::max(1, std::min(leaves, budget / 2)), nullptr, leaves);
  } else {
    LaunchDeviceLevelSelect(cfg, level, leaves, nullptr);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&state, device_state.data(), sizeof(state), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(journal.data(), device_journal.data(), journal.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(journal[2], rejected ? 1 : 0);
  ASSERT_EQ(journal[3], binding && selective ? 1 : 0);
  ASSERT_EQ(state.live_splits, static_cast<int>(expected.size()));
  if (expected.empty()) {
    EXPECT_EQ(state.done, 1);
    EXPECT_EQ(state.num_leaves, leaves);
    EXPECT_EQ(state.total_flat_blocks, 0);
    EXPECT_EQ(state.num_gaps, 0);
    EXPECT_EQ(journal[0], level);
    EXPECT_EQ(journal[1], cursor);
    return;
  }
  ASSERT_EQ(state.done, 0);
  EXPECT_EQ(state.num_leaves, leaves);
  EXPECT_EQ(state.split_info_base, cursor);
  EXPECT_EQ(journal[0], level + 1);
  EXPECT_EQ(journal[1], cursor + static_cast<int>(expected.size()));

  std::vector<CUDATreeBatchSplit> actual_batch(expected.size());
  std::vector<CUDAHybridApplyDescriptor> actual_descs(leaves);
  ASSERT_EQ(cudaMemcpy(actual_batch.data(), batch.data(), actual_batch.size() * sizeof(CUDATreeBatchSplit), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual_descs.data(), descs.data(), actual_descs.size() * sizeof(CUDAHybridApplyDescriptor), cudaMemcpyDeviceToHost), cudaSuccess);
  std::vector<bool> selected(leaves, false);
  std::vector<CUDASplitInfo> journaled(selective ? expected.size() : 0);
  if (selective) {
    ASSERT_EQ(cudaMemcpy(journaled.data(), split_info_journal.data() + cursor, journaled.size() * sizeof(CUDASplitInfo),
                         cudaMemcpyDeviceToHost), cudaSuccess);
  }
  for (size_t k = 0; k < expected.size(); ++k) {
    SCOPED_TRACE(::testing::Message() << "k=" << k);
    const int leaf = expected[k];
    selected[leaf] = true;
    if (selective) {
      // no tree record; the split's best-split record journaled at its split index
      EXPECT_EQ(journaled[k].leaf_index, leaf);
      EXPECT_EQ(journaled[k].gain, gains[leaf]);
    } else {
      EXPECT_EQ(actual_batch[k].leaf_index, leaf);
      EXPECT_EQ(actual_batch[k].num_leaves_at_split, leaves + static_cast<int>(k));
      EXPECT_EQ(actual_batch[k].split_info, best.data() + leaf);
    }
    EXPECT_EQ(actual_descs[k].left_leaf_index, leaf);
    EXPECT_EQ(actual_descs[k].right_leaf_index, leaves + static_cast<int>(k));
    EXPECT_EQ(actual_descs[k].leaf_data_start, host_starts[leaf]);
    const int* record = journal.data() + kDeviceLevelJournalHeader + kDeviceLevelJournalRecord * (cursor + k);
    EXPECT_EQ(record[0], leaf);
    EXPECT_EQ(record[1], 0);
    EXPECT_EQ(record[2], level);  // the record's level tag
  }
  int gap = 0;
  for (int leaf = 0; leaf < leaves; ++leaf) {
    if (selected[leaf]) continue;
    const auto& d = actual_descs[expected.size() + gap++];
    EXPECT_EQ(d.left_leaf_index, leaf);
    EXPECT_EQ(d.leaf_data_start, host_starts[leaf]);
    EXPECT_EQ(d.num_data_in_leaf, host_counts[leaf]);
  }
  EXPECT_EQ(state.num_gaps, gap);
  if (!final_depth || selective) return;
  // No child search at the final depth, while live_splits remains available to the tree record.
  LaunchDeviceLevelPairs(cfg, level, leaves, nullptr);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&state, device_state.data(), sizeof(state), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(state.live_splits, static_cast<int>(expected.size()));
  EXPECT_EQ(state.live_pairs, 0);
  EXPECT_EQ(state.construct_total_blocks, 0);
}
}  // namespace

TEST(CUDADeviceLevelBudget, EqualGainFinalCapsKeepLowerLeafTiesAndRightChildOrder) {
  // Original 255/depth8 and1023/depth10 final frontiers, including the full512-candidate block, and the
  // 2047/depth11 and 4095/depth12 ones up to the whole tree's full 2048-candidate block (kDeviceTreeSharedSplits);
  // past it (depth 13 and 14: 4096 and 8192 candidates) the select sorts in global scratch, block-looped
  for (const int leaves : {128, 512, 1024, 2048, 4096, 8192}) {
    SCOPED_TRACE(::testing::Message() << "leaves=" << leaves);
    CheckSelection(std::vector<double>(leaves, 7.0), std::vector<bool>(leaves, true), 2 * leaves - 1);
  }
}

TEST(CUDADeviceLevelBudget, GlobalSelectMatchesStableSortWithTiesAndInvalidCandidates) {
  // a few gain values (many exact ties across tiles), some invalid leaves, budgets that bind at different depths of
  // the frontier: the global-scratch select (4096 / 8192 / 16384 candidates) and the shared one (2048) against the
  // host's stable_sort
  for (const int leaves : {2048, 4096, 8192, 16384}) {
    std::vector<double> gains(leaves);
    std::vector<bool> valid(leaves);
    uint32_t x = 12345u + static_cast<uint32_t>(leaves);
    for (int leaf = 0; leaf < leaves; ++leaf) {
      x = 1664525u * x + 1013904223u;
      gains[leaf] = static_cast<double>((x >> 8) % 17) * 0.25;
      valid[leaf] = ((x >> 4) % 11) != 0;
    }
    for (const int extra : {1, leaves / 3, leaves - 1}) {
      SCOPED_TRACE(::testing::Message() << "leaves=" << leaves << " extra=" << extra);
      CheckSelection(gains, valid, leaves + extra);
    }
  }
}

TEST(CUDADeviceLevelBudget, FinalTopKAppliesInGainOrderAndPreservesTerminalGaps) {
  CheckSelection({2, 8, 4, 8, 1, 4, 8, 4}, std::vector<bool>(8, true), 11);
  // Seven candidates exercise padding in the cooperative sort, with ties at the cutoff.
  CheckSelection({2, 8, 4, 8, 1, 4, 8, 4}, {true, true, true, true, false, true, true, true}, 13);
}

TEST(CUDADeviceLevelBudget, NonbindingLevelsKeepAscendingOrderWithInvalidCandidates) {
  CheckSelection({2, 8, 4, 8, 1, 4, 8, 4}, std::vector<bool>(8, true), 16);
  CheckSelection({2, 8, 4, 8, 1, 4, 8, 4}, std::vector<bool>(8, true), 16, false);
  CheckSelection({2, 8, 4, 8, 1, 4, 8, 4}, {true, true, false, true, false, true, false, true}, 15);
}

TEST(CUDADeviceLevelBudget, EmptyExhaustedAndNonfinalBindingStopWithoutApply) {
  CheckSelection(std::vector<double>(8, 7.0), std::vector<bool>(8, false), 15);
  CheckSelection(std::vector<double>(8, 7.0), std::vector<bool>(8, true), 8);
  CheckSelection(std::vector<double>(8, 7.0), std::vector<bool>(8, true), 11, false);
}

TEST(CUDADeviceLevelBudget, SelectiveLevelsApplyWhileR1HoldsAndStopAtTheFirstBindingLevel) {
  // 8 leaves, 5 valid: 8 + 5 = 13 leaves fit a budget of 13 (A + S = 7 + 5 = 12 = B), not 12
  const std::vector<bool> five = {true, false, true, true, false, true, false, true};
  CheckSelection({2, 8, 4, 8, 1, 4, 8, 4}, five, 13, true, true);
  CheckSelection({2, 8, 4, 8, 1, 4, 8, 4}, five, 12, true, true);
  // no valid leaf never binds: the tree is complete
  CheckSelection(std::vector<double>(8, 7.0), std::vector<bool>(8, false), 8, true, true);
  // 1024 leaves, every one valid: 1024 splits, more than the register-batched block's 512, so the shared block selects
  // them (a level 10 of a tree of 4095 leaves; the register-batched block would stop with the error flag)
  CheckSelection(std::vector<double>(1024, 3.0), std::vector<bool>(1024, true), 2048, true, true);
  // every select capacity: register-batched (<= 1024 leaves), shared (<= 4096), global scratch; at the threshold and
  // one leaf short of it
  for (const int leaves : {600, 1024, 2000, 4096, 6000}) {
    std::vector<double> gains(leaves);
    std::vector<bool> valid(leaves);
    int num_valid = 0;
    uint32_t x = 777u + static_cast<uint32_t>(leaves);
    for (int leaf = 0; leaf < leaves; ++leaf) {
      x = 1664525u * x + 1013904223u;
      gains[leaf] = static_cast<double>((x >> 8) % 13);
      valid[leaf] = ((x >> 4) % 3) == 0;
      num_valid += valid[leaf] ? 1 : 0;
    }
    for (const int budget : {leaves + num_valid, leaves + num_valid - 1}) {
      SCOPED_TRACE(::testing::Message() << "leaves=" << leaves << " budget=" << budget);
      CheckSelection(gains, valid, budget, true, true);
    }
  }
}

TEST(CUDADeviceLevelBudget, BynodeChunkCapKeepsTaskOffsetsInIntRange) {
  // the per-node (offset, count) table holds int offsets 2 * slots + row_slot * max_tasks
  constexpr int64_t kIntMax = 2147483647;
  // depth 12 with few tasks: the unchunked layout (one row slot per leaf slot) fits, no cap
  EXPECT_EQ(Falcata::DeviceTreeBynodeChunkCap(4098, 400, 1024), 1024);
  // depth 17 unchunked: 131074 slots x 20000 tasks overflows, while every chunk below the widest search fits
  const int64_t slots = 131074;
  const int pairs = 32768;
  ASSERT_GT(2 * slots + slots * 20000, kIntMax);
  EXPECT_EQ(Falcata::DeviceTreeBynodeChunkCap(slots, 20000, pairs), pairs - 1);
  // 50000 tasks: the cap binds inside the range, at the largest chunk whose 1 + 2 P row slots fit
  const int64_t tasks = 50000;
  ASSERT_GT(2 * slots + slots * tasks, kIntMax);
  const int cap = Falcata::DeviceTreeBynodeChunkCap(slots, tasks, pairs);
  ASSERT_GE(cap, 1);
  EXPECT_LT(cap, pairs);
  EXPECT_LE(2 * slots + (1 + 2 * static_cast<int64_t>(cap)) * tasks, kIntMax);
  EXPECT_GT(2 * slots + (1 + 2 * static_cast<int64_t>(cap + 1)) * tasks, kIntMax);
  // not even one pair (3 row slots) fits: 0, the complete tree is refused (scratch_too_large)
  EXPECT_EQ(Falcata::DeviceTreeBynodeChunkCap(slots, kIntMax / 2, pairs), 0);
}
