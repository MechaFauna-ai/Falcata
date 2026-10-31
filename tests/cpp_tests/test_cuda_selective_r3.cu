/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cuda_runtime.h>
#include <Falcata/cuda/cuda_utils.hu>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <queue>
#include <random>
#include <vector>

#include <Falcata/cuda/cuda_split_info.hpp>

#include "../../src/treelearner/cuda/cuda_selective_r3.hpp"

namespace {
using Falcata::CUDASelectiveR3Config;
using Falcata::CUDASelectiveR3Record;
using Falcata::CUDASelectiveR3State;
using Falcata::CUDASplitInfo;

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t size) { EXPECT_EQ(cudaMalloc(&data_, std::max<size_t>(size, 1) * sizeof(T)), cudaSuccess); }
  ~DeviceBuffer() { EXPECT_EQ(cudaFree(data_), cudaSuccess); }
  T* data() { return data_; }
  void Up(const std::vector<T>& v) {
    if (!v.empty()) {
      ASSERT_EQ(cudaMemcpy(data_, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    }
  }
  std::vector<T> Down(size_t n) {
    std::vector<T> v(n);
    if (n > 0) {
      EXPECT_EQ(cudaMemcpy(v.data(), data_, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    }
    return v;
  }
 private:
  T* data_ = nullptr;
};

// ---- the host reference: CUDASingleGPUTreeLearner::SelectiveTieBreak and RunSelectiveLevel's three steps (the 256-lane
// emulation, std::priority_queue with the tied-set resolution). One difference in form: step 3 reads a created record's
// parent from the post-collapse leaf_parent where the learner uses the parent noted at discovery; they agree, since a
// selected candidate's parent is selected and so never collapsed. ----
int HostTieBreak(const std::vector<int>& tied_classic_leaves) {
  constexpr int kThreads = 256;
  std::array<int, kThreads> thread_winner;
  thread_winner.fill(-1);
  for (const int leaf : tied_classic_leaves) {
    const int t = leaf % kThreads;
    if (thread_winner[t] < 0 || leaf < thread_winner[t]) {
      thread_winner[t] = leaf;
    }
  }
  const auto reduce32 = [](std::array<int, 32> lanes) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      std::array<int, 32> next = lanes;
      for (int i = 0; i + offset < 32; ++i) {
        if (lanes[i] < 0 && lanes[i + offset] >= 0) {
          next[i] = lanes[i + offset];
        }
      }
      lanes = next;
    }
    return lanes[0];
  };
  std::array<int, 32> lanes;
  std::array<int, 32> warp_winner_lanes;
  warp_winner_lanes.fill(-1);
  for (int w = 0; w < kThreads / 32; ++w) {
    for (int l = 0; l < 32; ++l) {
      lanes[l] = thread_winner[w * 32 + l];
    }
    warp_winner_lanes[w] = reduce32(lanes);
  }
  return reduce32(warp_winner_lanes);
}

struct HostRecord {
  int parent, left_leaf, right_leaf, kid_left, kid_right;
  bool dead, selected;
  double gain;
};
struct HostState {
  std::vector<HostRecord> rec;
  std::vector<int> leaf_parent;
  std::vector<int> free;
  int num_allocated = 1;
};
struct LevelResult {
  int frontier = 0;
  std::vector<int> order, classic, collapsed, freed, apply_right;
};

LevelResult HostLevel(HostState* s, const std::vector<int>& children, const std::vector<char>& valid,
                      const std::vector<double>& gain, const int num_leaves) {
  const int budget = num_leaves - 1;
  LevelResult out;
  std::vector<int> f_leaf;
  std::vector<double> f_gain;
  std::vector<int> leaf_frontier(s->leaf_parent.size(), -1);
  for (const int leaf : children) {
    if (valid[leaf]) {
      leaf_frontier[leaf] = static_cast<int>(f_leaf.size());
      f_leaf.push_back(leaf);
      f_gain.push_back(gain[leaf]);
    }
  }
  out.frontier = static_cast<int>(f_leaf.size());
  for (HostRecord& r : s->rec) r.selected = false;
  struct HeapEntry { double gain; int classic_leaf; int id; };
  const auto worse = [](const HeapEntry& a, const HeapEntry& b) {
    return a.gain < b.gain || (a.gain == b.gain && a.classic_leaf > b.classic_leaf);
  };
  std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(worse)> heap(worse);
  if (!s->rec.empty()) {
    heap.push({s->rec[0].gain, 0, 0});
  } else if (!f_leaf.empty()) {
    heap.push({f_gain[0], 0, ~0});
  }
  const auto push_child = [&](const HostRecord& node, const bool left, const int classic_leaf) {
    const int kid = left ? node.kid_left : node.kid_right;
    if (kid >= 0) {
      heap.push({s->rec[kid].gain, classic_leaf, kid});
      return;
    }
    const int fidx = leaf_frontier[left ? node.left_leaf : node.right_leaf];
    if (fidx >= 0) heap.push({f_gain[fidx], classic_leaf, ~fidx});
  };
  std::vector<HeapEntry> tied;
  while (static_cast<int>(out.order.size()) < budget && !heap.empty()) {
    HeapEntry e = heap.top();
    heap.pop();
    if (!heap.empty() && heap.top().gain == e.gain) {
      tied.clear();
      tied.push_back(e);
      while (!heap.empty() && heap.top().gain == e.gain) {
        tied.push_back(heap.top());
        heap.pop();
      }
      std::vector<int> tied_leaves;
      for (const HeapEntry& c : tied) tied_leaves.push_back(c.classic_leaf);
      const int winner_leaf = HostTieBreak(tied_leaves);
      for (const HeapEntry& c : tied) {
        if (c.classic_leaf == winner_leaf) {
          e = c;
        } else {
          heap.push(c);
        }
      }
    }
    const int t = static_cast<int>(out.order.size());
    out.order.push_back(e.id);
    out.classic.push_back(e.classic_leaf);
    if (e.id >= 0) {
      s->rec[e.id].selected = true;
      const HostRecord node = s->rec[e.id];
      push_child(node, true, e.classic_leaf);
      push_child(node, false, t + 1);
    }
  }
  std::vector<int> stack;
  for (int id = 0; id < static_cast<int>(s->rec.size()); ++id) {
    HostRecord& node = s->rec[id];
    if (node.dead || node.selected) continue;
    out.collapsed.push_back(id);
    if (node.parent >= 0) {
      HostRecord& p = s->rec[node.parent];
      if (p.kid_left == id) {
        p.kid_left = -1;
      } else if (p.kid_right == id) {
        p.kid_right = -1;
      }
    }
    stack.assign(1, id);
    while (!stack.empty()) {
      HostRecord& m = s->rec[stack.back()];
      stack.pop_back();
      m.dead = true;
      s->leaf_parent[m.right_leaf] = -1;
      s->free.push_back(m.right_leaf);
      out.freed.push_back(m.right_leaf);
      if (m.kid_left >= 0) stack.push_back(m.kid_left);
      if (m.kid_right >= 0) stack.push_back(m.kid_right);
    }
    s->leaf_parent[node.left_leaf] = node.parent;
  }
  for (size_t t = 0; t < out.order.size(); ++t) {
    const int id = out.order[t];
    if (id >= 0) continue;
    const int leaf = f_leaf[~id];
    int right;
    if (!s->free.empty()) {
      right = s->free.back();
      s->free.pop_back();
    } else {
      right = s->num_allocated++;
    }
    EXPECT_LT(right, num_leaves);
    const int new_id = static_cast<int>(s->rec.size());
    const int parent = s->leaf_parent[leaf];
    s->rec.push_back({parent, leaf, right, -1, -1, false, true, f_gain[~id]});
    if (parent >= 0) {
      HostRecord& p = s->rec[parent];
      if (p.left_leaf == leaf) {
        p.kid_left = new_id;
      } else {
        p.kid_right = new_id;
      }
    }
    s->leaf_parent[leaf] = new_id;
    s->leaf_parent[right] = new_id;
    out.apply_right.push_back(right);
  }
  return out;
}

// one level on the device from the pre-level state; compared field by field with the host reference's result
void CheckLevel(const HostState& pre, const HostState& post, const LevelResult& want,
                const std::vector<int>& children, const std::vector<char>& valid, const std::vector<double>& gain,
                const int num_leaves, const bool force_global_heap = false) {
  const int budget = num_leaves - 1;
  const int leaf_cap = num_leaves + 2;
  const int n_rec = static_cast<int>(pre.rec.size());
  const int rec_cap = n_rec + budget + 1;
  std::vector<CUDASplitInfo> best(leaf_cap);
  for (int leaf = 0; leaf < leaf_cap; ++leaf) {
    best[leaf].is_valid = valid[leaf] != 0;
    best[leaf].gain = gain[leaf];
    best[leaf].leaf_index = leaf;
  }
  DeviceBuffer<CUDASplitInfo> d_best(leaf_cap);
  ASSERT_EQ(cudaMemcpy(d_best.data(), best.data(), leaf_cap * sizeof(CUDASplitInfo), cudaMemcpyHostToDevice),
            cudaSuccess);
  CUDASelectiveR3State hs{};
  hs.num_records = n_rec;
  hs.num_free = static_cast<int>(pre.free.size());
  hs.num_allocated = pre.num_allocated;
  hs.num_children = static_cast<int>(children.size());
  std::vector<CUDASelectiveR3Record> recs;
  for (const HostRecord& r : pre.rec) {
    recs.push_back({r.parent, r.left_leaf, r.right_leaf, r.kid_left, r.kid_right, r.dead ? 1 : 0, r.gain});
  }
  std::vector<int> free_leaves = pre.free;
  free_leaves.resize(leaf_cap, -7);
  DeviceBuffer<CUDASelectiveR3State> d_state(1);
  DeviceBuffer<CUDASelectiveR3Record> d_rec(rec_cap);
  DeviceBuffer<int> d_leaf_parent(leaf_cap), d_free(leaf_cap), d_children(children.size());
  DeviceBuffer<int> d_leaf_frontier(leaf_cap), d_frontier_leaf(leaf_cap), d_selected(rec_cap), d_stack(rec_cap);
  DeviceBuffer<double> d_frontier_gain(leaf_cap);
  DeviceBuffer<char> d_child_table(rec_cap * Falcata::kSelectiveR3ChildEntryBytes);
  DeviceBuffer<char> d_heap(num_leaves * Falcata::kSelectiveR3HeapEntryBytes);
  DeviceBuffer<int> d_order(budget), d_classic(budget), d_collapsed(rec_cap), d_freed(rec_cap), d_apply(budget);
  d_state.Up({hs});
  d_rec.Up(recs);
  d_leaf_parent.Up(pre.leaf_parent);
  d_free.Up(free_leaves);
  d_children.Up(children);
  CUDASelectiveR3Config cfg;
  cfg.state = d_state.data();
  cfg.records = d_rec.data();
  cfg.leaf_parent = d_leaf_parent.data();
  cfg.free_leaves = d_free.data();
  cfg.child_leaves = d_children.data();
  cfg.leaf_best = d_best.data();
  cfg.leaf_frontier = d_leaf_frontier.data();
  cfg.frontier_leaf = d_frontier_leaf.data();
  cfg.frontier_gain = d_frontier_gain.data();
  cfg.selected = d_selected.data();
  cfg.stack = d_stack.data();
  cfg.child_table = d_child_table.data();
  cfg.heap_global = d_heap.data();
  cfg.order_id = d_order.data();
  cfg.order_classic = d_classic.data();
  cfg.collapsed = d_collapsed.data();
  cfg.freed = d_freed.data();
  cfg.apply_right = d_apply.data();
  cfg.budget = budget;
  cfg.num_leaves = num_leaves;
  cfg.leaf_capacity = leaf_cap;
  cfg.record_capacity = rec_cap;
  Falcata::LaunchSelectiveR3(cfg, nullptr, /*shared_heap=*/!force_global_heap);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const CUDASelectiveR3State ds = d_state.Down(1)[0];
  ASSERT_EQ(ds.error, 0);
  ASSERT_EQ(ds.num_frontier, want.frontier);
  ASSERT_EQ(ds.num_order, static_cast<int>(want.order.size()));
  EXPECT_EQ(d_order.Down(ds.num_order), want.order);
  EXPECT_EQ(d_classic.Down(ds.num_order), want.classic);
  EXPECT_EQ(d_collapsed.Down(ds.num_collapsed), want.collapsed);
  EXPECT_EQ(d_freed.Down(ds.num_freed), want.freed);
  EXPECT_EQ(d_apply.Down(ds.num_apply), want.apply_right);
  EXPECT_EQ(d_free.Down(ds.num_free), post.free);
  EXPECT_EQ(ds.num_allocated, post.num_allocated);
  ASSERT_EQ(ds.num_records, static_cast<int>(post.rec.size()));
  const std::vector<CUDASelectiveR3Record> got = d_rec.Down(ds.num_records);
  for (int i = 0; i < ds.num_records; ++i) {
    const HostRecord& r = post.rec[i];
    EXPECT_EQ(got[i].parent, r.parent) << i;
    EXPECT_EQ(got[i].left_leaf, r.left_leaf) << i;
    EXPECT_EQ(got[i].right_leaf, r.right_leaf) << i;
    EXPECT_EQ(got[i].kid_left, r.kid_left) << i;
    EXPECT_EQ(got[i].kid_right, r.kid_right) << i;
    EXPECT_EQ(got[i].dead, r.dead ? 1 : 0) << i;
    EXPECT_EQ(got[i].gain, r.gain) << i;
  }
  const std::vector<int> leaf_parent = d_leaf_parent.Down(leaf_cap);
  for (int leaf = 0; leaf < post.num_allocated; ++leaf) {
    EXPECT_EQ(leaf_parent[leaf], post.leaf_parent[leaf]) << leaf;
  }
}

// grows whole trees level by level through the host reference, checking the device on every level. gain_of(leaf,
// level) returns the candidate's gain, or a negative value for an invalid one.
template <typename GainOf>
int GrowAndCheck(const int num_leaves, const int max_levels, GainOf gain_of, const bool force_global_heap = false) {
  const int leaf_cap = num_leaves + 2;
  HostState s;
  s.leaf_parent.assign(leaf_cap, -1);
  std::vector<int> children{0};
  int levels = 0;
  for (int level = 0; level < max_levels; ++level) {
    std::vector<char> valid(leaf_cap, 0);
    std::vector<double> gain(leaf_cap, 0.0);
    for (const int leaf : children) {
      const double g = gain_of(leaf, level);
      valid[leaf] = g >= 0.0;
      gain[leaf] = g >= 0.0 ? g : 0.0;
    }
    const HostState pre = s;
    const LevelResult want = HostLevel(&s, children, valid, gain, num_leaves);
    CheckLevel(pre, s, want, children, valid, gain, num_leaves, force_global_heap);
    ++levels;
    if (testing::Test::HasFatalFailure() || testing::Test::HasNonfatalFailure()) {
      ADD_FAILURE() << "level " << level;
      return levels;
    }
    int created = 0;
    children.clear();
    for (size_t i = pre.rec.size(); i < s.rec.size(); ++i) {
      children.push_back(s.rec[i].left_leaf);
      children.push_back(s.rec[i].right_leaf);
      ++created;
    }
    if (created == 0) {
      break;  // the final simulation: no frontier candidate selected
    }
  }
  return levels;
}
}  // namespace

TEST(CUDASelectiveR3, TieKeyIsTheReductionTieBreak) {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 4000; ++trial) {
    const int span = trial % 3 == 0 ? 300 : (trial % 3 == 1 ? 2000 : 70000);
    const int count = 1 + static_cast<int>(rng() % (trial % 5 == 0 ? 600 : 12));
    std::vector<int> leaves;
    for (int i = 0; i < count; ++i) leaves.push_back(static_cast<int>(rng() % span));
    std::sort(leaves.begin(), leaves.end());
    leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
    int best = leaves[0];
    for (const int leaf : leaves) {
      if (Falcata::SelectiveR3TieKey(leaf) < Falcata::SelectiveR3TieKey(best)) best = leaf;
    }
    ASSERT_EQ(best, HostTieBreak(leaves)) << "trial " << trial;
  }
  for (int leaf = 0; leaf < 200000; leaf += 7) {
    ASSERT_EQ(Falcata::SelectiveR3KeyLeaf(Falcata::SelectiveR3TieKey(leaf)), leaf);
  }
}

TEST(CUDASelectiveR3, AllGainsTiedAcrossManyLeaves) {
  // every candidate has the same gain: the pop order is decided by the tie-break alone, over > 256 tied leaves
  for (const int num_leaves : {31, 255, 600, 1500}) {
    GrowAndCheck(num_leaves, 40, [](int, int) { return 1.0; });
  }
}

TEST(CUDASelectiveR3, FewDistinctGainsWithInvalidCandidates) {
  for (const int num_leaves : {7, 63, 300, 1024, 2047}) {
    for (int seed = 0; seed < 4; ++seed) {
      std::mt19937 rng(1000 * num_leaves + seed);
      GrowAndCheck(num_leaves, 60, [&rng](int, int) {
        const unsigned u = rng() % 10;
        return u < 2 ? -1.0 : static_cast<double>(1 + rng() % 4);
      });
    }
  }
}

TEST(CUDASelectiveR3, DistinctGainsDeepChurn) {
  // continuous gains, deeper children often better than their parents (displacement and collapse every level)
  for (const int num_leaves : {15, 127, 4095}) {
    std::mt19937 rng(num_leaves);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    GrowAndCheck(num_leaves, 80, [&](int, int level) { return u(rng) < 0.1 ? -1.0 : u(rng) * (1.0 + 0.05 * level); });
  }
}

TEST(CUDASelectiveR3, AllInvalidLevelIsFinal) {
  // the root's children are all invalid: the second simulation re-selects the root alone and creates nothing
  const int levels = GrowAndCheck(15, 10, [](int, int level) { return level == 0 ? 2.0 : -1.0; });
  EXPECT_EQ(levels, 2);
  EXPECT_EQ(GrowAndCheck(15, 10, [](int, int) { return -1.0; }), 1);  // an invalid root: nothing selected
}

TEST(CUDASelectiveR3, BudgetExactlyMetAndExceededByOne) {
  // num_leaves 4 (budget 3): level 1 holds A = 1 applied + S = 2 valid = budget, every candidate selected, no collapse
  {
    HostState s;
    s.leaf_parent.assign(6, -1);
    std::vector<char> valid(6, 0);
    std::vector<double> gain(6, 0.0);
    valid[0] = 1;
    gain[0] = 5.0;
    HostState pre = s;
    LevelResult r0 = HostLevel(&s, {0}, valid, gain, 4);
    CheckLevel(pre, s, r0, {0}, valid, gain, 4);
    valid.assign(6, 1);
    gain = {3.0, 4.0, 0.0, 0.0, 0.0, 0.0};
    pre = s;
    const LevelResult r1 = HostLevel(&s, {0, 1}, valid, gain, 4);
    EXPECT_EQ(r1.order.size(), 3u);
    EXPECT_TRUE(r1.collapsed.empty());
    CheckLevel(pre, s, r1, {0, 1}, valid, gain, 4);
  }
  // num_leaves 3 (budget 2): A = 1 + S = 2 = budget + 1; the better child wins, the other is dormant
  {
    HostState s;
    s.leaf_parent.assign(5, -1);
    std::vector<char> valid(5, 1);
    std::vector<double> gain{5.0, 0.0, 0.0, 0.0, 0.0};
    HostState pre = s;
    LevelResult r0 = HostLevel(&s, {0}, valid, gain, 3);
    CheckLevel(pre, s, r0, {0}, valid, gain, 3);
    gain = {1.0, 2.0, 0.0, 0.0, 0.0};
    pre = s;
    const LevelResult r1 = HostLevel(&s, {0, 1}, valid, gain, 3);
    EXPECT_EQ(r1.order.size(), 2u);
    CheckLevel(pre, s, r1, {0, 1}, valid, gain, 3);
    // a tie at the budget's edge: both children equal, the tie-break picks classic leaf 0 over 1
    HostState s2;
    s2.leaf_parent.assign(5, -1);
    std::vector<double> gain2{5.0, 0.0, 0.0, 0.0, 0.0};
    pre = s2;
    r0 = HostLevel(&s2, {0}, valid, gain2, 3);
    CheckLevel(pre, s2, r0, {0}, valid, gain2, 3);
    gain2 = {2.0, 2.0, 0.0, 0.0, 0.0};
    pre = s2;
    const LevelResult r2 = HostLevel(&s2, {0, 1}, valid, gain2, 3);
    CheckLevel(pre, s2, r2, {0, 1}, valid, gain2, 3);
  }
  // a displaced subtree: a deep record outranked by a new sibling candidate collapses (free list LIFO)
  {
    std::mt19937 rng(5);
    GrowAndCheck(6, 30, [&rng](int, int level) { return level < 3 ? 10.0 - level : static_cast<double>(rng() % 20); });
  }
}

TEST(CUDASelectiveR3, WiderThanOneBlockAndGlobalHeap) {
  // levels of more than 256 (one block) and 1024 candidates; num_leaves 5000 takes the opted-in shared heap
  // (80 KB), 8000 the global heap (128 KB), and the forced global heap at 1500
  std::mt19937 rng(11);
  for (const int num_leaves : {5000, 8000}) {
    GrowAndCheck(num_leaves, 30, [&rng](int, int) { return rng() % 50 == 0 ? -1.0 : static_cast<double>(rng() % 3); });
  }
  GrowAndCheck(1500, 30, [&rng](int, int) { return static_cast<double>(rng() % 5); }, /*force_global_heap=*/true);
}
