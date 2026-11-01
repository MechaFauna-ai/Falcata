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

// the parallel R3's device state kept across the levels of one grown tree (P8c)
struct ParallelRun {
  explicit ParallelRun(const int num_leaves)
      : persist(1), order_buf(6 * (num_leaves + 2)), area(Falcata::R3ParallelAreaBytes(num_leaves)) {
    EXPECT_EQ(cudaMemset(persist.data(), 0, sizeof(Falcata::CUDASelectiveR3Persist)), cudaSuccess);
  }
  DeviceBuffer<Falcata::CUDASelectiveR3Persist> persist;
  DeviceBuffer<int> order_buf;
  DeviceBuffer<char> area;
  bool shared_area = true;
  int rebuild_every = 0;  // > 0: also rebuild the order from the records every k-th level
  int levels = 0;
  int merged = 0;        // levels with no resume at all (rebuild rounds included)
  int resumed = 0;       // levels with a resume (rebuild rounds included)
  int level_resumed = 0;    // levels whose own merge resumed (resume_at >= 0)
  int resumed_binding = 0;  // ... on a binding level (live records + frontier > budget)
  int resumed_last = 0;     // ... at the budget's last pop (resume_at == budget - 1)
  int rebuilt_rounds = 0;
};

// one level on the device from the pre-level state; compared field by field with the host reference's result. par:
// the parallel kernel with its kept order (else the sequential one)
void CheckLevel(const HostState& pre, const HostState& post, const LevelResult& want,
                const std::vector<int>& children, const std::vector<char>& valid, const std::vector<double>& gain,
                const int num_leaves, const bool force_global_heap = false, ParallelRun* par = nullptr) {
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
  DeviceBuffer<int> d_rec_pos(rec_cap), d_rec_round(rec_cap), d_cand(14 * leaf_cap);
  DeviceBuffer<double> d_cand_gain(3 * leaf_cap);
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
  if (par != nullptr) {
    cfg.persist = par->persist.data();
    cfg.order_buf = par->order_buf.data();
    cfg.rec_pos = d_rec_pos.data();
    cfg.rec_round = d_rec_round.data();
    cfg.cand = d_cand.data();
    cfg.cand_gain = d_cand_gain.data();
    cfg.par_global = par->area.data();
    cfg.rebuild = par->levels == 0 || (par->rebuild_every > 0 && par->levels % par->rebuild_every == 0);
    ASSERT_TRUE(Falcata::LaunchSelectiveR3Parallel(cfg, nullptr, par->shared_area));
  } else {
    Falcata::LaunchSelectiveR3(cfg, nullptr, /*shared_heap=*/!force_global_heap);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const CUDASelectiveR3State ds = d_state.Down(1)[0];
  ASSERT_EQ(ds.error, 0);
  if (par != nullptr) {
    ++par->levels;
    par->merged += ds.resumes == 0;
    par->resumed += ds.resumes > 0;
    int live = 0;
    for (const HostRecord& r : pre.rec) {
      live += r.dead ? 0 : 1;
    }
    EXPECT_LT(ds.resume_at, budget);
    par->level_resumed += ds.resume_at >= 0;
    par->resumed_binding += ds.resume_at >= 0 && live + ds.num_frontier > budget;
    par->resumed_last += ds.resume_at == budget - 1;
    par->rebuilt_rounds += ds.rounds;
  }
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
int GrowAndCheck(const int num_leaves, const int max_levels, GainOf gain_of, const bool force_global_heap = false,
                 ParallelRun* par = nullptr) {
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
    if (par != nullptr) {
      CheckLevel(pre, s, want, children, valid, gain, num_leaves, false, par);
    }
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
  // (each level also through the parallel kernel, its order kept between the levels of one tree)
  {
    ParallelRun par_4(4);
    HostState s;
    s.leaf_parent.assign(6, -1);
    std::vector<char> valid(6, 0);
    std::vector<double> gain(6, 0.0);
    valid[0] = 1;
    gain[0] = 5.0;
    HostState pre = s;
    LevelResult r0 = HostLevel(&s, {0}, valid, gain, 4);
    CheckLevel(pre, s, r0, {0}, valid, gain, 4);
    CheckLevel(pre, s, r0, {0}, valid, gain, 4, false, &par_4);
    valid.assign(6, 1);
    gain = {3.0, 4.0, 0.0, 0.0, 0.0, 0.0};
    pre = s;
    const LevelResult r1 = HostLevel(&s, {0, 1}, valid, gain, 4);
    EXPECT_EQ(r1.order.size(), 3u);
    EXPECT_TRUE(r1.collapsed.empty());
    CheckLevel(pre, s, r1, {0, 1}, valid, gain, 4);
    CheckLevel(pre, s, r1, {0, 1}, valid, gain, 4, false, &par_4);
  }
  // num_leaves 3 (budget 2): A = 1 + S = 2 = budget + 1; the better child wins, the other is dormant
  {
    ParallelRun par_3(3);
    ParallelRun par_tie(3);
    HostState s;
    s.leaf_parent.assign(5, -1);
    std::vector<char> valid(5, 1);
    std::vector<double> gain{5.0, 0.0, 0.0, 0.0, 0.0};
    HostState pre = s;
    LevelResult r0 = HostLevel(&s, {0}, valid, gain, 3);
    CheckLevel(pre, s, r0, {0}, valid, gain, 3);
    CheckLevel(pre, s, r0, {0}, valid, gain, 3, false, &par_3);
    gain = {1.0, 2.0, 0.0, 0.0, 0.0};
    pre = s;
    const LevelResult r1 = HostLevel(&s, {0, 1}, valid, gain, 3);
    EXPECT_EQ(r1.order.size(), 2u);
    CheckLevel(pre, s, r1, {0, 1}, valid, gain, 3);
    CheckLevel(pre, s, r1, {0, 1}, valid, gain, 3, false, &par_3);
    // a tie at the budget's edge: both children equal, the tie-break picks classic leaf 0 over 1
    HostState s2;
    s2.leaf_parent.assign(5, -1);
    std::vector<double> gain2{5.0, 0.0, 0.0, 0.0, 0.0};
    pre = s2;
    r0 = HostLevel(&s2, {0}, valid, gain2, 3);
    CheckLevel(pre, s2, r0, {0}, valid, gain2, 3);
    CheckLevel(pre, s2, r0, {0}, valid, gain2, 3, false, &par_tie);
    gain2 = {2.0, 2.0, 0.0, 0.0, 0.0};
    pre = s2;
    const LevelResult r2 = HostLevel(&s2, {0, 1}, valid, gain2, 3);
    CheckLevel(pre, s2, r2, {0, 1}, valid, gain2, 3);
    CheckLevel(pre, s2, r2, {0, 1}, valid, gain2, 3, false, &par_tie);
    EXPECT_EQ(par_tie.resumed, 0);  // the equal pair at the budget's edge: the merge's order is the tie key's
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

// ---- the parallel R3 (P8c): the same levels through the merge, its order kept on the device between them ----

TEST(CUDASelectiveR3Parallel, DistinctGainsEverySlotKind) {
  // continuous gains: no equal pair, so every level is merged; children often beat their parents (insertions right
  // after the parent, in the middle, at the end; displacement and collapse every level), budgets met and exceeded
  for (const int num_leaves : {2, 3, 4, 7, 15, 64, 127, 255, 1024}) {
    for (int seed = 0; seed < 3; ++seed) {
      std::mt19937 rng(100 * num_leaves + seed);
      std::uniform_real_distribution<double> u(0.0, 1.0);
      ParallelRun par(num_leaves);
      par.rebuild_every = seed == 2 ? 3 : 0;
      GrowAndCheck(num_leaves, 60, [&](int, int level) {
        return u(rng) < 0.1 ? -1.0 : u(rng) * (1.0 + (seed == 1 ? -0.01 : 0.05) * level);
      }, false, &par);
      EXPECT_EQ(par.resumed, 0) << num_leaves;
      if (seed == 2 && par.levels > 3) {
        EXPECT_GT(par.rebuilt_rounds, 0) << num_leaves;
      }
    }
  }
}

TEST(CUDASelectiveR3Parallel, CandidatesAllDisplacedOrAllFirst) {
  // gains falling with depth: once the budget binds no candidate is selected (all land past the budget); rising
  // with depth: every candidate pops right after its parent and the deepest records are displaced
  for (const double slope : {-1.0, 1.0}) {
    for (const int num_leaves : {31, 300}) {
      std::mt19937 rng(num_leaves);
      std::uniform_real_distribution<double> u(0.0, 1.0);
      ParallelRun par(num_leaves);
      GrowAndCheck(num_leaves, 40, [&](int, int level) { return 100.0 + slope * level + 0.5 * u(rng); }, false, &par);
      EXPECT_EQ(par.merged, par.levels);
    }
  }
}

TEST(CUDASelectiveR3Parallel, WideLevelsGlobalAreaAndManyLeaves) {
  // more candidates than the block's 512 threads; 5000 and 8000 leaves (the merge area in global memory), and the
  // global area forced at 2047; a rebuild from the records every other level at 3000
  std::mt19937 rng(23);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (const int num_leaves : {2047, 3000, 5000, 8000}) {
    ParallelRun par(num_leaves);
    par.shared_area = num_leaves != 2047;
    par.rebuild_every = num_leaves == 3000 ? 2 : 0;
    GrowAndCheck(num_leaves, 30, [&](int, int level) { return u(rng) < 0.02 ? -1.0 : u(rng) + 0.01 * level; }, false,
                 &par);
    EXPECT_EQ(par.merged, par.levels) << num_leaves;
  }
}

TEST(CUDASelectiveR3Parallel, EqualGainsResolvedByTheTieKey) {
  // few distinct gains: the merge's fixed tie order holds up to the first tie the tie key decides otherwise, then the
  // kernel runs best-first from there; all-equal gains over > 256 leaves (ties everywhere, also in rebuilds)
  for (const int num_leaves : {7, 63, 600, 2047}) {
    int resumed = 0;
    for (int seed = 0; seed < 3; ++seed) {
      std::mt19937 rng(7 * num_leaves + seed);
      ParallelRun par(num_leaves);
      par.rebuild_every = seed == 2 ? 2 : 0;
      GrowAndCheck(num_leaves, 50, [&rng](int, int level) {
        const unsigned v = rng() % 10;
        return v < 2 && level > 0 ? -1.0 : static_cast<double>(1 + rng() % 4);
      }, false, &par);
      resumed += par.resumed;
    }
    EXPECT_GT(resumed, 0) << num_leaves;
  }
  ParallelRun all_equal(1500);
  GrowAndCheck(1500, 40, [](int, int) { return 1.0; }, false, &all_equal);
  EXPECT_GT(all_equal.resumed, 0);
  // one equal pair late in a tree of distinct gains
  {
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int calls = 0;
    ParallelRun par(63);
    GrowAndCheck(63, 30, [&](int, int level) { return level == 4 && calls++ < 2 ? 0.75 : u(rng); }, false, &par);
    EXPECT_GT(par.merged, 0);
  }
}

TEST(CUDASelectiveR3Parallel, TwoValuedGainsTinyBudgets) {
  // two gains only, 2-33 leaves: ties at the budget's edge, between candidates and kept entries and among many; the
  // level's own resume must occur on binding levels and at the budget's last pop (review of #188)
  ParallelRun total(2);
  for (const int num_leaves : {2, 3, 4, 5, 6, 8, 9, 16, 17, 33}) {
    for (int seed = 0; seed < 40; ++seed) {
      std::mt19937 rng(1000 * num_leaves + seed);
      ParallelRun par(num_leaves);
      par.rebuild_every = seed % 4 == 3 ? 2 : 0;
      GrowAndCheck(num_leaves, 40, [&](int, int level) {
        if (rng() % 16 == 0 && level > 0) {
          return -1.0;
        }
        return (seed % 2) ? static_cast<double>(1 + rng() % 2) : static_cast<double>(1 + (rng() % 3 == 0));
      }, false, &par);
      ASSERT_FALSE(testing::Test::HasFailure()) << "num_leaves " << num_leaves << " seed " << seed;
      total.level_resumed += par.level_resumed;
      total.resumed_binding += par.resumed_binding;
      total.resumed_last += par.resumed_last;
    }
  }
  EXPECT_GT(total.resumed_binding, 0);
  EXPECT_GT(total.resumed_last, 0);
}
