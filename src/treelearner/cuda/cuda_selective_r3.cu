/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief R3, the device greedy simulation of one selective level (see cuda_selective_r3.hpp).
 */

#ifdef USE_CUDA

#include "cuda_selective_r3.hpp"

#include <Falcata/cuda/cuda_utils.hu>

#include <algorithm>
#include <climits>
#include <mutex>
#include <unordered_map>

namespace Falcata {

namespace {

constexpr int kR3Threads = 256;
constexpr int kR3None = INT_MIN;  // no child candidate (~INT_MIN = INT_MAX is no frontier index)

struct R3HeapEntry {
  double gain;
  uint32_t key;
  int id;
};
static_assert(sizeof(R3HeapEntry) == kSelectiveR3HeapEntryBytes, "heap entry size");

// a record's two child candidates (its applied child record, else its leaf's frontier candidate), gathered in parallel
// before the sequential simulation so a pop reads one 32-byte entry
struct __align__(32) R3Child {
  double gain[2];  // left, right
  int id[2];
  int pad[2];
};
static_assert(sizeof(R3Child) == kSelectiveR3ChildEntryBytes, "child entry size");

__device__ __forceinline__ bool R3Better(const R3HeapEntry& a, const R3HeapEntry& b) {
  return a.gain > b.gain || (a.gain == b.gain && a.key < b.key);
}

// 4-ary heap, one thread
template <typename Entry>
__device__ __forceinline__ void R3SiftUp(Entry* heap, int i, const Entry e) {
  while (i > 0) {
    const int p = (i - 1) >> 2;
    const Entry pe = heap[p];
    if (!R3Better(e, pe)) {
      break;
    }
    heap[i] = pe;
    i = p;
  }
  heap[i] = e;
}

template <typename Entry>
__device__ __forceinline__ void R3SiftDown(Entry* heap, const int n, int i, const Entry e) {
  while (true) {
    const int c = 4 * i + 1;
    if (c >= n) {
      break;
    }
    int best = c;
    Entry be = heap[c];
#pragma unroll
    for (int k = 1; k < 4; ++k) {
      if (c + k < n) {
        const Entry ce = heap[c + k];
        if (R3Better(ce, be)) {
          be = ce;
          best = c + k;
        }
      }
    }
    if (!R3Better(be, e)) {
      break;
    }
    heap[i] = be;
    i = best;
  }
  heap[i] = e;
}

// order-preserving block compaction of [0, n): emit(i, rank) for every i with pred(i); returns the count
template <int kThreads, typename Pred, typename Emit>
__device__ int R3BlockCompact(const int n, int* warp_counts, Pred pred, Emit emit) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int kWarps = kThreads / 32;
  int base = 0;
  for (int start = 0; start < n; start += kThreads) {
    const int i = start + static_cast<int>(threadIdx.x);
    const bool flag = i < n && pred(i);
    const unsigned ballot = __ballot_sync(0xffffffffu, flag);
    if (lane == 0) {
      warp_counts[warp] = __popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int sum = 0;
      for (int w = 0; w < kWarps; ++w) {
        const int c = warp_counts[w];
        warp_counts[w] = sum;
        sum += c;
      }
      warp_counts[kWarps] = sum;
    }
    __syncthreads();
    if (flag) {
      emit(i, base + warp_counts[warp] + __popc(ballot & ((1u << lane) - 1u)));
    }
    base += warp_counts[kWarps];
    __syncthreads();
  }
  return base;
}

template <bool kSharedHeap>
__global__ void __launch_bounds__(kR3Threads) SelectiveR3Kernel(const CUDASelectiveR3Config cfg) {
  extern __shared__ __align__(16) unsigned char r3_smem[];
  __shared__ int warp_counts[kR3Threads / 32 + 1];
  CUDASelectiveR3State* st = cfg.state;
  CUDASelectiveR3Record* records = cfg.records;
  R3Child* child = static_cast<R3Child*>(cfg.child_table);
  const int num_records = st->num_records;
  const int num_children = st->num_children;
  for (int i = threadIdx.x; i < cfg.leaf_capacity; i += kR3Threads) {
    cfg.leaf_frontier[i] = -1;
  }
  for (int r = threadIdx.x; r < num_records; r += kR3Threads) {
    cfg.selected[r] = 0;
  }
  __syncthreads();
  // the frontier: the valid children in child order (SelectiveDiscoverFrontier)
  const int num_frontier = R3BlockCompact<kR3Threads>(num_children, warp_counts,
    [&](const int i) { return cfg.leaf_best[cfg.child_leaves[i]].is_valid; },
    [&](const int i, const int f) {
      const int leaf = cfg.child_leaves[i];
      cfg.frontier_leaf[f] = leaf;
      cfg.frontier_gain[f] = cfg.leaf_best[leaf].gain;
      cfg.leaf_frontier[leaf] = f;
    });
  // each live record's child candidates (RunSelectiveLevel's push_child)
  for (int r = threadIdx.x; r < num_records; r += kR3Threads) {
    const CUDASelectiveR3Record rec = records[r];
    R3Child c = {{0.0, 0.0}, {kR3None, kR3None}, {0, 0}};
    for (int side = 0; side < 2 && !rec.dead; ++side) {
      const int kid = side == 0 ? rec.kid_left : rec.kid_right;
      const int f = kid >= 0 ? -1 : cfg.leaf_frontier[side == 0 ? rec.left_leaf : rec.right_leaf];
      if (kid >= 0 || f >= 0) {
        c.id[side] = kid >= 0 ? kid : ~f;
        c.gain[side] = kid >= 0 ? records[kid].gain : cfg.frontier_gain[f];
      }
    }
    child[r] = c;
  }
  __syncthreads();
  // ---- 1. the greedy simulation (one thread): replace-top keeps the heap's content equal to the host's after every
  // pop and push, and (gain, key) is a total order on the entries (one entry per classic leaf), so the pop sequence
  // is the host's
  if (threadIdx.x == 0) {
    R3HeapEntry* heap = kSharedHeap ? reinterpret_cast<R3HeapEntry*>(r3_smem)
                                    : static_cast<R3HeapEntry*>(cfg.heap_global);
    int n = 0;
    if (num_records > 0) {
      heap[0] = {records[0].gain, SelectiveR3TieKey(0), 0};
      n = 1;
    } else if (num_frontier > 0) {
      heap[0] = {cfg.frontier_gain[0], SelectiveR3TieKey(0), ~0};
      n = 1;
    }
    int t = 0;
    while (t < cfg.budget && n > 0) {
      const R3HeapEntry e = heap[0];
      const int classic = SelectiveR3KeyLeaf(e.key);
      cfg.order_id[t] = e.id;
      cfg.order_classic[t] = classic;
      // the popped record's children: the first replaces the root, the second is pushed
      R3HeapEntry kids[2];
      int num_kids = 0;
      if (e.id >= 0) {
        cfg.selected[e.id] = 1;
        const R3Child c = child[e.id];
        for (int side = 0; side < 2; ++side) {
          if (c.id[side] != kR3None) {
            kids[num_kids++] = {c.gain[side], SelectiveR3TieKey(side == 0 ? classic : t + 1), c.id[side]};
          }
        }
      }
      if (num_kids == 0) {
        --n;
      }
      if (n > 0) {
        R3SiftDown(heap, n, 0, num_kids > 0 ? kids[0] : heap[n]);
      }
      if (num_kids == 2) {
        R3SiftUp(heap, n++, kids[1]);
      }
      ++t;
    }
    st->num_order = t;
    st->num_frontier = num_frontier;
  }
  __syncthreads();
  // ---- 2. the displaced records. The host visits ids ascending and skips dead or selected ones; a record it collapses
  // as a top has a selected (or no) parent, every other displaced record is a descendant of one and dead by then
  const int num_collapsed = R3BlockCompact<kR3Threads>(num_records, warp_counts,
    [&](const int r) {
      const CUDASelectiveR3Record& rec = records[r];
      return !rec.dead && !cfg.selected[r] && (rec.parent < 0 || cfg.selected[rec.parent]);
    },
    [&](const int r, const int k) { cfg.collapsed[k] = r; });
  if (threadIdx.x != 0) {
    return;
  }
  int num_free = st->num_free;
  int num_freed = 0;
  for (int k = 0; k < num_collapsed; ++k) {
    const int id = cfg.collapsed[k];
    const CUDASelectiveR3Record top = records[id];
    if (top.parent >= 0) {
      CUDASelectiveR3Record* p = records + top.parent;
      if (p->kid_left == id) {
        p->kid_left = -1;
      } else if (p->kid_right == id) {
        p->kid_right = -1;
      }
    }
    int sp = 0;
    cfg.stack[sp++] = id;
    while (sp > 0) {
      CUDASelectiveR3Record* m = records + cfg.stack[--sp];
      m->dead = 1;
      const int right = m->right_leaf;
      cfg.leaf_parent[right] = -1;
      cfg.free_leaves[num_free++] = right;
      cfg.freed[num_freed++] = right;
      if (m->kid_left >= 0) {
        cfg.stack[sp++] = m->kid_left;
      }
      if (m->kid_right >= 0) {
        cfg.stack[sp++] = m->kid_right;
      }
    }
    cfg.leaf_parent[top.left_leaf] = top.parent;
  }
  // ---- 3. the selected frontier candidates' records, in pop order
  const int num_order = st->num_order;
  int num_allocated = st->num_allocated;
  int nr = num_records;
  int num_apply = 0;
  int error = 0;
  for (int t = 0; t < num_order; ++t) {
    const int id = cfg.order_id[t];
    if (id >= 0) {
      continue;
    }
    const int f = ~id;
    const int leaf = cfg.frontier_leaf[f];
    const int right = num_free > 0 ? cfg.free_leaves[--num_free] : num_allocated++;
    if (right >= cfg.num_leaves) {
      error = 1;
      break;
    }
    if (nr >= cfg.record_capacity) {
      error = 2;
      break;
    }
    const int parent = cfg.leaf_parent[leaf];
    CUDASelectiveR3Record rec;
    rec.parent = parent;
    rec.left_leaf = leaf;
    rec.right_leaf = right;
    rec.kid_left = -1;
    rec.kid_right = -1;
    rec.dead = 0;
    rec.gain = cfg.frontier_gain[f];
    records[nr] = rec;
    if (parent >= 0) {
      if (records[parent].left_leaf == leaf) {
        records[parent].kid_left = nr;
      } else {
        records[parent].kid_right = nr;
      }
    }
    cfg.leaf_parent[leaf] = nr;
    cfg.leaf_parent[right] = nr;
    cfg.apply_right[num_apply++] = right;
    ++nr;
  }
  st->num_records = nr;
  st->num_free = num_free;
  st->num_allocated = num_allocated;
  st->num_collapsed = num_collapsed;
  st->num_freed = num_freed;
  st->num_apply = num_apply;
  st->error = error;
}


// ---- the parallel R3 (P8c) ----

constexpr int kR3ParThreads = 512;
constexpr int kR3NotAvailable = -2;      // a candidate no live record holds (never pushed by the host)
constexpr int kR3Unpopped = INT_MAX / 2;  // a merged position past every budget
constexpr int kR3CountRanks = 1024;       // candidates ranked by counting up to this many

// a gain as an order-preserving integer (FP64 runs at 1/64 rate on consumer GPUs): larger gain, larger key; -0.0 and
// +0.0 map to one key, so equal keys are exactly the host's == on the doubles (NaN never occurs on a valid split)
__device__ __forceinline__ uint64_t R3Key(const double gain) {
  uint64_t b = static_cast<uint64_t>(__double_as_longlong(gain));
  b = (b << 1) == 0 ? 0 : b;
  return (b >> 63) ? ~b : b | (1ull << 63);
}

struct R3KeyEntry {  // the resume's heap entry: (key desc, tie key asc)
  uint64_t gain;
  uint32_t key;
  int id;
};

__device__ __forceinline__ bool R3Better(const R3KeyEntry& a, const R3KeyEntry& b) {
  return a.gain > b.gain || (a.gain == b.gain && a.key < b.key);
}

struct R3SortItem {
  uint64_t gain;
  int slot;  // INT_MAX: not available
  int k;
};

// the merge's order among candidates: slot, then gain desc, then index
__device__ __forceinline__ bool R3Before(const R3SortItem& a, const R3SortItem& b) {
  return a.slot < b.slot || (a.slot == b.slot && (a.gain > b.gain || (a.gain == b.gain && a.k < b.k)));
}

// in-place inclusive scan of a[0, len) (contiguous chunks per thread); returns the total
template <int kThreads>
__device__ int R3ScanInclusive(int* a, const int len, int* warp_scratch) {
  constexpr int kWarps = kThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int per = (len + kThreads - 1) / kThreads;
  const int lo = min(len, static_cast<int>(threadIdx.x) * per);
  const int hi = min(len, lo + per);
  int sum = 0;
  for (int i = lo; i < hi; ++i) {
    sum += a[i];
  }
  int x = sum;
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    x += lane >= o ? y : 0;
  }
  if (lane == 31) {
    warp_scratch[warp] = x;
  }
  __syncthreads();
  if (warp == 0) {
    int w = lane < kWarps ? warp_scratch[lane] : 0;
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, w, o);
      w += lane >= o ? y : 0;
    }
    if (lane < kWarps) {
      warp_scratch[lane] = w;
    }
  }
  __syncthreads();
  int run = x - sum + (warp > 0 ? warp_scratch[warp - 1] : 0);
  for (int i = lo; i < hi; ++i) {
    run += a[i];
    a[i] = run;
  }
  const int total = warp_scratch[kWarps - 1];
  __syncthreads();
  return total;
}

__device__ __forceinline__ int R3Pow2(const int n) {
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

// the min segment tree t[1, 2P) over v[0, n) (+inf past n)
template <int kThreads>
__device__ int R3BuildTree(uint64_t* t, const uint64_t* v, const int n) {
  const int P = R3Pow2(max(n, 1));
  for (int i = threadIdx.x; i < P; i += kThreads) {
    t[P + i] = i < n ? v[i] : ~0ull;
  }
  __syncthreads();
  for (int s = P >> 1; s >= 1; s >>= 1) {
    for (int i = s + threadIdx.x; i < 2 * s; i += kThreads) {
      const uint64_t a = t[2 * i];
      const uint64_t b = t[2 * i + 1];
      t[i] = b < a ? b : a;
    }
    __syncthreads();
  }
  return P;
}

// the first j in [l, n) with v[j] < x (kOrEqual: <= x), n if none
template <bool kOrEqual>
__device__ __forceinline__ int R3FirstBelow(const uint64_t* t, const int P, const int n, const int l, const uint64_t x) {
  if (l >= n) {
    return n;
  }
  const auto hit = [x](const uint64_t v) { return kOrEqual ? v <= x : v < x; };
  int i = l + P;
  while (!hit(t[i])) {
    while (i & 1) {
      i >>= 1;
    }
    if (i == 0) {
      return n;
    }
    ++i;
  }
  while (i < P) {
    i <<= 1;
    i += hit(t[i]) ? 0 : 1;
  }
  return min(i - P, n);
}

// Candidate arrays (cfg.cand, leaf_capacity each): merged position, creation index, parent (2 * position + right, -1
// the root, kR3NotAvailable), record id (rebuild). Merged-order arrays (after them, 2 x leaf_capacity each): parent,
// classic leaf, the two kids, the final position; their gains in cfg.cand_gain after the candidates'.
struct R3Arrays {
  int* c_pos;
  int* c_ci;
  int* c_parent;
  int* c_ident;
  int* m_parent;
  int* m_classic;
  int* m_kid;
  int* m_final;
  uint64_t* c_gain;  // gain keys (R3Key)
  uint64_t* m_gain;
  __device__ explicit R3Arrays(const CUDASelectiveR3Config& cfg) {
    const int cap = cfg.leaf_capacity;
    c_pos = cfg.cand;
    c_ci = cfg.cand + cap;
    c_parent = cfg.cand + 2 * cap;
    c_ident = cfg.cand + 3 * cap;
    m_parent = cfg.cand + 4 * cap;
    m_classic = cfg.cand + 6 * cap;
    m_kid = cfg.cand + 8 * cap;
    m_final = cfg.cand + 12 * cap;
    c_gain = reinterpret_cast<uint64_t*>(cfg.cand_gain);
    m_gain = reinterpret_cast<uint64_t*>(cfg.cand_gain) + cap;
  }
};

// The merge (Lemma 1) of m candidates into the order cur (n entries) under the fixed order (gain desc; at equal gain
// the old entries by position, then the candidates by index): np[j] = the old entry j's merged position, c_pos = a
// candidate's (slot + rank: slot = the first entry after its parent with a lower gain, rank = its index in (slot, gain
// desc, index)), c_ci = its rank. Returns the number of available candidates.
template <int kThreads>
__device__ int R3Merge(const CUDASelectiveR3Config& cfg, const R3Arrays& a, const int* cur_id, const int n, const int m,
                       uint64_t* area, int* np, int* scratch) {
  const int tid = threadIdx.x;
  for (int i = tid; i < n; i += kThreads) {
    a.m_gain[i] = R3Key(cfg.records[cur_id[i]].gain);  // staged for the tree
  }
  if (tid == 0) {
    scratch[0] = 0;
  }
  __syncthreads();
  const int P = R3BuildTree<kThreads>(area, a.m_gain, n);
  int available = 0;
  for (int k = tid; k < m; k += kThreads) {
    const int h = a.c_parent[k];
    available += h != kR3NotAvailable;
    a.c_pos[k] = h == kR3NotAvailable ? INT_MAX :
                 R3FirstBelow<false>(area, P, n, h < 0 ? 0 : (h >> 1) + 1, a.c_gain[k]);
  }
  atomicAdd(scratch, available);
  __syncthreads();
  // ranks in (slot, gain desc, index), the candidates staged in the area (the tree is not needed any more): by counting
  // for a few, else a block bitonic sort
  R3SortItem* item = reinterpret_cast<R3SortItem*>(area);
  const int N = m <= kR3CountRanks ? m : R3Pow2(m);
  for (int i = tid; i < N; i += kThreads) {
    item[i] = i < m ? R3SortItem{a.c_gain[i], a.c_pos[i], i} : R3SortItem{0, INT_MAX, INT_MAX};
    if (i < m) {
      a.c_ci[i] = 0;
    }
  }
  __syncthreads();
  if (m <= kR3CountRanks) {
    // every thread counts one slice of one candidate's predecessors
    const int parts = max(1, kThreads / max(m, 1));
    for (int w = tid; w < m * parts; w += kThreads) {
      const int k = w % m;
      const int part = w / m;
      const R3SortItem me = item[k];
      int rank = 0;
      for (int o = part * m / parts; o < (part + 1) * m / parts; ++o) {
        rank += R3Before(item[o], me);
      }
      atomicAdd(a.c_ci + k, rank);
    }
  } else {
    for (int size = 2; size <= N; size <<= 1) {
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        for (int i = tid; i < N / 2; i += kThreads) {
          const int lo = 2 * stride * (i / stride) + i % stride;
          const R3SortItem x = item[lo];
          const R3SortItem y = item[lo + stride];
          if (R3Before(y, x) == ((lo & size) == 0)) {
            item[lo] = y;
            item[lo + stride] = x;
          }
        }
        __syncthreads();
      }
    }
    for (int i = tid; i < m; i += kThreads) {
      a.c_ci[item[i].k] = i;
    }
  }
  __syncthreads();
  const int num_available = scratch[0];
  // old entry j moves to j + the candidates with slot <= j
  for (int j = tid; j <= n; j += kThreads) {
    np[j] = 0;
  }
  __syncthreads();
  for (int k = tid; k < m; k += kThreads) {
    if (a.c_pos[k] != INT_MAX) {
      atomicAdd(np + a.c_pos[k], 1);
    }
  }
  __syncthreads();
  R3ScanInclusive<kThreads>(np, n + 1, scratch);
  for (int j = tid; j < n; j += kThreads) {
    np[j] += j;
  }
  for (int k = tid; k < m; k += kThreads) {
    a.c_pos[k] = a.c_pos[k] == INT_MAX ? kR3Unpopped : a.c_pos[k] + a.c_ci[k];
  }
  __syncthreads();
  return num_available;
}

// a merged entry's classic leaf and parent from its old parent entry (h: 2 * old position + right, or -1)
__device__ __forceinline__ void R3NewLinks(const int h, const int* cur_classic, const int* np, int* classic,
                                          int* parent) {
  if (h < 0) {
    *classic = 0;
    *parent = -1;
    return;
  }
  const int pp = h >> 1;
  const int pc = cur_classic[pp];
  *classic = (h & 1) ? np[pp] + 1 : (pc == 0 ? 0 : np[pc - 1] + 1);
  *parent = 2 * np[pp] + (h & 1);
}

// The merged order M (total entries) is the host's run up to its first divergence d: the first step where an entry y
// with the popped entry's gain is available (parent popped, y not yet) and has the smaller tie key; every classic
// leaf involved up to d is exact. Ties M resolves the host's way are kept. From d on, best-first runs exactly (heap,
// tie keys) up to `limit` pops; np and c_pos are remapped to the final positions, c_ci recounted. Returns d (limit:
// none).
template <int kThreads>
__device__ int R3Resolve(const CUDASelectiveR3Config& cfg, const R3Arrays& a, const int* cur, const int n, const int m,
                         const int total, const int limit, uint64_t* area, int* np, int* scratch, int* s_flag) {
  const int tid = threadIdx.x;
  const int cap = cfg.leaf_capacity;
  for (int j = tid; j < n; j += kThreads) {
    int classic = 0;
    int parent = -1;
    R3NewLinks(cur[2 * cap + j], cur + cap, np, &classic, &parent);
    const int c = cur[cap + j];
    a.m_classic[np[j]] = c == 0 ? 0 : np[c - 1] + 1;
    a.m_parent[np[j]] = parent;
    a.m_gain[np[j]] = R3Key(cfg.records[cur[j]].gain);
  }
  for (int k = tid; k < m; k += kThreads) {
    if (a.c_pos[k] != kR3Unpopped) {
      R3NewLinks(a.c_parent[k], cur + cap, np, a.m_classic + a.c_pos[k], a.m_parent + a.c_pos[k]);
      a.m_gain[a.c_pos[k]] = a.c_gain[k];
    }
  }
  if (tid == 0) {
    s_flag[2] = limit;
    s_flag[3] = 0;
  }
  __syncthreads();
  const int H = min(total, limit);
  const int P = R3BuildTree<kThreads>(area, a.m_gain, H);
  for (int t = tid; t < total; t += kThreads) {
    const int lo = (a.m_parent[t] >> 1) + 1;  // -1 >> 1 = -1: the root, available from 0
    const int hi = min(t, H);
    if (lo >= hi) {
      continue;  // popped right after its parent
    }
    const uint64_t g = a.m_gain[t];
    const uint32_t key = SelectiveR3TieKey(a.m_classic[t]);
    for (int s = R3FirstBelow<true>(area, P, H, lo, g); s < hi; s = R3FirstBelow<true>(area, P, H, s + 1, g)) {
      if (key < SelectiveR3TieKey(a.m_classic[s])) {
        atomicMin(s_flag + 2, s);
        break;
      }
    }
  }
  __syncthreads();
  const int d = s_flag[2];
  if (d >= H) {
    return limit;
  }
  // ---- exact best-first from d: the heap holds the entries available at d (parent before d, own position >= d)
  R3KeyEntry* heap = reinterpret_cast<R3KeyEntry*>(area);
  for (int t = tid; t < total; t += kThreads) {
    a.m_kid[2 * t] = -1;
    a.m_kid[2 * t + 1] = -1;
    a.m_final[t] = t < d ? t : kR3Unpopped;
  }
  __syncthreads();
  for (int t = tid; t < total; t += kThreads) {
    const int h = a.m_parent[t];
    if (h >= 0) {
      a.m_kid[h] = t;
    }
    if (t >= d && (h >> 1) < d) {
      heap[atomicAdd(s_flag + 3, 1)] = {a.m_gain[t], SelectiveR3TieKey(a.m_classic[t]), t};
    }
  }
  __syncthreads();
  if (tid == 0) {
    int size = s_flag[3];
    for (int i = size / 2 - 1; i >= 0; --i) {
      R3SiftDown(heap, size, i, heap[i]);
    }
    for (int pos = d; pos < H && size > 0; ++pos) {
      const R3KeyEntry e = heap[0];
      const int classic = SelectiveR3KeyLeaf(e.key);
      a.m_final[e.id] = pos;
      R3KeyEntry kids[2];
      int num_kids = 0;
      for (int side = 0; side < 2; ++side) {
        const int kid = a.m_kid[2 * e.id + side];
        if (kid >= 0) {
          kids[num_kids++] = {a.m_gain[kid], SelectiveR3TieKey(side == 0 ? classic : pos + 1), kid};
        }
      }
      if (num_kids == 0) {
        --size;
      }
      if (size > 0) {
        R3SiftDown(heap, size, 0, num_kids > 0 ? kids[0] : heap[size]);
      }
      if (num_kids == 2) {
        R3SiftUp(heap, size++, kids[1]);
      }
    }
  }
  __syncthreads();
  int* flag = reinterpret_cast<int*>(area);  // the candidates' final positions, counted for their creation index
  for (int t = tid; t < H; t += kThreads) {
    flag[t] = 0;
  }
  for (int j = tid; j < n; j += kThreads) {
    np[j] = a.m_final[np[j]];
  }
  __syncthreads();
  for (int k = tid; k < m; k += kThreads) {
    if (a.c_pos[k] != kR3Unpopped) {
      a.c_pos[k] = a.m_final[a.c_pos[k]];
      if (a.c_pos[k] < H) {
        flag[a.c_pos[k]] = 1;
      }
    }
  }
  __syncthreads();
  R3ScanInclusive<kThreads>(flag, H, scratch);
  for (int k = tid; k < m; k += kThreads) {
    if (a.c_pos[k] < H) {
      a.c_ci[k] = flag[a.c_pos[k]] - 1;
    }
  }
  __syncthreads();
  return d;
}

// writes the order's first `limit` entries to nxt (and the level outputs when level)
template <int kThreads>
__device__ void R3WriteOrder(const CUDASelectiveR3Config& cfg, const R3Arrays& a, const int* cur, int* nxt,
                             const int n, const int m, const int limit, const bool level, const int* np) {
  const int cap = cfg.leaf_capacity;
  const int num_records = cfg.state->num_records;
  for (int j = threadIdx.x; j < n; j += kThreads) {
    const int t = np[j];
    if (t < limit) {
      const int c = cur[cap + j];
      const int classic = c == 0 ? 0 : np[c - 1] + 1;
      const int h = cur[2 * cap + j];
      nxt[t] = cur[j];
      nxt[cap + t] = classic;
      nxt[2 * cap + t] = h < 0 ? -1 : 2 * np[h >> 1] + (h & 1);
      if (level) {
        cfg.order_id[t] = cur[j];
        cfg.order_classic[t] = classic;
      }
    }
  }
  for (int k = threadIdx.x; k < m; k += kThreads) {
    const int t = a.c_pos[k];
    if (t >= limit) {
      continue;
    }
    int classic = 0;
    int parent = -1;
    R3NewLinks(a.c_parent[k], cur + cap, np, &classic, &parent);
    nxt[t] = level ? num_records + a.c_ci[k] : a.c_ident[k];
    nxt[cap + t] = classic;
    nxt[2 * cap + t] = parent;
    if (level) {
      cfg.order_id[t] = ~k;
      cfg.order_classic[t] = classic;
    }
  }
}

// the frontier: the valid children in child order (SelectiveDiscoverFrontier)
template <int kThreads>
__device__ int R3Frontier(const CUDASelectiveR3Config& cfg, int* warp_counts) {
  return R3BlockCompact<kThreads>(cfg.state->num_children, warp_counts,
    [&](const int i) { return cfg.leaf_best[cfg.child_leaves[i]].is_valid; },
    [&](const int i, const int f) {
      const int leaf = cfg.child_leaves[i];
      cfg.frontier_leaf[f] = leaf;
      cfg.frontier_gain[f] = cfg.leaf_best[leaf].gain;
    });
}

// the live records at depth d as merge candidates (their parents' positions from rec_pos); returns the count
template <int kThreads>
__device__ int R3DepthCandidates(const CUDASelectiveR3Config& cfg, const int d, int* warp_counts) {
  const CUDASelectiveR3Record* records = cfg.records;
  const int cap = cfg.leaf_capacity;
  return R3BlockCompact<kThreads>(cfg.state->num_records, warp_counts,
    [&](const int r) {
      const CUDASelectiveR3Record& rec = records[r];
      return !rec.dead && (rec.parent < 0 ? d == 0 : d > 0 && cfg.rec_round[rec.parent] == d - 1);
    },
    [&](const int r, const int k) {
      const CUDASelectiveR3Record& rec = records[r];
      cfg.rec_round[r] = d;
      reinterpret_cast<uint64_t*>(cfg.cand_gain)[k] = R3Key(rec.gain);
      cfg.cand[3 * cap + k] = r;
      cfg.cand[2 * cap + k] = rec.parent < 0 ? -1 : 2 * cfg.rec_pos[rec.parent] +
                                                   (records[rec.parent].left_leaf == rec.left_leaf ? 0 : 1);
    });
}

template <bool kSharedArea>
__global__ void __launch_bounds__(kR3ParThreads) SelectiveR3ParallelKernel(const CUDASelectiveR3Config cfg) {
  constexpr int kThreads = kR3ParThreads;
  extern __shared__ __align__(16) unsigned char r3_smem[];
  __shared__ int warp_counts[kThreads / 32 + 1];
  __shared__ int scratch[kThreads / 32 + 1];
  __shared__ int s_flag[4];
  const int tid = threadIdx.x;
  const int cap = cfg.leaf_capacity;
  const R3Arrays a(cfg);
  uint64_t* area = kSharedArea ? reinterpret_cast<uint64_t*>(r3_smem) : static_cast<uint64_t*>(cfg.par_global);
  int* np = reinterpret_cast<int*>(r3_smem + (kSharedArea ? R3ParallelAreaBytes(cfg.num_leaves) : 0));
  CUDASelectiveR3State* st = cfg.state;
  CUDASelectiveR3Persist* ps = cfg.persist;
  CUDASelectiveR3Record* records = cfg.records;
  const int num_records = st->num_records;
  const int budget = cfg.budget;
  const int num_frontier = R3Frontier<kThreads>(cfg, warp_counts);
  int parity = ps->parity;
  int n = ps->n;
  int rounds = 0;
  int resumes = 0;
  int error = 0;
  if (cfg.rebuild || !ps->valid) {
    // ---- the order from the records: the live records merged depth by depth from the root (Lemma 1 per depth)
    n = 0;
    for (int r = tid; r < num_records; r += kThreads) {
      cfg.rec_round[r] = -1;
    }
    __syncthreads();
    for (int d = 0;; ++d) {
      const int* cur = cfg.order_buf + parity * 3 * cap;
      int* nxt = cfg.order_buf + (parity ^ 1) * 3 * cap;
      const int m = R3DepthCandidates<kThreads>(cfg, d, warp_counts);
      if (m == 0) {
        break;
      }
      if (n + m > budget) {
        error = 3;  // more live records than the budget: not a state the selection produces
        break;
      }
      const int total = n + R3Merge<kThreads>(cfg, a, cur, n, m, area, np, scratch);
      resumes += R3Resolve<kThreads>(cfg, a, cur, n, m, total, total, area, np, scratch, s_flag) < total;
      R3WriteOrder<kThreads>(cfg, a, cur, nxt, n, m, total, false, np);
      __syncthreads();
      n = total;
      parity ^= 1;
      for (int t = tid; t < n; t += kThreads) {
        cfg.rec_pos[nxt[t]] = t;
      }
      __syncthreads();
      ++rounds;
    }
  }
  const int* cur = cfg.order_buf + parity * 3 * cap;
  int* nxt = cfg.order_buf + (parity ^ 1) * 3 * cap;
  // ---- the level's candidates: the live record holding the leaf with no child record there is the parent
  for (int i = tid; i < cap; i += kThreads) {
    np[i] = -1;
  }
  __syncthreads();
  for (int j = tid; j < n; j += kThreads) {
    const CUDASelectiveR3Record& rec = records[cur[j]];
    if (rec.kid_left < 0) {
      np[rec.left_leaf] = 2 * j;
    }
    if (rec.kid_right < 0) {
      np[rec.right_leaf] = 2 * j + 1;
    }
  }
  __syncthreads();
  for (int k = tid; k < num_frontier; k += kThreads) {
    const int holder = np[cfg.frontier_leaf[k]];
    // no records: the root candidate (frontier 0) only
    a.c_parent[k] = num_records == 0 ? (k == 0 ? -1 : kR3NotAvailable) : (holder >= 0 ? holder : kR3NotAvailable);
    a.c_gain[k] = R3Key(cfg.frontier_gain[k]);
  }
  __syncthreads();
  const int total = n + R3Merge<kThreads>(cfg, a, cur, n, num_frontier, area, np, scratch);
  const int resume_at = R3Resolve<kThreads>(cfg, a, cur, n, num_frontier, total, budget, area, np, scratch, s_flag);
  resumes += resume_at < budget;
  // ---- the order truncated at the budget (Lemma 2), its classic numbers, the level's outputs
  R3WriteOrder<kThreads>(cfg, a, cur, nxt, n, num_frontier, budget, true, np);
  const int num_order = min(budget, total);
  int created = 0;
  for (int k = tid; k < num_frontier; k += kThreads) {
    created += a.c_pos[k] < budget;
  }
  // ---- the collapse of the displaced entries (final position >= budget): the tops (parent selected) in ascending
  // record id, each subtree in the host's stack order (right subtree first: preorder offsets from the subtree sizes)
  int* size = reinterpret_cast<int*>(area);
  int* rkid = size + n;
  int* tops = rkid + n;
  int* top_size = tops + n;
  for (int j = tid; j < n; j += kThreads) {
    size[j] = 0;
    rkid[j] = -1;
  }
  if (tid == 0) {
    s_flag[0] = 0;
    s_flag[1] = 0;
  }
  __syncthreads();
  atomicAdd(s_flag + 1, created);
  const auto is_top = [&](const int j) {
    const int h = cur[2 * cap + j];
    return h < 0 || np[h >> 1] < budget;
  };
  for (int j = tid; j < n; j += kThreads) {
    if (np[j] >= budget) {
      if (is_top(j)) {
        tops[atomicAdd(s_flag, 1)] = j;
      } else if (cur[2 * cap + j] & 1) {
        rkid[cur[2 * cap + j] >> 1] = j;
      }
    }
  }
  __syncthreads();
  const int num_tops = s_flag[0];
  const int num_created = s_flag[1];
  const int num_free = st->num_free;
  int num_freed = 0;
  if (num_tops > 0) {  // (uniform)
    for (int j = tid; j < n; j += kThreads) {
      if (np[j] >= budget) {
        int y = j;
        while (true) {
          atomicAdd(size + y, 1);
          if (is_top(y)) {
            break;
          }
          y = cur[2 * cap + y] >> 1;
        }
      }
    }
    __syncthreads();
    for (int i = tid; i < num_tops; i += kThreads) {
      const int id = cur[tops[i]];
      int rank = 0;
      for (int o = 0; o < num_tops; ++o) {
        rank += cur[tops[o]] < id;
      }
      top_size[rank] = size[tops[i]];
      cfg.collapsed[rank] = id;
    }
    __syncthreads();
    num_freed = R3ScanInclusive<kThreads>(top_size, num_tops, scratch);
    for (int i = tid; i < num_tops; i += kThreads) {
      const int y = tops[i];
      int rank = 0;
      for (int o = 0; o < num_tops; ++o) {
        rank += cur[tops[o]] < cur[y];
      }
      size[y] = top_size[rank] - size[y];  // a top's size is not read again: its first freed index instead
    }
    __syncthreads();
    for (int j = tid; j < n; j += kThreads) {
      if (np[j] < budget) {
        continue;
      }
      int pre = 0;
      int y = j;
      while (!is_top(y)) {
        const int h = cur[2 * cap + y];
        const int p = h >> 1;
        pre += 1 + ((h & 1) == 0 && rkid[p] >= 0 ? size[rkid[p]] : 0);
        y = p;
      }
      const int index = size[y] + pre;
      const int id = cur[j];
      const CUDASelectiveR3Record rec = records[id];
      cfg.freed[index] = rec.right_leaf;
      cfg.free_leaves[num_free + index] = rec.right_leaf;
      cfg.leaf_parent[rec.right_leaf] = -1;
      records[id].dead = 1;
      if (y == j) {
        if (rec.parent >= 0) {
          CUDASelectiveR3Record* p = records + rec.parent;
          if (p->kid_left == id) {
            p->kid_left = -1;
          } else if (p->kid_right == id) {
            p->kid_right = -1;
          }
        }
        cfg.leaf_parent[rec.left_leaf] = rec.parent;
      }
    }
    __syncthreads();
  }
  // ---- the records of the selected candidates in pop order: the k-th takes the k-th leaf off the free list
  const int nf = num_free + num_freed;
  const bool fits = num_records + num_created <= cfg.record_capacity;
  error = error != 0 ? error : (fits ? 0 : 2);
  for (int k = tid; k < num_frontier && fits; k += kThreads) {
    if (a.c_pos[k] >= budget) {
      continue;
    }
    const int ci = a.c_ci[k];
    const int right = ci < nf ? cfg.free_leaves[nf - 1 - ci] : st->num_allocated + (ci - nf);
    if (right >= cfg.num_leaves) {
      error = 1;
      continue;
    }
    const int h = a.c_parent[k];
    const int parent = h < 0 ? -1 : cur[h >> 1];
    const int leaf = cfg.frontier_leaf[k];
    const int id = num_records + ci;
    records[id] = {parent, leaf, right, -1, -1, 0, cfg.frontier_gain[k]};
    if (parent >= 0) {
      if (h & 1) {
        records[parent].kid_right = id;
      } else {
        records[parent].kid_left = id;
      }
    }
    cfg.leaf_parent[leaf] = id;
    cfg.leaf_parent[right] = id;
    cfg.apply_right[ci] = right;
  }
  error = __syncthreads_or(error == 1) ? 1 : error;
  if (tid == 0) {
    st->num_frontier = num_frontier;
    st->num_order = num_order;
    st->num_collapsed = num_tops;
    st->num_freed = num_freed;
    st->num_apply = num_created;
    st->num_records = num_records + (fits ? num_created : 0);
    st->num_free = max(nf - num_created, 0);
    st->num_allocated += max(num_created - nf, 0);
    st->error = error;
    st->resumes = resumes;
    st->resume_at = resume_at < budget ? resume_at : -1;
    st->rounds = rounds;
    ps->n = num_order;
    ps->parity = parity ^ 1;
    ps->valid = error == 0;
  }
}

// the device's opt-in shared-memory limit (less the kernels' static shared memory and head room), and the dynamic
// shared-memory attribute of `kernel` raised to it, once per device (the attribute is per device)
template <typename Kernel>
int R3SharedLimit(Kernel kernel, std::unordered_map<int, int>* by_device) {
  static std::mutex mutex;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  auto it = by_device->find(device);
  if (it == by_device->end()) {
    int bytes = 0;
    CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    bytes -= 1024;
    if (bytes > 48 * 1024) {
      CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    }
    it = by_device->emplace(device, bytes > 48 * 1024 ? bytes : 47 * 1024).first;
  }
  return it->second;
}

}  // namespace

void LaunchSelectiveR3(const CUDASelectiveR3Config& config, cudaStream_t stream, const bool shared_heap) {
  static std::unordered_map<int, int> limit;
  const int max_shared_bytes = R3SharedLimit(SelectiveR3Kernel<true>, &limit);
  // the heap holds one entry per classic leaf: at most num_leaves (budget + 1)
  const size_t heap_bytes = static_cast<size_t>(config.num_leaves) * kSelectiveR3HeapEntryBytes;
  if (shared_heap && heap_bytes <= static_cast<size_t>(max_shared_bytes)) {
    SelectiveR3Kernel<true><<<1, kR3Threads, heap_bytes, stream>>>(config);
  } else {
    CHECK(config.heap_global != nullptr);
    SelectiveR3Kernel<false><<<1, kR3Threads, 0, stream>>>(config);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

bool LaunchSelectiveR3Parallel(const CUDASelectiveR3Config& config, cudaStream_t stream, const bool shared_area) {
  static std::unordered_map<int, int> limit_shared;
  static std::unordered_map<int, int> limit_global;
  const size_t area = R3ParallelAreaBytes(config.num_leaves);
  const size_t rest = static_cast<size_t>(config.leaf_capacity) * sizeof(int);
  if (shared_area && area + rest <= static_cast<size_t>(R3SharedLimit(SelectiveR3ParallelKernel<true>,
                                                                       &limit_shared))) {
    SelectiveR3ParallelKernel<true><<<1, kR3ParThreads, area + rest, stream>>>(config);
  } else if (rest <= static_cast<size_t>(R3SharedLimit(SelectiveR3ParallelKernel<false>, &limit_global))) {
    CHECK(config.par_global != nullptr);
    SelectiveR3ParallelKernel<false><<<1, kR3ParThreads, rest, stream>>>(config);
  } else {
    // the per-leaf index does not fit shared memory: the sequential kernel, and the order rebuilt next time
    LaunchSelectiveR3(config, stream, true);
    CUDASUCCESS_OR_FATAL(cudaMemsetAsync(&config.persist->valid, 0, sizeof(int), stream));
    return false;
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
  return true;
}

}  // namespace Falcata

#endif  // USE_CUDA
