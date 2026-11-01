/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief R3, one selective level on the device and, on the route, its apply (see cuda_selective_r3.hpp).
 */

#ifdef USE_CUDA

#include "cuda_selective_r3.hpp"

#include <Falcata/cuda/cuda_utils.hu>

#include <algorithm>
#include <climits>
#include <mutex>
#include <unordered_map>

#include "cuda_device_levels.cuh"

namespace Falcata {

namespace {

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

// the min segment tree t[1, 2P) over v[0, n) (+inf past n). tie (the resolve): beside it in tk[1, 2P) the largest tie key
// among the entries of a node that hold the node's min (SelectiveR3TieKey of classic[i])
template <int kThreads>
__device__ int R3BuildTree(uint64_t* t, const uint64_t* v, const int n, uint32_t* tk = nullptr,
                           const int* classic = nullptr) {
  const int P = R3Pow2(max(n, 1));
  for (int i = threadIdx.x; i < P; i += kThreads) {
    t[P + i] = i < n ? v[i] : ~0ull;
    if (tk != nullptr) {
      tk[P + i] = i < n ? SelectiveR3TieKey(classic[i]) : 0u;
    }
  }
  __syncthreads();
  for (int s = P >> 1; s >= 1; s >>= 1) {
    for (int i = s + threadIdx.x; i < 2 * s; i += kThreads) {
      const uint64_t a = t[2 * i];
      const uint64_t b = t[2 * i + 1];
      t[i] = b < a ? b : a;
      if (tk != nullptr) {
        tk[i] = a < b ? tk[2 * i] : (b < a ? tk[2 * i + 1] : max(tk[2 * i], tk[2 * i + 1]));
      }
    }
    __syncthreads();
  }
  return P;
}

// the resolve's divergence of an entry of gain g and tie key key, waiting on positions [l, r): the first position whose
// entry has gain g and a larger tie key, r if none. Every entry popped on [l, r) has gain >= g (it beat the waiting
// entry under the merge's order), so that is the first j >= l whose (gain, tie key) is below (g, key) in (gain asc, tie
// key desc), if it lies before r; a node holds such an entry iff its min is below g, or is g with a larger tie key at
// the min. O(log n): up from l to the first such node, then down to its first such leaf (R3FirstBelow's walk).
__device__ __forceinline__ int R3FirstTieAbove(const uint64_t* t, const uint32_t* tk, const int P, const int l,
                                               const int r, const uint64_t g, const uint32_t key) {
  const auto hit = [&](const int node) { return t[node] < g || (t[node] == g && tk[node] > key); };
  int i = l + P;
  while (!hit(i)) {
    while (i & 1) {
      i >>= 1;
    }
    if (i == 0) {
      return r;
    }
    ++i;
  }
  while (i < P) {
    i <<= 1;
    i += hit(i) ? 0 : 1;
  }
  return min(i - P, r);
}

// the first j in [l, n) with v[j] < x, n if none
__device__ __forceinline__ int R3FirstBelow(const uint64_t* t, const int P, const int n, const int l, const uint64_t x) {
  if (l >= n) {
    return n;
  }
  const auto hit = [x](const uint64_t v) { return v < x; };
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
// classic leaf, the two kids, the final position, then the resolve's tie-key tree (4 x leaf_capacity); their gains in
// cfg.cand_gain after the candidates'.
struct R3Arrays {
  int* c_pos;
  int* c_ci;
  int* c_parent;
  int* c_ident;
  int* m_parent;
  int* m_classic;
  int* m_kid;
  int* m_final;
  uint32_t* m_tiekey;
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
    m_tiekey = reinterpret_cast<uint32_t*>(cfg.cand + 14 * cap);
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
                 R3FirstBelow(area, P, n, h < 0 ? 0 : (h >> 1) + 1, a.c_gain[k]);
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
  const int P = R3BuildTree<kThreads>(area, a.m_gain, H, a.m_tiekey, a.m_classic);
  for (int t = tid; t < total; t += kThreads) {
    const int lo = (a.m_parent[t] >> 1) + 1;  // -1 >> 1 = -1: the root, available from 0
    const int hi = min(t, H);
    if (lo >= hi) {
      continue;  // popped right after its parent
    }
    const int s = R3FirstTieAbove(area, a.m_tiekey, P, lo, hi, a.m_gain[t], SelectiveR3TieKey(a.m_classic[t]));
    if (s < hi) {
      atomicMin(s_flag + 2, s);
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

// the frontier: the valid children in child order (SelectiveDiscoverFrontier). On the route the children are the last
// level's pairs' leaves, smaller then larger (the host loop's child list from FinishLevelBookkeeping's pairs)
template <int kThreads>
__device__ int R3Frontier(const CUDASelectiveR3Config& cfg, int* warp_counts) {
  const int nc = cfg.state->num_children;
  const auto child = [&](const int i) {
    if (nc >= 0) {
      return cfg.child_leaves[i];
    }
    const CUDAHybridPairDescriptor& pair = cfg.pair_descs[i >> 1];
    return (i & 1) ? pair.larger_leaf_index : pair.smaller_leaf_index;
  };
  return R3BlockCompact<kThreads>(nc >= 0 ? nc : 2 * cfg.dl_state->live_pairs, warp_counts,
    [&](const int i) { return cfg.leaf_best[child(i)].is_valid; },
    [&](const int i, const int f) {
      const int leaf = child(i);
      cfg.frontier_leaf[f] = leaf;
      cfg.frontier_gain[f] = cfg.leaf_best[leaf].gain;
    });
}

// the live records at depth d as merge candidates (their parents' positions from rec_pos); returns the count
template <int kThreads>
__device__ int R3DepthCandidates(const CUDASelectiveR3Config& cfg, const int d, int* warp_counts, int* nan_flag) {
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
      if (isnan(rec.gain)) {
        *nan_flag = 1;
      }
      cfg.cand[3 * cap + k] = r;
      cfg.cand[2 * cap + k] = rec.parent < 0 ? -1 : 2 * cfg.rec_pos[rec.parent] +
                                                   (records[rec.parent].left_leaf == rec.left_leaf ? 0 : 1);
    });
}

// the route: a stopped level applies nothing and its children are not searched (the device levels' stop)
__device__ __forceinline__ void R3StopDeviceLevel(CUDADeviceLevelState* dls) {
  dls->done = 1;
  dls->live_splits = 0;
  dls->live_num_splits = 0;
  dls->live_cat_splits = 0;
  dls->total_flat_blocks = 0;
  dls->num_gaps = 0;
  dls->gap_flat_blocks = 0;
}

// The route's apply of a level whose records exist, the device half of RunSelectiveLevel's collapse and record creation
// and of ApplyLevelBatchedSelective: the collapsed nodes revert to leaves with their pre-split windows (the host's
// leaf_num_data_ / leaf_data_start_ resets) and the freed leaves hold no rows (the terminal-region scan below covers
// the other leaves allocated before the level); each created record's payload, the best-split record as the host's
// readback copies it and the leaf's window; one apply descriptor per created record in pop order (SplitLevelBatched's,
// DeviceLevelSplitDescriptor), the terminal-region gaps (one per live leaf with rows the level does not split: they
// copy the rows SplitLevelBatched's gaps between the split windows copy) and the counts the apply reads. A level that
// created nothing completes the tree. mark: leaf_capacity ints of shared memory.
template <int kThreads>
__device__ void R3ApplyLevel(const CUDASelectiveR3Config& cfg, const int base, const int created, const int num_tops,
                             const int num_freed, const int gap_leaves, int* mark, int* scratch, int* warp_counts) {
  const int tid = threadIdx.x;
  CUDASelectiveR3State* st = cfg.state;
  CUDADeviceLevelState* dls = cfg.dl_state;
  if (created == 0) {
    if (tid == 0) {
      st->done = 1;
      R3StopDeviceLevel(dls);
    }
    return;
  }
  const CUDADeviceLevelConfig& dc = *cfg.dl_cfg;
  for (int i = tid; i < num_tops; i += kThreads) {
    const int id = cfg.collapsed[i];
    const int leaf = cfg.records[id].left_leaf;
    cfg.leaf_data_start[leaf] = cfg.payload[id].data_start;
    cfg.leaf_num_data[leaf] = cfg.payload[id].num_data;
  }
  for (int i = tid; i < num_freed; i += kThreads) {
    cfg.leaf_num_data[cfg.freed[i]] = 0;
  }
  for (int leaf = tid; leaf < gap_leaves; leaf += kThreads) {
    mark[leaf] = 0;
  }
  __syncthreads();
  int* blocks = cfg.cand + 12 * cfg.leaf_capacity;  // (the merge's m_final, free again)
  int* gap_leaf = blocks + cfg.leaf_capacity;
  static_assert(sizeof(CUDASplitInfo) % sizeof(uint64_t) == 0, "CUDASplitInfo is copied as 8-byte words");
  constexpr int kInfoWords = static_cast<int>(sizeof(CUDASplitInfo) / sizeof(uint64_t));
  for (int k = tid; k < created; k += kThreads) {
    const int leaf = cfg.apply_leaf[k];
    mark[leaf] = 1;
    const CUDASplitInfo* info = cfg.leaf_best + leaf;
    CUDASelectiveR3Payload* p = cfg.payload + base + k;
    const uint64_t* src = reinterpret_cast<const uint64_t*>(info);
    uint64_t* dst = reinterpret_cast<uint64_t*>(&p->info);
    for (int w = 0; w < kInfoWords; ++w) {
      dst[w] = src[w];
    }
    const data_size_t start = cfg.leaf_data_start[leaf];
    const data_size_t num = cfg.leaf_num_data[leaf];
    p->data_start = start;
    p->num_data = num;
    const CUDAHybridApplyDescriptor d = DeviceLevelSplitDescriptor(dc, info, leaf, cfg.apply_right[k], k, start, num);
    dc.apply_descs[k] = d;
    blocks[k] = d.num_blocks;
  }
  __syncthreads();
  const int total_flat_blocks = R3ScanInclusive<kThreads>(blocks, created, scratch);
  for (int k = tid; k < created; k += kThreads) {
    CUDAHybridApplyDescriptor& d = dc.apply_descs[k];
    d.flat_block_start = blocks[k] - d.num_blocks;
    d.block_offset_start = d.flat_block_start + k;
  }
  const int num_gaps = R3BlockCompact<kThreads>(gap_leaves, warp_counts,
    [&](const int leaf) { return mark[leaf] == 0 && cfg.leaf_num_data[leaf] > 0; },
    [&](const int leaf, const int g) { gap_leaf[g] = leaf; });
  constexpr int kChunk = SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
  __syncthreads();  // (blocks: the splits' offsets are read)
  for (int g = tid; g < num_gaps; g += kThreads) {
    blocks[g] = (cfg.leaf_num_data[gap_leaf[g]] + kChunk - 1) / kChunk;
  }
  __syncthreads();
  const int gap_flat_blocks = R3ScanInclusive<kThreads>(blocks, num_gaps, scratch);
  for (int g = tid; g < num_gaps; g += kThreads) {
    const int leaf = gap_leaf[g];
    const data_size_t num = cfg.leaf_num_data[leaf];
    CUDAHybridApplyDescriptor gd;
    memset(&gd, 0, sizeof(CUDAHybridApplyDescriptor));
    gd.left_leaf_index = leaf;
    gd.leaf_data_start = cfg.leaf_data_start[leaf];
    gd.num_data_in_leaf = num;
    gd.num_blocks = (num + dc.gap_block_dim - 1) / dc.gap_block_dim;
    gd.flat_block_start = total_flat_blocks + blocks[g] - (num + kChunk - 1) / kChunk;
    dc.apply_descs[created + g] = gd;
  }
  if (tid == 0) {
    dls->done = 0;  // (the device levels stopped at the first binding level: the route continues the tree)
    dls->live_splits = created;
    dls->live_num_splits = created;
    dls->split_info_base = 0;  // (the apply's per-split counts: no route level reads them back)
    dls->live_cat_splits = 0;
    dls->cat_split_base = 0;
    dls->total_flat_blocks = total_flat_blocks;
    dls->num_gaps = num_gaps;
    dls->gap_flat_blocks = gap_flat_blocks;
    dls->num_leaves = st->num_allocated;
    dls->levels_applied += 1;
    st->levels += 1;
  }
}

template <bool kSharedArea>
__global__ void __launch_bounds__(kR3ParThreads) SelectiveR3ParallelKernel(const CUDASelectiveR3Config cfg) {
  constexpr int kThreads = kR3ParThreads;
  extern __shared__ __align__(16) unsigned char r3_smem[];
  __shared__ int warp_counts[kThreads / 32 + 1];
  __shared__ int scratch[kThreads / 32 + 1];
  __shared__ int s_flag[4];
  __shared__ int s_nan;
  __shared__ int s_displaced;
  const int tid = threadIdx.x;
  const int cap = cfg.leaf_capacity;
  if (cfg.state->done) {
    return;  // the route: an earlier level completed the tree or stopped it
  }
  const R3Arrays a(cfg);
  if (tid == 0) {
    s_nan = 0;
  }
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
      const int m = R3DepthCandidates<kThreads>(cfg, d, warp_counts, &s_nan);
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
    if (isnan(cfg.frontier_gain[k])) {
      s_nan = 1;  // R3Key would rank a NaN above +inf, the host's compares would not: reported, not decided
    }
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
    s_displaced = 0;
  }
  __syncthreads();
  atomicAdd(s_flag + 1, created);
  const auto is_top = [&](const int j) {
    const int h = cur[2 * cap + j];
    return h < 0 || np[h >> 1] < budget;
  };
  int displaced = 0;
  for (int j = tid; j < n; j += kThreads) {
    if (np[j] >= budget) {
      ++displaced;
      if (is_top(j)) {
        tops[atomicAdd(s_flag, 1)] = j;
      } else if (cur[2 * cap + j] & 1) {
        rkid[cur[2 * cap + j] >> 1] = j;
      }
    }
  }
  atomicAdd(&s_displaced, displaced);
  __syncthreads();
  const int num_tops = s_flag[0];
  const int num_created = s_flag[1];
  const int num_free = st->num_free;
  // ---- checked before the level changes anything (the route then hands the tree back to the host's loop): every
  // displaced record frees one leaf, the k-th record takes the k-th leaf off the free list, then fresh ones
  const int nf = num_free + s_displaced;
  error = error != 0 ? error : (s_nan != 0 ? 4 : (num_records + num_created > cfg.record_capacity ? 2 :
          (st->num_allocated + max(num_created - nf, 0) > cfg.num_leaves ? 1 : 0)));
  if (error != 0) {  // (uniform)
    if (tid == 0) {
      st->num_frontier = num_frontier;  // (num_apply keeps the last applied level's records: its children)
      st->num_order = 0;
      st->num_collapsed = 0;
      st->num_freed = 0;
      st->error = error;
      st->done = 1;
      ps->valid = 0;
      if (cfg.dl_state != nullptr) {
        R3StopDeviceLevel(cfg.dl_state);
      }
    }
    return;
  }
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
  for (int k = tid; k < num_frontier; k += kThreads) {
    if (a.c_pos[k] >= budget) {
      continue;
    }
    const int ci = a.c_ci[k];
    const int right = ci < nf ? cfg.free_leaves[nf - 1 - ci] : st->num_allocated + (ci - nf);
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
    cfg.apply_leaf[ci] = leaf;
  }
  const int gap_leaves = st->num_allocated;
  __syncthreads();
  if (tid == 0) {
    st->num_frontier = num_frontier;
    st->num_order = num_order;
    st->num_collapsed = num_tops;
    st->num_freed = num_freed;
    st->num_apply = num_created;
    st->num_records = num_records + num_created;
    st->num_free = max(nf - num_created, 0);
    st->num_allocated += max(num_created - nf, 0);
    st->gap_leaves = gap_leaves;
    st->created_total += num_created;
    st->freed_total += num_freed;
    st->error = 0;
    st->resumes = resumes;
    st->resume_at = resume_at < budget ? resume_at : -1;
    st->rounds = rounds;
    ps->n = num_order;
    ps->parity = parity ^ 1;
    ps->valid = 1;
  }
  if (cfg.dl_cfg != nullptr) {
    // the route: the level's device state and apply descriptors (np, the per-leaf index, is free again)
    R3ApplyLevel<kThreads>(cfg, num_records, num_created, num_tops, num_freed, gap_leaves, np, scratch, warp_counts);
  }
}

// The route's seed: the host's state after the hand-off level (packed, kSelectiveR3SeedHeader) into the arrays, the
// device levels' records' best-split records from their journal, then the hand-off level's apply
__global__ void __launch_bounds__(kR3ParThreads) SelectiveR3SeedKernel(const CUDASelectiveR3Config cfg, const int* seed,
                                                                      const CUDASplitInfo* journal, const int journaled) {
  constexpr int kThreads = kR3ParThreads;
  extern __shared__ int r3_mark[];
  __shared__ int warp_counts[kThreads / 32 + 1];
  __shared__ int scratch[kThreads / 32 + 1];
  const int tid = threadIdx.x;
  const int cap = cfg.leaf_capacity;
  const int n = seed[0];
  const int num_free = seed[1];
  const int created = seed[4];
  const int num_tops = seed[5];
  const int num_freed = seed[6];
  const int order = seed[7];
  const int* p = seed + kSelectiveR3SeedHeader;
  constexpr int kRecordInts = static_cast<int>(sizeof(CUDASelectiveR3Record) / sizeof(int));
  for (int i = tid; i < n * kRecordInts; i += kThreads) {
    reinterpret_cast<int*>(cfg.records)[i] = p[i];
  }
  p += n * kRecordInts;
  constexpr int kInfoWords = static_cast<int>(sizeof(CUDASplitInfo) / sizeof(uint64_t));
  for (int i = tid; i < n; i += kThreads) {
    cfg.payload[i].data_start = p[2 * i];
    cfg.payload[i].num_data = p[2 * i + 1];
    if (i < journaled) {
      for (int w = 0; w < kInfoWords; ++w) {
        reinterpret_cast<uint64_t*>(&cfg.payload[i].info)[w] = reinterpret_cast<const uint64_t*>(journal + i)[w];
      }
    }
  }
  p += 2 * n;
  for (int i = tid; i < cap; i += kThreads) {
    cfg.leaf_parent[i] = p[i];
  }
  p += cap;
  for (int i = tid; i < num_free; i += kThreads) {
    cfg.free_leaves[i] = p[i];
  }
  p += num_free;
  for (int i = tid; i < created; i += kThreads) {
    cfg.apply_leaf[i] = p[i];
    cfg.apply_right[i] = p[created + i];
  }
  p += 2 * created;
  for (int i = tid; i < num_tops; i += kThreads) {
    cfg.collapsed[i] = p[i];
  }
  p += num_tops;
  for (int i = tid; i < num_freed; i += kThreads) {
    cfg.freed[i] = p[i];
  }
  p += num_freed;
  for (int i = tid; i < order; i += kThreads) {
    cfg.order_buf[i] = p[3 * i];
    cfg.order_buf[cap + i] = p[3 * i + 1];
    cfg.order_buf[2 * cap + i] = p[3 * i + 2];
  }
  if (tid == 0) {
    CUDASelectiveR3State s = {};
    s.num_records = n;
    s.num_free = num_free;
    s.num_allocated = seed[2];
    s.num_children = -1;
    s.num_order = order;
    s.num_collapsed = num_tops;
    s.num_freed = num_freed;
    s.num_apply = created;
    s.resume_at = -1;
    s.gap_leaves = seed[3];
    *cfg.state = s;  // (the totals count the routed levels: the host counted the hand-off level)
    cfg.persist->n = order;
    cfg.persist->parity = 0;
    cfg.persist->valid = 1;
  }
  __syncthreads();
  R3ApplyLevel<kThreads>(cfg, n - created, created, num_tops, num_freed, seed[3], r3_mark, scratch, warp_counts);
}

__global__ void SelectiveR3FlagKernel(const CUDASelectiveR3State* st, int* flag) {
  flag[1] = st->levels;
  flag[2] = st->error;
  flag[3] = st->num_order;
  flag[4] = st->num_allocated;
  flag[5] = st->num_records;
  flag[0] = st->done ? 0 : 1;
}

__global__ void SelectiveR3GatherKernel(const CUDASelectiveR3Config cfg, CUDASelectiveR3Final* out) {
  constexpr int kInfoWords = static_cast<int>(sizeof(CUDASplitInfo) / sizeof(uint64_t));
  for (int t = threadIdx.x; t < cfg.state->num_order; t += blockDim.x) {
    const int id = cfg.order_id[t];
    for (int w = 0; w < kInfoWords; ++w) {
      reinterpret_cast<uint64_t*>(&out[t].info)[w] = reinterpret_cast<const uint64_t*>(&cfg.payload[id].info)[w];
    }
    out[t].left_leaf = cfg.records[id].left_leaf;
    out[t].right_leaf = cfg.records[id].right_leaf;
    out[t].classic = cfg.order_classic[t];
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

namespace {
// the kernels' shared-memory limits per device (R3SharedLimit), set by SelectiveR3Fits before any launch is captured
std::unordered_map<int, int> r3_limit_shared;
std::unordered_map<int, int> r3_limit_global;
std::unordered_map<int, int> r3_limit_seed;
}  // namespace

bool SelectiveR3Fits(const int num_leaves) {
  R3SharedLimit(SelectiveR3ParallelKernel<true>, &r3_limit_shared);
  R3SharedLimit(SelectiveR3SeedKernel, &r3_limit_seed);
  const size_t index = (static_cast<size_t>(num_leaves) + 2) * sizeof(int);
  return index <= static_cast<size_t>(R3SharedLimit(SelectiveR3ParallelKernel<false>, &r3_limit_global));
}

void LaunchSelectiveR3Parallel(const CUDASelectiveR3Config& config, cudaStream_t stream, const bool shared_area) {
  const size_t area = R3ParallelAreaBytes(config.num_leaves);
  const size_t rest = static_cast<size_t>(config.leaf_capacity) * sizeof(int);
  if (shared_area && area + rest <= static_cast<size_t>(R3SharedLimit(SelectiveR3ParallelKernel<true>,
                                                                       &r3_limit_shared))) {
    SelectiveR3ParallelKernel<true><<<1, kR3ParThreads, area + rest, stream>>>(config);
  } else {
    CHECK(rest <= static_cast<size_t>(R3SharedLimit(SelectiveR3ParallelKernel<false>, &r3_limit_global)));
    CHECK(config.par_global != nullptr);
    SelectiveR3ParallelKernel<false><<<1, kR3ParThreads, rest, stream>>>(config);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

void LaunchSelectiveR3Seed(const CUDASelectiveR3Config& config, const int* seed, const CUDASplitInfo* journal,
                           const int journaled, cudaStream_t stream) {
  const size_t mark = static_cast<size_t>(config.leaf_capacity) * sizeof(int);
  CHECK(mark <= static_cast<size_t>(R3SharedLimit(SelectiveR3SeedKernel, &r3_limit_seed)));
  SelectiveR3SeedKernel<<<1, kR3ParThreads, mark, stream>>>(config, seed, journal, journaled);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

void LaunchSelectiveR3Flag(const CUDASelectiveR3Config& config, int* flag, cudaStream_t stream) {
  SelectiveR3FlagKernel<<<1, 1, 0, stream>>>(config.state, flag);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

void LaunchSelectiveR3Gather(const CUDASelectiveR3Config& config, CUDASelectiveR3Final* out, cudaStream_t stream) {
  SelectiveR3GatherKernel<<<1, 256, 0, stream>>>(config, out);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

}  // namespace Falcata

#endif  // USE_CUDA
