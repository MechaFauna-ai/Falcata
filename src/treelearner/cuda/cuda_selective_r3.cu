/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief R3, the device greedy simulation of one selective level (see cuda_selective_r3.hpp).
 */

#ifdef USE_CUDA

#include "cuda_selective_r3.hpp"

#include <Falcata/cuda/cuda_utils.hu>

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
__device__ __forceinline__ void R3SiftUp(R3HeapEntry* heap, int i, const R3HeapEntry e) {
  while (i > 0) {
    const int p = (i - 1) >> 2;
    const R3HeapEntry pe = heap[p];
    if (!R3Better(e, pe)) {
      break;
    }
    heap[i] = pe;
    i = p;
  }
  heap[i] = e;
}

__device__ __forceinline__ void R3SiftDown(R3HeapEntry* heap, const int n, int i, const R3HeapEntry e) {
  while (true) {
    const int c = 4 * i + 1;
    if (c >= n) {
      break;
    }
    int best = c;
    R3HeapEntry be = heap[c];
#pragma unroll
    for (int k = 1; k < 4; ++k) {
      if (c + k < n) {
        const R3HeapEntry ce = heap[c + k];
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
template <typename Pred, typename Emit>
__device__ int R3BlockCompact(const int n, int* warp_counts, Pred pred, Emit emit) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int kWarps = kR3Threads / 32;
  int base = 0;
  for (int start = 0; start < n; start += kR3Threads) {
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
  const int num_frontier = R3BlockCompact(num_children, warp_counts,
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
  const int num_collapsed = R3BlockCompact(num_records, warp_counts,
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

}  // namespace

void LaunchSelectiveR3(const CUDASelectiveR3Config& config, cudaStream_t stream, const bool shared_heap) {
  // the device's opt-in shared-memory limit, and the kernel's dynamic shared-memory attribute raised to it, once
  // per device (the attribute is per device)
  static std::mutex mutex;
  static std::unordered_map<int, int> max_shared_by_device;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  int max_shared_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = max_shared_by_device.find(device);
    if (it == max_shared_by_device.end()) {
      int bytes = 0;
      CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
      bytes -= 1024;  // the kernel's static shared memory and head room
      if (bytes > 48 * 1024) {
        CUDASUCCESS_OR_FATAL(cudaFuncSetAttribute(SelectiveR3Kernel<true>,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
      }
      it = max_shared_by_device.emplace(device, bytes > 48 * 1024 ? bytes : 47 * 1024).first;
    }
    max_shared_bytes = it->second;
  }
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

}  // namespace Falcata

#endif  // USE_CUDA
