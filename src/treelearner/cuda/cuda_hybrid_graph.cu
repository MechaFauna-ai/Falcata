/*!
 * Copyright (c) 2026 ExaBoost contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief device-side level controller of the graphs-L1.5 hybrid prefix (see
 *  cuda_hybrid_graph.hpp). One controller run replays exactly the host level
 *  loop's decisions between two levels: CollectSplittableLeaves (leaf-ascending
 *  is_valid scan), ArbitrateLevelBudget (incl. the final partial level's
 *  stable gain sort), ApplyLevelBatched's descriptor building (tree batch
 *  splits + partition apply descriptors + terminal-region gap copies), and
 *  EnqueueLevelBestSplitSearchSpeculative's pair descriptors -- then resizes
 *  ITS unrolled level body's kernels through device-updatable node handles.
 *  Stopping sets the loop state's done flag; later bodies' controllers exit
 *  immediately and keep their body nodes disabled.
 */

#ifdef USE_CUDA

#include "cuda_hybrid_graph.hpp"

#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED

#include <algorithm>
#include <cfloat>

#include "cuda_data_partition.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_leaf_splits.hpp"

namespace LightGBM {

namespace {

/*! \brief level body B applies at most min(2^B, kMax) splits (each level at
 *  most doubles the leaf count from 1), so its controller block only needs to
 *  cover that bound: one thread per split for the descriptor build, one per
 *  gap slot (splits + 1), one per bitonic-sort lane (NextPow2(splits)), and
 *  one per device-updatable node. 2x the split bound covers all four (the
 *  1024-thread top body relies on the learner's num_leaves <= 2047 gate for
 *  the splits + 1 gap slots, as before); the 64 floor covers the node slice
 *  and keeps two full warps for the scan/max reductions. Small-level bodies
 *  drop from 32 warps to 2 through the ~20 block-wide barriers, which is most
 *  of the controller's fixed cost. */
__host__ __device__ constexpr int ControllerThreadsForBody(const int body_index) {
  return body_index >= 10 ? kHybridGraphMaxSplitsPerLevel :
    (body_index <= 4 ? 64 : 2 * (1 << body_index));
}

__device__ __forceinline__ int NextPow2(int n) {
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

/*! \brief block-wide exclusive prefix sum of \p val (warp-shuffle two-level
 *  scan, 2 block syncs); returns the exclusive prefix, total in *s_total.
 *  s_warp must hold >= 32 ints. */
__device__ int BlockExclusiveScan(const int val, int* s_warp, int* s_total) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int num_warps = static_cast<int>(blockDim.x) >> 5;
  int incl = val;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, incl, offset);
    if (lane >= offset) {
      incl += other;
    }
  }
  if (lane == 31) {
    s_warp[warp] = incl;
  }
  __syncthreads();
  if (warp == 0) {
    const int w = lane < num_warps ? s_warp[lane] : 0;
    int winc = w;
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, winc, offset);
      if (lane >= offset) {
        winc += other;
      }
    }
    if (lane < num_warps) {
      s_warp[lane] = winc - w;  // exclusive warp base
    }
    if (lane == num_warps - 1) {
      *s_total = winc;
    }
  }
  __syncthreads();
  return s_warp[warp] + incl - val;
}

/*! \brief block-wide max of \p val (warp shuffles, 3 block syncs); result in
 *  *s_out on every thread's next read after the final sync. The entry sync
 *  protects s_warp: a preceding BlockExclusiveScan's tail read of s_warp may
 *  otherwise race this function's first write under independent thread
 *  scheduling (racecheck-verified). */
__device__ void BlockMax(const int val, int* s_warp, int* s_out) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int num_warps = static_cast<int>(blockDim.x) >> 5;
  __syncthreads();
  int v = val;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const int other = __shfl_down_sync(0xffffffffu, v, offset);
    if (other > v) {
      v = other;
    }
  }
  if (lane == 0) {
    s_warp[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    int m = lane < num_warps ? s_warp[lane] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
      const int other = __shfl_down_sync(0xffffffffu, m, offset);
      if (other > m) {
        m = other;
      }
    }
    if (lane == 0) {
      *s_out = m;
    }
  }
  __syncthreads();
}

/*! \brief in-block bitonic sort of (key ascending) int pairs padded to p
 *  (pad key == INT_MAX); payload follows the keys */
__device__ void BitonicSortIntAsc(int* keys, int* payload, const int p) {
  const int i = static_cast<int>(threadIdx.x);
  for (int size = 2; size <= p; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      if (i < p) {
        const int j = i ^ stride;
        if (j > i && j < p) {
          const bool ascending = ((i & size) == 0);
          const bool swap = ascending ? (keys[i] > keys[j]) : (keys[i] < keys[j]);
          if (swap) {
            const int tk = keys[i]; keys[i] = keys[j]; keys[j] = tk;
            const int tp = payload[i]; payload[i] = payload[j]; payload[j] = tp;
          }
        }
      }
    }
  }
  __syncthreads();
}

/*! \brief "a sorts before b" for the final partial level: descending gain,
 *  ties to the smaller leaf index -- exactly the host's stable_sort over the
 *  leaf-ascending candidate list */
__device__ __forceinline__ bool GainBefore(const double gain_a, const int leaf_a,
                                           const double gain_b, const int leaf_b) {
  return gain_a > gain_b || (gain_a == gain_b && leaf_a < leaf_b);
}

__device__ void BitonicSortGainDesc(double* gains, int* leaves, const int p) {
  const int i = static_cast<int>(threadIdx.x);
  for (int size = 2; size <= p; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      if (i < p) {
        const int j = i ^ stride;
        if (j > i && j < p) {
          const bool ascending = ((i & size) == 0);
          // "ascending" here means GainBefore-order
          const bool in_order = GainBefore(gains[i], leaves[i], gains[j], leaves[j]);
          if (ascending != in_order) {
            const double tg = gains[i]; gains[i] = gains[j]; gains[j] = tg;
            const int tl = leaves[i]; leaves[i] = leaves[j]; leaves[j] = tl;
          }
        }
      }
    }
  }
  __syncthreads();
}

/*! \brief enable/disable a body node, skipping the (relatively expensive)
 *  device graph update when the node is already in the requested state */
__device__ __forceinline__ void SetNodeEnabledCached(CUDAHybridGraphLoopState* st,
                                                     const int node_index,
                                                     const bool enabled) {
  if (st->node_enabled[node_index] == static_cast<int>(enabled)) {
    return;
  }
  st->node_enabled[node_index] = static_cast<int>(enabled);
  const cudaError_t e = cudaGraphKernelNodeSetEnabled(st->nodes[node_index].node, enabled);
  if (e != cudaSuccess) {
    atomicOr(reinterpret_cast<unsigned int*>(&st->journal[2]), 1u << st->nodes[node_index].role);
  }
}

/*! \brief set a node's grid, skipping the (relatively expensive) device graph
 *  update when the last-set grid already matches: each unrolled level body
 *  owns its nodes, so n-stable roles hit the cache across trees */
__device__ __forceinline__ void SetNodeGridCached(CUDAHybridGraphLoopState* st,
                                                  const int node_index,
                                                  const dim3 grid) {
  CUDAHybridGraphNodeSlot& slot = st->nodes[node_index];
  if (slot.grid_x == static_cast<int>(grid.x) &&
      slot.grid_y == static_cast<int>(grid.y) &&
      slot.grid_z == static_cast<int>(grid.z)) {
    return;
  }
  slot.grid_x = static_cast<int>(grid.x);
  slot.grid_y = static_cast<int>(grid.y);
  slot.grid_z = static_cast<int>(grid.z);
  const cudaError_t e = cudaGraphKernelNodeSetGridDim(slot.node, grid);
  if (e != cudaSuccess) {
    atomicOr(reinterpret_cast<unsigned int*>(&st->journal[2]), 1u << (slot.role + 16));
  }
}

__global__ void HybridGraphLevelControllerKernel(CUDAHybridGraphLoopState* st,
                                                 const int body_index) {
  constexpr int kMax = kHybridGraphMaxSplitsPerLevel;
  __shared__ int s_leaves[kMax];
  __shared__ double s_gains[kMax];
  __shared__ int s_total;
  __shared__ int s_rkey[kMax];      // region sort keys (data start)
  __shared__ int s_rpay[kMax];      // region sort payload (applied index)
  __shared__ int s_warp[32];        // warp partials of the scans/reductions
  __shared__ int s_max_blocks;
  __shared__ int s_max_bound;
  __shared__ int s_max_gap_blocks;

  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);
  // this body's device-updatable node slice (the epilogue body has none)
  const int node_base = body_index * kHybridGraphMaxNodes;
  const int own_nodes = body_index < st->max_depth ? st->num_nodes : 0;
  if (st->done) {
    // an earlier body stopped the level loop: keep this body's nodes disabled
    // (cached: steady-state same-shape trees issue no device graph updates)
    if (tid < own_nodes) {
      SetNodeEnabledCached(st, node_base + tid, false);
    }
    return;
  }
  const int level = st->level;
  const int cur_leaves = st->num_leaves;
  const int split_cursor = st->journal_split_cursor;

  if (tid == 0) {
    if (level == 0) {
      st->journal[2] = 0;  // sticky error flags of this tree's loop
    } else {
      // the previous level's out buffer holds the new main layout: swap at
      // entry (level 0 uses the buffers exactly as the host wrote them)
      data_size_t* tmp = st->main_indices;
      st->main_indices = st->out_indices;
      st->out_indices = tmp;
    }
  }
  __syncthreads();

  // ---- 1. CollectSplittableLeaves: leaf-ascending is_valid scan ----
  // contiguous per-thread leaf chunks keep the ascending order under the scan
  const int chunk = (cur_leaves + nt - 1) / nt;
  const int chunk_begin = tid * chunk;
  const int chunk_end = min(chunk_begin + chunk, cur_leaves);
  int my_count = 0;
  for (int leaf = chunk_begin; leaf < chunk_end; ++leaf) {
    if (st->leaf_best_split_info[leaf].is_valid) {
      ++my_count;
    }
  }
  const int my_offset = BlockExclusiveScan(my_count, s_warp, &s_total);
  if (s_total > kMax || (nt < kMax && s_total >= nt)) {
    // cannot happen inside the gated regime (body B's splittable count is
    // structurally <= 2^B < ControllerThreadsForBody(B), and <= kMax); guard
    // the shared staging and the one-thread-per-split phases anyway
    if (tid == 0) {
      atomicOr(reinterpret_cast<unsigned int*>(&st->journal[2]), 0x80000000u);
      st->journal[0] = level;
      st->journal[1] = split_cursor;
      st->done = 1;
    }
    if (tid < own_nodes) {
      SetNodeEnabledCached(st, node_base + tid, false);
    }
    return;
  }
  int write_pos = my_offset;
  for (int leaf = chunk_begin; leaf < chunk_end; ++leaf) {
    const CUDASplitInfo* info = &st->leaf_best_split_info[leaf];
    if (info->is_valid) {
      s_leaves[write_pos] = leaf;
      s_gains[write_pos] = info->gain;
      ++write_pos;
    }
  }
  __syncthreads();
  int n = s_total;

  // ---- 2. ArbitrateLevelBudget ----
  bool apply_level = n > 0;
  bool final_partial = false;
  if (body_index >= st->max_depth) {
    // epilogue body: the loop must already be over (the last level's children
    // sit at max_depth, so their speculative searches were invalid); n > 0
    // here would break the unroll's level bound -- flag it for the host
    if (n > 0 && tid == 0) {
      atomicOr(reinterpret_cast<unsigned int*>(&st->journal[2]), 0x40000000u);
    }
    apply_level = false;
  }
  if (n > 0 && cur_leaves + n > st->num_leaves_budget) {
    const int remaining = st->num_leaves_budget - cur_leaves;
    // every splittable leaf of the prefix sits at depth == level (valid leaves
    // split immediately), so the host's per-leaf children_depth_capped check
    // reduces to the level test
    const bool children_depth_capped = st->max_depth > 0 && (level + 1) >= st->max_depth;
    if (!children_depth_capped || remaining <= 0) {
      apply_level = false;  // leave the budget-bound level to the leaf-wise tail
    } else {
      const int p = NextPow2(n);
      if (tid >= n && tid < p) {
        s_gains[tid] = -DBL_MAX;
        s_leaves[tid] = 2147483647;
      }
      BitonicSortGainDesc(s_gains, s_leaves, p);
      n = remaining;
      final_partial = true;
    }
  }

  if (!apply_level) {
    // stop: journal header, this body's nodes off, done flag set (each thread
    // disables one node; SetNodeEnabledCached skips already-disabled ones;
    // later bodies' controllers disable their own nodes on the done path)
    if (tid == 0) {
      st->journal[0] = level;
      st->journal[1] = split_cursor;
      st->done = 1;
    }
    if (tid < own_nodes) {
      SetNodeEnabledCached(st, node_base + tid, false);
    }
    return;
  }

  // ---- 3. per-split descriptors (one thread per split; n <= blockDim.x) ----
  const uint8_t child_below_max_depth =
    (st->max_depth <= 0 || (level + 1) < st->max_depth) ? 1 : 0;
  int my_blocks = 0;
  data_size_t my_parent_num = 0;
  if (tid < n) {
    const int leaf = s_leaves[tid];
    const int right = cur_leaves + tid;
    const CUDASplitInfo* info = st->leaf_best_split_info + leaf;
    const int inner_feature = info->inner_feature_index;
    const uint32_t threshold = info->threshold;
    const CUDAHybridGraphFeatureMeta m = st->feature_meta[inner_feature];
    const CUDAHybridGraphFeatureSource src = st->feature_source[inner_feature];
    // tree-structure batch split (mirror of ApplyLevelBatched's host build)
    CUDATreeBatchSplit& ts = st->tree_batch_splits[tid];
    ts.leaf_index = leaf;
    ts.num_leaves_at_split = right;
    ts.real_feature_index = m.real_feature_index;
    ts.real_threshold = st->real_thresholds[m.real_threshold_offset + threshold];
    ts.missing_type = m.missing_type;
    ts.split_info = info;
    // partition apply descriptor (mirror of SplitLevelBatched's host math)
    const data_size_t parent_num_data = st->leaf_num_data[leaf];
    const data_size_t parent_data_start = st->leaf_data_start[leaf];
    uint32_t th = threshold + m.min_bin;
    uint32_t t_zero_bin = m.min_bin + m.default_bin;
    if (m.most_freq_bin == 0) {
      --th;
      --t_zero_bin;
    }
    uint8_t split_default_to_left = 0;
    uint8_t split_missing_default_to_left = 0;
    int default_leaf_index = right;
    int missing_default_leaf_index = right;
    if (m.most_freq_bin <= threshold) {
      split_default_to_left = 1;
      default_leaf_index = leaf;
    }
    if (m.missing_is_zero || m.missing_is_na) {
      if (info->default_left) {
        split_missing_default_to_left = 1;
        missing_default_leaf_index = leaf;
      }
    }
    CUDAHybridApplyDescriptor& d = st->apply_descs[tid];
    d.column_data = src.column_data;
    d.best_split_info = info;
    d.smaller_leaf_splits = st->pair_slots + 2 * tid;
    d.larger_leaf_splits = st->pair_slots + 2 * tid + 1;
    d.leaf_data_start = parent_data_start;
    d.num_data_in_leaf = parent_num_data;
    d.block_offset_start = 0;  // filled after the scan below
    d.num_blocks = (parent_num_data + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
      SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    d.left_leaf_index = leaf;
    d.right_leaf_index = right;
    d.th = th;
    d.t_zero_bin = t_zero_bin;
    d.max_bin = m.max_bin;
    d.min_bin = m.min_bin;
    d.default_leaf_index = default_leaf_index;
    d.missing_default_leaf_index = missing_default_leaf_index;
    d.split_default_to_left = split_default_to_left;
    d.split_missing_default_to_left = split_missing_default_to_left;
    d.packed_row_stride = src.packed_row_stride;
    d.bit_type = src.bit_type;
    d.packed_shift = src.packed_shift;
    d.min_is_max = (m.min_bin < m.max_bin) ? 0 : 1;
    d.missing_is_zero = m.missing_is_zero;
    d.missing_is_na = m.missing_is_na;
    d.mfb_is_zero = m.mfb_is_zero;
    d.mfb_is_na = m.mfb_is_na;
    d.max_bin_to_left = (m.max_bin <= th) ? 1 : 0;
    d.use_min_bin = m.use_min_bin;
    my_blocks = d.num_blocks;
    my_parent_num = parent_num_data;
    // journal record
    st->journal[kHybridGraphJournalHeader + 2 * (split_cursor + tid)] = leaf;
    st->journal[kHybridGraphJournalHeader + 2 * (split_cursor + tid) + 1] = inner_feature;
    // next level's speculative pair descriptor (search skipped after the
    // final partial level)
    if (!final_partial) {
      CUDAHybridPairDescriptor& pd = st->pair_descs[tid];
      pd.smaller_struct = st->pair_slots + 2 * tid;
      pd.larger_struct = st->pair_slots + 2 * tid + 1;
      pd.smaller_leaf_index = leaf;
      pd.larger_leaf_index = right;
      pd.num_data_in_smaller_leaf = 0;
      pd.num_data_in_larger_leaf = 0;
      pd.construct_valid = 1;
      pd.smaller_valid = child_below_max_depth;
      pd.larger_valid = child_below_max_depth;
      pd.parent_num_bits = 0;
      pd.smaller_num_bits = 0;
      pd.larger_num_bits = 0;
    }
  }
  // block-offset scan: split k's region is [scan_k, scan_k + num_blocks_k + 1)
  const int block_offset_start = BlockExclusiveScan(tid < n ? my_blocks + 1 : 0, s_warp, &s_total);
  if (tid < n) {
    st->apply_descs[tid].block_offset_start = block_offset_start;
  }

  // level maxima: partition data-grid x extent and the construct bound
  BlockMax(tid < n ? my_blocks : 0, s_warp, &s_max_blocks);
  BlockMax(tid < n ? static_cast<int>(my_parent_num / 2) : 0, s_warp, &s_max_bound);
  const int max_num_blocks = s_max_blocks;
  const data_size_t max_smaller_leaf_bound = static_cast<data_size_t>(s_max_bound);

  // ---- 4. terminal-region gap copies (mirror of SplitLevelBatched) ----
  // sort the split regions by data start, then emit one gap descriptor per
  // uncovered hole of [0, root_num_data), in ascending order like the host
  const int p = NextPow2(n);
  if (tid < p) {
    if (tid < n) {
      s_rkey[tid] = static_cast<int>(st->leaf_data_start[s_leaves[tid]]);
      s_rpay[tid] = tid;
    } else {
      s_rkey[tid] = 2147483647;
      s_rpay[tid] = -1;
    }
  }
  __syncthreads();
  BitonicSortIntAsc(s_rkey, s_rpay, p);
  // gap slot i sits before sorted region i (i == n: after the last region).
  // n <= budget/2 <= 1023 (the learner gates num_leaves <= 2047), so the
  // n + 1 slots fit the block's threads.
  int my_gap = 0;
  data_size_t gap_start = 0;
  data_size_t gap_num = 0;
  if (tid <= n) {
    const data_size_t region_start = tid < n ?
      static_cast<data_size_t>(s_rkey[tid]) : st->root_num_data;
    data_size_t prev_end = 0;
    if (tid > 0) {
      const int prev_applied = s_rpay[tid - 1];
      prev_end = static_cast<data_size_t>(s_rkey[tid - 1]) +
        st->leaf_num_data[s_leaves[prev_applied]];
    }
    if (region_start > prev_end) {
      my_gap = 1;
      gap_start = prev_end;
      gap_num = region_start - prev_end;
    }
  }
  const int gap_pos = BlockExclusiveScan(my_gap, s_warp, &s_total);
  const int num_gaps = s_total;
  int my_gap_blocks = 0;
  if (my_gap) {
    // gap descriptors live at a fixed base so the gap kernel's frozen
    // descriptor parameter (captured as descs + kHybridGraphGapDescBase)
    // stays valid for every level
    CUDAHybridApplyDescriptor& g = st->apply_descs[kHybridGraphGapDescBase + gap_pos];
    // zero-init like the host's memset gap descriptors
    memset(&g, 0, sizeof(CUDAHybridApplyDescriptor));
    g.leaf_data_start = gap_start;
    g.num_data_in_leaf = gap_num;
    g.num_blocks = (gap_num + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
      SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    my_gap_blocks = g.num_blocks;
  }
  BlockMax(my_gap_blocks, s_warp, &s_max_gap_blocks);
  const int max_gap_blocks = s_max_gap_blocks;

  // ---- 5. resize + arm the body nodes (one thread per node; the device
  // graph updates dominate the controller's cost when serialized) ----
  // graphs A2: every role except the construct kernel is bit-invariant under
  // grid ENLARGEMENT along its n-sized (and per-descriptor-guarded blocks-)
  // axes -- the kernels guard on the loop state's live counts (cur_num_splits
  // / cur_num_gaps, written below) or on the descriptor's own num_blocks and
  // exit before any other read or write. Rounding those extents up to their
  // power-of-two bucket makes the cached grid stable across levels/trees whose
  // counts share a bucket, so the (expensive, ~4.5us) SetGridDim device graph
  // updates fire only on bucket flips while idle blocks stay < 2x the live
  // range. The construct kernel derives its row grouping from its exact grid
  // (gridDim.z pairs, formula-sized y), so it keeps exact device updates.
  if (tid < own_nodes) {
    const CUDAHybridGraphNodeSlot slot = st->nodes[node_base + tid];
    const unsigned int un = static_cast<unsigned int>(n);
    const unsigned int qn = static_cast<unsigned int>(NextPow2(n));
    bool enabled = true;
    dim3 grid(1, 1, 1);
    switch (slot.role) {
      case kHybridGraphNodeTreeSplit:
      case kHybridGraphNodeAggregate:
      case kHybridGraphNodeTreeStructure:
        grid = dim3(qn, 1, 1);
        break;
      case kHybridGraphNodeGenBitVector:
      case kHybridGraphNodeSplitInner:
        grid = dim3(static_cast<unsigned int>(NextPow2(max_num_blocks)), qn, 1);
        break;
      case kHybridGraphNodeCopyGaps:
        enabled = num_gaps > 0;
        grid = enabled ?
          dim3(static_cast<unsigned int>(NextPow2(max_gap_blocks)),
               static_cast<unsigned int>(NextPow2(num_gaps)), 1) :
          dim3(1, 1, 1);
        break;
      case kHybridGraphNodeConstruct:
        // pow2 bucket like the rest: the kernel computes its exact row
        // grouping from the level's actual sizes (using the loop state's live
        // pair count instead of gridDim.z) and idle y/z blocks exit before any
        // write, so the frozen grid only needs to be an upper bound. Quantized
        // training (num_grad_quant_bins > 0) includes the packed int32
        // shared-histogram overflow guard, so the bucket always covers the
        // kernel's device-computed (guarded) row grouping.
        enabled = !final_partial;
        grid = dim3(static_cast<unsigned int>(st->construct_grid_x),
                    static_cast<unsigned int>(NextPow2(HybridBatchedConstructGridDimYQuant(
                      max_smaller_leaf_bound, n, st->construct_block_dim_y,
                      st->construct_min_grid_dim_y, st->construct_min_rows_per_thread,
                      st->construct_saturation_floor, st->num_grad_quant_bins))), qn);
        break;
      case kHybridGraphNodeSearchPairY:
        enabled = !final_partial;
        grid = dim3(static_cast<unsigned int>(slot.static_x), qn, 1);
        break;
      case kHybridGraphNodeFind:
        enabled = !final_partial;
        grid = dim3(static_cast<unsigned int>(st->find_grid_x), qn, 2);
        break;
      case kHybridGraphNodeSyncLevel:
        enabled = !final_partial;
        grid = dim3(2, qn, static_cast<unsigned int>(slot.static_x));
        break;
      case kHybridGraphNodeSyncAllBlocks:
        enabled = !final_partial;
        grid = dim3(2, qn, 1);
        break;
      default:
        break;
    }
    SetNodeEnabledCached(st, node_base + tid, enabled);
    if (enabled) {
      SetNodeGridCached(st, node_base + tid, grid);
    }
  }
  if (tid == 0) {
    // live counts of THIS body's kernels (the pow2-bucketed grids' idle blocks
    // guard on these; visible to the body kernels because they are ordered
    // after this controller by the graph's event edges)
    st->cur_num_splits = n;
    st->cur_num_gaps = num_gaps;
    // deferred split-info slab base of THIS level's tree-structure kernel
    st->split_info_base = split_cursor;
    st->journal[3 + level] = n;
    st->journal_split_cursor = split_cursor + n;
    st->num_leaves = cur_leaves + n;
    st->level = level + 1;
    if (final_partial) {
      st->journal[0] = level + 1;
      st->journal[1] = split_cursor + n;
      st->done = 1;
    }
  }
}

}  // anonymous namespace

void CaptureHybridGraphControllerKernel(cudaStream_t stream,
                                        CUDAHybridGraphLoopState* state,
                                        const int body_index) {
  // block dims are per-node-instantiation constants of the unrolled graph, so
  // each body's bound is baked at capture (the epilogue controller at
  // body_index == max_depth never applies splits; its body-sized block is
  // just an upper bound)
  HybridGraphLevelControllerKernel<<<1, ControllerThreadsForBody(body_index), 0, stream>>>(
    state, body_index);
}

}  // namespace LightGBM

#endif  // EXABOOST_HYBRID_GRAPH_SUPPORTED
#endif  // USE_CUDA
