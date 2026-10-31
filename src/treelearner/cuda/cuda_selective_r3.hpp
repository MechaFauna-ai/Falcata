/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief R3: one binding level of a selective (leaf-wise, num_leaves-bound) tree's greedy simulation on the device.
 *
 *  The kernel reproduces CUDASingleGPUTreeLearner::RunSelectiveLevel's three steps from the same state:
 *   1. the greedy simulation over the live applied records and the level's frontier: pops in (gain desc, then the
 *      FindBestFromAllSplitsKernel tie-break of SelectiveTieBreak) order while fewer than the budget are popped; the
 *      t-th pop keeps its classic leaf for the left child and hands t + 1 to the right child;
 *   2. the collapse of the displaced applied records in ascending record id, each subtree depth-first (the host's
 *      stack order), every right-child leaf pushed onto the LIFO free list;
 *   3. the records of the selected frontier candidates in pop order, right-child leaves popped from the free list
 *      (else fresh).
 *  SelectiveTieBreak is a fixed total order on classic leaves (SelectiveR3TieKey: the shuffle-down reduction keeps the
 *  receiving lane on a tie, so the occupied lane, and then warp, with the smallest bit-reversed index wins, and the
 *  smallest leaf within a lane), so the simulation is a heap on (gain desc, key asc) for any number of tied leaves.
 *  P8a runs it in shadow mode only (FALCATA_DEBUG=diag): the host simulation decides, the learner compares.
 */
#ifndef FALCATA_TREELEARNER_CUDA_CUDA_SELECTIVE_R3_HPP_
#define FALCATA_TREELEARNER_CUDA_CUDA_SELECTIVE_R3_HPP_

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <Falcata/cuda/cuda_utils.hu>

#include <cstddef>
#include <cstdint>

#include <Falcata/cuda/cuda_split_info.hpp>

namespace Falcata {

/*! \brief one applied split record (CUDASingleGPUTreeLearner::SelectiveApplied's simulation fields) */
struct CUDASelectiveR3Record {
  int parent;
  int left_leaf;
  int right_leaf;
  int kid_left;
  int kid_right;
  int dead;
  double gain;
};

/*! \brief counts in and out of one level */
struct CUDASelectiveR3State {
  // in (the level's start); num_records / num_free / num_allocated are updated in place
  int num_records;
  int num_free;
  int num_allocated;
  int num_children;
  // out
  int num_frontier;
  int num_order;
  int num_collapsed;  // top-level collapsed records
  int num_freed;      // leaves pushed onto the free list, in push order
  int num_apply;      // records created for selected frontier candidates
  int error;          // 1: a right child past num_leaves - 1, 2: record capacity
};

struct CUDASelectiveR3Config {
  CUDASelectiveR3State* state;
  CUDASelectiveR3Record* records;  // [record_capacity], updated in place
  int* leaf_parent;                // [leaf_capacity], updated in place
  int* free_leaves;                // [leaf_capacity] LIFO stack, updated in place
  const int* child_leaves;         // [num_children]: the children of the last level's applied splits (root: {0})
  const CUDASplitInfo* leaf_best;  // the device per-leaf best-split cache (is_valid, gain)
  // scratch
  int* leaf_frontier;     // [leaf_capacity]
  int* frontier_leaf;     // [leaf_capacity]
  double* frontier_gain;  // [leaf_capacity]
  int* selected;          // [record_capacity]
  int* stack;             // [record_capacity]
  void* child_table;      // [record_capacity] x 32 bytes
  void* heap_global;      // [num_leaves] x 16 bytes, when the heap does not fit shared memory
  // out
  int* order_id;          // [budget]: >= 0 record id, < 0 ~frontier index
  int* order_classic;     // [budget]
  int* collapsed;         // [record_capacity]: top-level collapsed record ids, ascending
  int* freed;             // [record_capacity]
  int* apply_right;       // [budget]
  int budget;             // num_leaves - 1
  int num_leaves;
  int leaf_capacity;
  int record_capacity;
};

/*! \brief the tie-break order of FindBestFromAllSplitsKernel (256 threads, strided scan, 8-warp shuffle-down with
 *  keep-on-tie, see SelectiveTieBreak) as a key: among equal gains the smallest key wins */
__host__ __device__ inline uint32_t SelectiveR3TieKey(const int classic_leaf) {
  const uint32_t leaf = static_cast<uint32_t>(classic_leaf);
  const uint32_t w = (leaf >> 5) & 7u;
  const uint32_t l = leaf & 31u;
  const uint32_t w_rev = ((w & 1u) << 2) | (w & 2u) | ((w >> 2) & 1u);
  const uint32_t l_rev = ((l & 1u) << 4) | ((l & 2u) << 2) | (l & 4u) | ((l >> 2) & 2u) | ((l >> 4) & 1u);
  return (w_rev << 28) | (l_rev << 23) | (leaf >> 8);
}

__host__ __device__ inline int SelectiveR3KeyLeaf(const uint32_t key) {
  const uint32_t w_rev = key >> 28;
  const uint32_t l_rev = (key >> 23) & 31u;
  const uint32_t w = ((w_rev & 1u) << 2) | (w_rev & 2u) | ((w_rev >> 2) & 1u);
  const uint32_t l = ((l_rev & 1u) << 4) | ((l_rev & 2u) << 2) | (l_rev & 4u) | ((l_rev >> 2) & 2u) |
                     ((l_rev >> 4) & 1u);
  return static_cast<int>(((key & ((1u << 23) - 1u)) << 8) | (w << 5) | l);
}

/*! \brief bytes of the scratch the config's void pointers need */
constexpr size_t kSelectiveR3ChildEntryBytes = 32;
constexpr size_t kSelectiveR3HeapEntryBytes = 16;

/*! \brief one level on `stream` (one block). The heap goes to shared memory when it fits the device's opt-in limit
 *  (and shared_heap, false only in tests), else to heap_global (which must then be non-null). */
void LaunchSelectiveR3(const CUDASelectiveR3Config& config, cudaStream_t stream, const bool shared_heap = true);

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_SELECTIVE_R3_HPP_
