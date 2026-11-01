/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief R3: the binding levels of a selective (leaf-wise, num_leaves-bound) tree on the device (P8a-P8c, routed in P8b).
 *
 *  A level reproduces CUDASingleGPUTreeLearner::RunSelectiveLevel's three steps from the same state:
 *   1. the greedy simulation over the live applied records and the level's frontier: pops in (gain desc, then the
 *      FindBestFromAllSplitsKernel tie-break of SelectiveTieBreak) order while fewer than the budget are popped; the
 *      t-th pop keeps its classic leaf for the left child and hands t + 1 to the right child;
 *   2. the collapse of the displaced applied records in ascending record id, each subtree depth-first (the host's
 *      stack order), every right-child leaf pushed onto the LIFO free list;
 *   3. the records of the selected frontier candidates in pop order, right-child leaves popped from the free list
 *      (else fresh).
 *  SelectiveTieBreak is a fixed total order on classic leaves (SelectiveR3TieKey: the shuffle-down reduction keeps the
 *  receiving lane on a tie, so the occupied lane, and then warp, with the smallest bit-reversed index wins, and the
 *  smallest leaf within a lane), so the simulation is best-first on (gain desc, key asc) for any number of tied leaves.
 *
 *  The kernel (P8c, LaunchSelectiveR3Parallel) produces those results without a simulation. It keeps the last level's
 *  pop order on the device (frontier picks renamed to the records they became: the best-first run over the live
 *  records) and inserts the level's frontier candidates into it: under a fixed total order a candidate pops right
 *  before the first entry after its parent whose gain is lower than its own (or at the end), candidates in one slot in
 *  gain order, and the level's selection is the first num_leaves - 1 entries (runs/device_default/P8C_PARALLEL_R3.md
 *  has the proof). Equal gains: the merge uses a fixed tie order, which is the host's run up to the first step where an
 *  available entry of the popped entry's gain has the smaller tie key; from there (found in parallel) the kernel runs
 *  best-first exactly.
 *
 *  The route (P8b, CUDASelectiveR3Config::dl_cfg set): every level after a selective tree's first binding level runs
 *  here inside the device levels. The kernel then also applies its result to the device state the host's selective
 *  loop keeps on the host (R3ApplyLevel): the collapsed nodes' windows, the freed leaves' row counts, each created
 *  record's payload (its best-split record and window) and the level's apply descriptors and terminal-region gaps for
 *  the batched apply. The host simulates the hand-off level once and seeds the state (SelectiveR3Seed).
 */
#ifndef FALCATA_TREELEARNER_CUDA_CUDA_SELECTIVE_R3_HPP_
#define FALCATA_TREELEARNER_CUDA_CUDA_SELECTIVE_R3_HPP_

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/meta.h>

#include <cstddef>
#include <cstdint>

#include <Falcata/cuda/cuda_split_info.hpp>

namespace Falcata {

struct CUDADeviceLevelConfig;
struct CUDADeviceLevelState;
struct CUDAHybridPairDescriptor;

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

/*! \brief the route's payload of a record: the split's best-split record when it was created and its parent's window
 *  (SelectiveApplied::info / data_start / num_data) */
struct CUDASelectiveR3Payload {
  CUDASplitInfo info;
  data_size_t data_start;
  data_size_t num_data;
};

/*! \brief counts in and out of one level */
struct CUDASelectiveR3State {
  // in (the level's start); num_records / num_free / num_allocated are updated in place
  int num_records;
  int num_free;
  int num_allocated;
  int num_children;   // < 0: the route's children, read from the last level's pair descriptors
  // out
  int num_frontier;
  int num_order;
  int num_collapsed;  // top-level collapsed records
  int num_freed;      // leaves pushed onto the free list, in push order
  int num_apply;      // records created for selected frontier candidates
  int error;          // 1: a right child past num_leaves - 1, 2: record capacity, 3: more live records than the
                      // budget (rebuild), 4: a NaN gain (its integer keys would not order it as the host). Errors 1, 2
                      // and 4 are found before the level changes any state (the route hands the tree back to the host)
  // parallel R3 only
  int resumes;        // merges that ran best-first from a tie the merge's fixed order resolved differently
  int resume_at;      // the level's first such pop (-1: the merge was exact)
  int rounds;         // merges that rebuilt the order from the records (0: the order was kept from the last level)
  // route only (P8b)
  int done;           // the tree is complete (a level created no record) or stopped with an error: later kernels skip
  int gap_leaves;     // the leaves allocated before the level's records (the terminal-region scan's range)
  int levels;         // levels applied on the route (the hand-off level included)
  int created_total;  // records created on the route
  int freed_total;    // leaves freed on the route
};

/*! \brief the parallel R3's order, kept on the device across the levels of a tree */
struct CUDASelectiveR3Persist {
  int n;         // entries in the order
  int parity;    // the half of order_buf that holds it
  int valid;     // 0: rebuild it from the records
};

struct CUDASelectiveR3Config {
  CUDASelectiveR3State* state;
  CUDASelectiveR3Record* records;  // [record_capacity], updated in place
  int* leaf_parent;                // [leaf_capacity], updated in place
  int* free_leaves;                // [leaf_capacity] LIFO stack, updated in place
  const int* child_leaves;         // [num_children]: the children of the last level's applied splits (root: {0})
  const CUDASplitInfo* leaf_best;  // the device per-leaf best-split cache (is_valid, gain)
  CUDASelectiveR3Persist* persist;
  int* order_buf;         // 2 x 3 x leaf_capacity: record id, classic leaf, parent (2 * position + right, -1 none)
  // scratch
  int* frontier_leaf;     // [leaf_capacity]
  double* frontier_gain;  // [leaf_capacity]
  int* rec_pos;           // [record_capacity]: a record's position in the order
  int* rec_round;         // [record_capacity]: a record's depth while the order is rebuilt
  int* cand;              // 18 x leaf_capacity: per candidate and per merged entry (R3Arrays)
  double* cand_gain;      // 3 x leaf_capacity: the candidates' gains, then the merged entries' gains
  void* par_global;       // [R3ParallelAreaBytes] when the merge area does not fit shared memory
  // out
  int* order_id;          // [budget]: >= 0 record id, < 0 ~frontier index
  int* order_classic;     // [budget]
  int* collapsed;         // [leaf_capacity]: top-level collapsed record ids, ascending
  int* freed;             // [leaf_capacity]
  int* apply_right;       // [budget]: the created records' right leaves, in pop order
  int* apply_leaf;        // [budget]: their (left) leaves
  // the route (P8b; null otherwise)
  CUDASelectiveR3Payload* payload;          // [record_capacity]
  const CUDADeviceLevelConfig* dl_cfg;      // device memory: the levels' descriptors, slots and feature tables
  CUDADeviceLevelState* dl_state;           // the device levels' state (live pairs in, the level's counts out)
  const CUDAHybridPairDescriptor* pair_descs;  // the last level's children (num_children < 0)
  data_size_t* leaf_data_start;             // the partition's windows
  data_size_t* leaf_num_data;
  int rebuild;            // 1: rebuild the order from the records (the first level of a tree)
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

/*! \brief bytes of par_global: the merge area (2 x pow2(num_leaves) 8-byte words) */
__host__ __device__ inline size_t R3ParallelAreaBytes(const int num_leaves) {
  size_t p = 1;
  while (p < static_cast<size_t>(num_leaves)) {
    p <<= 1;
  }
  return 2 * p * sizeof(double);
}

/*! \brief whether one level of a num_leaves tree fits the kernel: its per-leaf index (leaf_capacity ints) in shared
 *  memory (the merge area may go to global memory). False above ~24 K leaves on sm_120 */
bool SelectiveR3Fits(const int num_leaves);

/*! \brief one level on `stream` (one block). The merge area goes to shared memory when it fits (and shared_area, false
 *  only in tests), else to par_global. Requires SelectiveR3Fits(num_leaves). On the route the level is also applied
 *  to the device state (see the file comment) */
void LaunchSelectiveR3Parallel(const CUDASelectiveR3Config& config, cudaStream_t stream, const bool shared_area = true);

/*! \brief the route's seed (one block, after the seed's H2D into `seed`): the host's state after the hand-off level
 *  (counts, records, the records' windows, leaf parents, free list, the level's outputs and the order, packed as
 *  SelectiveR3SeedLayout) unpacked into the config's arrays; the records journaled by the device levels before it
 *  (ids < journaled) take their best-split record from `journal`; then the hand-off level applied (R3ApplyLevel) */
void LaunchSelectiveR3Seed(const CUDASelectiveR3Config& config, const int* seed, const CUDASplitInfo* journal,
                           const int journaled, cudaStream_t stream);

/*! \brief the route's flag after a body (mapped host memory, 8 ints): [0] 1 while the tree continues, [1] levels
 *  applied, [2] the error, [3] the last level's pops, [4] allocated leaves, [5] records */
void LaunchSelectiveR3Flag(const CUDASelectiveR3Config& config, int* flag, cudaStream_t stream);

/*! \brief the final level's records in pop order, gathered for the tree end's readback */
struct CUDASelectiveR3Final {
  CUDASplitInfo info;
  int left_leaf;
  int right_leaf;
  int classic;
  int pad;
};
void LaunchSelectiveR3Gather(const CUDASelectiveR3Config& config, CUDASelectiveR3Final* out, cudaStream_t stream);

/*! \brief the packed seed (ints): [0, kSeedHeader) counts {records, free, allocated, gap_leaves, created, collapsed,
 *  freed, order, unused}, then records (sizeof(CUDASelectiveR3Record) / 4 ints each), the records' windows (2 ints
 *  each), leaf parents (leaf_capacity), the free list, the created records' leaves and right leaves, the collapsed
 *  record ids, the freed leaves, and the order (3 ints per entry) */
constexpr int kSelectiveR3SeedHeader = 9;

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_SELECTIVE_R3_HPP_
