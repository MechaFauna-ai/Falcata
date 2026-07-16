/*!
 * Copyright (c) 2026 ExaBoost contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief Device-driven level loop ("graphs L1") for the hybrid depth-wise
 *  prefix: the whole depth-limited prefix of a tree runs as ONE host graph
 *  launch + ONE readback. The per-level pipeline (apply level L, search level
 *  L+1) is UNROLLED into max_depth straight-line level bodies plus one
 *  epilogue controller (L1.5; the depth-limited gate bounds the level count,
 *  so no conditional-WHILE relaunch separates the levels). A device-side
 *  controller kernel at the start of each level body replays the host loop's
 *  decisions (which leaves split, right-child index assignment, budget
 *  arbitration incl. the final partial level's gain sort, apply/pair
 *  descriptor building, gap regions) and resizes ITS level's body kernels
 *  through device-updatable graph node handles
 *  (cudaGraphKernelNodeSetGridDim / SetEnabled), so no host round trip
 *  separates the levels. Early tree completion sets a device-side done flag:
 *  the remaining levels' controllers exit immediately and keep their body
 *  nodes disabled (enabled state is cached per node, so steady-state
 *  same-shape trees issue no redundant device graph updates).
 *
 *  Requires CUDA >= 12.4 (device-updatable kernel nodes);
 *  the learner probes support at runtime and falls back to the host level loop
 *  (bit-for-bit the previous behavior) when unsupported or when
 *  EXABOOST_GRAPH_LEVEL_LOOP=0.
 */

#ifndef LIGHTGBM_TREELEARNER_CUDA_CUDA_HYBRID_GRAPH_HPP_
#define LIGHTGBM_TREELEARNER_CUDA_CUDA_HYBRID_GRAPH_HPP_

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <LightGBM/meta.h>

#if CUDART_VERSION >= 12040
#define EXABOOST_HYBRID_GRAPH_SUPPORTED 1
#endif  // CUDART_VERSION >= 12040

#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED

#include <LightGBM/cuda/cuda_utils.hu>

#include <cstdint>
#include <vector>

#include <LightGBM/cuda/cuda_split_info.hpp>
#include <LightGBM/cuda/cuda_tree.hpp>

namespace LightGBM {

struct CUDAHybridApplyDescriptor;
struct CUDAHybridPairDescriptor;
struct CUDALeafSplitsStruct;

/*! \brief static (per-dataset) split metadata of one inner feature, mirroring
 *  the host-side math of CUDADataPartition::SplitLevelBatched and the
 *  CUDATree::SplitBatch input fields */
struct CUDAHybridGraphFeatureMeta {
  uint32_t default_bin;
  uint32_t most_freq_bin;
  /*! \brief already single-feature-in-column adjusted (1 when single) */
  uint32_t min_bin;
  uint32_t max_bin;
  int real_feature_index;
  int missing_type;
  /*! \brief offset of this feature's bin values in the real-threshold table */
  uint32_t real_threshold_offset;
  uint8_t missing_is_zero;
  uint8_t missing_is_na;
  uint8_t mfb_is_zero;
  uint8_t mfb_is_na;
  /*! \brief 0 when the feature is alone in its column (min_bin forced to 1) */
  uint8_t use_min_bin;
  uint8_t padding_[3];
};

/*! \brief per-tree data source of one inner feature's column (the packed
 *  compact matrix is per-tree and double-buffered, so pointers change per
 *  tree); rebuilt and uploaded before every graph launch */
struct CUDAHybridGraphFeatureSource {
  const void* column_data;
  int packed_row_stride;
  uint8_t packed_shift;
  uint8_t bit_type;  // 4 == packed compact source
  uint8_t padding_[2];
};

/*! \brief roles of the WHILE-body kernel nodes the controller resizes.
 *  graphs A2: every dynamic extent below is frozen at its power-of-two bucket
 *  (NextPow2), so the SetGridDim device graph update fires only on bucket
 *  flips; the kernels guard on the loop state's live counts (cur_num_splits /
 *  cur_num_gaps) or on their descriptor's own num_blocks and exit idle blocks
 *  before any read or write */
enum CUDAHybridGraphNodeRole : int {
  kHybridGraphNodeTreeSplit = 0,   // grid (p2(n), 1, 1)
  kHybridGraphNodeGenBitVector,    // grid (p2(max_blocks), p2(n))
  kHybridGraphNodeAggregate,       // grid (p2(n), 1, 1)
  kHybridGraphNodeSplitInner,      // grid (p2(max_blocks), p2(n))
  kHybridGraphNodeTreeStructure,   // grid (p2(n), 1, 1)
  kHybridGraphNodeCopyGaps,        // grid (p2(max_gap_blocks), p2(num_gaps)); disabled when no gaps
  kHybridGraphNodeConstruct,       // grid (static_x, p2(formula_y), p2(n))
  kHybridGraphNodeSearchPairY,     // grid (static_x, p2(n)): fix/subtract family
  kHybridGraphNodeFind,            // grid (find_grid_x, p2(n), 2)
  kHybridGraphNodeSyncLevel,       // grid (2, p2(n), static_x /* blocks per leaf */)
  kHybridGraphNodeSyncAllBlocks,   // grid (2, p2(n))
};

struct CUDAHybridGraphNodeSlot {
  cudaGraphDeviceNode_t node;
  int role;
  /*! \brief role-dependent static grid extent (see role comments) */
  int static_x;
  /*! \brief controller-side cache of the node's last-set grid (0 == not set
   *  yet): each unrolled level body owns its nodes, so a level whose grid
   *  does not change across trees (7 of ~10 roles only depend on the level's
   *  split count) skips the SetGridDim device graph update entirely */
  int grid_x;
  int grid_y;
  int grid_z;
  int padding_;
};

/*! \brief journal layout (ints): [0] applied levels, [1] total splits,
 *  [2] sticky controller error flags, [3 + L] splits of level L, then per
 *  split (in global applied order) {left leaf, inner feature} pairs at
 *  kHybridGraphJournalHeader */
constexpr int kHybridGraphMaxLevels = 32;
constexpr int kHybridGraphJournalHeader = 3 + kHybridGraphMaxLevels;
constexpr int kHybridGraphMaxNodes = 16;
/*! \brief unrolled level bodies per graph: the depth-limited gate bounds
 *  max_depth by 11 (2^max_depth <= num_leaves + 1 <= 2048), plus one
 *  epilogue-controller body that only stops the loop */
constexpr int kHybridGraphMaxBodies = 12;
/*! \brief max splits of one level the controller's shared-memory staging
 *  supports; the learner gates the graph path on num_leaves <= 2 * this */
constexpr int kHybridGraphMaxSplitsPerLevel = 1024;
/*! \brief gap (terminal-region copy) descriptors live at this FIXED offset of
 *  the apply-descriptor buffer inside the graph loop: the gap kernel's
 *  descriptor base is frozen at capture, while the host path keeps its
 *  per-level `descs + num_splits` base */
constexpr int kHybridGraphGapDescBase = kHybridGraphMaxSplitsPerLevel;

/*! \brief the device-visible loop state: static config + per-tree inputs
 *  written by the host, per-level fields owned by the controller kernel */
struct CUDAHybridGraphLoopState {
  // ---- per-tree mutable prefix (host-written before every graph launch;
  //      keep these fields FIRST so one small H2D covers them) ----
  /*! \brief current main data-index buffer (level L reads main, writes out;
   *  the controller swaps them at the start of every level > 0) */
  data_size_t* main_indices;
  data_size_t* out_indices;
  int level;
  int num_leaves;           // current tree leaf count
  int split_info_base;      // deferred split-info slab base of the current level
  int journal_split_cursor;
  data_size_t root_num_data;
  int find_grid_x;          // num_used_tasks of this tree (or num_tasks)
  /*! \brief set by the controller that stops the level loop; later bodies'
   *  controllers exit immediately (host resets it to 0 before every launch) */
  int done;
  /*! \brief graphs A2: live split count of the CURRENT level body, written by
   *  the controller before its body kernels run. Most body kernels' grids are
   *  frozen at a power-of-two bucket of n instead of being device-updated per
   *  level; blocks beyond the live range read this and exit immediately. */
  int cur_num_splits;
  /*! \brief live gap-copy count of the current level body (same scheme) */
  int cur_num_gaps;

  // ---- static config (host-written once at graph build) ----
  int max_depth;
  int num_leaves_budget;
  // construct grid-y formula constants (see HybridBatchedConstructGridDimY)
  int construct_grid_x;
  int construct_block_dim_y;
  int construct_min_grid_dim_y;
  int construct_min_rows_per_thread;
  int construct_saturation_floor;
  /*! \brief quantized training only (0 otherwise): the controller and the
   *  quantized construct kernel's device row-grouping replica apply the packed
   *  int32 shared-histogram overflow guard (QuantConstructMaxRowsPerThread)
   *  with this bin count, and the quantized body kernels derive the per-leaf
   *  histogram bit widths from the device-resident leaf counts with it
   *  (exactly the host's SetNumBitsInHistogramBin thresholds) */
  int num_grad_quant_bins;

  // ---- fixed device buffers ----
  const CUDASplitInfo* leaf_best_split_info;
  const data_size_t* leaf_num_data;
  const data_size_t* leaf_data_start;
  CUDATreeBatchSplit* tree_batch_splits;
  CUDAHybridApplyDescriptor* apply_descs;
  CUDAHybridPairDescriptor* pair_descs;
  CUDALeafSplitsStruct* pair_slots;
  const CUDAHybridGraphFeatureMeta* feature_meta;
  const CUDAHybridGraphFeatureSource* feature_source;
  const double* real_thresholds;
  int* journal;

  // ---- device-updatable node handles (host-written after instantiate) ----
  /*! \brief per-body node slices at stride kHybridGraphMaxNodes: level L's
   *  body nodes live at [L * kHybridGraphMaxNodes, ... + num_nodes) (the
   *  epilogue body has none) */
  CUDAHybridGraphNodeSlot nodes[kHybridGraphMaxNodes * kHybridGraphMaxBodies];
  /*! \brief controller-side cache of each node's enabled state (captured
   *  nodes start enabled); lets the controller skip redundant SetEnabled
   *  device graph updates */
  int node_enabled[kHybridGraphMaxNodes * kHybridGraphMaxBodies];
  /*! \brief body nodes per level (uniform across the unrolled level bodies) */
  int num_nodes;
};

/*! \brief appends the node most recently captured on \p stream to \p nodes
 *  (stream-capture bookkeeping: the capture dependency set right after a
 *  launch is exactly that launch's node). Returns false on any error. */
inline bool AppendCapturedNode(cudaStream_t stream, std::vector<cudaGraphNode_t>* nodes) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const cudaGraphNode_t* deps = nullptr;
  size_t num_deps = 0;
#if CUDART_VERSION >= 13000
  const cudaGraphEdgeData* edge_data = nullptr;
  if (cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &deps, &edge_data, &num_deps) != cudaSuccess) {
    return false;
  }
#else
  if (cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &deps, &num_deps) != cudaSuccess) {
    return false;
  }
#endif  // CUDART_VERSION >= 13000
  if (status != cudaStreamCaptureStatusActive || num_deps != 1) {
    return false;
  }
  nodes->push_back(deps[0]);
  return true;
}

/*! \brief launches the level-controller kernel of unrolled body \p body_index
 *  on \p stream (used inside the level-body capture; body_index == max_depth
 *  is the epilogue controller that only stops the loop; defined in
 *  cuda_hybrid_graph.cu) */
void CaptureHybridGraphControllerKernel(cudaStream_t stream,
                                        CUDAHybridGraphLoopState* state,
                                        int body_index);

}  // namespace LightGBM

#endif  // EXABOOST_HYBRID_GRAPH_SUPPORTED

namespace LightGBM {

/*! \brief optional graph loop-state parameter of the batched apply kernels:
 *  nullptr on every host-launched path (bit-for-bit the previous behavior),
 *  the loop state on the graph-captured launches. Collapses to an opaque,
 *  always-null pointer when built against CUDA < 12.4. */
#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
typedef const CUDAHybridGraphLoopState* CUDAHybridGraphLoopStateOpt;
#else
typedef const void* CUDAHybridGraphLoopStateOpt;
#endif  // EXABOOST_HYBRID_GRAPH_SUPPORTED

#ifdef __CUDACC__

__device__ __forceinline__ const data_size_t* HybridGraphMainIndices(
    CUDAHybridGraphLoopStateOpt gstate, const data_size_t* host_value) {
#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  return gstate != nullptr ? gstate->main_indices : host_value;
#else
  return host_value;
#endif
}

__device__ __forceinline__ data_size_t* HybridGraphOutIndices(
    CUDAHybridGraphLoopStateOpt gstate, data_size_t* host_value) {
#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  return gstate != nullptr ? gstate->out_indices : host_value;
#else
  return host_value;
#endif
}

/*! \brief deferred split-info slab slot base of the current level (0 on the
 *  host-launched per-level path, cumulative inside the graph loop) */
__device__ __forceinline__ int HybridGraphSplitInfoBase(
    CUDAHybridGraphLoopStateOpt gstate) {
#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  return gstate != nullptr ? gstate->split_info_base : 0;
#else
  return 0;
#endif
}

/*! \brief graphs A2 idle-block guard: true when this block sits beyond the
 *  current level's live split count (grids are frozen at a power-of-two
 *  bucket of n inside the graph loop; the excess blocks must exit before any
 *  other read or write). Always false on the host-launched exact-grid path. */
__device__ __forceinline__ bool HybridGraphBeyondLiveSplits(
    CUDAHybridGraphLoopStateOpt gstate, const unsigned int n_index) {
#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  return gstate != nullptr && n_index >= static_cast<unsigned int>(gstate->cur_num_splits);
#else
  (void)n_index;
  return false;
#endif
}

/*! \brief graphs A2 idle-block guard of the gap-copy kernel (live gap count) */
__device__ __forceinline__ bool HybridGraphBeyondLiveGaps(
    CUDAHybridGraphLoopStateOpt gstate, const unsigned int gap_index) {
#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  return gstate != nullptr && gap_index >= static_cast<unsigned int>(gstate->cur_num_gaps);
#else
  (void)gap_index;
  return false;
#endif
}

/*! \brief live pair count of the current graph level body (the batched
 *  construct kernel's device row-grouping must use it instead of gridDim.z
 *  once the grid is frozen at a pow2 bucket); the host-launched exact-grid
 *  path keeps its grid-derived value */
__device__ __forceinline__ int HybridGraphLivePairCount(
    CUDAHybridGraphLoopStateOpt gstate, const int host_value) {
#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  return gstate != nullptr ? gstate->cur_num_splits : host_value;
#else
  return host_value;
#endif
}

/*! \brief whether this launch runs inside the graph loop (the quantized body
 *  kernels then derive descriptor-independent values on-device) */
__device__ __forceinline__ bool HybridGraphActive(CUDAHybridGraphLoopStateOpt gstate) {
  return gstate != nullptr;
}

/*! \brief per-leaf histogram bit width of quantized training, derived from the
 *  (device-resident, exact) leaf row count: EXACTLY the host's
 *  GradientDiscretizer::SetNumBitsInHistogramBin thresholds, so the decision
 *  is bit-identical to the host bookkeeping. Only meaningful inside the graph
 *  loop (gstate != nullptr). */
__device__ __forceinline__ uint8_t HybridGraphQuantHistBits(
    CUDAHybridGraphLoopStateOpt gstate, const data_size_t num_data_in_leaf) {
#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
  const uint64_t max_stat_per_bin = static_cast<uint64_t>(num_data_in_leaf) *
    static_cast<uint64_t>(gstate->num_grad_quant_bins);
  return max_stat_per_bin < 256 ? 8 : (max_stat_per_bin < 65536 ? 16 : 32);
#else
  (void)gstate;
  (void)num_data_in_leaf;
  return 0;
#endif
}

#endif  // __CUDACC__

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_TREELEARNER_CUDA_CUDA_HYBRID_GRAPH_HPP_
