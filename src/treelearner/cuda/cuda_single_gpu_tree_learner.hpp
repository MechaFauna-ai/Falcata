/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_

#include <memory>
#include <string>
#include <vector>

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_device_tree_reason.hpp>

#include "cuda_leaf_splits.hpp"
#include "cuda_feature_parallel.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_data_partition.hpp"
#include "cuda_best_split_finder.hpp"
#include "cuda_device_levels.hpp"

#include "cuda_gradient_discretizer.hpp"
#include "../serial_tree_learner.h"

namespace Falcata {

#define CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE (1024)

class CUDASingleGPUTreeLearner: public SerialTreeLearner, public NCCLInfo {
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
  friend struct WholeMapOnlyFaultPeer;
#endif

 public:
  explicit CUDASingleGPUTreeLearner(const Config* config, const bool boosting_on_cuda);

  ~CUDASingleGPUTreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data,
                         bool is_constant_hessian) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override;

  void AddPredictionToScore(const Tree* tree, double* out_score) const override;

  void RestoreBagOrder() override;

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                       data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt, const double* train_score) const override;

  void ResetConfig(const Config* config) override;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) const override;

  void ResetBoostingOnGPU(const bool boosting_on_gpu) override;

  void SetNCCLInfo(
    ncclComm_t nccl_communicator,
    int nccl_gpu_rank,
    int local_gpu_rank,
    int gpu_device_id,
    data_size_t global_num_data) override;

  /*! \brief enable feature-parallel multi-GPU: this learner constructs and
   *  searches only inner features with index %% num_ranks == rank; per-level
   *  winners are merged host-side through the shared state. Mutually
   *  exclusive with SetNCCLInfo (no communicator, full data per rank). */
  void SetFeatureParallel(int rank, int num_ranks, FeatureParallelMergeState* state) {
    fp_rank_ = rank;
    fp_num_ranks_ = num_ranks;
    fp_merge_state_ = state;
  }

  // ---- cuda_plan key forest_batched: this learner grows trees of a random forest alongside other learners, one tree
  // per host thread (CUDAForestTrainer) ----
  /*! \brief Before Init: mark the learner a forest slot. With private_column_view the per-tree column views go to a
   *  view of the dataset's columns of its own (CUDAColumnData::InitViewOf) instead of the dataset's, which the
   *  booster's own learner (slot 0) keeps; with row_data_source the bin matrix (CUDARowData) is that learner's,
   *  not a copy; no compact prefill, no graph loop (its stream capture would forbid the other threads' API calls). */
  void EnableForestSlot(bool private_column_view, const CUDASingleGPUTreeLearner* row_data_source = nullptr) {
    forest_slot_ = true;
    forest_private_column_view_ = private_column_view;
    forest_row_data_source_ = row_data_source;
  }
  /*! \brief The level buffers that grow with the widest level seen, sized now for any level the tree shape allows
   *  (num_leaves / 2 + 2 pairs, fewer under max_depth), each up to an eighth of the device: a growth during training
   *  is a device synchronize, which stalls every forest slot, and holds the old block beside the new one (a forest's
   *  slots growing together took the device to its limit: numerai's level output is ~2 GiB per slot). Presized, the
   *  buffers count in the slot's Init footprint (CUDAForestTrainer::EnsureSlots) */
  void PresizeLevelBuffers();
  /*! \brief The booster learner leaves forest mode (its trainer is gone): graph loop and compact prefill as before */
  void LeaveForest() {
    forest_slot_ = false;
    forest_next_sample_ready_ = false;
  }
  /*! \brief cuda_plan key forest_single_job_device: the next Train is the only tree of its group (no other slot thread
   *  runs until it returns), so it trains as the sequential learner does: the flows forest slots exclude (graph
   *  capture, the device_levels prefix and the complete device tree) are open to it. The forest trainer sets it around
   *  one Train; the learner stays a forest slot otherwise (the test hooks keep counting its trees). */
  void SetForestSoloTree(bool solo) { forest_solo_ = solo; }
  /*! \brief The per-tree counters the next Train runs under: the tree part of the bynode node keys (bynode) and the
   *  quantized discretizer's tree counter (stochastic rounding noise, error-feedback slot). Sequential training
   *  advances each by one per tree; the forest trainer gives every tree the values sequential training would. */
  struct TreeCounters {
    uint64_t bynode = 0;
    int discretizer = 0;
  };
  TreeCounters tree_counters() const;
  void SetTreeCounters(const TreeCounters& counters);
  /*! \brief The next Train's feature_fraction sample (ColSampler::SetByTreeSample), drawn in tree order by the
   *  forest trainer from the booster learner's sampler, instead of this learner's own draw */
  void SetNextTreeSample(const std::vector<int>& used_feature_indices) {
    forest_next_sample_ = used_feature_indices;
    forest_next_sample_ready_ = true;
  }
  /*! \brief The sampler whose stream gives every tree its feature_fraction sample (the forest trainer draws from the
   *  booster learner's, in tree order) */
  ColSampler* mutable_col_sampler() { return &col_sampler_; }
  /*! \brief gpu device this learner trains on */
  int gpu_device_id() const { return gpu_device_id_; }

  /*! \brief Log::Fatal unless the multi-GPU flows can train this config: feature-parallel and quantized data-parallel
   *  ride the hybrid level flow only. Train checks it per tree (num_splits_done: the forced splits it applied);
   *  NCCLGBDT::Init checks it once after Init on the caller's thread, since a Fatal inside Train runs on a per-rank
   *  std::thread and terminates the process instead of raising. A no-op for single-GPU training. */
  void CheckMultiGPUFlowUsable(int num_splits_done) const;

 protected:
  bool IsCUDATreeLearner() const override { return true; }

  void BeforeTrain() override;

  // ---- linear tree support ----
  // Upload raw (un-binned) numerical feature columns to the GPU (feature-major).
  // Only done when config_->linear_tree is set; mirrors the host Dataset::raw_data_.
  void InitLinearTreeCUDA(const Dataset* train_data);
  // Fit a linear model (const + sum coeff*rawfeat) at each leaf of the just-built
  // tree, matching the CPU LinearTreeLearner. Heavy XtHX/Xtg accumulation runs on
  // the GPU; the small per-leaf solve runs on the host (Eigen) for CPU parity.
  void CalculateLinearCUDA(CUDATree* tree, const score_t* gradients, const score_t* hessians, bool is_first_tree);
  // Build the device per-leaf linear model arrays from the tree's CURRENT (post-shrinkage)
  // coefficients and upload them, so the score kernel matches the tree exactly (as the CPU
  // LinearTreeLearner does by reading tree->LeafConst/LeafCoeffs at score time).
  void BuildAndUploadLinearScoreModel(const Tree* tree) const;
  // Launch the per-leaf XtHX (upper-triangle) + Xtg accumulation kernel.
  void LaunchCalcLinearGramKernel(
    const int* cuda_data_index_to_leaf_index, const score_t* gradients, const score_t* hessians,
    const int* leaf_num_feat, const int* leaf_feat_offset, const int* leaf_feat_col,
    const int* leaf_xthx_offset, const int* leaf_xtg_offset, int num_leaves,
    double* cuda_xthx, double* cuda_xtg, int* cuda_leaf_nonzero);
  // Launch the linear score-update kernel (score += const + sum coeff*rawfeat, NaN -> leaf_output).
  void LaunchLinearAddScoreKernel(
    const int* cuda_data_index_to_leaf_index, const double* cuda_leaf_const,
    const double* cuda_leaf_coeff, const int* cuda_leaf_coeff_offset, const int* cuda_leaf_coeff_col,
    const int* leaf_num_coeff, const double* cuda_leaf_output, double* score) const;

  void ReduceLeafStat(CUDATree* old_tree, const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf) const;

  void LaunchReduceLeafStatKernel(const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf,
    const int* leaf_parent, const int* left_child, const int* right_child,
    const int num_leaves, const data_size_t num_data, double* cuda_leaf_value, const double shrinkage_rate) const;

  void ConstructBitsetForCategoricalSplit(const CUDASplitInfo* best_split_info);

  void LaunchConstructBitsetForCategoricalSplitKernel(const CUDASplitInfo* best_split_info);

  /*! \brief build ALL of a level's categorical split bitsets (real + inner)
   *  into per-split slab regions + write their word lengths; one launch */
  void LaunchBatchConstructCatBitsetsKernel(const int num_cat_splits);

  void AllocateBitset();

  // per-split loop: draw and upload the per-node masks of the pair's two leaves (shared stream, or the keyed draw
  // under level_bynode_masks)
  void SelectFeatureByNode(const Tree* tree, const int smaller_leaf_index, const int larger_leaf_index);

  // ---- cuda_plan key level_bynode_masks ----
  // the keyed mask of a leaf (ColSampler::GetByNodeKeyed with the leaf's node key) into out (num_features bytes);
  // FALCATA_DEBUG=bynode keeps a host copy per leaf
  void DrawKeyedNodeMask(const int leaf, int8_t* out, std::vector<int>* inner_features = nullptr);
  // level flows: draw one row per leaf (leaves < 0 skipped) and upload them with the per-leaf row table on the
  // level's histogram stream; a no-op unless bynode_level_masks_
  void UploadLevelNodeMasks(const std::vector<int>& leaves);
  // node keys: the left child keeps the parent's leaf index, the right child takes right_leaf; a child's key is the
  // parent's mixed with its side, so a node's key is a function of its path from the root (no-op unless keyed)
  void BynodeKeySplit(const int leaf, const int right_leaf) {
    if (!bynode_keyed_) {
      return;
    }
    const uint64_t parent = bynode_node_key_[leaf];
    bynode_node_key_[leaf] = ColSampler::MixKey(parent ^ 0x6A09E667F3BCC908ULL);
    bynode_node_key_[right_leaf] = ColSampler::MixKey(parent ^ 0xBB67AE8584CAA73BULL);
  }
  // FALCATA_DEBUG=bynode: the BYNODE line of a split applied at leaf (feature, the mask the leaf was searched with)
  void DebugPrintBynodeSplit(const int leaf, const int inner_feature) const;

  // Apply a split (device CUDASplitInfo + host-known feature/threshold) to the tree
  // and the data partition. Shared by the main loop and the forced-split pre-pass.
  // Returns the new (right) leaf index.
  int ApplySplit(CUDATree* tree, const CUDASplitInfo* best_split_info, const int leaf_index);
  // cuda_plan key split_counts_early: when the previous level's readback returned before its partition kernels and
  // tree record finished (ApplyInFlight), order the histogram stream after the partition and the finder's stream
  // after the record, and clear the flag; nothing otherwise
  void WaitForInFlightApply();
  // Record tree_record_done_event_ (created on first use) on the tree stream, after every record kernel of the level
  // is enqueued there; the next sync kernel on the finder's stream waits on it.
  void RecordTreeRecordDone(const CUDATree* tree);

  // ---- hybrid growth: depth-wise (level-batched) prefix + leaf-wise tail ----
  // While the leaf budget cannot bind, every positive-gain frontier leaf will be
  // split by leaf-wise growth eventually, so whole levels are grown with one
  // device synchronization instead of one full pipeline sync per split. Split
  // decisions are node-local, hence the resulting tree equals the leaf-wise tree
  // whenever the budget does not bind; the tail preserves exact leaf-wise
  // semantics once it might.
  bool HybridGrowthUsable() const;
  // cuda_plan key level_bynode_masks: per-node feature masks do not keep the level flows off (single GPU, both
  // batched phases available); with select_features_by_node_ and this false, the per-split loop runs
  bool LevelFlowsTakeNodeMasks() const;
  // cuda_plan key leaf_map_no_zero: whether this tree's flow makes the tree start's map zero store dead
  bool LeafMapZeroDead() const;
  // Whether the hybrid prefix will run the single-sync (speculative) level
  // pipeline (see TrainLevelWisePrefixOneSync); requires both batched phases
  // and non-quantized training.
  bool UseOneSyncPrefix() const;
  // Lazily perform the root-sum readback (and the root leaf-output init) that
  // BeforeTrain deferred for the single-sync flow. No-op when not deferred.
  void EnsureRootSumsReadBack(CUDATree* tree);
  // Enqueue histogram construction + best-split search for one sibling pair
  // (device work only; no host synchronization).
  void EnqueuePairBestSplitSearch(const CUDATree* tree,
    const CUDALeafSplitsStruct* smaller_struct, const CUDALeafSplitsStruct* larger_struct,
    int smaller_leaf_index, int larger_leaf_index, bool synchronize = true,
    int pipeline = 0, int parent_leaf_index = -1);
  // One sibling pair of a level awaiting its best-split search. parent is the
  // pair's parent leaf index (== the split's left leaf; -1 for the root pair):
  // the quantized path stores the parent's histogram bit width at that index.
  // Without selective growth right > left always holds, so parent ==
  // min(smaller, larger); with selective growth right-child indices are
  // recycled and the explicit field is required.
  struct HybridPendingPair {
    int smaller;
    int larger;
    int parent;
    const CUDALeafSplitsStruct* smaller_struct;
    const CUDALeafSplitsStruct* larger_struct;
  };
  // Batched replacement for the per-pair EnqueuePairBestSplitSearch loop of one
  // level: builds per-pair descriptors (validity flags, leaf sizes, quantized
  // histogram bit widths), uploads them in a single H2D copy, then runs ONE
  // batched construct/fix/subtract launch (histogram constructor) and ONE batched
  // find + sync launch (best split finder) covering every pair. Device work only;
  // the caller synchronizes via SyncAllLeafBestSplitsToHost.
  // vec_plane_slab is the level's per-plane leaf-splits slab in vector mode
  // (nullptr in scalar mode).
  void EnqueueLevelBestSplitSearch(const CUDATree* tree,
    const std::vector<HybridPendingPair>& pairs,
    CUDALeafSplitsStruct* vec_plane_slab = nullptr);
  // One applied split of a level: host-assigned child leaf indices plus the pair
  // struct slots the batched apply kernels fill (smaller/larger designation is
  // decided on-device; the host learns it from the deferred readback).
  struct HybridAppliedSplit {
    int left;
    int right;
    CUDALeafSplitsStruct* smaller_slot;
    CUDALeafSplitsStruct* larger_slot;
  };
  // Speculative variant of EnqueueLevelBestSplitSearch used by the single-sync
  // flow: enqueued right after the level's batched apply, BEFORE the child
  // statistics are read back. Descriptors carry only host-known data (struct slot
  // pointers, the max_depth gate); the batched kernels derive leaf indices,
  // sizes, hessian sums and the min_data/min_hessian gates from the child structs
  // the apply kernels write (ordered via the partition's apply-done event). The
  // construct grid is sized by the parents' sizes (smaller child <= parent / 2)
  // and the kernel restores the exact row grouping from the device-side actual
  // sizes, so results are bit-identical to the classic two-sync flow.
  // tree_record_done: an event recorded on the tree stream after the level's
  // tree record (nullptr: the flow records nothing on the tree stream before
  // its readback); the find/sync stream waits on it, since the record reads the
  // split leaves' cached best splits that the children's sync overwrites.
  void EnqueueLevelBestSplitSearchSpeculative(const CUDATree* tree,
    const std::vector<HybridAppliedSplit>& applied,
    cudaEvent_t tree_record_done);
  // Collect all leaves with a valid cached best-split candidate, mirroring the
  // device cache into the host split arrays (the leaf-wise tail may apply any of
  // them later).
  void CollectSplittableLeaves(const CUDATree* tree, std::vector<int>* splittable);
  /*! \brief feature-parallel: deterministic per-leaf winner merge across the
   *  rank threads (host buffers), then re-upload to the device leaf cache so
   *  the batched apply reads the winning split infos */
  void FeatureParallelMergeLevel(int num_leaves);
  // Leaf-budget arbitration of one level: returns false when the budget binds
  // and the level must defer to the leaf-wise tail; truncates splittable to a
  // final partial level (*final_partial_level = true) when every child would sit
  // at max_depth. Also applies the FALCATA_HYBRID_MAXSPLITS debug cap.
  bool ArbitrateLevelBudget(const CUDATree* tree, std::vector<int>* splittable,
    bool* final_partial_level);
  // Batched apply of one level's splits: tree-structure update (SplitBatch) +
  // partition (SplitLevelBatched), zero host synchronization; fills *applied.
  void ApplyLevelBatched(CUDATree* tree, const std::vector<int>& splittable,
    std::vector<HybridAppliedSplit>* applied, const bool final_level = false);
  // Shared per-level readback bookkeeping: consume the deferred split info of one
  // level's applied splits (child counts/starts/sums), update the host leaf
  // arrays, quantized histogram bit widths and smaller/larger designation, count
  // the splits, and (when next_pairs is non-null) emit the next level's pairs.
  void FinishLevelBookkeeping(const std::vector<HybridAppliedSplit>& applied,
    const std::vector<int>& batch_info,
    std::vector<HybridPendingPair>* next_pairs,
    int* num_splits);
  // the same over 18 ints per applied split at batch_info (the selective prefix's hand-off: the prefix's
  // split-batch readback in place, without a per-level slice copy)
  void FinishLevelBookkeeping(const std::vector<HybridAppliedSplit>& applied,
    const int* batch_info,
    std::vector<HybridPendingPair>* next_pairs,
    int* num_splits);
  // Grow full levels until the next level could exceed num_leaves. Returns the
  // number of splits applied; afterwards every current leaf has a valid cached
  // best-split candidate and smaller_/larger_leaf_index_ name the last applied pair.
  int TrainLevelWisePrefix(CUDATree* tree);
  // Single-sync variant of TrainLevelWisePrefix: enqueues each level's APPLY and
  // the children's SEARCH back to back, then reads best splits + deferred split
  // info back with ONE device synchronization per level (the classic flow pays
  // two). Non-quantized batched-kernels + batched-apply configurations only.
  int TrainLevelWisePrefixOneSync(CUDATree* tree);

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  // ---- graphs L1: device-driven level loop (see cuda_hybrid_graph.hpp) ----
  // One cached instantiated graph per per-tree pointer key (the compact column
  // view is double-buffered, so two keys alternate in the steady state).
  struct HybridGraphInstance {
    std::vector<const void*> key;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDAHybridGraphLoopState* state_dev = nullptr;  // raw cudaMalloc
    ~HybridGraphInstance();
  };
  // Static config gate of the graph loop (depth-limited one-sync prefix only,
  // controller capacity limits, no debug modes, the graph_loop plan key).
  bool HybridGraphPrefixUsable() const;
  // cuda_plan key forest_count_hist: this tree's gradient codes for the count construct (nullptr: not this tree)
  const CUDACountHistCodes* CountHistCodesForTree();
  // a tree that threw mid-way (tree_in_progress_ still set at the next Train): reset what the next tree trusts
  void AbortedTreeReset();
  // test hook FALCATA_TEST_FOREST_ABORT_TREE (see the definition)
  void MaybeInjectTreeFailure(int num_leaves);
  // Lazy one-time setup: static feature tables, journal/staging buffers, and
  // worst-case capacity preallocation of every captured device buffer.
  bool SetupHybridGraphStatics();
  // Build (capture + instantiate) one graph instance for the current per-tree
  // pointer key. Returns false (and rolls back) on any failure.
  bool BuildHybridGraphInstance(CUDATree* tree, HybridGraphInstance* instance);
  // The per-tree pointer/flag key a graph instance is valid for.
  void ComputeHybridGraphKey(CUDATree* tree, std::vector<const void*>* key) const;
  // Cached-or-new instance for this tree; nullptr -> host loop fallback.
  HybridGraphInstance* GetOrBuildHybridGraph(CUDATree* tree);
  // The graph-driven prefix: root search (host, as usual), ONE graph launch,
  // ONE readback + host bookkeeping replay. Returns the number of splits, or
  // -1 when no instance is available (caller falls back to the host loop).
  int TrainLevelWisePrefixGraph(CUDATree* tree);
  void ReleaseHybridGraphs();
  // ---- cuda_plan key device_levels (cuda_device_levels.hpp) ----
  /*! \brief cuda_plan key device_tree (with device_levels): the complete device tree runs for this config, i.e. its
   *  depth bound holds (positive max_depth <= kDeviceTreeMaxDepth = 12 and 2^max_depth <= num_leaves + 1) and its
   *  presized scratch passed the free-memory gate (EvaluateDeviceTreeScratchGate) */
  bool DeviceTreeRequested() const;
  // How many of the classic quantized prefix's first levels run device-driven for this tree: the levels L with
  // 2^(L + 1) <= num_leaves (the budget cannot bind) and L + 1 < max_depth (the children are searched), when every
  // kernel of the flow has a device-count variant; 0 = the host loop as before
  int DeviceLevelsToRun(bool batched_level_kernels, bool batched_level_apply);
  /*! \brief DeviceLevelsToRun refused: record why (device_levels_refusal_) and, under FALCATA_DEBUG=diag for a flow
   *  outside the complete device tree, report the prefix's own fallback once */
  void LogDeviceLevelsFallback(DeviceTreeReason reason);
  /*! \brief FALCATA_DEBUG=diag: why the complete device tree did not grow this tree (the first refusal in the order
   *  of FALCATA_DEVICE_TREE_REASONS that holds), from the learner's state after the tree's growth */
  DeviceTreeReason DeviceTreeRefusal() const;
  /*! \brief FALCATA_DEBUG=diag, at the end of a tree the complete device tree did not grow: the reason line
   *  `device_tree: fallback (<reason>) route=<route>[ prefix=<reason>]`, printed when it differs from this learner's
   *  last one */
  void LogDeviceTreeFallback();
  // ---- cuda_plan key device_bynode_masks ----
  /*! \brief the complete device tree takes this learner's per-node masks: keyed masks in the level flows (no
   *  interaction constraints, single GPU, the batched phases), no forest slot and no FALCATA_DEBUG=bynode (its BYNODE
   *  lines come from the host draws) */
  bool DeviceBynodeMasksUsable() const;
  /*! \brief before a complete device tree's root search: (re)build the device sample inputs and buffers, upload the
   *  tree's candidates where they changed, draw the root's mask on the side stream and point the finder at the device
   *  rows; device_bynode_active_ until EndDeviceBynode */
  void BeginDeviceBynode();
  void EndDeviceBynode();
  /*! \brief the masks of the root (level < 0) or of level \p level's children on the side stream, forked off \p stream
   *  (after the apply); JoinDeviceBynode orders the next level find after them */
  void EnqueueDeviceBynodeMasks(cudaStream_t stream, const int level, const int max_splits);
  void JoinDeviceBynode();
  // one-time buffers and static tables of the device levels; false disables them
  bool SetupDeviceLevels(const int num_levels);
  /*! \brief pairs of the widest child search of num_levels device levels (whole tree: 2^(num_levels - 2)) */
  int DeviceLevelSearchPairs(int num_levels) const;
  /*! \brief cuda_plan key device_tree: before the first tree's level flow (again after ReleaseHybridGraphs, e.g. a
   *  ResetConfig), whether a whole tree's presized scratch fits half of free device memory; when it does not,
   *  DeviceTreeRequested() is false (the prefix flow) */
  void EvaluateDeviceTreeScratchGate();
  // after the root level's search is enqueued: num_levels device-driven levels (apply + children's search), one
  // readback, the host bookkeeping replayed; host_leaf_best_splits_ then hold the last search's result and *pairs
  // the last level's pairs, as after the host loop's own readback. Returns the splits applied.
  // record_split_infos (cuda_plan key selective_device_prefix): the select also journals every applied split's
  // best-split record, read back with the journal into pinned_selective_prefix_split_infos_
  int TrainDeviceLevels(CUDATree* tree, const int num_levels, std::vector<HybridPendingPair>* pairs,
                        bool record_split_infos = false);
  // cuda_plan key selective_device_prefix: the selective flow's first levels as the device_levels prefix (the root
  // search, then TrainDeviceLevels), and the selective records rebuilt from its journal. True when it ran: the host
  // loop then continues with the readback of the last prefix level's search (host_leaf_best_splits_, *pairs).
  bool TrainSelectiveDevicePrefix(CUDATree* tree, std::vector<HybridPendingPair>* pairs);
  // the device levels' launches and stream edges (capturable: nothing on the legacy stream, nothing read back)
  void EnqueueDeviceLevels(CUDATree* tree, const CUDADeviceLevelConfig& cfg, const int num_levels,
                           const bool inline_all_map, const bool final_map_only);
  // cuda_plan key device_level_graph: launch the levels as a cached graph (captured on a key's first tree); false
  // when this tree cannot use one (the caller enqueues the levels)
  bool LaunchDeviceLevelGraph(CUDATree* tree, const CUDADeviceLevelConfig& cfg, const int num_levels,
                              const bool inline_all_map, const bool final_map_only);
  struct DeviceLevelGraph {
    std::vector<const void*> key;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~DeviceLevelGraph();
  };
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

  // ---- selective (grow-then-prune) hybrid growth: exact leaf-wise equivalence
  // in budget-limited configs (num_leaves << 2^max_depth) ----
  //
  // Leaf-wise growth is greedy selection: repeatedly apply the max-gain
  // candidate among AVAILABLE candidates (available = parent split already
  // applied) until num_leaves - 1 splits or no positive gain. Key monotonicity
  // property: newly discovered (deeper) candidates can only DISPLACE currently
  // selected candidates from the greedy selection, never promote unselected
  // ones (a new candidate inserts into the greedy sequence right after its
  // parent's selection and consumes budget earlier, shifting the tail out).
  // Hence, per level: run the greedy selection over all live applied splits +
  // the newly discovered frontier candidates, apply ONLY the selected frontier
  // candidates (unselected ones are permanently dormant), and eagerly collapse
  // applied splits displaced from the selection (their whole subtree is
  // displaced by availability closure). At termination the live applied splits
  // equal the classic leaf-wise selection, in the classic greedy order given by
  // the final simulation; the tree is then rebuilt host-side in that order
  // (classic node/leaf numbering => bit-identical model files in deterministic
  // configs). Eager collapse recycles hybrid leaf indices (and their histogram
  // slots), so live leaves never exceed num_leaves and no device buffer grows.

  // One applied split record of the selective growth phase.
  struct SelectiveApplied {
    int parent;             // id of the parent applied record (-1 for the root split)
    int left_leaf;          // hybrid leaf that was split (left child keeps it)
    int right_leaf;         // hybrid right child leaf (recycled index possible)
    int kid_left;           // applied child record at left_leaf (-1 none)
    int kid_right;          // applied child record at right_leaf (-1 none)
    bool dead;              // collapsed (pruned) -- never selectable again
    bool selected;          // in the current greedy selection (scratch)
    double gain;
    data_size_t data_start;  // the split node's window in the main index array
    data_size_t num_data;
    // captured host copy of the device CUDASplitInfo at apply time (categorical
    // pointers scrubbed by SyncAllLeafBestSplitsToHost)
    CUDASplitInfo info;
    // categorical splits: inner threshold bins snapshotted from the device slab
    // at record creation (the slab slot is reused once the leaf is re-searched,
    // so SelectiveFinalize cannot read it later)
    std::vector<uint32_t> cat_inner;
    // vector-leaf mode: kNumVecPayloadFields * T doubles snapshotted from the
    // per-leaf payload slab, under the same reuse discipline as cat_inner.
    // SelectiveFinalize replays both children's per-target outputs from it.
    std::vector<double> vec_payload;
  };
  // One frontier candidate (discovered this level, not yet applied).
  struct SelectiveFrontier {
    int parent;             // applied record that created this leaf (-1 root)
    int leaf;               // hybrid leaf index
    double gain;
  };

  // Resolve the fp32 histogram mode (FALCATA_FP32_HIST) against the consumers
  // its layout does not cover and hand the result to the best split finder.
  void SyncHistFP32();
  // Whether the budget-limited selective mode governs this tree's growth.
  bool UseSelectiveGrowth() const;
  // Exact host replica of FindBestFromAllSplitsKernel's equal-gain tie-break:
  // the winner among equal-gain candidates is decided by the kernel's
  // strided per-thread scan + shuffle-down reductions (keep-current-on-tie),
  // NOT by the smallest leaf index. Returns the winning classic leaf index.
  int SelectiveTieBreak(const std::vector<int>& tied_classic_leaves) const;
  // Enqueue the root level's batched search of the single-sync flow (descriptor
  // with device-derived gating, exact host-known construct grid). Shared by the
  // exact-fit one-sync prefix and the selective one-sync loop.
  void EnqueueRootLevelSearchOneSync(bool reuse_fused_root = false);
  // Reset the per-tree selective growth state.
  void SelectiveReset();
  // Collect the new frontier candidates from the given child leaves (children of
  // last level's applied splits; the root leaf on the first level).
  void SelectiveDiscoverFrontier(const std::vector<int>& child_leaves);
  // One level's host-side selection step: greedy simulation over live applied
  // splits + the current frontier (budget num_leaves - 1, gains descending,
  // ties to the smaller classic leaf index, matching FindBestFromAllSplits),
  // then eager collapse of displaced applied splits (device window rewrite +
  // histogram slot recycling) and creation of the applied records for the
  // selected frontier candidates (right-child indices from the free list).
  // Returns the number of frontier candidates to apply this level.
  int RunSelectiveLevel();
  // Batched partition-only apply of the selected frontier splits (the CUDA tree
  // object is not touched during selective growth; the final tree is rebuilt
  // host-side after pruning). Fills *applied for the readback bookkeeping.
  void ApplyLevelBatchedSelective(std::vector<HybridAppliedSplit>* applied);
  // The two growth loops (same level pipelines as the exact-fit prefix).
  void TrainSelectiveOneSync(CUDATree* tree);
  void TrainSelectiveTwoSync(CUDATree* tree);
  // Rebuild the final (pruned) tree host-side in classic greedy order, remap the
  // partition's data-index-to-leaf-index to final numbering and upload the final
  // per-leaf data layout.
  void SelectiveFinalize(CUDATree* tree);
  // Full selective training of one tree (grow + prune + finalize).
  void TrainSelective(CUDATree* tree);
  // Leaf depth during hybrid growth: the CUDA tree's host mirror on the exact-fit
  // paths, the selective host array when selective growth is active.
  int GrowthLeafDepth(const CUDATree* tree, const int leaf) const {
    return selective_active_ ? sel_leaf_depth_[leaf] : tree->leaf_depth(leaf);
  }

  // Apply forcedsplits_filename before the main split loop, mirroring
  // SerialTreeLearner::ForceSplits (numerical features, single-GPU, non-quantized).
  // Returns the number of forced splits applied.
  int ForceSplitsCUDA(CUDATree* tree, int* num_splits_done);

  #ifdef DEBUG
  void CheckSplitValid(
    const int left_leaf, const int right_leaf);
  #endif  // DEBUG

  void RenewDiscretizedTreeLeaves(CUDATree* cuda_tree);

  void LaunchCalcLeafValuesGivenGradStat(CUDATree* cuda_tree, const data_size_t* num_data_in_leaf);

  void NCCLReduceHistogram(const int pipeline);

  /*! \brief (re)build the compacted list of histogram bins this tree can read,
   *  from the histogram constructor's per-tree feature sample. Returns true
   *  when compaction is worthwhile (sampling active and a real reduction). */
  bool BuildNCCLReduceBinIndex();

  /*! \brief gather the used bins of `hist` into `cuda_nccl_reduce_buf_`
   *  (2 doubles per bin), and the inverse. */
  void LaunchNCCLGatherBins(const hist_t* hist, double* out, int num_bins, cudaStream_t stream);
  void LaunchNCCLScatterBins(const double* in, hist_t* hist, int num_bins, cudaStream_t stream);

  /*! \brief one grouped all-reduce for a whole level's smaller-leaf
   *  histograms (replaces one collective per split; the level path is
   *  latency-bound, so this is the dominant multi-GPU optimization).
   *  Histogram locations are read on-device from the pair descriptors'
   *  smaller_struct->hist_in_leaf -- the pointer the construct kernel itself
   *  wrote through -- because the level flow does not maintain the host-side
   *  leaf_to_hist_index_map_. */
  void NCCLReduceLevelHistograms(const int num_pairs);
  void LaunchNCCLGatherLevel(const CUDAHybridPairDescriptor* descs, const int* idx,
                             double* out, int num_bins, int num_pairs, cudaStream_t stream);
  void LaunchNCCLScatterLevel(const double* in, const int* idx,
                              const CUDAHybridPairDescriptor* descs,
                              int num_bins, int num_pairs, cudaStream_t stream);
  /*! \brief integer (quantized) variants: per-pair 16/32-bit histograms are
   *  widened into uniform int64 (grad32,hess32) lanes so one ncclInt64 sum
   *  serves the level; half the payload of the fp64-pair reduce, exact sums */
  void LaunchNCCLGatherLevelQuant(const CUDAHybridPairDescriptor* descs, const int* idx,
                                  int64_t* out, int num_bins, int num_pairs,
                                  cudaStream_t stream);
  void LaunchNCCLScatterLevelQuant(const int64_t* in, const int* idx,
                                   const CUDAHybridPairDescriptor* descs,
                                   int num_bins, int num_pairs, cudaStream_t stream);

  /*! \brief compacted bin indices this tree reads (empty = reduce everything) */
  CUDAVector<int> cuda_nccl_reduce_bin_idx_;
  /*! \brief staging buffer for the compacted all-reduce (2 doubles per bin) */
  CUDAVector<double> cuda_nccl_reduce_buf_;
  int nccl_reduce_num_bins_ = 0;

  // Per-tree compact column data: transposes the on-GPU row-major bin matrix into
  // a column-major buffer for partition kernels, avoiding the 17 GB per-column
  // allocation in CUDAColumnData.
  void BuildCompactColumnView();
  // Lazy fallback of the packed split read (FALCATA_SPLIT_PACKED_READ): the
  // batched apply path reads the packed compact matrix directly, so the
  // column-major gather is skipped in BuildCompactColumnView; any classic
  // per-split Split() (leaf-wise tail, forced splits, batched-apply fallback
  // env) still needs the plain per-column buffer, so ApplySplit calls this
  // first to run the deferred gather once for the tree.
  /*! \brief Grow the per-tree compact column buffer, refusing with a sized
   *  message rather than a bare cudaMalloc OOM when it cannot fit. */
  void EnsureCompactColumnBuffer(const size_t needed_bytes, const int num_compact_cols,
                                 const data_size_t num_data);

  void EnsureClassicColumnView();

  // ---- colmajor_fill: when to make the column-major copy and how much to leave free after it ----
  /*! \brief device bytes training can still allocate once the first tree is done (see the definition) */
  size_t ColMajorFillReserveBytes() const;
  /*! \brief called at the start of every tree: lets the first tree run, then decides once */
  void MaybeInitColMajorFill();
  bool colmajor_fill_pending_ = false;
  int colmajor_fill_trees_seen_ = 0;

  // ---- colmajor_direct: which view holds the bin matrix (CUDAHistogramConstructor::ChooseViewRegime) ----
  /*! \brief device bytes training can still allocate once the regime's view exists (see the definition) */
  size_t DirectViewReserveBytes() const;
  /*! \brief whether the split kernels read the packed compact matrix in place (BuildCompactColumnView); otherwise
   *  they get the one-byte-per-value view of the sampled columns */
  bool PackedSplitReadUsable() const;
  /*! \brief whether training builds that one-byte split view at some tree (see the definition) */
  bool BuildsOneByteSplitView() const;
  /*! \brief whether every tree runs the quantized per-leaf construct, which reads the whole matrix (see the
   *  definition) */
  bool TreesReadFullMatrix() const;
  /*! \brief choose the regime for the current data and feature_fraction (CUDAHistogramConstructor::ChooseViewRegime:
   *  the static rule, mask whenever the compact regime does not fit); why goes to the log */
  void InitViewRegime(const char* why);
  /*! \brief the feature_fraction the regime was chosen for (ResetConfig re-chooses only when it changes) */
  double view_decided_ff_ = -1.0;

  CUDAVector<uint8_t> compact_column_buffer_;
  std::vector<int> compact_column_to_orig_;        // [slot] -> original column index
  std::vector<int> orig_column_to_compact_slot_;   // [col] -> slot, or -1
  // Per-slot precomputed metadata for the new (slot, row)-launched transpose:
  CUDAVector<size_t> cuda_slot_p_byte_;
  CUDAVector<int> cuda_slot_p_stride_;
  CUDAVector<int> cuda_slot_col_in_p_;
  uint64_t compact_col_signature_ = 0;
  // Published-view generation when compact_col_signature_'s view was installed;
  // a mismatch means someone replaced the table and we must reinstall.
  uint64_t col_view_generation_seen_ = 0;
  // packed split read: this tree's column view is the packed compact matrix
  // (no column-major gather ran yet; see EnsureClassicColumnView)
  bool compact_packed_view_active_ = false;
  // gather inputs stashed for the lazy fallback (device pointer + 4-bit flag)
  const uint8_t* compact_gather_src_ = nullptr;
  bool compact_gather_src_is_4bit_ = false;
  // whether the installed column view reads colmajor_direct's full view (a cache hit must keep it alive)
  bool column_view_reads_full_view_ = false;

  // Mirror of BasicLeafConstraints::Update for the CUDA path. Given a just-applied
  // numerical split on `inner_feature_index`, the parent leaf (`left_leaf`) and the
  // newly created leaf (`right_leaf`), and the (clamped) child outputs, update the
  // host-side per-leaf [min,max] constraint arrays.
  void UpdateLeafConstraints(
    const int left_leaf, const int right_leaf, const int inner_feature_index,
    const double left_value, const double right_value, const bool is_numerical_split);

  // ---- vector-leaf (multi-target) training ----
  // Number of targets T; > 1 engages the vector flow: T histogram planes per
  // leaf slot, the summed-gain vector finder, and per-target leaf outputs.
  // Rides the classic per-split loop and both hybrid TWO-SYNC level flows: the
  // exact-fit level-batched prefix and selective grow-then-prune (the one-sync
  // and graph flows are excluded).
  int vec_num_targets_ = 0;
  // Per-plane leaf-splits structs: slots [0, T) mirror the smaller leaf with
  // plane t's gradient sums / histogram pointer, slots [T, 2T) the larger leaf.
  // This is also the ROOT level's plane slab of the hybrid prefix (pair 0's
  // smaller role at [0, T), larger role at [T, 2T)).
  CUDAVector<CUDALeafSplitsStruct> vec_plane_structs_;
  // Hybrid level prefix: per-plane leaf-splits structs of a whole level, at
  // [(2 * pair + role) * T + t] over the same pair/role order as
  // hybrid_pair_slots_ (role 0 = smaller, 1 = larger). Plane 0 of each role is
  // a copy of the primary slot the partition wrote.
  CUDAVector<CUDALeafSplitsStruct> vec_level_plane_slots_;
  // Fan-out inputs of one level: the parent split info and left-child leaf
  // index of every applied split (device copies).
  std::vector<const CUDASplitInfo*> host_vec_fanout_infos_;
  std::vector<int> host_vec_fanout_left_leaves_;
  CUDAVector<const CUDASplitInfo*> cuda_vec_fanout_infos_;
  CUDAVector<int> cuda_vec_fanout_left_leaves_;
  // Per-target root sums (host copies of the per-plane InitValues readbacks).
  std::vector<double> vec_root_sum_gradients_;
  std::vector<double> vec_root_sum_hessians_;
  // Refuse dataset/config shapes the vector flow does not cover.
  void CheckVectorLeafSupported() const;
  // Root init of the plane structs (BeforeTrain tail).
  void VectorInitRootPlanes(const data_size_t* leaf_splits_init_indices,
                            const data_size_t root_num_data);
  // Per-pair search: per-plane construct + subtract, then the vector finder.
  void EnqueuePairBestSplitSearchVector(const CUDATree* tree,
    const int smaller_leaf_index, const int larger_leaf_index);
  // Refresh the plane structs from the just-applied split (partition wrote the
  // primary smaller/larger structs; planes add per-target sums/values and the
  // plane histogram pointers).
  void LaunchVectorPlaneFanOutKernel(const CUDASplitInfo* best_split_info,
    const int left_leaf_index);
  // Level-batched counterpart: refresh every plane struct of a level's applied
  // splits from the primary slots the batched apply wrote.
  void VectorLevelPlaneFanOut(const std::vector<HybridAppliedSplit>& applied);
  void LaunchVectorLevelPlaneFanOutKernel(const int num_pairs);
  // Per-plane histogram phase + vector find of one level (the vector branch of
  // EnqueueLevelBestSplitSearch); plane_slab holds this level's plane structs.
  void EnqueueLevelHistogramsAndFindVector(const int num_pairs,
    const data_size_t max_num_data_in_smaller_leaf,
    CUDALeafSplitsStruct* plane_slab);

  // cuda_plan key forest_batched (EnableForestSlot, SetTreeCount, SetNextTreeSample)
  bool forest_slot_ = false;
  // cuda_plan key forest_single_job_device (SetForestSoloTree): this tree is its group's only one
  bool forest_solo_ = false;
  /*! \brief whether other slot threads may launch while this tree trains: a forest slot not training a group's only
   *  tree. Their legacy-stream launches would join this thread's stream capture, so no capture and no device-driven
   *  level flow then. */
  bool ForestSlotShared() const { return forest_slot_ && !forest_solo_; }
  bool forest_private_column_view_ = false;
  // the learner whose bin matrix (CUDARowData, read-only after its Init) this slot's histogram constructor shares
  const CUDASingleGPUTreeLearner* forest_row_data_source_ = nullptr;
  std::unique_ptr<CUDAColumnData> private_column_view_;
  std::vector<int> forest_next_sample_;
  bool forest_next_sample_ready_ = false;
  /*! \brief the column view this learner installs its per-tree views into: its private one (forest slots) or the
   *  dataset's */
  CUDAColumnData* ColumnView() const {
    return private_column_view_ ? private_column_view_.get() :
      const_cast<CUDAColumnData*>(train_data_->cuda_column_data());
  }

  // number of threads on CPU
  int num_threads_;

  // FALCATA_FIXEDPOINT_QUANT: non-stochastic near-lossless quant mode.
  // Resolved once in Init so the histogram constructor and the gradient
  // discretizer share the same effective quant bin count.
  bool fixedpoint_quant_ = false;
  // Outlier-robust gradient scale within fixed-point mode (default ON when
  // fixedpoint_quant_ is on; disabled by cuda_plan=auto,robust_scale:off).
  bool fixedpoint_robust_scale_ = false;
  // Compact-view prefill: the previous Train() already drew this tree's
  // column sample (BeforeTrain must not draw again; cuda_plan compact_prefill)
  bool next_tree_col_sample_ready_ = false;

  /*! \brief quantized training: the batched construct's saturation floor for this device (160 x SM count / 170,
   *  clamped to [10, 5120]; 0 = the plan's value), see Init */
  int construct_saturation_floor_ = 0;
  int effective_quant_bins_ = 0;

  // CUDA components for tree training

  // leaf splits information for smaller and larger leaves
  std::unique_ptr<CUDALeafSplits> cuda_smaller_leaf_splits_;
  std::unique_ptr<CUDALeafSplits> cuda_larger_leaf_splits_;
  // hybrid growth phase-1 state: per-split (smaller, larger) struct slots for the
  // pairs of one level (2 slots per applied split, reused across levels), and the
  // host copy of the device per-leaf best-split cache. The slots are one device
  // slab of bare CUDALeafSplitsStructs (they are only ever written/read by the
  // split kernels through their struct pointers); allocating num_leaves + 2
  // CUDALeafSplits objects instead costs ~3 cudaMallocs each (~3000 calls,
  // ~3.7 ms) inside the first Train() call.
  bool use_hybrid_growth_ = false;
  CUDAVector<CUDALeafSplitsStruct> hybrid_pair_slots_;
  std::vector<CUDASplitInfo> host_leaf_best_splits_;
  std::vector<CUDAHybridPairDescriptor> host_hybrid_pair_descs_;
  CUDAVector<CUDAHybridPairDescriptor> cuda_hybrid_pair_descs_;
  std::vector<CUDATreeBatchSplit> host_tree_batch_splits_;
  std::vector<CUDAHybridApplySplitInput> host_apply_split_inputs_;
  /*! \brief the final level's split-inner wrote the leaf map inline; the
   *  tree-end materialize covers only hybrid_map_residual_leaves_ */
  bool hybrid_map_final_written_ = false;
  std::vector<int> hybrid_map_residual_leaves_;
  /*! \brief cuda_plan key skip_empty_tail: ArbitrateLevelBudget's last level splits every candidate leaf, the leaf
   *  budget does not bind and every child sits at max_depth; set by the level prefix when it ended on such a level */
  bool level_completes_tree_ = false;
  bool prefix_completes_tree_ = false;
  /*! \brief cuda_plan key device_tree: the whole tree applied no split and its classic root search (the host level
   *  flows' first leaf-wise tail search) found one; Train's tail continues from that cached search */
  bool device_tree_root_tail_ = false;
  /*! \brief cuda_plan key dirty_final_slots: the tree's leaf count before the final level that completed the level
   *  prefix (prefix_completes_tree_), -1 otherwise */
  int prefix_final_leaves_before_ = -1;
  /*! \brief cuda_plan key readback_kernel: mapped pinned staging of the tree end's readbacks (the tree's pooled slab
   *  in CUDATree::ToHost, the exact leaf counts), its device alias (nullptr where the device cannot map it) and size */
  void* readback_staging_ = nullptr;
  void* readback_staging_device_ = nullptr;
  size_t readback_staging_bytes_ = 0;
  /*! \brief readback_staging_ of at least `bytes` bytes, or false (key off or no mapping: use cudaMemcpy) */
  bool EnsureReadbackStaging(size_t bytes);
  /*! \brief cuda_plan key early_leaf_map: the residual leaves' map pass was launched with the final level, when
   *  the tree had hybrid_map_early_num_leaves_ leaves (a leaf-wise tail split after it re-runs the pass) */
  bool hybrid_map_early_written_ = false;
  int hybrid_map_early_num_leaves_ = 0;
  /*! \brief cuda_plan key final_readback_first: the early map pass is due, launched by FlushEarlyLeafMap from the
   *  tree end's ToHost right after the tree's device copy (if no flush runs, the tree end writes the residual leaves
   *  as without early_leaf_map) */
  bool hybrid_map_early_pending_ = false;
  void FlushEarlyLeafMap(const CUDATree* tree);
  /*! \brief cuda_plan key final_readback_first: the tree end's slab and leaf-count readbacks launched ahead of the
   *  final level's split batch readback (PrefetchTreeReadback): the tree's leaf count then (0: none), and whether
   *  the counts were copied too (behind the slab in readback_staging_, as in Train's early_to_host) */
  int prefetched_tree_num_leaves_ = 0;
  bool prefetched_tree_counts_ = false;
  void PrefetchTreeReadback(const CUDATree* tree);
  // hybrid growth: single-sync (speculative) level pipeline
  // (FALCATA_HYBRID_ONE_SYNC, default on; "0" keeps the classic two-sync flow)
  bool use_hybrid_one_sync_ = false;
  bool selective_active_ = false;
  /*! \brief the controller that grew the current tree (FALCATA_DEBUG=diag reason line), set where each one runs */
  GrowthRoute growth_route_ = GrowthRoute::kClassic;
  /*! \brief why DeviceLevelsToRun refused a device_levels prefix this tree (kNone: it did not, or was not asked) */
  DeviceTreeReason device_levels_refusal_ = DeviceTreeReason::kNone;
  /*! \brief the last reason line this learner printed (LogDeviceTreeFallback prints only changes) */
  std::string last_device_tree_fallback_line_;
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  // ---- graphs L1 state ----
  bool hybrid_graph_statics_ready_ = false;
  bool hybrid_graph_disabled_ = false;   // sticky off after build failures
  int hybrid_graph_build_failures_ = 0;
  std::vector<std::unique_ptr<HybridGraphInstance>> hybrid_graph_cache_;
  CUDAVector<CUDAHybridGraphFeatureMeta> cuda_hybrid_graph_feature_meta_;
  CUDAVector<CUDAHybridGraphFeatureSource> cuda_hybrid_graph_feature_source_;
  CUDAVector<double> cuda_hybrid_graph_real_thresholds_;
  CUDAVector<int> cuda_hybrid_graph_journal_;
  std::vector<CUDAHybridGraphFeatureSource> host_hybrid_graph_feature_source_;
  // pinned staging: per-tree state prefix + journal readback
  CUDAHybridGraphLoopState* pinned_hybrid_graph_state_ = nullptr;
  int* pinned_hybrid_graph_journal_ = nullptr;
  size_t hybrid_graph_journal_size_ = 0;
  // static config mirror written into every instance's device state
  CUDAHybridGraphLoopState hybrid_graph_state_template_;
  std::vector<int> hybrid_graph_splittable_scratch_;
  std::vector<int> hybrid_graph_batch_info_;
  std::vector<HybridAppliedSplit> hybrid_graph_applied_scratch_;
  // ---- cuda_plan key device_levels state ----
  bool device_tree_completed_ = false;
  bool device_levels_logged_ = false;
  bool device_levels_fallback_logged_ = false;
  bool device_levels_ready_ = false;
  bool device_levels_disabled_ = false;
  /*! \brief EvaluateDeviceTreeScratchGate's decision: -1 not yet made, 0 over the free-memory gate, 1 fits */
  int device_tree_scratch_gate_ = -1;
  /*! \brief pairs SetupDeviceLevels sized the per-pair search scratch for */
  int device_level_max_pairs_ = 0;
  CUDAVector<CUDADeviceLevelState> cuda_device_level_state_;
  // the select / pairs kernels' inputs in device memory (the apply's pairs block reads them there), uploaded per tree
  CUDAVector<CUDADeviceLevelConfig> cuda_device_level_config_;
  CUDADeviceLevelConfig host_device_level_config_;
  CUDAVector<int> cuda_device_level_journal_;
  // cuda_plan key selective_device_prefix: the applied splits' best-split records (device journal, pinned readback)
  CUDAVector<CUDASplitInfo> cuda_selective_prefix_split_infos_;
  void* pinned_selective_prefix_split_infos_ = nullptr;
  size_t pinned_selective_prefix_split_infos_count_ = 0;
  bool selective_prefix_logged_ = false;
  int* pinned_device_level_journal_ = nullptr;
  size_t device_level_journal_size_ = 0;
  CUDAVector<CUDAHybridGraphFeatureMeta> cuda_device_level_feature_meta_;
  CUDAVector<CUDAHybridGraphFeatureSource> cuda_device_level_feature_source_;
  CUDAVector<double> cuda_device_level_real_thresholds_;
  std::vector<CUDAHybridGraphFeatureSource> host_device_level_feature_source_;
  cudaEvent_t device_level_select_event_ = nullptr;
  std::vector<int> device_level_batch_info_;
  std::vector<int> device_level_batch_slice_;
  std::vector<HybridAppliedSplit> device_level_applied_;
  std::vector<std::unique_ptr<DeviceLevelGraph>> device_level_graphs_;
  bool device_level_graph_disabled_ = false;
  int device_level_graph_hits_ = 0;
  int device_level_graph_misses_ = 0;
  // cuda_plan key device_level_graph_update: a cached exec refused an in-place update (a topology change)
  bool device_level_graph_update_failed_ = false;
  int device_level_graph_update_hits_ = 0;
  int device_level_graph_update_misses_ = 0;
  // the cudaGraphExecUpdateResult of the last refused update (-1: none), for FALCATA_DEBUG=diag
  int device_level_graph_update_last_refusal_ = -1;
  // ---- cuda_plan key device_bynode_masks state ----
  CUDADeviceBynodeConfig device_bynode_cfg_ = {};
  // the row table (a pointer per leaf slot, padded to 256 bytes), then the rows
  CUDAVector<int8_t> cuda_bynode_rows_;
  size_t bynode_table_bytes_ = 0;
  CUDAVector<int> cuda_bynode_tasks_;
  CUDAVector<uint64_t> cuda_bynode_keys_;
  CUDAVector<int> cuda_bynode_sample_;
  CUDAVector<int> cuda_bynode_task_inner_;
  // the sample inputs last uploaded (CUDADeviceBynodeConfig::sample's layout), and this tree's
  std::vector<int> host_bynode_sample_;
  std::vector<int> bynode_sample_scratch_;
  std::vector<int> bynode_candidates_scratch_;
  // the side stream the mask kernels run on beside the construct, its fork and join events
  cudaStream_t bynode_stream_ = nullptr;
  cudaEvent_t bynode_fork_event_ = nullptr;
  cudaEvent_t bynode_join_event_ = nullptr;
  bool bynode_join_pending_ = false;
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  std::vector<SelectiveApplied> sel_applied_;
  std::vector<SelectiveFrontier> sel_frontier_;
  std::vector<int> sel_leaf_parent_;    // hybrid leaf -> applied record id (-1 root)
  std::vector<int> sel_leaf_depth_;
  std::vector<int> sel_leaf_classic_;   // classic leaf index (last simulation)
  std::vector<int> sel_leaf_frontier_;  // hybrid leaf -> frontier idx (per level)
  std::vector<char> sel_leaf_live_;
  std::vector<int> sel_free_leaves_;    // recycled hybrid leaf indices (LIFO)
  std::vector<int> sel_order_;          // last simulation: greedy order (applied ids, ~frontier idx)
  std::vector<int> sel_order_classic_leaf_;
  std::vector<int> sel_level_apply_;    // applied record ids to apply this level
  std::vector<CUDACollapseWindow> sel_collapse_windows_;
  std::vector<int> sel_freed_hist_slots_;
  // vector-leaf mode: this level's host snapshot of the per-leaf payload slab
  // (kNumVecPayloadFields * T doubles per hybrid leaf), read once per level
  // right after the best-split readback
  std::vector<double> sel_leaf_vec_payloads_;
  int sel_num_allocated_ = 0;           // peak hybrid leaf count (dirty hist slots)
  int sel_num_splits_ = 0;              // applied splits surviving readback bookkeeping
  int sel_last_peak_ = 0;               // sel_num_allocated_ of the finished tree
  // churn statistics (FALCATA_HYBRID_DEBUG): applied / displaced split totals
  int64_t sel_stat_applied_ = 0;
  int64_t sel_stat_displaced_ = 0;
  int64_t sel_stat_levels_ = 0;
  int64_t sel_stat_trees_ = 0;
  int64_t sel_stat_sim_ns_ = 0;
  // reusable device slab for CUDATree's per-tree arrays (see CUDATree ctor)
  CUDAVector<uint8_t> cuda_tree_pool_buffer_;
  // reusable growth buffers for the categorical split record of the tree being grown (see CUDATreeCatArena)
  CUDATreeCatArena cuda_tree_cat_arena_;
  // whether BeforeTrain deferred the root-sum readback (single-sync flow); the
  // readback (plus the root leaf-output init) then happens lazily in
  // EnsureRootSumsReadBack on the paths that need host root sums
  bool root_sums_deferred_ = false;
  /*! \brief cuda_plan key async_tree_start: kept host copies of BuildCompactColumnView's local slot tables */
  TreeStartUploads tree_start_uploads_;
  // data partition that partitions data indices into different leaves
  std::unique_ptr<CUDADataPartition> cuda_data_partition_;
  // for histogram construction
  std::unique_ptr<CUDAHistogramConstructor> cuda_histogram_constructor_;
  // cuda_plan key forest_count_hist: codes per gradient array (shared with the forest slots) and the diag line's state
  std::shared_ptr<CUDACountHistCache> count_hist_cache_;
  bool count_hist_logged_ = false;
  // a tree started and not completed (Train threw): the next Train resets what it would take from it
  bool tree_in_progress_ = false;
  // FALCATA_TEST_FOREST_ABORT_TREE: trees the booster learner started in forest mode, and whether the hook fired
  int forest_main_trees_started_ = 0;
  bool injected_tree_failure_ = false;
  // cuda_plan key tree_meta_batch: staging of the tree start's metadata uploads (see cuda_meta_batch.hpp)
  TreeStartMetaBatch tree_meta_batch_;
  // for best split information finding, given the histograms
  std::unique_ptr<CUDABestSplitFinder> cuda_best_split_finder_;
  // gradient discretizer for quantized training
  std::unique_ptr<CUDAGradientDiscretizer> cuda_gradient_discretizer_;

  std::vector<int> leaf_best_split_feature_;
  // FALCATA_DEBUG=bynode only: the per-node feature mask each leaf was last searched with
  std::vector<std::vector<int8_t>> debug_bynode_masks_;
  std::vector<uint32_t> leaf_best_split_threshold_;
  std::vector<uint8_t> leaf_best_split_default_left_;
  std::vector<data_size_t> leaf_num_data_;
  std::vector<data_size_t> leaf_data_start_;
  std::vector<double> leaf_sum_gradients_;
  std::vector<double> leaf_sum_hessians_;
  // A completed, narrowly gated whole tree may omit host child statistics and discretizer bit updates.
  // These caches then cannot feed host growth/CPU renewal. BeforeTrain rebuilds the active root before
  // clearing this flag; subsequent ordinary splits populate every newly active child before it is read.
  bool whole_host_leaf_cache_stale_ = false;
  // host-side per-leaf monotone constraints (basic method). [-DBL_MAX, +DBL_MAX]
  // means unconstrained. Empty/unused when monotone_constraints is not set.
  std::vector<double> leaf_constraint_min_;
  std::vector<double> leaf_constraint_max_;
  // per-inner-feature monotone constraint, mapped from the real-indexed
  // config_->monotone_constraints; empty when monotone_constraints is not set.
  std::vector<int8_t> monotone_constraints_;
  bool use_monotone_constraints_;
  int smaller_leaf_index_;
  int larger_leaf_index_;
  int best_leaf_index_;
  int num_cat_threshold_;
  bool has_categorical_feature_;
  // whether need to select features by node
  bool select_features_by_node_;
  // cuda_plan key level_bynode_masks: per-node masks keyed by (feature_fraction_seed, tree count, node path) in
  // every flow (feature_fraction_bynode < 1, no interaction constraints)
  bool bynode_keyed_ = false;
  // ... and the level flows run with them (the batched level kernels and apply are available)
  bool bynode_level_masks_ = false;
  // cuda_plan key device_bynode_masks: this tree's masks are drawn on the device (BeginDeviceBynode .. EndDeviceBynode)
  bool device_bynode_active_ = false;
  // trees this learner has trained (the tree part of the node keys)
  uint64_t bynode_tree_count_ = 0;
  // per leaf index: the key of the node the leaf currently is
  std::vector<uint64_t> bynode_node_key_;
  // per-split loop: host rows of the keyed masks of the pair's two leaves
  std::vector<int8_t> bynode_pair_masks_[2];
  // level flows: the leaves of the level's mask rows
  std::vector<int> bynode_level_leaves_;
  // level flows: each row's sampled inner features, rows in bynode_level_leaves_ order (CSR)
  std::vector<int> bynode_row_feature_begin_;
  std::vector<int> bynode_row_features_;
  std::vector<int> bynode_one_row_features_;
  // level flows: the leaves the level searches, before UploadLevelNodeMasks filters them
  std::vector<int> level_search_leaves_;

  std::vector<int> categorical_bin_to_value_;
  std::vector<int> categorical_bin_offsets_;

  mutable CUDAVector<double> cuda_leaf_gradient_stat_buffer_;
  mutable CUDAVector<double> cuda_leaf_hessian_stat_buffer_;
  mutable data_size_t refit_num_data_;
  uint32_t* cuda_bitset_;
  size_t cuda_bitset_len_;
  /*! \brief level-batched categorical bitset construction: per-split slab
   *  strides (words) + device staging; ONE kernel + ONE length readback per
   *  level replaces the per-split 4-kernel/2-sync construction swarm */
  size_t batch_cat_bitset_stride_ = 0;
  size_t batch_cat_bitset_inner_stride_ = 0;
  CUDAVector<const CUDASplitInfo*> cuda_batch_cat_infos_;
  CUDAVector<int> cuda_batch_cat_feats_;
  CUDAVector<uint32_t> cuda_batch_cat_bitset_;
  CUDAVector<uint32_t> cuda_batch_cat_bitset_inner_;
  CUDAVector<uint64_t> cuda_batch_cat_lens_;
  std::vector<const CUDASplitInfo*> host_batch_cat_infos_;
  std::vector<int> host_batch_cat_feats_;
  std::vector<uint64_t> host_batch_cat_lens_;
  uint32_t* cuda_bitset_inner_;
  size_t cuda_bitset_inner_len_;
  size_t* cuda_block_bitset_len_buffer_;
  int* cuda_categorical_bin_to_value_;
  int* cuda_categorical_bin_offsets_;

  /*! \brief gradients on CUDA */
  CUDAVector<score_t> cuda_gradients_;
  /*! \brief hessians on CUDA */
  CUDAVector<score_t> cuda_hessians_;
  /*! \brief whether boosting is done on CUDA */
  bool boosting_on_cuda_;

  // members used in multi-GPU training
  /*! \brief cuda stream for nccl operations */
  cudaStream_t nccl_stream_;
  // ---- feature-parallel multi-GPU (tree_learner=feature) ----
  /*! \brief rank of this learner's feature stripe; -1 = feature parallel off */
  int fp_rank_ = -1;
  int fp_num_ranks_ = 0;
  FeatureParallelMergeState* fp_merge_state_ = nullptr;
  /*! \brief per-tree feature mask: col_sampler sample intersected with the
   *  rank stripe (inner feature %% fp_num_ranks_ == fp_rank_) */
  std::vector<int8_t> fp_feature_mask_;
  /*! \brief scratch for the deterministic per-leaf winner merge */
  std::vector<CUDASplitInfo> fp_merge_tmp_;
  /*! \brief scatter-complete marker of the level all-reduce: the deferred
   *  fix/subtract on the histogram stream waits on this instead of a
   *  host-blocking stream sync (one host round trip per level saved) */
  cudaEvent_t nccl_reduce_done_event_ = nullptr;
  /*! \brief categorical-slab consumption markers: recorded after the batched
   *  bitset build (legacy stream) and the tree's categorical recording (tree
   *  stream); the next level's best-split sync waits on both so its per-leaf
   *  threshold deep-copies cannot overwrite entries still being consumed */
  cudaEvent_t cat_consumed_default_event_ = nullptr;
  cudaEvent_t cat_consumed_tree_event_ = nullptr;
  /*! \brief recorded on the tree stream after a level's tree record (SplitBatch / SplitCategorical) where no device
   *  synchronize separates it from the next level's sync kernel: the one-sync prefix's speculative search, and the
   *  classic prefix under cuda_plan key split_counts_early (a non-final level); the next level's find/sync stream
   *  waits on it: the record reads the cuda_leaf_best_split_info_ entries that sync kernel overwrites. Created on
   *  first use and recorded by RecordTreeRecordDone. */
  cudaEvent_t tree_record_done_event_ = nullptr;
  /*! \brief index map from leaf index to histogram index */
  std::vector<int> leaf_to_hist_index_map_;
  /*! \brief number of total histogram bins */
  int num_total_bin_;
  /*! \brief global number of data in the leaves across */
  std::vector<data_size_t> global_num_data_in_leaf_;

  #ifdef DEBUG
  /*! \brief gradients on CPU */
  std::vector<score_t> host_gradients_;
  /*! \brief hessians on CPU */
  std::vector<score_t> host_hessians_;
  #endif  // DEBUG

  // ---- linear tree support ----
  // Raw un-binned numerical feature values on device, feature-major:
  // cuda_raw_data_[col * num_data_ + row]. col indexes numerical features in
  // dataset feature order (== host Dataset::numeric_feature_map_ order).
  CUDAVector<float> cuda_raw_data_;
  int num_numeric_features_ = 0;
  // inner feature index -> numeric column in cuda_raw_data_ (-1 if categorical)
  std::vector<int> inner_feature_to_numeric_col_;
  // device scratch for the per-leaf gram accumulation
  CUDAVector<double> cuda_linear_xthx_;
  CUDAVector<double> cuda_linear_xtg_;
  CUDAVector<int> cuda_linear_nonzero_;
  // device copies of the fitted per-leaf linear model, used by the score kernel.
  // Rebuilt from the (post-shrinkage) tree at score time, so mutable/const-built.
  mutable CUDAVector<double> cuda_linear_leaf_const_;
  mutable CUDAVector<double> cuda_linear_leaf_coeff_;
  mutable CUDAVector<int> cuda_linear_leaf_coeff_offset_;
  mutable CUDAVector<int> cuda_linear_leaf_coeff_col_;
  mutable CUDAVector<int> cuda_linear_leaf_num_coeff_;
  mutable CUDAVector<double> cuda_linear_leaf_output_;
  mutable bool last_tree_is_linear_ = false;
};

}  // namespace Falcata

#else  // USE_CUDA

// When GPU support is not compiled in, quit with an error message

namespace Falcata {

class CUDASingleGPUTreeLearner: public SerialTreeLearner {
 public:
    #pragma warning(disable : 4702)
    explicit CUDASingleGPUTreeLearner(const Config* tree_config, const bool /*boosting_on_cuda*/) : SerialTreeLearner(tree_config) {
      Log::Fatal("CUDA Tree Learner was not enabled in this build.\n"
                 "Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs)");
    }
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_
