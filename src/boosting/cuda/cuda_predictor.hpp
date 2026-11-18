/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_HPP_
#define FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_HPP_

#ifdef USE_CUDA

#include <Falcata/boosting.h>
#include <Falcata/cuda/cuda_utils.hu>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace Falcata {

/*!
 * \brief The native CUDA predictor: evaluates a model's trees on dense rows on the device.
 *
 * Routing is exact in both summation modes: a float64 input is compared with the double thresholds; a float32 input x
 * is compared with the largest float not above the threshold t, which decides (double)x <= t exactly for every float x
 * (floats are a subset of doubles). Values with |x| <= kZeroThreshold are zero, NaN and zero go where
 * Tree::NumericalDecision sends them, categorical splits follow Tree::CategoricalDecision. Linear leaves add
 * const + coeff_i * x_i with separate rounding of each product and sum, as the CPU build (-ffp-contract=off) does. So
 * every row reaches the CPU predictor's leaves, and the two modes differ only in how the leaf values are added:
 *  - deterministic (opt-in, CUDAPredictRequest::deterministic): each output's raw score is the sum in tree order and
 *    in double precision, exactly as GBDT::PredictRawInRange adds it (__dadd_rn, no contraction), so the raw scores
 *    are bit-identical to the CPU predictor's;
 *  - default: the canonical pairwise sum (PairwisePlan), which lets the trees of one row be spread over many
 *    threads. It is a fixed order that depends only on the iteration range, never on the batch, the plan or the
 *    device, so a row gets the same result in any batch, run and GPU; it differs from the in-order sum by rounding.
 * The normal (transformed) prediction is the raw score passed through Boosting::ConvertRawToPrediction on the host, the
 * same code PredictInRange runs.
 */

/*! \brief flag bits of PackedNode::feature_flags; the feature index uses the bits below kNodeFeatureMask */
constexpr uint32_t kNodeCategorical = 1u << 31;
constexpr uint32_t kNodeDefaultLeft = 1u << 30;
constexpr int kNodeMissingShift = 28;           // two bits: MissingType::None / Zero / NaN
constexpr uint32_t kNodeFeatureMask = (1u << 28) - 1;

/*!
 * \brief One internal node. Children are global indices into the forest: >= 0 an internal node, < 0 the leaf ~child
 *        (a global leaf index). A categorical node keeps its index into the bitset ranges in the bits of threshold.
 */
template <typename T>
struct PackedNode {
  T threshold;
  uint32_t feature_flags;
  int32_t left;
  int32_t right;
};

/*!
 * \brief The compact layout the default mode walks (when the model fits it): each tree is a run of slots in depth-first
 *        order, a node's left child in the next slot and its right child `offset` slots on; a leaf's slot holds its
 *        value (a double), so a walk ends with the load that reaches the leaf. A slot is 8 bytes for float thresholds
 *        (CompactSlot<float>: threshold or categorical range index, meta) and 16 for double thresholds. meta holds the
 *        feature in its low `feature_bits` bits, the right child's offset up to bit 26, then the flags below. Models
 *        with linear leaves, vector leaves, or a feature index or subtree that does not fit stay on PackedNode only;
 *        the layout changes no result.
 */
constexpr uint32_t kSlotLeftLeaf = 1u << 26;
constexpr uint32_t kSlotRightLeaf = 1u << 27;
constexpr int kSlotMissingShift = 28;           // two bits, as kNodeMissingShift
constexpr uint32_t kSlotDefaultLeft = 1u << 30;
constexpr uint32_t kSlotCategorical = 1u << 31;
constexpr uint32_t kSlotLowMask = (1u << 26) - 1;   // feature and offset

template <typename T>
struct CompactSlot;
template <>
struct __align__(8) CompactSlot<float> {
  uint32_t lo;     // threshold bits / categorical range index, or the leaf value's low word
  uint32_t meta;   // or the leaf value's high word
};
template <>
struct __align__(16) CompactSlot<double> {
  double value;    // threshold, categorical range index (low word), or leaf value
  uint32_t meta;
  uint32_t pad;
};

/*! \brief device pointers of one packed forest (every tree of the model) */
template <typename T>
struct DeviceForestView {
  const PackedNode<T>* nodes = nullptr;
  const int32_t* tree_root = nullptr;         // [num_trees]: the root node, or ~leaf for a single-leaf tree
  const int32_t* tree_leaf_begin = nullptr;   // [num_trees]: global index of the tree's leaf 0
  const double* leaf_values = nullptr;        // [total_leaves * leaf_dim]
  const uint32_t* cat_words = nullptr;        // categorical bitsets
  const int64_t* cat_range = nullptr;         // [2 * num_cat_nodes]: word begin, word count
  const uint8_t* tree_is_linear = nullptr;    // [num_trees]
  const int64_t* linear_begin = nullptr;      // [total_leaves + 1] into linear_coeff / linear_feature
  const double* linear_const = nullptr;       // [total_leaves]
  const double* linear_coeff = nullptr;
  const int32_t* linear_feature = nullptr;
  const int32_t* node_count = nullptr;        // [total_nodes]: data count of each internal node (TreeSHAP)
  const int32_t* leaf_count = nullptr;        // [total_leaves]: data count of each leaf (TreeSHAP)
  const double* tree_expected = nullptr;      // [num_trees]: Tree::ExpectedValue (TreeSHAP)
  const CompactSlot<T>* slots = nullptr;      // the compact layout, or nullptr when the model does not fit it
  const int32_t* slot_start = nullptr;        // [num_trees]: the tree's first slot; ~slot for a single-leaf tree
  int feature_bits = 0;
  int leaf_dim = 1;
  int max_depth = 0;                          // edges on the longest root-to-leaf path
  bool has_categorical = false;
  bool has_linear = false;
};

/*! \brief one dense input matrix on the device: element (r, c) at data[r * row_stride + c * col_stride] */
template <typename T>
struct DeviceRows {
  const T* data = nullptr;
  int64_t num_row = 0;
  int32_t num_col = 0;
  int64_t row_stride = 0;
  int64_t col_stride = 0;
};

/*! \brief what a CUDA prediction measured: device bytes it held and how it split a host input */
struct CUDAPredictStats {
  int64_t forest_bytes = 0;     // the packed model, cached across calls
  int64_t scratch_bytes = 0;    // input chunk and output buffers of this call
  int64_t num_chunks = 0;       // host inputs: number of row chunks copied to the device
  int64_t chunk_rows = 0;
};

/*!
 * \brief How the deterministic mode computes raw scores; both kernels add each output's leaf values in tree order.
 *  - kTileWs (the plan's choice): a block stages `tile_rows` rows in shared memory; warp 0 keeps their running sums
 *    (lane r adds row r's leaf values in tree order) while the other warps walk the next tile of `tile_trees` trees
 *    for every (row, tree) pair, `ilp` walks in flight per thread; the tiles' leaf values are double-buffered.
 *  - kRow: a thread per row walks the trees, `ilp` of them at a time; for models with so many outputs per row that
 *    a block cannot hold its rows' sums.
 * Measured alternatives that lost (benchmarks/predict_gpu.py rounds, docs/performance.md): a thread per row for
 * every shape, tiles summed by one thread per row after a barrier, an 8-byte depth-first node layout, tree nodes
 * staged in shared memory, and two passes (all leaf values, then the sums) for small batches.
 */
struct PredictPlan {
  enum Kernel { kRow, kTileWs } kernel = kTileWs;
  int ilp = 2;
  int tile_rows = 8;
  int tile_trees = 64;
  bool stage_rows = true;   // rows in shared memory (else read from global memory)
  int block_threads = 256;  // warp 0 sums, the others walk trees
};

/*! \brief iterations whose leaf values the default mode adds in tree order before the pairwise tree */
constexpr int kPairwiseGroupIterations = 8;

/*!
 * \brief The default (fast) mode. The canonical pairwise sum of one output of one row: the iterations of the range
 *        are cut into groups of kPairwiseGroupIterations (the last one may be shorter); each group's leaf values are
 *        added in tree order starting from 0.0; the group sums are then added by the complete binary tree over the
 *        group index (groups (0,1), (2,3), ..., then pairs of those, and so on), a missing partner counting as 0.0.
 *        Adding 0.0 changes nothing (no partial sum is ever -0.0, since every sum starts from +0.0), so any
 *        decomposition into aligned subtrees gives the same bits. The kernel computes such subtrees:
 *  - a block takes `tile_rows` rows and `out_chunk` outputs, and an aligned span of `span_groups` groups (all of them
 *    when `num_spans` is 1; otherwise each span's subtree goes to a scratch array and a second kernel adds the spans
 *    by the same tree);
 *  - it walks the span in waves of `wave_groups` groups: one task per (row, output, group) walks the group's trees
 *    (`ilp` at a time) and adds them in order; aligned segments of `seg` group sums are added by warp shuffles; the
 *    segment sums enter a binary-counter stack (`levels` deep) per (row, output) that keeps the aligned subtree sums.
 */
struct PairwisePlan {
  int tile_rows = 8;
  int out_chunk = 1;
  int wave_groups = 32;
  int seg = 32;
  int span_groups = 32;
  int num_spans = 1;
  int levels = 1;
  int ilp = 8;
  bool stage_rows = true;
  bool compact = true;      // walk the compact layout when the model has it
  int stage_cols = 0;       // a staged row covers max(input columns, stage_cols): the model's features
  int group_lanes = 1;      // lanes that share one group's walks (2, 4 or 8 for small batches)
  int block_threads = 256;
};

/*! \brief launchers (cuda_predictor.cu); out is row-major [num_row][num_output] doubles */
template <typename TIn, typename TThr>
void LaunchPredictRawKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                            int num_iteration, int num_tree_per_iteration, int num_output, bool vector_leaf,
                            const PredictPlan& plan, double* out, cudaStream_t stream);

/*!
 * \brief One model's prediction in a batched launch (LaunchPredictPairwiseJobs): the default mode on rows of its own,
 *        over its own iteration range, with its own plan; the job's blocks start at `block_begin` of the flattened grid
 *        and its (row, output) cells at `cell_begin` of the combine pass.
 */
template <typename T>
struct PairwiseJob {
  DeviceForestView<T> forest;
  DeviceRows<T> rows;
  int tree_begin = 0;
  int num_iteration = 0;
  int num_tree_per_iteration = 1;
  PairwisePlan plan;
  double* out = nullptr;
  double* partials = nullptr;
  int64_t block_begin = 0;
  int64_t cell_begin = 0;
};

/*!
 * \brief Launches a batch of jobs that share the input type, the layout (compact when `compact`) and whether rows
 *        are staged; `device_jobs` is the jobs' table in device memory (block_begin and cell_begin filled, ascending),
 *        `jobs` the same table on the host. Each job's result equals its own LaunchPredictPairwiseKernel's.
 */
template <typename T>
void LaunchPredictPairwiseJobs(const std::vector<PairwiseJob<T>>& jobs, const PairwiseJob<T>* device_jobs,
                               bool has_categorical, bool compact, bool stage, size_t shared_bytes,
                               cudaStream_t stream);

/*!
 * \brief the default mode's launcher (scalar leaves): out is row-major [num_row][num_tree_per_iteration] doubles;
 *        `partials` holds plan.num_spans * num_row * num_tree_per_iteration doubles when plan.num_spans > 1
 */
template <typename TIn, typename TThr>
void LaunchPredictPairwiseKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                                 int num_iteration, int num_tree_per_iteration, const PairwisePlan& plan, double* out,
                                 double* partials, cudaStream_t stream);
size_t PairwiseSharedBytes(const PairwisePlan& plan, int num_col, size_t input_elem);

/*! \brief out is row-major [num_row][num_tree] int32 leaf indices of the trees [tree_begin, tree_begin + num_tree) */
template <typename TIn, typename TThr>
void LaunchPredictLeafKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                             int num_tree, int32_t* out, cudaStream_t stream);

/*!
 * \brief Feature contributions (TreeSHAP, Tree::PredictContrib) of rows [0, rows.num_row): out is row-major
 *        [num_row][num_tree_per_iteration][num_feature + 1]; scratch holds the path buffers of the threads in flight
 *        (ContribScratchElements per row and output)
 */
template <typename TIn, typename TThr>
void LaunchPredictContribKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                                int num_iteration, int num_tree_per_iteration, int num_feature, double* out,
                                void* scratch, cudaStream_t stream);
size_t ContribScratchBytes(int max_depth);

/*! \brief shared memory a kTileWs launch needs, and the most the device grants a block */
size_t TileWsSharedBytes(const PredictPlan& plan, int num_col, size_t input_elem, int num_tree_per_iteration);
size_t MaxSharedBytesPerBlock();

class CUDAPackedForest;
class CUDAPredictor;

/*! \brief One model's part of a CUDAPredictor::PredictMany request: its rows, its iteration range, its output */
struct CUDAPredictJob {
  const Boosting* boosting = nullptr;
  CUDAPredictor* predictor = nullptr;   // the booster's predictor, which keeps its packed forests
  Boosting::PredictRange range;
  const void* data = nullptr;
  int64_t num_row = 0;
  int32_t num_col = 0;
  int64_t row_stride = 0;
  int64_t col_stride = 0;
  void* out = nullptr;                  // doubles [num_row][num_output], where the input is (host or device)
};

/*! \brief A request to CUDAPredictor::Predict */
struct CUDAPredictRequest {
  const void* data = nullptr;
  int data_type = 0;            // C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
  int64_t num_row = 0;
  int32_t num_col = 0;
  int64_t row_stride = 0;       // in elements; host inputs must be C or F contiguous
  int64_t col_stride = 0;
  bool on_device = false;       // data and out are device pointers (else host pointers)
  int predict_type = 0;         // C_API_PREDICT_NORMAL, _RAW_SCORE or _LEAF_INDEX
  int device_id = -1;           // host inputs: the device to use (-1: the current device)
  cudaStream_t stream = nullptr;  // device inputs: the stream the input was written on
  bool deterministic = false;   // raw scores: sum in tree order (bit-identical to the CPU), else the pairwise sum
  void* out = nullptr;          // doubles [num_row][num_output], or int32 [num_row][num_tree] for leaf indices
};

/*!
 * \brief The predictor a Booster keeps: packs the model once per (device, threshold precision) and evaluates any
 *        iteration range from that one copy. Invalidate() must be called whenever the model changes.
 */
class CUDAPredictor {
 public:
  CUDAPredictor();
  ~CUDAPredictor();
  void Predict(const Boosting* boosting, const Boosting::PredictRange& range, const CUDAPredictRequest& request,
               CUDAPredictStats* stats);
  void Invalidate();
  /*!
   * \brief Several models' normal or raw-score predictions in a few launches. `common` gives what every job shares:
   *        data_type, on_device (device inputs on one device), stream, predict_type, device_id and deterministic.
   *        The default mode batches every job whose model has scalar leaves and no linear leaves into launches of the
   *        pairwise kernel (each job keeps its own plan, so its result equals its own Predict's); every other job,
   *        and every job of the deterministic mode, runs through Predict one after the other.
   */
  static void PredictMany(const std::vector<CUDAPredictJob>& jobs, const CUDAPredictRequest& common,
                          CUDAPredictStats* stats);
  /*! \brief a call's device scratch (the default mode's span sums): kept per device across calls, one call at a time */
  class Scratch;

 private:
  std::shared_ptr<const CUDAPackedForest> GetForest(const Boosting* boosting, int device_id, bool double_thresholds,
                                                    bool compact);
  std::mutex mutex_;
  // (device, double thresholds, compact layout)
  std::map<std::tuple<int, bool, bool>, std::shared_ptr<const CUDAPackedForest>> forests_;
  std::mutex scratch_mutex_;
  std::map<std::pair<int, int>, std::unique_ptr<CUDAVector<char>>> scratch_;   // (device, stream lane)
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_HPP_
