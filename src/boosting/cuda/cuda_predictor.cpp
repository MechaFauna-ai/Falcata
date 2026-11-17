/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#include "cuda_predictor.hpp"

#ifdef USE_CUDA

#include <Falcata/c_api.h>
#include <Falcata/falcata_plan.h>
#include <Falcata/tree.h>
#include <Falcata/utils/log.h>
#include <Falcata/utils/openmp_wrapper.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Falcata {

/*! \brief reads the tree arrays the device layout is packed from (a friend of Tree) */
struct TreeDevicePacker {
  static int NumLeaves(const Tree& t) { return t.num_leaves_; }
  static int NumCat(const Tree& t) { return t.num_cat_; }
  static int LeafDim(const Tree& t) { return t.leaf_value_dim_; }
  static bool IsLinear(const Tree& t) { return t.is_linear_; }
  static const std::vector<int>& SplitFeature(const Tree& t) { return t.split_feature_; }
  static const std::vector<double>& Threshold(const Tree& t) { return t.threshold_; }
  static const std::vector<int8_t>& DecisionType(const Tree& t) { return t.decision_type_; }
  static const std::vector<int>& LeftChild(const Tree& t) { return t.left_child_; }
  static const std::vector<int>& RightChild(const Tree& t) { return t.right_child_; }
  static const std::vector<int>& CatBoundaries(const Tree& t) { return t.cat_boundaries_; }
  static const std::vector<uint32_t>& CatThreshold(const Tree& t) { return t.cat_threshold_; }
  static double LeafValue(const Tree& t, int leaf, int k) {
    return t.leaf_value_dim_ > 1 ? t.leaf_values_vec_[static_cast<size_t>(leaf) * t.leaf_value_dim_ + k]
                                 : t.leaf_value_[leaf];
  }
  static double LeafConst(const Tree& t, int leaf) { return t.leaf_const_[leaf]; }
  static const std::vector<double>& LeafCoeff(const Tree& t, int leaf) { return t.leaf_coeff_[leaf]; }
  static const std::vector<int>& LeafFeatures(const Tree& t, int leaf) { return t.leaf_features_[leaf]; }
  static int NodeCount(const Tree& t, int node) { return t.internal_count_[node]; }
  static int LeafCount(const Tree& t, int leaf) { return t.leaf_count_[leaf]; }
  static double ExpectedValue(const Tree& t) { return t.ExpectedValue(); }
};

namespace {

// the largest float that is not above t: for every float x, (double)x <= t exactly when x <= FloatThresholdBelow(t)
float FloatThresholdBelow(double t) {
  float f = static_cast<float>(t);
  if (static_cast<double>(f) > t) {
    f = std::nextafter(f, -std::numeric_limits<float>::infinity());
  }
  return f;
}

template <typename T>
T ThresholdFor(double t);
template <>
float ThresholdFor<float>(double t) { return FloatThresholdBelow(t); }
template <>
double ThresholdFor<double>(double t) { return t; }

template <typename T>
void SetThresholdBits(T* threshold, uint32_t bits);
template <>
void SetThresholdBits<float>(float* threshold, uint32_t bits) { std::memcpy(threshold, &bits, sizeof(bits)); }
template <>
void SetThresholdBits<double>(double* threshold, uint32_t bits) {
  // the low word on the device is __double2loint: the low 32 bits of the IEEE pattern
  const uint64_t pattern = bits;
  std::memcpy(threshold, &pattern, sizeof(pattern));
}

template <typename T>
int64_t VectorBytes(const CUDAVector<T>& v) {
  return static_cast<int64_t>(v.Size() * sizeof(T));
}

// restores the calling thread's current device: a prediction must not move the caller's (e.g. CuPy's) device
class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) {
    CUDASUCCESS_OR_FATAL(cudaGetDevice(&previous_));
    if (device_id >= 0 && device_id != previous_) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    }
  }
  ~DeviceGuard() { cudaSetDevice(previous_); }

 private:
  int previous_ = 0;
};

// a non-blocking stream for the chunks of one host input, destroyed however the call ends
class OwnedStream {
 public:
  OwnedStream() { CUDASUCCESS_OR_FATAL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~OwnedStream() {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

}  // namespace

/*! \brief every tree of a model, packed for one device and one threshold precision; immutable once built */
class CUDAPackedForest {
 public:
  CUDAPackedForest(const Boosting* boosting, bool double_thresholds) : double_thresholds_(double_thresholds) {
    Boosting::PredictRange all;
    all.start_iteration = 0;
    all.num_iteration = boosting->GetCurrentIteration();
    const Boosting::TreeRangeView view = boosting->TreesInRange(all);
    num_trees_ = static_cast<int>(view.trees.size());
    leaf_dim_ = view.vector_leaf ? view.num_output : 1;
    if (double_thresholds) {
      Pack<double>(view, &nodes_double_);
    } else {
      Pack<float>(view, &nodes_float_);
    }
  }

  template <typename T>
  DeviceForestView<T> View() const;

  int64_t Bytes() const {
    return VectorBytes(nodes_float_) + VectorBytes(nodes_double_) + VectorBytes(tree_root_) +
           VectorBytes(tree_leaf_begin_) + VectorBytes(leaf_values_) + VectorBytes(node_count_) +
           VectorBytes(leaf_count_) + VectorBytes(tree_expected_) + VectorBytes(cat_words_) +
           VectorBytes(cat_range_) + VectorBytes(tree_is_linear_) + VectorBytes(linear_begin_) +
           VectorBytes(linear_const_) + VectorBytes(linear_coeff_) + VectorBytes(linear_feature_);
  }

  int num_trees() const { return num_trees_; }
  int max_depth() const { return max_depth_; }

 private:
  template <typename T>
  void Pack(const Boosting::TreeRangeView& view, CUDAVector<PackedNode<T>>* device_nodes) {
    using P = TreeDevicePacker;
    std::vector<PackedNode<T>> nodes;
    std::vector<int32_t> tree_root, tree_leaf_begin;
    int64_t num_leaves_total = 0;
    std::vector<double> leaf_values;
    std::vector<int32_t> node_count, leaf_count;
    std::vector<double> tree_expected;
    std::vector<uint32_t> cat_words;
    std::vector<int64_t> cat_range;
    std::vector<uint8_t> tree_is_linear;
    std::vector<int64_t> linear_begin(1, 0);
    std::vector<double> linear_const, linear_coeff;
    std::vector<int32_t> linear_feature;
    for (const Tree* tree : view.trees) {
      const Tree& t = *tree;
      const int num_leaves = P::NumLeaves(t);
      const bool linear = P::IsLinear(t);
      if (P::LeafDim(t) != leaf_dim_) {
        Log::Fatal("CUDA predictor: a tree stores %d values per leaf, the model %d", P::LeafDim(t), leaf_dim_);
      }
      // children become global indices: internal nodes offset by the nodes before this tree, leaves by the leaves
      const int64_t node_base = static_cast<int64_t>(nodes.size());
      const int64_t leaf_base = num_leaves_total;
      if (node_base + num_leaves > std::numeric_limits<int32_t>::max() ||
          leaf_base + num_leaves > std::numeric_limits<int32_t>::max()) {
        Log::Fatal("CUDA predictor: the model has more than 2^31 nodes or leaves");
      }
      auto global = [node_base, leaf_base](int child) {
        return static_cast<int32_t>(child >= 0 ? node_base + child : ~(leaf_base + ~child));
      };
      tree_root.push_back(num_leaves > 1 ? static_cast<int32_t>(node_base) : static_cast<int32_t>(~leaf_base));
      tree_leaf_begin.push_back(static_cast<int32_t>(leaf_base));
      if (num_leaves > 1) {
        const auto& feature = P::SplitFeature(t);
        const auto& threshold = P::Threshold(t);
        const auto& decision = P::DecisionType(t);
        const auto& left = P::LeftChild(t);
        const auto& right = P::RightChild(t);
        for (int i = 0; i < num_leaves - 1; ++i) {
          PackedNode<T> n;
          if (feature[i] < 0 || static_cast<uint32_t>(feature[i]) > kNodeFeatureMask) {
            Log::Fatal("CUDA predictor: split feature %d is outside [0, %u]", feature[i], kNodeFeatureMask);
          }
          n.feature_flags = static_cast<uint32_t>(feature[i]);
          n.left = global(left[i]);
          n.right = global(right[i]);
          // Tree::GetLeaf takes the categorical branch only when the tree has categorical splits at all
          const bool categorical = P::NumCat(t) > 0 && (decision[i] & kCategoricalMask);
          if (categorical) {
            const auto& boundaries = P::CatBoundaries(t);
            const auto& bits = P::CatThreshold(t);
            const int cat_idx = static_cast<int>(threshold[i]);
            const uint32_t range = static_cast<uint32_t>(cat_range.size() / 2);
            cat_range.push_back(static_cast<int64_t>(cat_words.size()));
            cat_range.push_back(boundaries[cat_idx + 1] - boundaries[cat_idx]);
            cat_words.insert(cat_words.end(), bits.begin() + boundaries[cat_idx],
                             bits.begin() + boundaries[cat_idx + 1]);
            n.threshold = T(0);
            SetThresholdBits<T>(&n.threshold, range);
            n.feature_flags |= kNodeCategorical;
          } else {
            n.threshold = ThresholdFor<T>(threshold[i]);
            const uint32_t missing = static_cast<uint32_t>((decision[i] >> 2) & 3);
            n.feature_flags |= missing << kNodeMissingShift;
            if (decision[i] & kDefaultLeftMask) {
              n.feature_flags |= kNodeDefaultLeft;
            }
          }
          nodes.push_back(n);
          node_count.push_back(P::NodeCount(t, i));
        }
        // depth of the deepest leaf: a walk is never longer, which sizes nothing but is reported to the planner
        std::vector<std::pair<int, int>> stack(1, std::make_pair(0, 1));
        while (!stack.empty()) {
          const auto cur = stack.back();
          stack.pop_back();
          max_depth_ = std::max(max_depth_, cur.second);
          if (left[cur.first] >= 0) stack.emplace_back(left[cur.first], cur.second + 1);
          if (right[cur.first] >= 0) stack.emplace_back(right[cur.first], cur.second + 1);
        }
      }
      num_leaves_total += num_leaves;
      for (int leaf = 0; leaf < num_leaves; ++leaf) {
        for (int k = 0; k < leaf_dim_; ++k) {
          leaf_values.push_back(P::LeafValue(t, leaf, k));
        }
        leaf_count.push_back(P::LeafCount(t, leaf));
        if (linear) {
          const auto& coeff = P::LeafCoeff(t, leaf);
          const auto& feats = P::LeafFeatures(t, leaf);
          linear_const.push_back(P::LeafConst(t, leaf));
          linear_coeff.insert(linear_coeff.end(), coeff.begin(), coeff.end());
          linear_feature.insert(linear_feature.end(), feats.begin(), feats.end());
        } else {
          linear_const.push_back(0.0);
        }
        linear_begin.push_back(static_cast<int64_t>(linear_coeff.size()));
      }
      tree_is_linear.push_back(linear ? 1 : 0);
      // TreeSHAP's base value; vector-leaf trees have none (their contributions are refused before any kernel)
      tree_expected.push_back(leaf_dim_ == 1 ? P::ExpectedValue(t) : 0.0);
      has_linear_ = has_linear_ || linear;
    }
    has_categorical_ = !cat_range.empty();
    if (cat_range.size() / 2 > std::numeric_limits<uint32_t>::max()) {
      Log::Fatal("CUDA predictor: more than 2^32 categorical splits");
    }
    for (int32_t f : linear_feature) {
      if (f < 0) {
        Log::Fatal("CUDA predictor: a linear leaf uses feature %d", f);
      }
    }
    // a CUDAVector of size 0 holds no pointer; kernels never read an empty array
    if (!nodes.empty()) device_nodes->InitFromHostVector(nodes);
    if (!tree_root.empty()) {
      tree_root_.InitFromHostVector(tree_root);
      tree_leaf_begin_.InitFromHostVector(tree_leaf_begin);
    }
    if (!leaf_values.empty()) leaf_values_.InitFromHostVector(leaf_values);
    if (!node_count.empty()) node_count_.InitFromHostVector(node_count);
    if (!leaf_count.empty()) leaf_count_.InitFromHostVector(leaf_count);
    if (!tree_expected.empty()) tree_expected_.InitFromHostVector(tree_expected);
    if (has_categorical_) {
      if (!cat_words.empty()) cat_words_.InitFromHostVector(cat_words);
      cat_range_.InitFromHostVector(cat_range);
    }
    if (has_linear_) {
      tree_is_linear_.InitFromHostVector(tree_is_linear);
      linear_begin_.InitFromHostVector(linear_begin);
      linear_const_.InitFromHostVector(linear_const);
      if (!linear_coeff.empty()) {
        linear_coeff_.InitFromHostVector(linear_coeff);
        linear_feature_.InitFromHostVector(linear_feature);
      }
    }
  }

  template <typename T>
  DeviceForestView<T> FillView(const CUDAVector<PackedNode<T>>& nodes) const {
    DeviceForestView<T> v;
    v.nodes = nodes.RawData();
    v.tree_root = tree_root_.RawData();
    v.tree_leaf_begin = tree_leaf_begin_.RawData();
    v.max_depth = max_depth_;
    v.leaf_values = leaf_values_.RawData();
    v.node_count = node_count_.RawData();
    v.leaf_count = leaf_count_.RawData();
    v.tree_expected = tree_expected_.RawData();
    v.cat_words = cat_words_.RawData();
    v.cat_range = cat_range_.RawData();
    v.tree_is_linear = tree_is_linear_.RawData();
    v.linear_begin = linear_begin_.RawData();
    v.linear_const = linear_const_.RawData();
    v.linear_coeff = linear_coeff_.RawData();
    v.linear_feature = linear_feature_.RawData();
    v.leaf_dim = leaf_dim_;
    v.has_categorical = has_categorical_;
    v.has_linear = has_linear_;
    return v;
  }

  bool double_thresholds_;
  int num_trees_ = 0;
  int leaf_dim_ = 1;
  int max_depth_ = 0;
  bool has_categorical_ = false;
  bool has_linear_ = false;
  CUDAVector<PackedNode<float>> nodes_float_;
  CUDAVector<PackedNode<double>> nodes_double_;
  CUDAVector<int32_t> tree_root_;
  CUDAVector<int32_t> tree_leaf_begin_;
  CUDAVector<double> leaf_values_;
  CUDAVector<int32_t> node_count_;
  CUDAVector<int32_t> leaf_count_;
  CUDAVector<double> tree_expected_;
  CUDAVector<uint32_t> cat_words_;
  CUDAVector<int64_t> cat_range_;
  CUDAVector<uint8_t> tree_is_linear_;
  CUDAVector<int64_t> linear_begin_;
  CUDAVector<double> linear_const_;
  CUDAVector<double> linear_coeff_;
  CUDAVector<int32_t> linear_feature_;
};

template <>
DeviceForestView<float> CUDAPackedForest::View<float>() const { return FillView(nodes_float_); }
template <>
DeviceForestView<double> CUDAPackedForest::View<double>() const { return FillView(nodes_double_); }

CUDAPredictor::CUDAPredictor() {}

CUDAPredictor::~CUDAPredictor() {}

void CUDAPredictor::Invalidate() {
  std::lock_guard<std::mutex> lock(mutex_);
  forests_.clear();
}

std::shared_ptr<const CUDAPackedForest> CUDAPredictor::GetForest(const Boosting* boosting, int device_id,
                                                                 bool double_thresholds) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& slot = forests_[std::make_pair(device_id, double_thresholds)];
  if (slot == nullptr) {
    slot = std::make_shared<const CUDAPackedForest>(boosting, double_thresholds);
  }
  return slot;
}

namespace {

// the kernel for a launch: row-parallel when the rows alone fill the device, else tree-parallel tiles; FALCATA_DEBUG
// predict_* tokens override (benchmarks; every kernel gives the same result)
PredictPlan ChoosePlan(int64_t num_row, int num_col, int num_tree_per_iteration, size_t elem) {
  PredictPlan plan;
  int device = 0, sms = 1;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  const int64_t row_threads = num_row * num_tree_per_iteration;
  plan.kernel = row_threads >= static_cast<int64_t>(sms) * 1024 ? PredictPlan::kRow : PredictPlan::kTile;
  const FalcataDebugOptions& debug = FalcataDebug();
  if (debug.predict_kernel == "row" || debug.predict_kernel == "ilp") {
    plan.kernel = PredictPlan::kRow;
  } else if (debug.predict_kernel == "tile") {
    plan.kernel = PredictPlan::kTile;
  } else if (!debug.predict_kernel.empty()) {
    Log::Fatal("FALCATA_DEBUG predict_kernel=%s: expected row, ilp or tile", debug.predict_kernel.c_str());
  }
  if (debug.predict_kernel == "row") {
    plan.ilp = 1;
  }
  if (debug.predict_ilp > 0) {
    plan.ilp = debug.predict_ilp;
  }
  if (debug.predict_tile_rows > 0) {
    plan.tile_rows = std::min(debug.predict_tile_rows, 256);
  }
  const size_t budget = std::min<size_t>(MaxSharedBytesPerBlock(), size_t(96) << 10);
  plan.stage_rows = TileSharedBytes(plan, num_col, elem) <= budget;
  while (TileSharedBytes(plan, num_col, elem) > budget && plan.tile_trees > 32) {
    plan.tile_trees /= 2;
  }
  return plan;
}

/*! \brief what a launch needs beyond the rows: the contribution kernel's path buffers and the model's width */
struct RunContext {
  int num_feature = 0;
  void* contrib_scratch = nullptr;
  int64_t contrib_rows = 0;   // rows the scratch holds path buffers for
};

template <typename T>
void RunRows(const CUDAPackedForest& forest, const Boosting::TreeRangeView& view, int tree_begin,
             int num_iteration, int predict_type, const DeviceRows<T>& rows, void* out, cudaStream_t stream,
             const RunContext& ctx) {
  const DeviceForestView<T> fv = forest.View<T>();
  if (predict_type == C_API_PREDICT_LEAF_INDEX) {
    LaunchPredictLeafKernel<T, T>(fv, rows, tree_begin, num_iteration * view.num_tree_per_iteration,
                                  reinterpret_cast<int32_t*>(out), stream);
  } else if (predict_type == C_API_PREDICT_CONTRIB) {
    const int64_t width = static_cast<int64_t>(view.num_tree_per_iteration) * (ctx.num_feature + 1);
    for (int64_t r0 = 0; r0 < rows.num_row; r0 += ctx.contrib_rows) {
      DeviceRows<T> part = rows;
      part.data = rows.data + r0 * rows.row_stride;
      part.num_row = std::min<int64_t>(ctx.contrib_rows, rows.num_row - r0);
      LaunchPredictContribKernel<T, T>(fv, part, tree_begin, num_iteration, view.num_tree_per_iteration,
                                       ctx.num_feature, reinterpret_cast<double*>(out) + r0 * width,
                                       ctx.contrib_scratch, stream);
    }
  } else {
    const PredictPlan plan = ChoosePlan(rows.num_row, rows.num_col, view.num_tree_per_iteration, sizeof(T));
    LaunchPredictRawKernel<T, T>(fv, rows, tree_begin, num_iteration, view.num_tree_per_iteration, view.num_output,
                                 view.vector_leaf, plan, reinterpret_cast<double*>(out), stream);
  }
}

void ConvertOnHost(const Boosting* boosting, const Boosting::PredictRange& range, double* out, int64_t num_row,
                   int num_output) {
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int64_t i = 0; i < num_row; ++i) {
    boosting->ConvertRawToPrediction(out + i * num_output, range);
  }
}

}  // namespace

void CUDAPredictor::Predict(const Boosting* boosting, const Boosting::PredictRange& range,
                            const CUDAPredictRequest& req, CUDAPredictStats* stats) {
  if (req.data_type != C_API_DTYPE_FLOAT32 && req.data_type != C_API_DTYPE_FLOAT64) {
    Log::Fatal("CUDA predictor: the input must be float32 or float64");
  }
  const bool contrib = req.predict_type == C_API_PREDICT_CONTRIB;
  if (contrib && boosting->IsLinear()) {
    Log::Fatal("Predicting SHAP feature contributions is not implemented for linear trees.");
  }
  const bool is_double = req.data_type == C_API_DTYPE_FLOAT64;
  const size_t elem = is_double ? sizeof(double) : sizeof(float);
  int device_id = req.device_id;
  if (req.on_device) {
    cudaPointerAttributes attr;
    CUDASUCCESS_OR_FATAL(cudaPointerGetAttributes(&attr, req.data));
    if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
      Log::Fatal("CUDA predictor: the input is not in device memory");
    }
    device_id = attr.device;
  }
  DeviceGuard guard(device_id);
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device_id));
  const Boosting::TreeRangeView view = boosting->TreesInRange(range);
  const int tree_begin = range.start_iteration * view.num_tree_per_iteration;
  const int num_output = req.predict_type == C_API_PREDICT_LEAF_INDEX
                             ? range.num_iteration * view.num_tree_per_iteration
                             : contrib ? view.num_tree_per_iteration * (view.num_feature + 1) : view.num_output;
  const size_t out_elem = req.predict_type == C_API_PREDICT_LEAF_INDEX ? sizeof(int32_t) : sizeof(double);
  const int expected = boosting->NumPredictOneRow(range.start_iteration, range.num_iteration,
                                                  req.predict_type == C_API_PREDICT_LEAF_INDEX, contrib);
  if (expected != num_output) {
    Log::Fatal("CUDA predictor: %d values per row, the model's prediction has %d", num_output, expected);
  }
  const bool convert = req.predict_type == C_API_PREDICT_NORMAL && boosting->HasOutputConversion();

  std::shared_ptr<const CUDAPackedForest> forest = GetForest(boosting, device_id, is_double);
  stats->forest_bytes = forest->Bytes();
  if (forest->num_trees() < tree_begin + static_cast<int>(view.trees.size())) {
    Log::Fatal("CUDA predictor: the packed model has %d trees, the prediction needs %d", forest->num_trees(),
               tree_begin + static_cast<int>(view.trees.size()));
  }
  if (req.num_row == 0) {
    return;
  }
  RunContext ctx;
  ctx.num_feature = view.num_feature;
  CUDAVector<char> contrib_scratch;
  if (contrib) {
    // path buffers for the (row, output) threads of one launch, at most a quarter of the free memory (and 1 GiB)
    size_t free_bytes = 0, total_bytes = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t per_row = ContribScratchBytes(forest->max_depth()) * view.num_tree_per_iteration;
    const size_t budget = std::min<size_t>(std::max<size_t>(free_bytes / 4, size_t(64) << 20), size_t(1) << 30);
    ctx.contrib_rows = std::max<int64_t>(1, std::min<int64_t>(req.num_row, static_cast<int64_t>(budget / per_row)));
    contrib_scratch.Resize(static_cast<size_t>(ctx.contrib_rows) * per_row);
    ctx.contrib_scratch = contrib_scratch.RawData();
    stats->scratch_bytes += VectorBytes(contrib_scratch);
  }

  if (req.on_device) {
    cudaStream_t stream = req.stream;
    if (is_double) {
      DeviceRows<double> rows{reinterpret_cast<const double*>(req.data), req.num_row, req.num_col, req.row_stride,
                              req.col_stride};
      RunRows(*forest, view, tree_begin, range.num_iteration, req.predict_type, rows, req.out, stream, ctx);
    } else {
      DeviceRows<float> rows{reinterpret_cast<const float*>(req.data), req.num_row, req.num_col, req.row_stride,
                             req.col_stride};
      RunRows(*forest, view, tree_begin, range.num_iteration, req.predict_type, rows, req.out, stream, ctx);
    }
    if (convert) {
      // the transform is the host's ConvertRawToPrediction, so the normal prediction equals the CPU predictor's
      const size_t n = static_cast<size_t>(req.num_row) * num_output;
      std::vector<double> host(n);
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(host.data(), req.out, n * sizeof(double), cudaMemcpyDeviceToHost, stream));
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(stream));
      ConvertOnHost(boosting, range, host.data(), req.num_row, num_output);
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(req.out, host.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    }
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(stream));
    stats->num_chunks = 1;
    stats->chunk_rows = req.num_row;
    return;
  }

  // host input: rows go to the device in chunks; each chunk's input and output buffers are the call's only
  // device memory beside the cached model
  const bool row_major = req.col_stride == 1 && req.row_stride == req.num_col;
  const bool col_major = req.row_stride == 1 && req.col_stride == req.num_row;
  if (!row_major && !col_major) {
    Log::Fatal("CUDA predictor: a host input must be C- or F-contiguous");
  }
  size_t free_bytes = 0, total_bytes = 0;
  CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_bytes, &total_bytes));
  const size_t row_bytes = std::max<size_t>(1, static_cast<size_t>(req.num_col)) * elem + num_output * out_elem;
  const size_t budget = std::min<size_t>(std::max<size_t>(free_bytes / 4, size_t(64) << 20), size_t(1) << 30);
  int64_t chunk_rows = std::max<int64_t>(1, std::min<int64_t>(req.num_row, budget / row_bytes));
  if (FalcataDebug().predict_chunk_rows > 0) {
    chunk_rows = std::min<int64_t>(req.num_row, FalcataDebug().predict_chunk_rows);
  }
  const int64_t num_chunks = (req.num_row + chunk_rows - 1) / chunk_rows;
  CUDAVector<char> in_buf(static_cast<size_t>(chunk_rows) * req.num_col * elem + 1);
  CUDAVector<char> out_buf(static_cast<size_t>(chunk_rows) * num_output * out_elem + 1);
  stats->scratch_bytes += VectorBytes(in_buf) + VectorBytes(out_buf);
  stats->num_chunks = num_chunks;
  stats->chunk_rows = chunk_rows;
  OwnedStream owned_stream;
  cudaStream_t stream = owned_stream.get();
  const char* src = reinterpret_cast<const char*>(req.data);
  char* dst = reinterpret_cast<char*>(req.out);
  for (int64_t c = 0; c < num_chunks; ++c) {
    const int64_t r0 = c * chunk_rows;
    const int64_t nr = std::min<int64_t>(chunk_rows, req.num_row - r0);
    if (req.num_col > 0) {
      if (row_major) {
        CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(in_buf.RawData(), src + r0 * req.num_col * elem,
                                             static_cast<size_t>(nr) * req.num_col * elem, cudaMemcpyHostToDevice,
                                             stream));
      } else {
        CUDASUCCESS_OR_FATAL(cudaMemcpy2DAsync(in_buf.RawData(), static_cast<size_t>(nr) * elem, src + r0 * elem,
                                               static_cast<size_t>(req.num_row) * elem,
                                               static_cast<size_t>(nr) * elem, req.num_col, cudaMemcpyHostToDevice,
                                               stream));
      }
    }
    const int64_t rs = row_major ? req.num_col : 1;
    const int64_t cs = row_major ? 1 : nr;
    if (is_double) {
      DeviceRows<double> rows{reinterpret_cast<const double*>(in_buf.RawData()), nr, req.num_col, rs, cs};
      RunRows(*forest, view, tree_begin, range.num_iteration, req.predict_type, rows, out_buf.RawData(), stream, ctx);
    } else {
      DeviceRows<float> rows{reinterpret_cast<const float*>(in_buf.RawData()), nr, req.num_col, rs, cs};
      RunRows(*forest, view, tree_begin, range.num_iteration, req.predict_type, rows, out_buf.RawData(), stream, ctx);
    }
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(dst + r0 * num_output * out_elem, out_buf.RawData(),
                                         static_cast<size_t>(nr) * num_output * out_elem, cudaMemcpyDeviceToHost,
                                         stream));
  }
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(stream));
  if (convert) {
    ConvertOnHost(boosting, range, reinterpret_cast<double*>(req.out), req.num_row, num_output);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
