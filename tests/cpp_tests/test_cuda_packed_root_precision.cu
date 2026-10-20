/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Standalone exact-source CUDA regression. The parent compiler defines CONTROLLER_SOURCE to the frozen
 * src/treelearner/cuda/cuda_leaf_splits.cu. This calls the production Kernel4, not a copied reduction.
 */
#ifndef CONTROLLER_SOURCE
#define CONTROLLER_SOURCE "../../src/treelearner/cuda/cuda_leaf_splits.cu"
#endif
#include CONTROLLER_SOURCE

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17) -- standalone CUDA regression uses C++17.
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace PackedRootQA {
using Falcata::CUDALeafSplitsStruct;
using Falcata::data_size_t;
using Falcata::hist_t;
constexpr int kMaxChunks = 2051;
constexpr int kGuard = 8;
constexpr unsigned char kCanary = 0xa7;
bool failed_cuda = false;

void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void CUDA(cudaError_t status) {
  if (status != cudaSuccess) {
    failed_cuda = true;
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <typename T>
class Guarded {
 public:
  explicit Guarded(int size) : size_(size) {
    static_assert(std::is_trivially_copyable<T>::value, "probe buffers require trivial CUDA ABI");
    const size_t bytes = static_cast<size_t>(size_ + 2 * kGuard) * sizeof(T);
    CUDA(cudaMalloc(reinterpret_cast<void**>(&raw_), bytes));
    CUDA(cudaMemset(raw_, kCanary, bytes));
  }
  ~Guarded() {
    if (raw_ != nullptr && !failed_cuda) {
      const auto status = cudaFree(raw_);
      if (status != cudaSuccess) {
        failed_cuda = true;
        std::cerr << "CUDA cleanup: " << cudaGetErrorString(status) << '\n';
      }
    }
  }
  T* get() const { return raw_ + kGuard; }
  void Put(const std::vector<T>& values) {
    Require(values.size() == static_cast<size_t>(size_), "upload extent");
    CUDA(cudaMemcpy(get(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> Read() const {
    const size_t guard_bytes = kGuard * sizeof(T);
    const size_t data_bytes = static_cast<size_t>(size_) * sizeof(T);
    std::vector<unsigned char> bytes(data_bytes + 2 * guard_bytes);
    CUDA(cudaMemcpy(bytes.data(), raw_, bytes.size(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < guard_bytes; ++i) {
      Require(bytes[i] == kCanary && bytes[guard_bytes + data_bytes + i] == kCanary,
              "CUDA input/output canary changed");
    }
    std::vector<T> values(size_);
    std::memcpy(values.data(), bytes.data() + guard_bytes, data_bytes);
    return values;
  }

 private:
  int size_;
  T* raw_ = nullptr;
};

uint64_t Bits(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int64_t Pack(int64_t gradient, uint32_t hessian) {
  // Multiplication avoids shifting a negative signed integer in the host oracle.
  return gradient * (int64_t{1} << 32) + hessian;
}

int64_t Part(int64_t total, int chunks, int index) {
  const int64_t remainder = total % chunks;
  const int64_t extra = index < (remainder < 0 ? -remainder : remainder) ? (remainder < 0 ? -1 : 1) : 0;
  return total / chunks + extra;
}

struct Case {
  const char* name;
  int64_t gradient;
  uint32_t hessian;
  bool loses_precision;
};

int Run(bool expect_loss) {
  static_assert(sizeof(int64_t) == sizeof(double), "production shared scratch lane size");
  const uint32_t umax = std::numeric_limits<uint32_t>::max();
  const std::vector<Case> cases{
      {"zero", 0, 0, false},
      {"hessian_one", 0, 1, false},
      {"hessian_two", 0, 2, false},
      {"positive_2p53_minus1", 2097151, umax, false},
      {"positive_2p53", 2097152, 0, false},
      {"positive_2p53_plus1", 2097152, 1, true},
      {"positive_2p53_plus2", 2097152, 2, false},
      {"negative_2p53_plus1", -2097152, 1, false},
      {"negative_2p53", -2097152, 0, false},
      {"negative_2p53_minus1", -2097153, umax, true},
      {"negative_2p53_minus2", -2097153, umax - 1, false},
      {"wide_negative_aligned", -15728640, 33554432, false},
      {"wide_negative_plus1", -15728640, 33554433, true},
      {"wide_negative_plus2", -15728640, 33554434, true},
      {"wide_positive_aligned", 15728640, 33554432, false},
      {"wide_positive_plus1", 15728640, 33554433, true},
      {"wide_positive_plus2", 15728640, 33554434, true}};
  Guarded<double> gradients(kMaxChunks), hessians(kMaxChunks);
  Guarded<int64_t> packed(kMaxChunks);
  Guarded<CUDALeafSplitsStruct> output(1);
  Guarded<data_size_t> indices(1);
  Guarded<hist_t> histogram(1);
  const std::vector<data_size_t> index_values{91};
  const std::vector<hist_t> histogram_values{static_cast<hist_t>(0.25)};
  indices.Put(index_values);
  histogram.Put(histogram_values);
  int checked = 0, discriminating = 0;
  for (const auto& c : cases) {
    const int64_t expected = Pack(c.gradient, c.hessian);
    const int64_t legacy = static_cast<int64_t>(static_cast<double>(expected));
    Require((legacy != expected) == c.loses_precision, "predeclared precision witness is invalid");
    for (int chunks : {1, 33, 1024, 1025, kMaxChunks}) {
      for (int block : {256, NUM_THREADS_PER_BLOCK_LEAF_SPLITS}) {
        std::vector<double> grad(kMaxChunks, -17.25), hess(kMaxChunks, 9.5);
        std::vector<int64_t> gh(kMaxChunks, 73);
        int64_t integer_total = 0;
        for (int i = 0; i < chunks; ++i) {
          const int64_t g = Part(c.gradient, chunks, i);
          const auto h = static_cast<uint32_t>(Part(c.hessian, chunks, i));
          grad[i] = static_cast<double>(g) * 0.125;
          hess[i] = static_cast<double>(h) * 0.00390625;
          gh[i] = Pack(g, h);
          integer_total += gh[i];
        }
        Require(integer_total == expected, "independent host integer sum differs");
        const double expected_gradient = static_cast<double>(c.gradient) * 0.125;
        const double expected_hessian = static_cast<double>(c.hessian) * 0.00390625;
        gradients.Put(grad);
        hessians.Put(hess);
        packed.Put(gh);
        CUDALeafSplitsStruct initial{};
        initial.leaf_index = -123;
        initial.sum_of_gradients_hessians = 97;
        output.Put({initial});
        const data_size_t rows = 131072;
        // L2=1 keeps the zero-hessian control finite without changing either total lane.
        Falcata::CUDAInitValuesKernel4<<<1, block>>>(
            0.0, 1.0, 0.0, chunks, gradients.get(), hessians.get(), packed.get(), rows,
            indices.get(), histogram.get(), output.get());
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());  // stop immediately on any launch/async error
        const auto actual = output.Read()[0];
        const auto actual_grad = gradients.Read();
        const auto actual_hess = hessians.Read();
        const auto actual_packed = packed.Read();
        // Emit raw evidence before value assertions, including on an unpatched precision failure.
        std::cout << "{\"case\":\"" << c.name << "\",\"chunks\":" << chunks << ",\"block\":" << block
                  << ",\"mode\":\"" << (expect_loss ? "precision_loss_control" : "exact")
                  << "\",\"packed_expected\":\"" << expected << "\",\"packed_actual\":\""
                  << actual.sum_of_gradients_hessians << "\",\"legacy_prediction\":\"" << legacy
                  << "\",\"gradient_bits\":\"" << std::hex << Bits(actual.sum_of_gradients)
                  << "\",\"gradient_expected_bits\":\"" << Bits(expected_gradient)
                  << "\",\"hessian_bits\":\"" << Bits(actual.sum_of_hessians)
                  << "\",\"hessian_expected_bits\":\"" << Bits(expected_hessian)
                  << std::dec << "\",\"discriminating\":" << (c.loses_precision ? "true" : "false") << "}\n";
        std::cout.flush();
        Require(Bits(actual.sum_of_gradients) == Bits(expected_gradient), "root double gradient changed");
        Require(Bits(actual.sum_of_hessians) == Bits(expected_hessian), "root double hessian changed");
        grad[0] = expected_gradient;
        hess[0] = expected_hessian;
        Require(std::memcmp(actual_grad.data(), grad.data(), grad.size() * sizeof(double)) == 0,
                "double gradient partials/readback differ");
        Require(std::memcmp(actual_hess.data(), hess.data(), hess.size() * sizeof(double)) == 0,
                "double hessian partials/readback differ");
        Require(actual_packed == gh, "packed input partials changed");
        Require(indices.Read() == index_values && histogram.Read() == histogram_values, "pointed-to backing changed");
        Require(actual.leaf_index == 0 && actual.num_data_in_leaf == rows &&
                    actual.data_indices_in_leaf == indices.get() && actual.hist_in_leaf == histogram.get(),
                "root metadata/pointer outputs differ");
        Require(actual.sum_of_gradients_hessians == (expect_loss ? legacy : expected),
                "actual Kernel4 packed root differs from selected exact/old-loss contract");
        if (c.loses_precision) ++discriminating;
        ++checked;
      }
    }
  }
  Require(checked == 170 && discriminating == 60, "incomplete kernel witness matrix");
  std::cout << "PASS packed_root_precision cases=170 discriminating=60 exact_controls=110 double_fields_exact=170 mode="
            << (expect_loss ? "precision_loss_control" : "exact") << '\n';
  return checked;
}
}  // namespace PackedRootQA

int main(int argc, char** argv) {
  try {
    // Parent owns source/library pins, queue identity, STOPs and sanitizer policy. No CUDA API precedes release.
    const char* job = std::getenv("GPUQ_JOB_ID");
    char* end = nullptr;
    const int64_t jid = job == nullptr ? 0 : std::strtoll(job, &end, 10);
    const bool released = argc >= 2 && std::string(argv[1]) == "--parent-released-gpu";
    const bool expect_loss = argc == 3 && std::string(argv[2]) == "--expect-precision-loss";
    const bool exact = argc == 2 || (argc == 3 && std::string(argv[2]) == "--expect-exact");
    const char* stop = std::getenv("FALCATA_GPU_STOP_PATH");
    if (!released || (!expect_loss && !exact) || jid <= 0 || end == nullptr || *end != '\0' ||
        (stop != nullptr && std::filesystem::exists(stop))) {
      std::cerr << "HOLD: parent queue and explicit GPU release required; no CUDA API called\n";
      return 90;
    }
    PackedRootQA::Require(PackedRootQA::Run(expect_loss) == 170 && !PackedRootQA::failed_cuda,
                          "case count or CUDA cleanup failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL packed_root_precision " << error.what() << '\n';
    return 1;
  }
}
