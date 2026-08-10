/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_CONSTRUCT_JIT_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_CONSTRUCT_JIT_HPP_

#ifdef USE_CUDA

#include <cstdint>
#include <string>
#include <unordered_map>

namespace Falcata {

// Runtime NVRTC JIT of a shape-specialized quantized dense construct kernel.
//
// Motivation (docs/design/nvrtc-construct-jit.md): the quantized dense construct
// kernel is the dominant GPU cost of quantized CUDA training. The AOT kernel loads
// per-column bin offsets, partition column counts and loop bounds at runtime. A
// kernel JIT-compiled with the dataset's exact shape baked in as compile-time
// constants (single BINS for uniform low-bin data, per-partition column count,
// row stride, shared histogram size, 16- vs 32-bit hist) removes those loads and
// lets the compiler size / unroll the zero and flush loops exactly.
//
// The JIT is a PERF-ONLY fast path. Its histogram output must be bit-identical to
// the AOT kernel (integer atomics are order-invariant, so a specialization that
// visits the same (bin, gradient) pairs produces identical sums). To make that a
// guarantee rather than a hope, the first launch is VALIDATED against the AOT
// kernel's histogram; on any mismatch (or any NVRTC/driver failure, or an
// unsupported shape) the module is discarded and the caller falls back to the AOT
// kernel permanently for that signature. The JIT is on by default
// (cuda_plan key construct_jit); set construct_jit:off to force AOT.
//
// Scope of the specialized kernel: the host-launched (two-sync), non-graph,
// non-4bit, dense compact-quant path -- exactly numerai's shape. Every other
// shape (graph loop, 4-bit packed, sparse, large-bin) uses the AOT kernel; the
// JIT reports "unsupported" and the dispatch is unchanged.

struct ConstructJITShapeKey {
  int bins = 0;                 // uniform per-column bin count (0 = non-uniform -> unsupported)
  int num_partitions = 0;
  int cols_per_partition = 0;   // uniform compact columns per partition
  int shared_hist_size = 0;
  int use_16bit_hist = 0;
  int is_4bit = 0;              // compact bin matrix is 4-bit nibble-packed
  int sm_major = 0;
  int sm_minor = 0;

  bool operator==(const ConstructJITShapeKey& o) const {
    return bins == o.bins && num_partitions == o.num_partitions &&
           cols_per_partition == o.cols_per_partition &&
           shared_hist_size == o.shared_hist_size &&
           use_16bit_hist == o.use_16bit_hist && is_4bit == o.is_4bit &&
           sm_major == o.sm_major && sm_minor == o.sm_minor;
  }
  std::string Signature() const;
};

// A compiled + cached module. func == nullptr means "compiled/validated failed;
// permanently fall back to AOT for this signature".
struct ConstructJITEntry {
  void* module = nullptr;        // CUmodule
  void* func = nullptr;          // CUfunction: single-pair self-test kernel (nullptr => fall back)
  void* func_batched = nullptr;  // CUfunction: live batched kernel (nullptr => fall back)
  bool attempted = false;        // compile has been attempted (success or fail)
  bool validated = false;        // bit-identity vs AOT confirmed
  double compile_ms = 0.0;       // one-time NVRTC compile wall (for reporting)
};

class CUDAConstructJIT {
 public:
  CUDAConstructJIT() = default;
  ~CUDAConstructJIT();

  // Master enable (FALCATA_CONSTRUCT_JIT != "0"). Cached.
  static bool Enabled();

  // NVRTC + driver actually usable on this host (lib present, context creatable).
  // Cached; false => the whole JIT is a no-op.
  static bool Available();

  // Look up (compiling on first request) the specialized CUfunction for a shape.
  // Returns nullptr if the JIT is disabled/unavailable, the shape is unsupported,
  // or compilation failed -- the caller then uses the AOT kernel. Not yet marked
  // validated: the caller must Validate() before trusting it.
  void* GetOrCompile(const ConstructJITShapeKey& key, double* compile_ms_out);

  // The compiled batched CUfunction for a shape, regardless of validation (used
  // by the self-test to launch it before recording the verdict). nullptr if not
  // compiled. Prefer GetBatchedIfValidated on the live path.
  void* GetBatchedFunc(const ConstructJITShapeKey& key) const;

  // The live batched CUfunction for a shape (compiled together with the
  // single-pair kernel by GetOrCompile). Returns nullptr unless the signature's
  // module compiled AND the shape was validated bit-identical to the AOT kernel.
  void* GetBatchedIfValidated(const ConstructJITShapeKey& key) const;

  // Has this signature's kernel been confirmed bit-identical to the AOT kernel?
  bool IsValidated(const ConstructJITShapeKey& key) const;
  // Record the validation verdict (caller compares JIT vs AOT histograms once).
  void SetValidated(const ConstructJITShapeKey& key, bool ok);

 private:
  std::string BuildKernelSource(const ConstructJITShapeKey& key) const;
  std::unordered_map<std::string, ConstructJITEntry> cache_;
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_CONSTRUCT_JIT_HPP_
