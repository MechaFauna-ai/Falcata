/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_RANDOM_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_RANDOM_HPP_

#ifdef USE_CUDA

#ifndef USE_ROCM
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace Falcata {

/*!
* \brief A wrapper for random generator
*/
class CUDARandom {
 public:
  /*!
  * \brief Set specific seed
  */
  __device__ void SetSeed(int seed) {
    x = seed;
  }
  /*!
  * \brief Generate random integer from a 15-bit draw (RandInt16 returns [0, 32767], as CPU Random::NextShort)
  * \param lower_bound lower bound
  * \param upper_bound upper bound
  * \return The random integer between [lower_bound, upper_bound); uniform only when upper_bound - lower_bound
  *         divides 32768, and never above lower_bound + 32767
  */
  __device__ inline int NextShort(int lower_bound, int upper_bound) {
    return (RandInt16()) % (upper_bound - lower_bound) + lower_bound;
  }

  /*!
  * \brief Generate random integer, int32 range
  * \param lower_bound lower bound
  * \param upper_bound upper bound
  * \return The random integer between [lower_bound, upper_bound)
  */
  __device__ inline int NextInt(int lower_bound, int upper_bound) {
    return (RandInt32()) % (upper_bound - lower_bound) + lower_bound;
  }

  /*!
  * \brief Generate random float data
  * \return The random float between [0.0, 1.0)
  */
  __device__ inline float NextFloat() {
    // get random float in [0,1)
    return static_cast<float>(RandInt16()) / (32768.0f);
  }

  /*!
  * \brief The state k draws after state x: the generator's affine step x <- 214013 x + 2531011 (mod 2^32), composed
  *        with itself by squaring, so the k-th draw costs O(log k) instead of k steps
  */
  __host__ __device__ static inline unsigned int AdvanceState(unsigned int x, unsigned int k) {
    unsigned int mul = 214013u;
    unsigned int add = 2531011u;
    while (k != 0) {
      if (k & 1u) {
        x = mul * x + add;
      }
      add = mul * add + add;
      mul *= mul;
      k >>= 1;
    }
    return x;
  }

  /*!
  * \brief The value NextFloat returns when its step lands on state x
  */
  __host__ __device__ static inline float FloatFromState(unsigned int x) {
    return static_cast<float>(static_cast<int>((x >> 16) & 0x7FFF)) / (32768.0f);
  }

 private:
  __device__ inline int RandInt16() {
    x = (214013 * x + 2531011);
    return static_cast<int>((x >> 16) & 0x7FFF);
  }

  __device__ inline int RandInt32() {
    x = (214013 * x + 2531011);
    return static_cast<int>(x & 0x7FFFFFFF);
  }

  unsigned int x = 123456789;
};


}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_RANDOM_HPP_
