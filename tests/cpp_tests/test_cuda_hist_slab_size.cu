/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <gtest/gtest.h>

#include <cstdint>

#include "../../src/treelearner/cuda/cuda_histogram_constructor.hpp"

// cuda_hist_ holds a (grad, hess) pair per bin, per leaf, per plane. origin/master sized it with
// static_cast<size_t>(num_total_bin_ * 2 * num_leaves_) in int: 2.2M total bins (e.g. 8600 features x 255
// bins) at 1023 leaves wrapped to 206232704 elements instead of 4501200000, a 22x smaller buffer the
// construct kernels then overran instead of an out-of-memory error.
TEST(CUDAHistSlab, ElementCountPastInt32) {
  EXPECT_EQ(Falcata::CUDAHistSlabElements(2200000, 1023, 1), static_cast<size_t>(4501200000ULL));
  EXPECT_EQ(Falcata::CUDAHistSlabElements(2200000, 1023, 3), static_cast<size_t>(13503600000ULL));
  EXPECT_EQ(Falcata::CUDAHistSlabElements(255 * 28, 255, 1), static_cast<size_t>(255 * 28 * 2 * 255));
  static_assert(Falcata::CUDAHistSlabElements(1, 1, 1) == 2, "one bin, one leaf: one (grad, hess) pair");
}
