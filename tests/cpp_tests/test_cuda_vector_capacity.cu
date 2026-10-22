/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root.
 */
#include <Falcata/cuda/cuda_utils.hu>
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

// CUDAVector's capacity (the categorical tree-record buffers, CUDATreeCatArena): PushBack grows by doubling and
// keeps the content, ClearKeepCapacity restarts the content in place, Swap carries the capacity along, and every
// other mutator keeps capacity == size.
namespace {
using Falcata::CUDAVector;

std::vector<uint32_t> Iota(size_t n, uint32_t start) {
  std::vector<uint32_t> v(n);
  std::iota(v.begin(), v.end(), start);
  return v;
}

// append host values through a device staging buffer, as the tree appends the learner's bitset scratch
void Append(CUDAVector<uint32_t>* vec, const std::vector<uint32_t>& values) {
  CUDAVector<uint32_t> staging;
  staging.InitFromHostVector(values);
  vec->PushBack(staging.RawData(), values.size());
}

TEST(CUDAVectorCapacity, PushBackDoublesAndKeepsContent) {
  CUDAVector<uint32_t> vec;
  EXPECT_EQ(vec.Capacity(), 0u);
  std::vector<uint32_t> expected;
  int reallocations = 0;
  for (int i = 0; i < 40; ++i) {
    const std::vector<uint32_t> part = Iota(3 + i % 5, 1000u * i);
    const uint32_t* before = vec.RawData();
    const size_t capacity_before = vec.Capacity();
    Append(&vec, part);
    expected.insert(expected.end(), part.begin(), part.end());
    if (vec.RawData() != before) {
      ++reallocations;
      // a reallocation at least doubles the storage
      EXPECT_GE(vec.Capacity(), 2 * capacity_before);
    } else {
      EXPECT_EQ(vec.Capacity(), capacity_before);
    }
    ASSERT_EQ(vec.Size(), expected.size());
    EXPECT_GE(vec.Capacity(), vec.Size());
  }
  EXPECT_EQ(vec.ToHost(), expected);
  // 40 appends (204 elements) from empty: O(log n) reallocations, not one per append
  EXPECT_LE(reallocations, 10);
}

TEST(CUDAVectorCapacity, ReserveThenAppendWithinCapacityDoesNotReallocate) {
  CUDAVector<uint32_t> vec;
  vec.Reserve(256);
  EXPECT_EQ(vec.Size(), 0u);
  EXPECT_EQ(vec.Capacity(), 256u);
  const uint32_t* storage = vec.RawData();
  std::vector<uint32_t> expected;
  for (int i = 0; i < 16; ++i) {
    const std::vector<uint32_t> part = Iota(16, 100u * i);
    Append(&vec, part);
    expected.insert(expected.end(), part.begin(), part.end());
  }
  EXPECT_EQ(vec.RawData(), storage);
  EXPECT_EQ(vec.Capacity(), 256u);
  EXPECT_EQ(vec.ToHost(), expected);
  // a smaller Reserve is a no-op
  vec.Reserve(8);
  EXPECT_EQ(vec.RawData(), storage);
  EXPECT_EQ(vec.Capacity(), 256u);
}

TEST(CUDAVectorCapacity, ClearKeepCapacityRestartsInPlace) {
  CUDAVector<uint32_t> vec;
  Append(&vec, Iota(100, 0));
  const uint32_t* storage = vec.RawData();
  const size_t capacity = vec.Capacity();
  vec.ClearKeepCapacity();
  EXPECT_EQ(vec.Size(), 0u);
  EXPECT_EQ(vec.Capacity(), capacity);
  EXPECT_EQ(vec.RawData(), storage);
  EXPECT_TRUE(vec.ToHost().empty());
  // the next tree's record overwrites the stale words in place
  Append(&vec, Iota(60, 7));
  EXPECT_EQ(vec.RawData(), storage);
  EXPECT_EQ(vec.ToHost(), Iota(60, 7));
}

TEST(CUDAVectorCapacity, SwapCarriesCapacity) {
  // the tree takes the learner's arena buffer in its constructor and gives it back at its end
  CUDAVector<uint32_t> arena;
  arena.Reserve(512);
  const uint32_t* storage = arena.RawData();
  CUDAVector<uint32_t> tree;
  tree.Swap(&arena);
  EXPECT_EQ(arena.Capacity(), 0u);
  EXPECT_EQ(arena.RawData(), nullptr);
  EXPECT_EQ(tree.Capacity(), 512u);
  tree.ClearKeepCapacity();
  Append(&tree, Iota(300, 1));
  EXPECT_EQ(tree.RawData(), storage);
  tree.Swap(&arena);
  EXPECT_EQ(tree.Capacity(), 0u);
  EXPECT_EQ(tree.Size(), 0u);
  EXPECT_EQ(arena.RawData(), storage);
  EXPECT_EQ(arena.Capacity(), 512u);
  EXPECT_EQ(arena.ToHost(), Iota(300, 1));
}

TEST(CUDAVectorCapacity, OtherMutatorsKeepCapacityEqualToSize) {
  CUDAVector<uint32_t> vec;
  Append(&vec, Iota(5, 0));
  ASSERT_GT(vec.Capacity(), 0u);
  vec.Resize(3);  // exact reallocation, as before: bounds per-tree memory
  EXPECT_EQ(vec.Capacity(), 3u);
  EXPECT_EQ(vec.ToHost(), Iota(3, 0));
  vec.Resize(10);
  EXPECT_EQ(vec.Capacity(), 10u);
  vec.ResizeDiscard(4);
  EXPECT_EQ(vec.Capacity(), 4u);
  vec.InitFromHostVector(Iota(6, 9));
  EXPECT_EQ(vec.Capacity(), 6u);
  vec.Clear();
  EXPECT_EQ(vec.Size(), 0u);
  EXPECT_EQ(vec.Capacity(), 0u);
  // a view: an append past its end copies it into owned storage first (as Resize does)
  CUDAVector<uint32_t> owner;
  owner.InitFromHostVector(Iota(4, 50));
  CUDAVector<uint32_t> view;
  view.SetView(owner.RawData(), owner.Size());
  EXPECT_TRUE(view.IsView());
  EXPECT_EQ(view.Capacity(), 4u);
  Append(&view, Iota(2, 90));
  EXPECT_FALSE(view.IsView());
  EXPECT_NE(view.RawData(), owner.RawData());
  std::vector<uint32_t> expected = Iota(4, 50);
  expected.push_back(90);
  expected.push_back(91);
  EXPECT_EQ(view.ToHost(), expected);
  EXPECT_EQ(owner.ToHost(), Iota(4, 50));
}

}  // namespace
