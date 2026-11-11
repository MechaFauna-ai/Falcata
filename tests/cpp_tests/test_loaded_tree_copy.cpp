/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/tree.h>

#include <cstring>
#include <new>
#include <string>

namespace {

// Builds a Tree from model text in storage pre-filled with a non-boolean byte, so a
// member the text constructor forgets to set holds an invalid bool / garbage int.
// Copying such a tree (GBDT::MergeFrom does, for every init_model tree) loaded an
// indeterminate bool: UBSan reports "load of value 190, which is not a valid value
// for type 'bool'" in Tree::Tree(const Tree&).
void CheckCopyOfLoadedTree(const Falcata::Tree& source) {
  const std::string text = source.ToString();
  alignas(Falcata::Tree) unsigned char storage[sizeof(Falcata::Tree)];
  std::memset(storage, 0xBE, sizeof(storage));
  size_t used = 0;
  auto* loaded = new (storage) Falcata::Tree(text.c_str(), &used);
  {
    Falcata::Tree copied(*loaded);
    EXPECT_EQ(copied.ToString(), text);
    EXPECT_EQ(copied.num_leaves(), source.num_leaves());
    EXPECT_FALSE(copied.is_linear());
  }
  loaded->~Tree();
}

}  // namespace

TEST(LoadedTreeCopy, ConstantTree) {
  Falcata::Tree source(4, false, false);
  source.SetLeafOutput(0, 0.25);
  CheckCopyOfLoadedTree(source);
}

TEST(LoadedTreeCopy, SplitTree) {
  Falcata::Tree source(4, false, false);
  source.Split(0, 0, 0, 1, 0.5, -1, 1, 1, 1, 1, 1, 1, Falcata::MissingType::None, true);
  CheckCopyOfLoadedTree(source);
}
