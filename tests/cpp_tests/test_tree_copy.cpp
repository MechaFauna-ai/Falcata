/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/c_api.h>
#include <Falcata/tree.h>

#include <memory>
#include <string>

using Falcata::Tree;

static const char kTree[] =
  "Tree=0\n"
  "num_leaves=3\n"
  "num_cat=0\n"
  "split_feature=0 1\n"
  "split_gain=4 2\n"
  "threshold=0.5 1.5\n"
  "decision_type=2 2\n"
  "left_child=1 -1\n"
  "right_child=-3 -2\n"
  "leaf_value=-1 0.5 2\n"
  "leaf_weight=1 2 3\n"
  "leaf_count=1 2 3\n"
  "internal_value=0 0.1\n"
  "internal_weight=6 3\n"
  "internal_count=6 3\n"
  "is_linear=0\n"
  "shrinkage=1\n"
  "\n\n";

// A tree read from a model string used to leave track_branch_features_ (and
// max_leaves_) uninitialized, and every copy of a loaded tree -- ShuffleModels,
// MergeFrom (refit, an in-memory init_model), FitByExistingTree -- read the
// indeterminate bool (UBSan: "load of value 190, which is not a valid value
// for type 'bool'"). Heap-allocated so ASan's malloc fill makes it deterministic.
TEST(TreeCopy, CopyOfLoadedTreeIsWellDefined) {
  size_t used = 0;
  std::unique_ptr<Tree> loaded(new Tree(kTree, &used));
  ASSERT_EQ(loaded->num_leaves(), 3);
  std::unique_ptr<Tree> copied(new Tree(*loaded));
  EXPECT_EQ(copied->ToString(), loaded->ToString());
}

TEST(TreeCopy, ShuffleModelsOfLoadedModel) {
  std::string model =
      "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=1\n"
      "objective=regression\nfeature_names=a b\nfeature_infos=[0:1] [0:2]\n\n";
  for (int i = 0; i < 3; ++i) {
    std::string t(kTree);
    t.replace(0, 6, "Tree=" + std::to_string(i));
    model += t;
  }
  model += "end of trees\n";
  int num_iterations = 0;
  BoosterHandle booster = nullptr;
  ASSERT_EQ(FLC_BoosterLoadModelFromString(model.c_str(), &num_iterations, &booster), 0) << FLC_GetLastError();
  ASSERT_EQ(num_iterations, 3);
  EXPECT_EQ(FLC_BoosterShuffleModels(booster, 0, -1), 0) << FLC_GetLastError();
  int after = 0;
  EXPECT_EQ(FLC_BoosterGetCurrentIteration(booster, &after), 0);
  EXPECT_EQ(after, 3);
  EXPECT_EQ(FLC_BoosterFree(booster), 0);
}
