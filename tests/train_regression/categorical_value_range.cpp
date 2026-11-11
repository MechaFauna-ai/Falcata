// Copyright (c) 2026 The Falcata developers. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root.
#include <Falcata/tree.h>
#include <cassert>
#include <limits>
#include <iostream>
#include <unordered_map>
int main(int argc, char**) {
  Falcata::Tree tree(2, false, false);
  uint32_t bits = 3;
  tree.SplitCategorical(0, 0, 0, &bits, 1, &bits, 1, 3.0, 7.0, 1, 1, 1.0, 1.0, 1.0f, Falcata::MissingType::None);
  if (argc > 1) {
    std::cout << tree.ToIfElse(0, false);
    return 0;
  }
  for (double value : {2147483647.5, 2147483648.0, -2147483649.0, 1e100, -1e100, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), -1.0}) {
    assert(tree.Predict(&value) == 7.0);
    assert(tree.PredictByMap({{0, value}}) == 7.0);
  }
  double value = 1.9;
  assert(tree.Predict(&value) == 3.0);
  value = -0.5;
  assert(tree.Predict(&value) == 3.0);
}
