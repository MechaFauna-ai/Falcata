// Copyright (c) 2026 The Falcata developers. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root.
#include <Falcata/tree.h>
#include <cassert>
#include <string>
int main() {
  Falcata::Tree tree(2, false, false);
  const double value = 1.2345678901234567;
  tree.AsConstantTree(value);
  const std::string code = tree.ToIfElse(0, false);
  size_t position = 0;
  int count = 0;
  while ((position = code.find("return ", position)) != std::string::npos) {
    position += 7;
    assert(std::stod(code.substr(position)) == value);
    ++count;
  }
  assert(count == 2);
}
