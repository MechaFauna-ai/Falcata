// Copyright (c) 2026 The Falcata developers. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root.
// Writes a C++ program that checks the if-else export of one-split trees
// against Tree::Predict, for every missing type and default direction.
#include <Falcata/tree.h>

#include <cstdio>
#include <limits>
#include <string>
#include <vector>

int main() {
  const std::vector<std::string> thresholds = {"0.5", "-0.5", "1.0000000180025095e-35",
                                               "-1.0000000180025095e-35", "inf"};
  // missing type None, Zero and NaN, each with default right and default left
  const std::vector<int> decision_types = {0, 2, 4, 6, 8, 10};
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> values = {std::numeric_limits<double>::quiet_NaN(), 0.0, -0.0, 1e-40, -1e-40,
                                      0.25, 0.5, 0.75, -0.25, -0.5, -2.0, 2.0, inf, -inf};
  std::string code;
  std::string checks;
  int index = 0;
  for (const auto& threshold : thresholds) {
    for (int decision_type : decision_types) {
      const std::string model =
          "num_leaves=2\nnum_cat=0\nsplit_feature=0\nsplit_gain=1\nthreshold=" + threshold +
          "\ndecision_type=" + std::to_string(decision_type) +
          "\nleft_child=-1\nright_child=-2\nleaf_value=-1 1\nleaf_weight=1 1\nleaf_count=1 1"
          "\ninternal_value=0\ninternal_weight=2\ninternal_count=2\nis_linear=0\nshrinkage=1\n\n";
      size_t used_len = 0;
      Falcata::Tree tree(model.c_str(), &used_len);
      code += tree.ToIfElse(index, false);
      for (size_t i = 0; i < values.size(); ++i) {
        const double expected = tree.Predict(&values[i]);
        checks += "  check(" + std::to_string(index) + ", PredictTree" + std::to_string(index) + ", " +
                  std::to_string(i) + ", " + std::to_string(static_cast<int>(expected)) + ");\n";
      }
      ++index;
    }
  }
  std::printf(
      "#include <Falcata/tree.h>\n#include <cmath>\n#include <cstdint>\n#include <cstdio>\n"
      "#include <cstdlib>\n#include <limits>\n#include <unordered_map>\n#include <vector>\n"
      "namespace Falcata {\n%s}\n"
      "using namespace Falcata;\n"
      "static const double kInf = std::numeric_limits<double>::infinity();\n"
      "static const double kValues[] = {std::numeric_limits<double>::quiet_NaN(), 0.0, -0.0, 1e-40, -1e-40,"
      " 0.25, 0.5, 0.75, -0.25, -0.5, -2.0, 2.0, kInf, -kInf};\n"
      "static int failures = 0;\n"
      "static void check(int tree, double (*predict)(const double*), int value, int expected) {\n"
      "  const double got = predict(&kValues[value]);\n"
      "  if (got != expected) {\n"
      "    std::printf(\"tree %%d value %%g: if-else %%g, Tree::Predict %%d\\n\", tree, kValues[value], got, expected);\n"
      "    ++failures;\n"
      "  }\n"
      "}\n"
      "int main() {\n%s  return failures == 0 ? 0 : 1;\n}\n",
      code.c_str(), checks.c_str());
  return 0;
}
