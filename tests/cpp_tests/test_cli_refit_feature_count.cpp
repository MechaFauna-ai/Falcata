/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/application.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void WriteCsv(const std::string& path, int num_columns) {
  std::ofstream out(path);
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < num_columns; ++j) {
      out << (j ? "," : "") << ((i * (j + 7) * 31) % 97) / 97.0;
    }
    out << "\n";
  }
}

void RunCli(std::vector<std::string> args) {
  args.insert(args.begin(), "falcata");
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(&arg[0]);
  }
  Falcata::Application app(static_cast<int>(argv.size()), argv.data());
  app.Run();
}

}  // namespace

// task=refit predicts leaf indices with the model's parser and then loads the file as a Dataset. A file with one
// column fewer than label + model features passed the first step (its width matched the model's feature count, so
// it was read as label-less features) and loaded with one feature fewer: the trees then indexed features past the
// Dataset (heap-buffer-overflow in FeatureImportance while saving the refit model).
TEST(CliRefit, RejectsDataWithFewerFeaturesThanTheModel) {
  const std::string dir = ::testing::TempDir();
  const std::string train = dir + "refit_train.csv";
  const std::string narrow = dir + "refit_narrow.csv";
  const std::string model = dir + "refit_model.txt";
  const std::string refit = dir + "refit_out.txt";
  const std::string result = dir + "refit_leaves.txt";
  WriteCsv(train, 4);   // label + 3 features
  WriteCsv(narrow, 3);  // label + 2 features
  RunCli({"task=train", "data=" + train, "num_iterations=3", "min_data_in_leaf=2", "verbose=-1",
          "output_model=" + model});
  try {
    RunCli({"task=refit", "data=" + narrow, "input_model=" + model, "output_model=" + refit,
            "output_result=" + result, "verbose=-1"});
    FAIL() << "refit accepted data with fewer features than the model";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("The refit data has 2 features"), std::string::npos) << e.what();
  }
  for (const auto& path : {train, narrow, model, refit, result}) {
    std::remove(path.c_str());
  }
}
