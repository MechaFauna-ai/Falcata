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

void WriteCsv(const std::string& path, const std::string& header, const std::vector<int>& columns) {
  std::ofstream out(path);
  out << header << "\n";
  for (int i = 0; i < 100; ++i) {
    const double x[3] = {((i * 7 * 31) % 97) / 97.0, ((i * 11 * 31) % 89) / 89.0, ((i * 13 * 31) % 83) / 83.0};
    out << 5 * x[0];
    for (const int c : columns) {
      out << "," << x[c];
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

// The CLI loads every validation file with the training data's DatasetLoader, whose header names are the training
// file's: the validation file's own header must be read to notice that it orders the features differently (the data
// is read by position, so the features were silently scrambled).
TEST(CliValidHeader, RejectsReorderedColumns) {
  const std::string dir = ::testing::TempDir();
  const std::string train = dir + "vh_train.csv";
  const std::string valid = dir + "vh_valid.csv";
  const std::string model = dir + "vh_model.txt";
  WriteCsv(train, "label,a,b,c", {0, 1, 2});
  WriteCsv(valid, "label,c,b,a", {2, 1, 0});
  try {
    RunCli({"task=train", "data=" + train, "valid=" + valid, "header=true", "num_iterations=2", "min_data_in_leaf=2",
            "verbose=-1", "output_model=" + model});
    FAIL() << "a validation file with reordered columns was accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("Column \"c\" is feature 0"), std::string::npos) << e.what();
  }
  WriteCsv(valid, "label,a,b,c", {0, 1, 2});
  RunCli({"task=train", "data=" + train, "valid=" + valid, "header=true", "num_iterations=2", "min_data_in_leaf=2",
          "verbose=-1", "output_model=" + model});
  for (const auto& path : {train, valid, model}) {
    std::remove(path.c_str());
  }
}
