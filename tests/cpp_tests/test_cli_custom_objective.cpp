/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/application.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void WriteCsv(const std::string& path) {
  std::ofstream out(path);
  for (int i = 0; i < 100; ++i) {
    out << (i % 2) << "," << ((i * 7 * 31) % 97) / 97.0 << "," << ((i * 11 * 31) % 97) / 97.0 << "\n";
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

void ExpectCustomObjectiveError(const std::vector<std::string>& args) {
  try {
    RunCli(args);
    FAIL() << "the command line accepted a custom objective";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("needs gradients computed by the caller"), std::string::npos) << e.what();
  }
}

}  // namespace

// objective=custom has no objective function (the caller supplies the gradients, which the command line cannot):
// task=train and task=refit dereferenced the missing objective.
TEST(CliObjective, CustomObjectiveIsAClearError) {
  const std::string dir = ::testing::TempDir();
  const std::string data = dir + "cli_custom_objective.csv";
  const std::string model = dir + "cli_custom_objective_model.txt";
  WriteCsv(data);
  for (const std::string objective : {"custom", "none"}) {
    SCOPED_TRACE(objective);
    ExpectCustomObjectiveError({"task=train", "data=" + data, "objective=" + objective, "num_iterations=2",
                                "verbose=-1", "output_model=" + model});
  }
  RunCli({"task=train", "data=" + data, "objective=binary", "num_iterations=2", "min_data_in_leaf=2", "verbose=-1",
          "output_model=" + model});
  ExpectCustomObjectiveError({"task=refit", "data=" + data, "input_model=" + model, "objective=custom",
                              "output_model=" + dir + "cli_custom_objective_refit.txt",
                              "output_result=" + dir + "cli_custom_objective_leaves.txt", "verbose=-1"});
}
