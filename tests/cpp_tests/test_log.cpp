/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/utils/log.h>

#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::mutex g_lines_mutex;
std::vector<std::string> g_lines;

void Collect(const char* msg) {
  std::lock_guard<std::mutex> lock(g_lines_mutex);
  g_lines.emplace_back(msg);
}

bool Logged(const std::string& text) {
  std::lock_guard<std::mutex> lock(g_lines_mutex);
  for (const auto& line : g_lines) {
    if (line.find(text) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// restores the process-wide defaults after each test
class LogThreads : public testing::Test {
 protected:
  void SetUp() override {
    std::lock_guard<std::mutex> lock(g_lines_mutex);
    g_lines.clear();
  }
  void TearDown() override {
    Falcata::Log::ResetLogLevel(Falcata::LogLevel::Info);
    Falcata::Log::ResetCallBack(nullptr);
  }
};

}  // namespace

// The level and callback were per thread only, so a thread that never set
// them (an OpenMP worker, any thread the library starts) logged at Info to
// stdout whatever verbosity and callback the caller had installed.
TEST_F(LogThreads, WorkerThreadsFollowTheLastSetting) {
  Falcata::Log::ResetLogLevel(Falcata::LogLevel::Warning);
  Falcata::Log::ResetCallBack(Collect);
  std::thread([] {
    EXPECT_EQ(Falcata::Log::CurrentLevel(), Falcata::LogLevel::Warning);
    EXPECT_EQ(Falcata::Log::CurrentCallBack(), &Collect);
    Falcata::Log::Info("worker info must be dropped");
    Falcata::Log::Warning("worker warning goes to the callback");
  }).join();
  EXPECT_FALSE(Logged("worker info must be dropped"));
  EXPECT_TRUE(Logged("worker warning goes to the callback"));
}

// A thread that sets its own level and callback keeps them.
TEST_F(LogThreads, ThreadOwnSettingWins) {
  Falcata::Log::ResetCallBack(Collect);
  std::thread([] {
    Falcata::Log::ResetLogLevel(Falcata::LogLevel::Fatal);
    Falcata::Log::Warning("thread-own fatal level drops this");
    EXPECT_EQ(Falcata::Log::CurrentLevel(), Falcata::LogLevel::Fatal);
  }).join();
  EXPECT_FALSE(Logged("thread-own fatal level drops this"));
  // this thread set its own level before, so the other thread's newer
  // setting does not change it
  Falcata::Log::ResetLogLevel(Falcata::LogLevel::Info);
  std::thread([] { Falcata::Log::ResetLogLevel(Falcata::LogLevel::Fatal); }).join();
  EXPECT_EQ(Falcata::Log::CurrentLevel(), Falcata::LogLevel::Info);
  Falcata::Log::Info("main thread keeps info");
  EXPECT_TRUE(Logged("main thread keeps info"));
}
