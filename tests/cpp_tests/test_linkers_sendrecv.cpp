/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

// the socket transport (not MPI) over POSIX sockets: Linux only
#if defined(__linux__) && defined(USE_SOCKET)

#include <gtest/gtest.h>

#include <Falcata/config.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../../src/network/linkers.h"

namespace {

// a port nothing listens on right now
int FreePort() {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  socklen_t len = sizeof(addr);
  getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
  close(fd);
  return ntohs(addr.sin_port);
}

// n ranks of one job in this process, connected over loopback
std::vector<std::unique_ptr<Falcata::Linkers>> MakeRanks(int n) {
  std::vector<int> ports;
  std::string machines;
  for (int i = 0; i < n; ++i) {
    ports.push_back(FreePort());
    machines += (i ? ",127.0.0.1 " : "127.0.0.1 ") + std::to_string(ports.back());
  }
  auto make = [&machines, n](int port) {
    Falcata::Config config;
    config.num_machines = n;
    config.machines = machines;
    config.local_listen_port = port;
    config.time_out = 1;
    return std::unique_ptr<Falcata::Linkers>(new Falcata::Linkers(config));
  };
  std::vector<std::future<std::unique_ptr<Falcata::Linkers>>> pending;
  for (int i = 1; i < n; ++i) {
    pending.push_back(std::async(std::launch::async, make, ports[i]));
  }
  std::vector<std::unique_ptr<Falcata::Linkers>> ranks;
  ranks.push_back(make(ports[0]));
  for (auto& f : pending) {
    ranks.push_back(f.get());
  }
  return ranks;
}

void MakePair(std::unique_ptr<Falcata::Linkers>* rank0, std::unique_ptr<Falcata::Linkers>* rank1) {
  auto ranks = MakeRanks(2);
  *rank0 = std::move(ranks[0]);
  *rank1 = std::move(ranks[1]);
}

// Runs f on a detached thread and returns its result, or "timeout" if it has not finished after the given time.
// f must own everything it touches (capture shared_ptrs by value): on a timeout the thread is left running, so a
// hang fails the test instead of hanging the test binary.
template <typename F>
std::string RunWithTimeout(F f, int seconds) {
  auto promise = std::make_shared<std::promise<std::string>>();
  std::future<std::string> result = promise->get_future();
  std::thread([promise, f]() mutable { promise->set_value(f()); }).detach();
  if (result.wait_for(std::chrono::seconds(seconds)) != std::future_status::ready) {
    return "timeout";
  }
  return result.get();
}

}  // namespace

// A failure while SendRecv waited for its receive left the std::thread that
// sends still joinable, and a failure inside that thread escaped it: either
// way std::terminate. A receive from a peer that closed its socket looped on
// recv() returning 0 forever. A dead peer now surfaces as an exception.
TEST(LinkersSendRecv, PeerGoneIsAnErrorNotAnAbortOrAHang) {
  signal(SIGPIPE, SIG_IGN);  // as the Python interpreter does
  for (const int send_len : {1000, 4 << 20}) {  // below and above the size SendRecv sends from a thread
    SCOPED_TRACE(send_len);
    std::unique_ptr<Falcata::Linkers> rank0_owner, rank1;
    MakePair(&rank0_owner, &rank1);
    rank1.reset();  // the peer dies
    std::shared_ptr<Falcata::Linkers> rank0(std::move(rank0_owner));
    auto send = std::make_shared<std::vector<char>>(send_len, 1);
    auto recv = std::make_shared<std::vector<char>>(send_len);
    const std::string result = RunWithTimeout([rank0, send, recv, send_len]() {
      try {
        rank0->SendRecv(1, send->data(), send_len, 1, recv->data(), send_len);
        return std::string("returned");
      } catch (const std::runtime_error&) {
        return std::string("threw");
      }
    }, 30);
    EXPECT_EQ(result, "threw") << "(\"timeout\": SendRecv hangs)";
  }
}

// A send to a dead rank while receiving from another, live but silent rank:
// the failed send also aborts the receive, so the error comes back at once
// rather than after the receive timeout.
TEST(LinkersSendRecv, FailedSendAbortsAReceiveFromAnotherRank) {
  signal(SIGPIPE, SIG_IGN);
  auto ranks = MakeRanks(3);
  ranks[1].reset();  // the send peer dies; rank 2 stays up and sends nothing
  std::shared_ptr<Falcata::Linkers> rank0(std::move(ranks[0]));
  const int len = 4 << 20;
  auto send = std::make_shared<std::vector<char>>(len, 1);
  auto recv = std::make_shared<std::vector<char>>(len);
  const std::string result = RunWithTimeout([rank0, send, recv, len]() {
    try {
      rank0->SendRecv(1, send->data(), len, 2, recv->data(), len);
      return std::string("returned");
    } catch (const std::runtime_error& e) {
      return std::string(e.what()).find("send") != std::string::npos ? std::string("threw the send error")
                                                                      : std::string("threw ") + e.what();
    }
  }, 30);
  EXPECT_EQ(result, "threw the send error");
}

// The other way round: the receive peer dies while the send to a live rank
// that does not read is blocked. The receive failure is what is reported, not
// the send the abort interrupted.
TEST(LinkersSendRecv, FailedReceiveIsReportedNotTheSendItAborted) {
  signal(SIGPIPE, SIG_IGN);
  auto ranks = MakeRanks(3);
  ranks[1].reset();  // the receive peer dies; rank 2 stays up and reads nothing
  std::shared_ptr<Falcata::Linkers> rank0(std::move(ranks[0]));
  const int len = 16 << 20;  // more than the socket buffers hold, so the send blocks
  auto send = std::make_shared<std::vector<char>>(len, 1);
  auto recv = std::make_shared<std::vector<char>>(len);
  const std::string result = RunWithTimeout([rank0, send, recv, len]() {
    try {
      rank0->SendRecv(2, send->data(), len, 1, recv->data(), len);
      return std::string("returned");
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
  }, 30);
  EXPECT_NE(result.find("rank 1 closed the connection"), std::string::npos) << result;
}

// A healthy exchange still works, from the thread and the direct path.
TEST(LinkersSendRecv, ExchangesBothWays) {
  std::unique_ptr<Falcata::Linkers> rank0, rank1;
  MakePair(&rank0, &rank1);
  for (const int len : {1000, 4 << 20}) {
    std::vector<char> s0(len, 'a'), s1(len, 'b'), r0(len), r1(len);
    auto other = std::async(std::launch::async, [&]() { rank1->SendRecv(0, s1.data(), len, 0, r1.data(), len); });
    rank0->SendRecv(1, s0.data(), len, 1, r0.data(), len);
    other.get();
    EXPECT_EQ(r0, s1);
    EXPECT_EQ(r1, s0);
  }
}

// The listening thread Construct starts threw from inside the std::thread on a
// bad handshake (here: a connection announcing rank 99), so std::terminate
// ended the process; the failure now reaches the Linkers constructor.
TEST(LinkersSendRecv, BadHandshakeFailsTheSetup) {
  const int p0 = FreePort(), p1 = FreePort();
  Falcata::Config config;
  config.num_machines = 2;
  config.machines = "127.0.0.1 " + std::to_string(p0) + ",127.0.0.1 " + std::to_string(p1);
  config.local_listen_port = p1;  // rank 1 waits for rank 0 to connect
  config.time_out = 1;
  auto setup_promise = std::make_shared<std::promise<std::string>>();
  std::future<std::string> setup = setup_promise->get_future();
  std::thread([setup_promise, config]() {
    try {
      Falcata::Linkers linkers(config);
      setup_promise->set_value("constructed");
    } catch (const std::runtime_error&) {
      setup_promise->set_value("threw");
    }
  }).detach();  // see RunWithTimeout: a hang fails the test, it does not block it
  // play rank 0, badly
  int fd = -1;
  for (int attempt = 0; attempt < 100; ++attempt) {
    fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(p1));
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      break;
    }
    close(fd);
    fd = -1;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ASSERT_GE(fd, 0);
  const int bad_rank = 99;
  ASSERT_EQ(send(fd, &bad_rank, sizeof(bad_rank), 0), static_cast<ssize_t>(sizeof(bad_rank)));
  ASSERT_EQ(setup.wait_for(std::chrono::seconds(30)), std::future_status::ready);
  EXPECT_EQ(setup.get(), "threw");
  close(fd);
  // the failed setup closed its listener: nothing accepts connections on the port any more
  const int again = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(p1));
  EXPECT_NE(connect(again, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  close(again);
}

#endif  // defined(__linux__) && defined(USE_SOCKET)
