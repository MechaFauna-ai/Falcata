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

// two ranks of one job in this process, connected over loopback
void MakePair(std::unique_ptr<Falcata::Linkers>* rank0, std::unique_ptr<Falcata::Linkers>* rank1) {
  const int p0 = FreePort(), p1 = FreePort();
  const std::string machines = "127.0.0.1 " + std::to_string(p0) + ",127.0.0.1 " + std::to_string(p1);
  auto make = [&machines](int port) {
    Falcata::Config config;
    config.num_machines = 2;
    config.machines = machines;
    config.local_listen_port = port;
    config.time_out = 1;
    return std::unique_ptr<Falcata::Linkers>(new Falcata::Linkers(config));
  };
  auto f1 = std::async(std::launch::async, make, p1);
  *rank0 = make(p0);
  *rank1 = f1.get();
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
    std::unique_ptr<Falcata::Linkers> rank0, rank1;
    MakePair(&rank0, &rank1);
    rank1.reset();  // the peer dies
    std::vector<char> send(send_len, 1), recv(send_len);
    auto result = std::async(std::launch::async, [&]() {
      try {
        rank0->SendRecv(1, send.data(), send_len, 1, recv.data(), send_len);
        return std::string("returned");
      } catch (const std::runtime_error&) {
        return std::string("threw");
      }
    });
    ASSERT_EQ(result.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "SendRecv hangs";
    EXPECT_EQ(result.get(), "threw");
  }
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
  auto setup = std::async(std::launch::async, [&config]() {
    try {
      Falcata::Linkers linkers(config);
      return std::string("constructed");
    } catch (const std::runtime_error&) {
      return std::string("threw");
    }
  });
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
