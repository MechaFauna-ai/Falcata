/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/config.h>

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "../../src/network/linkers.h"

namespace {

int OpenDescriptors() {
  int count = 0;
  DIR* dir = opendir("/proc/self/fd");
  while (readdir(dir) != nullptr) {
    ++count;
  }
  closedir(dir);
  return count;
}

}  // namespace

// TcpSocket was a copyable handle whose destructor never closed the
// descriptor: every socket not closed by hand on every path leaked, and a copy
// kept the number of a descriptor another copy had closed (and the system
// could reuse).
TEST(TcpSocket, OwnsItsDescriptor) {
  static_assert(!std::is_copy_constructible<Falcata::TcpSocket>::value, "TcpSocket must not be copyable");
  const int before = OpenDescriptors();
  {
    Falcata::TcpSocket socket;
    EXPECT_FALSE(socket.IsClosed());
    EXPECT_EQ(OpenDescriptors(), before + 1);
  }
  EXPECT_EQ(OpenDescriptors(), before);

  Falcata::TcpSocket a;
  Falcata::TcpSocket b(std::move(a));
  EXPECT_TRUE(a.IsClosed());
  EXPECT_FALSE(b.IsClosed());
  Falcata::TcpSocket c;
  c = std::move(b);  // c's own descriptor is closed, b's moves over
  EXPECT_TRUE(b.IsClosed());
  EXPECT_EQ(OpenDescriptors(), before + 1);
}

// A Linkers whose setup fails (here: its port is taken) left its listening
// socket open: a caller retrying the network setup leaked one descriptor per
// attempt.
TEST(TcpSocket, FailedLinkersSetupLeavesNoDescriptor) {
  // hold a port
  const int holder = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = 0;
  ASSERT_EQ(bind(holder, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  ASSERT_EQ(listen(holder, 1), 0);
  socklen_t len = sizeof(addr);
  getsockname(holder, reinterpret_cast<sockaddr*>(&addr), &len);
  const int port = ntohs(addr.sin_port);

  Falcata::Config config;
  config.num_machines = 2;
  config.machines = "127.0.0.1 " + std::to_string(port) + ",127.0.0.1 " + std::to_string(port + 1);
  config.local_listen_port = port;
  const int before = OpenDescriptors();
  for (int attempt = 0; attempt < 3; ++attempt) {
    EXPECT_THROW(Falcata::Linkers linkers(config), std::runtime_error);
  }
  EXPECT_EQ(OpenDescriptors(), before);
  close(holder);
}
