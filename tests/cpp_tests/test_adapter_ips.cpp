/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

// the socket transport only (an MPI build has no TcpSocket)
#ifdef USE_SOCKET

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "../../src/network/socket_wrapper.hpp"

namespace {

// the members of the Windows IP_ADAPTER_INFO / IP_ADDR_STRING records that
// TcpSocket::GetLocalIpList reads (iphlpapi.h)
struct AddrString {
  AddrString* Next;
  struct {
    char String[16];
  } IpAddress;
};

struct AdapterInfo {
  AdapterInfo* Next;
  AddrString IpAddressList;
};

}  // namespace

// The Windows GetLocalIpList took only IpAddressList (the first address of
// each adapter) and never walked IpAddressList.Next, so a machine listed in
// the machine list by a secondary address could not find its own rank.
TEST(AdapterIps, EveryAddressOfEveryAdapter) {
  AddrString second_b{nullptr, {"10.0.0.7"}};
  AddrString second_a{&second_b, {"192.168.1.20"}};
  AdapterInfo eth1{nullptr, {nullptr, {"172.16.0.3"}}};
  AdapterInfo eth0{&eth1, {&second_a, {"192.168.1.10"}}};
  std::unordered_set<std::string> ips;
  Falcata::CollectAdapterIps(&eth0, &ips);
  EXPECT_EQ(ips, std::unordered_set<std::string>({"192.168.1.10", "192.168.1.20", "10.0.0.7", "172.16.0.3"}));

  std::unordered_set<std::string> none;
  Falcata::CollectAdapterIps(static_cast<const AdapterInfo*>(nullptr), &none);
  EXPECT_TRUE(none.empty());
}

#endif  // USE_SOCKET
