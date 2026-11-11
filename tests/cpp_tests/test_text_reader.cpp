/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/utils/text_reader.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// ReadContent returns the whole file, across its 16 MiB read chunks.
TEST(TextReader, ReadContentReturnsTheWholeFile) {
  const std::string path = ::testing::TempDir() + "falcata_read_content.txt";
  for (const size_t size : {size_t{0}, size_t{1}, size_t{4096}, size_t{16} * 1024 * 1024,
                            size_t{16} * 1024 * 1024 + 12345}) {
    std::string content(size, '\0');
    for (size_t i = 0; i < size; ++i) {
      content[i] = static_cast<char>('a' + (i * 7919) % 26);
    }
    {
      std::ofstream f(path, std::ios::binary);
      f.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    Falcata::TextReader<size_t> reader(path.c_str(), false);
    size_t len = 12345;
    const std::vector<char> got = reader.ReadContent(&len);
    ASSERT_EQ(len, size);
    ASSERT_EQ(got.size(), size);
    EXPECT_TRUE(std::string(got.begin(), got.end()) == content) << size;
  }
  std::remove(path.c_str());
}
