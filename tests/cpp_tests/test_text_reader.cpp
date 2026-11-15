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
    if (size >= size_t{16} * 1024 * 1024) {
      // reaching the end does not grow the result by another chunk
      EXPECT_LT(got.capacity(), 2 * size) << size;
    }
  }
  std::remove(path.c_str());
}

namespace {

std::vector<std::string> SerialLines(const std::string& path) {
  Falcata::TextReader<int> reader(path.c_str(), false);
  std::vector<std::string> lines;
  const int n = reader.ReadAllAndProcess(
      [&lines](int, const char* buffer, size_t size) { lines.emplace_back(buffer, size); });
  EXPECT_EQ(n, static_cast<int>(lines.size()));
  return lines;
}

std::vector<std::string> ParallelLines(const std::string& path) {
  Falcata::TextReader<int> reader(path.c_str(), false);
  std::vector<std::string> lines;
  const int n = reader.ReadAllAndProcessParallel(
      [&lines](int, const std::vector<std::string>& chunk) { lines.insert(lines.end(), chunk.begin(), chunk.end()); });
  EXPECT_EQ(n, static_cast<int>(lines.size()));
  return lines;
}

}  // namespace

// A blank line is not a data row: one in the middle of a chunk was skipped, but one at the start of the file
// (or a \r\n blank line whose break straddled the 16 MiB chunk boundary) came out as an empty line, which the
// parsers turn into an extra all-zero row with label 0.
TEST(TextReader, BlankLinesAreSkippedWhereverTheyFall) {
  const std::string path = ::testing::TempDir() + "falcata_text_reader_blank_lines.txt";
  const auto write = [&path](const std::string& content) {
    std::ofstream f(path, std::ios::binary);
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
  };
  const std::vector<std::string> expected = {"1,2", "0,3"};
  for (const std::string& content : {std::string("\r\n1,2\r\n0,3\r\n"), std::string("\n\n1,2\n0,3\n"),
                                     std::string("\r\r\n\n1,2\n\r\n0,3"), std::string("1,2\r\n\r\n0,3\r\n")}) {
    write(content);
    EXPECT_EQ(SerialLines(path), expected);
    EXPECT_EQ(ParallelLines(path), expected);
  }
  // a \r\n\r\n blank line right after the first chunk's last full line: the second chunk starts with "\r\n"
  const size_t block = 16 * 1024 * 1024;
  const std::string line = "1,2\r\n";
  std::string content = std::string(block - (block / line.size() - 1) * line.size() - 2, '7') + "\r\n";
  size_t num_lines = 1;
  while (content.size() < block) {
    content += line;
    ++num_lines;
  }
  ASSERT_EQ(content.size(), block);
  content += "\r\n0,3\r\n";
  ++num_lines;
  write(content);
  for (const auto& lines : {SerialLines(path), ParallelLines(path)}) {
    ASSERT_EQ(lines.size(), num_lines);
    EXPECT_EQ(lines.front(), std::string(block - (block / line.size() - 1) * line.size() - 2, '7'));
    EXPECT_EQ(lines.back(), "0,3");
  }
  std::remove(path.c_str());
}
