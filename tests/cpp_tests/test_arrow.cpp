/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Oliver Borchert
 */

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include <nanoarrow/nanoarrow.hpp>

#include "../../src/arrow/array.hpp"

using Falcata::ArrowChunkedArray;

namespace {

// Build an ArrowArrayStream from a schema and a list of chunk arrays. Takes ownership of the
// passed schema and chunks.
nanoarrow::UniqueArrayStream MakeStream(nanoarrow::UniqueSchema schema,
                                        std::vector<nanoarrow::UniqueArray> chunks) {
  nanoarrow::UniqueArrayStream stream;
  nanoarrow::VectorArrayStream(schema.get(), std::move(chunks)).ToArrayStream(stream.get());
  return stream;
}

nanoarrow::UniqueSchema MakePrimitiveSchema(ArrowType type) {
  nanoarrow::UniqueSchema schema;
  EXPECT_EQ(ArrowSchemaInitFromType(schema.get(), type), NANOARROW_OK);
  return schema;
}

nanoarrow::UniqueSchema MakeStructSchema(const std::vector<ArrowType>& field_types) {
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  EXPECT_EQ(ArrowSchemaSetTypeStruct(schema.get(), field_types.size()), NANOARROW_OK);
  for (size_t i = 0; i < field_types.size(); ++i) {
    EXPECT_EQ(ArrowSchemaSetType(schema->children[i], field_types[i]), NANOARROW_OK);
  }
  return schema;
}

template <typename T>
nanoarrow::UniqueArray MakePrimitiveArray(ArrowType type, const std::vector<T>& values,
                                          const std::vector<int64_t>& null_indices = {},
                                          int64_t offset = 0) {
  nanoarrow::UniqueArray array;
  EXPECT_EQ(ArrowArrayInitFromType(array.get(), type), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  size_t null_idx_pos = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    if (null_idx_pos < null_indices.size() &&
        null_indices[null_idx_pos] == static_cast<int64_t>(i)) {
      EXPECT_EQ(ArrowArrayAppendNull(array.get(), 1), NANOARROW_OK);
      ++null_idx_pos;
    } else {
      if (type == NANOARROW_TYPE_BOOL) {
        EXPECT_EQ(ArrowArrayAppendInt(array.get(), values[i] ? 1 : 0), NANOARROW_OK);
      } else {
        EXPECT_EQ(ArrowArrayAppendDouble(array.get(), static_cast<double>(values[i])),
                  NANOARROW_OK);
      }
    }
  }
  EXPECT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);

  // Apply slicing offset (tests the consumer's handling of `array->offset`).
  if (offset > 0) {
    array->offset += offset;
    array->length -= offset;
  }
  return array;
}

}  // namespace

TEST(ArrowChunkedArrayTest, GetLength) {
  // Single chunk
  {
    auto schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
    std::vector<nanoarrow::UniqueArray> chunks;
    chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {1, 2}));
    ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());
    ASSERT_EQ(chunked_array.get_length(), 2);
  }

  // Multiple chunks
  {
    auto schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
    std::vector<nanoarrow::UniqueArray> chunks;
    chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {1, 2}));
    chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {3, 4, 5, 6}));
    ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());
    ASSERT_EQ(chunked_array.get_length(), 6);
  }

  // Sliced chunk via offset
  {
    auto schema = MakePrimitiveSchema(NANOARROW_TYPE_BOOL);
    std::vector<nanoarrow::UniqueArray> chunks;
    chunks.emplace_back(
        MakePrimitiveArray<bool>(NANOARROW_TYPE_BOOL, {true, false, true, true}, {}, 1));
    ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());
    ASSERT_EQ(chunked_array.get_length(), 3);
  }
}

TEST(ArrowChunkedArrayTest, GetFields) {
  auto schema = MakeStructSchema({NANOARROW_TYPE_FLOAT, NANOARROW_TYPE_FLOAT});

  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  std::vector<float> dat1 = {1, 2, 3};
  std::vector<float> dat2 = {4, 5, 6};
  for (size_t i = 0; i < dat1.size(); ++i) {
    ASSERT_EQ(ArrowArrayAppendDouble(array->children[0], dat1[i]), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayAppendDouble(array->children[1], dat2[i]), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayFinishElement(array.get()), NANOARROW_OK);
  }
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);

  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(std::move(array));
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  ASSERT_EQ(chunked_array.get_length(), 3);
  ASSERT_EQ(chunked_array.get_num_fields(), 2);

  int32_t first0 = 0, first1 = 0;
  chunked_array.view().field(0).visit<int32_t>([&](auto v) { first0 = *v.begin(); });
  chunked_array.view().field(1).visit<int32_t>([&](auto v) { first1 = *v.begin(); });
  ASSERT_EQ(first0, 1);
  ASSERT_EQ(first1, 4);
}

TEST(ArrowChunkedArrayTest, IteratorArithmetic) {
  auto schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {1, 2}));
  chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {3, 4, 5, 6}));
  chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {7}));
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  chunked_array.view().visit<int32_t>([](auto v) {
    auto it = v.begin();
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 2);
    ++it;
    EXPECT_EQ(*it, 3);
    it += 2;
    EXPECT_EQ(*it, 5);
    it += 2;
    EXPECT_EQ(*it, 7);

    auto begin = v.begin();
    EXPECT_EQ(begin[0], 1);
    EXPECT_EQ(begin[1], 2);
    EXPECT_EQ(begin[2], 3);
    EXPECT_EQ(begin[6], 7);

    auto end = v.end();
    EXPECT_EQ(end - it, 1);
    EXPECT_EQ(end - v.begin(), 7);
  });
}

TEST(ArrowChunkedArrayTest, BooleanIterator) {
  auto schema = MakePrimitiveSchema(NANOARROW_TYPE_BOOL);
  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(MakePrimitiveArray<bool>(NANOARROW_TYPE_BOOL, {false, true, false}, {2}));
  chunks.emplace_back(MakePrimitiveArray<bool>(
      NANOARROW_TYPE_BOOL, {false, false, false, false, true, true, true, true, false, true}, {},
      1));
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  chunked_array.view().visit<float>([](auto v) {
    auto it = v.begin();
    // First chunk
    EXPECT_EQ(*it, 0);
    EXPECT_EQ(*(++it), 1);
    EXPECT_TRUE(std::isnan(*(++it)));

    // Second chunk
    EXPECT_EQ(*(++it), 0);
    it += 3;
    EXPECT_EQ(*it, 1);
    it += 4;
    EXPECT_EQ(*it, 0);
    EXPECT_EQ(*(++it), 1);

    EXPECT_EQ(++it, v.end());
  });
}

TEST(ArrowChunkedArrayTest, OffsetAndValidity) {
  auto schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(
      MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {0, 1, 2, 3, 4, 5, 6}, {2, 3}, 2));
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  chunked_array.view().visit<double>([](auto v) {
    auto it = v.begin();
    EXPECT_TRUE(std::isnan(*it));
    EXPECT_TRUE(std::isnan(*(++it)));
    EXPECT_EQ(it[2], 4);
    EXPECT_EQ(it[4], 6);
  });
}

namespace {

nanoarrow::UniqueArray MakeStructArray(const nanoarrow::UniqueSchema& schema,
                                       const std::vector<std::vector<double>>& columns) {
  nanoarrow::UniqueArray array;
  EXPECT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  for (size_t i = 0; i < columns[0].size(); ++i) {
    for (size_t j = 0; j < columns.size(); ++j) {
      EXPECT_EQ(ArrowArrayAppendDouble(array->children[j], columns[j][i]), NANOARROW_OK);
    }
    EXPECT_EQ(ArrowArrayFinishElement(array.get()), NANOARROW_OK);
  }
  EXPECT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);
  return array;
}

nanoarrow::UniqueArray MakeStructArray(ArrowSchema* schema, const std::vector<double>& dat0,
                                       const std::vector<double>& dat1) {
  nanoarrow::UniqueArray array;
  EXPECT_EQ(ArrowArrayInitFromSchema(array.get(), schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  for (size_t i = 0; i < dat0.size(); ++i) {
    EXPECT_EQ(ArrowArrayAppendDouble(array->children[0], dat0[i]), NANOARROW_OK);
    EXPECT_EQ(ArrowArrayAppendDouble(array->children[1], dat1[i]), NANOARROW_OK);
    EXPECT_EQ(ArrowArrayFinishElement(array.get()), NANOARROW_OK);
  }
  EXPECT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);
  return array;
}

void NoopRelease(struct ArrowArray* array) { array->release = nullptr; }

// Hand a corrupted shallow copy of `chunk` (whose release does nothing, so `chunk` still frees the
// real buffers) to the constructor taking an array of chunks, and expect it to be rejected.
void ExpectRejected(const nanoarrow::UniqueSchema& schema, const nanoarrow::UniqueArray& chunk,
                    const std::function<void(struct ArrowArray*)>& corrupt) {
  struct ArrowArray shallow = *chunk.get();
  shallow.release = &NoopRelease;
  corrupt(&shallow);
  nanoarrow::UniqueSchema schema_copy;
  ASSERT_EQ(ArrowSchemaDeepCopy(schema.get(), schema_copy.get()), NANOARROW_OK);
  struct ArrowSchema raw_schema;
  schema_copy.move(&raw_schema);
  EXPECT_THROW(ArrowChunkedArray chunked_array(1, &shallow, &raw_schema), std::exception);
}

std::vector<double> ReadField(const ArrowChunkedArray& chunked_array, int64_t field_idx) {
  std::vector<double> out;
  chunked_array.view().field(field_idx).visit<double>([&](auto v) {
    for (auto it = v.begin(); it != v.end(); ++it) out.push_back(*it);
    EXPECT_EQ(v.end() - v.begin(), static_cast<int64_t>(out.size()));
  });
  return out;
}

}  // namespace

// Chunks are validated against the schema before their buffers are read: each of these used to be
// read out of bounds (or through a NULL pointer) by the dataset, set_field and predict paths.
TEST(ArrowChunkedArrayTest, MalformedChunksAreRejected) {
  for (auto type : {NANOARROW_TYPE_DOUBLE, NANOARROW_TYPE_FLOAT, NANOARROW_TYPE_BOOL}) {
    auto schema = MakePrimitiveSchema(type);
    auto chunk = MakePrimitiveArray<double>(type, {1, 2, 3, 4});
    ExpectRejected(schema, chunk, [](struct ArrowArray* a) { a->length = -1; });
    ExpectRejected(schema, chunk, [](struct ArrowArray* a) { a->offset = -3; });
    ExpectRejected(schema, chunk, [](struct ArrowArray* a) { a->n_buffers = 1; });
    ExpectRejected(schema, chunk, [](struct ArrowArray* a) { a->buffers = nullptr; });
    const void* null_buffers[2] = {nullptr, nullptr};
    ExpectRejected(schema, chunk, [&](struct ArrowArray* a) { a->buffers = null_buffers; });
  }
}

TEST(ArrowChunkedArrayTest, MalformedStructChunksAreRejected) {
  auto schema = MakeStructSchema({NANOARROW_TYPE_DOUBLE, NANOARROW_TYPE_DOUBLE});
  auto chunk = MakeStructArray(schema, {{1, 2, 3}, {4, 5, 6}});
  // fewer children than the schema has fields, a NULL children array and a NULL child
  ExpectRejected(schema, chunk, [](struct ArrowArray* a) { a->n_children = 1; });
  ExpectRejected(schema, chunk, [](struct ArrowArray* a) { a->children = nullptr; });
  struct ArrowArray* children[2] = {chunk->children[0], nullptr};
  ExpectRejected(schema, chunk, [&](struct ArrowArray* a) { a->children = children; });
  // a child shorter than its struct parent
  struct ArrowArray short_child = *chunk->children[1];
  short_child.length = 1;
  struct ArrowArray* short_children[2] = {chunk->children[0], &short_child};
  ExpectRejected(schema, chunk, [&](struct ArrowArray* a) { a->children = short_children; });

  // the stream constructor validates too
  chunk->children[1]->length = 1;
  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(std::move(chunk));
  auto stream = MakeStream(std::move(schema), std::move(chunks));
  EXPECT_THROW(ArrowChunkedArray chunked_array(stream.get()), std::exception);
}

TEST(ArrowChunkedArrayTest, NegativeChunkCountIsRejected) {
  auto schema = MakePrimitiveSchema(NANOARROW_TYPE_DOUBLE);
  struct ArrowSchema raw_schema;
  schema.move(&raw_schema);
  EXPECT_THROW(ArrowChunkedArray chunked_array(-1, nullptr, &raw_schema), std::exception);
}

TEST(ArrowChunkedArrayTest, SlicedStructFields) {
  // A sliced struct keeps its offset and length on the parent and leaves its children whole, so
  // element i of a field is element (parent offset + i) of the child. The fields used to be read
  // with the child's own offset and length: the wrong rows, and more of them than the struct has.
  auto schema = MakeStructSchema({NANOARROW_TYPE_DOUBLE, NANOARROW_TYPE_DOUBLE});
  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(MakeStructArray(schema.get(), {0, 1, 2, 3, 4, 5}, {10, 11, 12, 13, 14, 15}));
  chunks.back()->offset = 2;  // rows 2, 3, 4
  chunks.back()->length = 3;
  chunks.emplace_back(MakeStructArray(schema.get(), {6, 7, 8}, {16, 17, 18}));
  chunks.back()->children[1]->offset = 1;  // the child's own slice adds to the parent's
  chunks.back()->children[1]->length = 2;
  chunks.back()->offset = 1;  // row 1 only: element 1 of child 0, element 2 of child 1's buffer
  chunks.back()->length = 1;
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  ASSERT_EQ(chunked_array.get_length(), 4);
  EXPECT_EQ(ReadField(chunked_array, 0), (std::vector<double>{2, 3, 4, 7}));
  EXPECT_EQ(ReadField(chunked_array, 1), (std::vector<double>{12, 13, 14, 18}));

  // random access through a concatenated view of both fields (the init_score path)
  std::vector<ArrowChunkedArray::View> fields = {chunked_array.view().field(0),
                                                 chunked_array.view().field(1)};
  ArrowChunkedArray::View(fields).visit<double>([](auto v) {
    ASSERT_EQ(v.end() - v.begin(), 8);
    EXPECT_EQ(v.begin()[3], 7);
    EXPECT_EQ(v.begin()[7], 18);
  });

  // a visitor copied out of visit() outlives the field view (the Arrow predict accessors)
  std::function<double(int64_t)> accessor;
  chunked_array.view().field(1).visit<double>([&](auto v) {
    accessor = [v](int64_t i) { return v.begin()[i]; };
  });
  EXPECT_EQ(accessor(3), 18);
}

TEST(ArrowChunkedArrayTest, StructChildShorterThanParentSlice) {
  // the parent's rows 1 and 2 need three elements of each child; child 0 has two. The chunk is
  // rejected when it is validated against the schema, before any field is read.
  auto schema = MakeStructSchema({NANOARROW_TYPE_FLOAT, NANOARROW_TYPE_DOUBLE});
  auto array = MakeStructArray(schema.get(), {0, 1, 2}, {10, 11, 12});
  array->children[0]->length = 2;
  array->offset = 1;
  array->length = 2;
  ExpectRejected(schema, array, [](struct ArrowArray*) {});
}
