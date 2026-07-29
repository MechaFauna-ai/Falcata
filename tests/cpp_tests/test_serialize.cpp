/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <Falcata/utils/byte_buffer.h>
#include <Falcata/utils/log.h>
#include <Falcata/c_api.h>
#include <Falcata/dataset.h>

#include <iostream>
#include <string>

using Falcata::ByteBuffer;
using Falcata::Dataset;
using Falcata::Log;
using Falcata::TestUtils;

TEST(Serialization, JustWorks) {
  // Load some test data
  DatasetHandle dataset_handle;
  const char* params = "max_bin=15";
  int result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.test", params, &dataset_handle);
  EXPECT_EQ(0, result) << "LoadDatasetFromExamples result code: " << result;

  Dataset* dataset;
  bool succeeded = true;
  std::string exceptionText("");
  try {
    dataset = static_cast<Dataset*>(dataset_handle);

    // Serialize the reference
    ByteBufferHandle buffer_handle;
    int32_t buffer_len;
    result = FLC_DatasetSerializeReferenceToBinary(dataset_handle, &buffer_handle, &buffer_len);
    EXPECT_EQ(0, result) << "FLC_DatasetSerializeReferenceToBinary result code: " << result;

    ByteBuffer* buffer = nullptr;
    Dataset* deserialized_dataset = nullptr;
    try {
      buffer = static_cast<ByteBuffer*>(buffer_handle);

      // Deserialize the reference
      DatasetHandle deserialized_dataset_handle;
      result = FLC_DatasetCreateFromSerializedReference(buffer->Data(),
                                                         static_cast<int32_t>(buffer->GetSize()),
                                                         dataset->num_data(),
                                                         0,  // num_classes
                                                         params,
                                                         &deserialized_dataset_handle);
      EXPECT_EQ(0, result) << "FLC_DatasetCreateFromSerializedReference result code: " << result;

      // Confirm 1 successful API call
      deserialized_dataset = static_cast<Dataset*>(deserialized_dataset_handle);
      EXPECT_EQ(dataset->num_data(), deserialized_dataset->num_data());
    } catch (std::exception& ex) {
      succeeded = false;
      exceptionText = std::string(ex.what());
    }

    // Free memory
    if (buffer) {
      result = FLC_ByteBufferFree(buffer);
      EXPECT_EQ(0, result) << "FLC_ByteBufferFree result code: " << result;
    }
    if (deserialized_dataset) {
      result = FLC_DatasetFree(deserialized_dataset);
      EXPECT_EQ(0, result) << "FLC_DatasetFree result code: " << result;
    }
  } catch (std::exception& ex) {
    succeeded = false;
    exceptionText = std::string(ex.what());
  }

  if (dataset) {
    result = FLC_DatasetFree(dataset);
    EXPECT_EQ(0, result) << "FLC_DatasetFree result code: " << result;
  }

  if (!succeeded) {
    FAIL() << "Test Serialization failed with exception: " << exceptionText;
  }
}
