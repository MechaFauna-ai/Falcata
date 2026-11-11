// Copyright (c) 2026 The Falcata developers. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root.
#include <Falcata/c_api.h>
#include <cassert>
#include <cstring>
#include <new>

#include "src/boosting/dart.hpp"
int main() {
  double data[40];
  for (int i = 0; i < 40; ++i) data[i] = i;
  DatasetHandle handle = nullptr;
  assert(FLC_DatasetCreateFromMat(data, C_API_DTYPE_FLOAT64, 40, 1, 1, "min_data_in_leaf=1 num_threads=1", nullptr, &handle) == 0);
  alignas(Falcata::DART) unsigned char storage[sizeof(Falcata::DART)];
  std::memset(storage, 2, sizeof(storage));
  auto* dart = new (storage) Falcata::DART();
  Falcata::Config config;
  config.num_threads = 1;
  config.min_data_in_leaf = 1;
  dart->Init(&config, static_cast<Falcata::Dataset*>(handle), nullptr, {});
  int64_t len;
  const double* scores = dart->GetTrainingScore(&len);
  assert(len == 40);
  for (int i = 0; i < 40; ++i) assert(scores[i] == 0.0);
  assert(dart->GetTrainingScore(&len) == scores);
  dart->~DART();
  assert(FLC_DatasetFree(handle) == 0);
}
