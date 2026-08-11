/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_BAGGING_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_BAGGING_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/meta.h>

namespace Falcata {

/*!
 * \brief Draw a bagging sample on the device.
 *
 * Used instead of the host sampler for device_type=cuda: the host version
 * walks every row through a stateful RNG and copies the whole index array over
 * PCIe on each re-bagging iteration -- work proportional to the DATASET rather
 * than to the sample, and the dominant cost of bagged training.
 *
 * \param num_data          rows in the training set
 * \param bagging_fraction  probability a row is kept
 * \param iter              boosting iteration; part of the Philox counter, so
 *                          each re-bag draws an independent sample
 * \param seed              bagging seed
 * \param scratch           device scratch, resized as needed (flags + scan)
 * \param block_buffer      device scratch for the scan's per-block sums
 * \param out_indices       device buffer of at least num_data entries, filled
 *                          as [in-bag | out-of-bag]
 * \return number of in-bag rows
 */
data_size_t CUDABaggingSample(data_size_t num_data,
                              double bagging_fraction,
                              int iter,
                              int seed,
                              CUDAVector<data_size_t>* scratch,
                              CUDAVector<data_size_t>* block_buffer,
                              data_size_t* out_indices);

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_BAGGING_HPP_
