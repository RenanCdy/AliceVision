// This file is part of the AliceVision project.
// Copyright (c) 2018 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <aliceVision/depthMap/BufPtr.hpp>
#include <aliceVision/depthMap/cuda/host/memory.hpp>

#include <set>

namespace aliceVision {
namespace depthMap {
namespace __sycl {
#define MAX_CONSTANT_GAUSS_SCALES 10
#define MAX_CONSTANT_GAUSS_MEM_SIZE 128

/*********************************************************************************
 * global / constant data structures
 *********************************************************************************/

inline float getGauss(int scale, int idx, int* d_gaussianArrayOffset, float* d_gaussianArray)
{
    return d_gaussianArray[d_gaussianArrayOffset[scale] + idx];
}

/**
 * @brief Create Gaussian array in device constant memory.
 * @param[in] cudaDeviceId the cuda device id
 * @param[in] scales the number of pre-computed Gaussian scales
 */
extern void cuda_createConstantGaussianArray(sycl::queue& stream, int scales);
} 

/**
 * @brief Downscale with Gaussian blur the given frame.
 * @param[out] out_downscaledImg_dmp the downscaled image in device memory
 * @param[in] in_img_tex the cuda texture object of the input full size image
 * @param[in] downscale the downscale factor to apply
 * @param[in] gaussRadius the Gaussian radius
 * @param[in] stream the CUDA stream for gpu execution
 */
extern void cuda_downscaleWithGaussianBlur(CudaDeviceMemoryPitched<CudaRGBA, 2>& out_downscaledImg_dmp,
                                           CudaDeviceMemoryPitched<CudaRGBA, 2>& in_img_tex, int downscale,
                                           int gaussRadius, sycl::queue& stream);


} // namespace depthMap
} // namespace aliceVision

