// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sycl/sycl.hpp>
#include "customDataType.dp.hpp"

namespace aliceVision {
namespace depthMap {
namespace __sycl {

/**
 * @struct DeviceCameraParams
 * @brief Support class to maintain useful camera parameters in gpu memory.
 */
struct DeviceCameraParams
{
    float P[12];
    float iP[9];
    float R[9];
    float iR[9];
    float K[9];
    float iK[9];
    custom_sycl::custom_float3 C;
    custom_sycl::custom_float3 XVect;
    custom_sycl::custom_float3 YVect;
    custom_sycl::custom_float3 ZVect;
};

// global / constant data structures

#define ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS 100

// extern __constant__ DeviceCameraParams constantCameraParametersArray_d[ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS];
extern DeviceCameraParams* cameraParametersArray_d;

} // namespace __sycl
} // namespace depthMap
} // namespace aliceVision
