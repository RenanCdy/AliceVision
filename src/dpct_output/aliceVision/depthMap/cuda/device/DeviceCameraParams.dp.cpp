// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "DeviceCameraParams.hpp"

namespace aliceVision {
namespace depthMap {

static dpct::constant_memory<DeviceCameraParams, 1>
    constantCameraParametersArray_d(ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS);

} // namespace depthMap
} // namespace aliceVision

