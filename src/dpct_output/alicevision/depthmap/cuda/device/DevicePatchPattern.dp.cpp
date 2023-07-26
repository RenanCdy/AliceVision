// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "DevicePatchPattern.hpp"

namespace aliceVision {
namespace depthMap {

static dpct::constant_memory<DevicePatchPattern, 0> constantPatchPattern_d;

} // namespace depthMap
} // namespace aliceVision
