// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sycl/sycl.hpp>
// #include "aliceVision/depthMap/cuda/device/operators.dp.hpp"

// mn MATRIX ADDRESSING: mxy = x*n+y (x-row,y-col), (m-number of rows, n-number of columns)

namespace aliceVision {
namespace depthMap {

inline sycl::float3 M3x4mulV3(const float* M3x4, const sycl::float3& V)
{
    return sycl::float3(M3x4[0] * V.x() + M3x4[3] * V.y() + M3x4[6] * V.z() + M3x4[9],
                        M3x4[1] * V.x() + M3x4[4] * V.y() + M3x4[7] * V.z() + M3x4[10],
                        M3x4[2] * V.x() + M3x4[5] * V.y() + M3x4[8] * V.z() + M3x4[11]);
}

inline sycl::float2 project3DPoint(const float* M3x4, const sycl::float3& V)
{
    // without optimization
    // const float3 p = M3x4mulV3(M3x4, V);
    // return make_float2(p.x / p.z, p.y / p.z);

    sycl::float3 p = M3x4mulV3(M3x4, V);
    const float pzInv = 1.0f / p.z();
    return sycl::float2(p.x() * pzInv, p.y() * pzInv);
}

} // namespace depthMap
} // namespace aliceVision
