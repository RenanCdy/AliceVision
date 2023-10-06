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

inline sycl::float3 operator * (float k, sycl::float3 v)
{
    return sycl::float3(k*v[0], k*v[1], k*v[2]);
}

inline sycl::float3 operator * (sycl::float3 v, float k)
{
    return k * v;
}

inline float size(const sycl::float3& a)
{
    return sycl::sqrt(a.x() * a.x() + a.y() * a.y() + a.z() * a.z());
}

inline float size(const sycl::float2& a)
{
    return sycl::sqrt(a.x() * a.x() + a.y() * a.y());
}

inline float dot(const sycl::float3& a, const sycl::float3& b)
{
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

inline float dot(const sycl::float2& a, const sycl::float2& b)
{
    return a.x() * b.x() + a.y() * b.y();
}

inline float dist(const sycl::float3& a, const sycl::float3& b)
{
    return size(a -b);
}

inline float dist(const sycl::float2& a, const sycl::float2& b)
{
    return size(a -b);
}

inline sycl::float3 cross(const sycl::float3& a, const sycl::float3& b)
{
    return sycl::float3(a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(), a.x() * b.y() - a.y() * b.x());
}

inline sycl::float3 M3x4mulV3(const float* M3x4, const sycl::float3& V)
{
    return sycl::float3(M3x4[0] * V.x() + M3x4[3] * V.y() + M3x4[6] * V.z() + M3x4[9],
                        M3x4[1] * V.x() + M3x4[4] * V.y() + M3x4[7] * V.z() + M3x4[10],
                        M3x4[2] * V.x() + M3x4[5] * V.y() + M3x4[8] * V.z() + M3x4[11]);
}

inline sycl::float3 M3x3mulV2(const float* M3x3, const sycl::float2& V)
{
    return sycl::float3(M3x3[0] * V.x() + M3x3[3] * V.y() + M3x3[6], M3x3[1] * V.x() + M3x3[4] * V.y() + M3x3[7],
                        M3x3[2] * V.x() + M3x3[5] * V.y() + M3x3[8]);
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

inline float pointLineDistance3D(const sycl::float3& point, const sycl::float3& linePoint,
                                 const sycl::float3& lineVectNormalized)
{
    return size(cross(lineVectNormalized, linePoint - point));
}

inline sycl::float3 linePlaneIntersect(const sycl::float3& linePoint, const sycl::float3& lineVect,
                                       const sycl::float3& planePoint, const sycl::float3& planeNormal)
{
    const float k = (dot(planePoint, planeNormal) - dot(planeNormal, linePoint)) / dot(planeNormal, lineVect);
    return linePoint + k * lineVect;
}

/**
 * @brief Sigmoid function filtering
 * @note f(x) = min + (max-min) * \frac{1}{1 + e^{10 * (x - mid) / width}}
 * @see https://www.desmos.com/calculator/1qvampwbyx
 */
inline float sigmoid(float zeroVal, float endVal, float sigwidth, float sigMid, float xval)
{
    return zeroVal + (endVal - zeroVal) * (1.0f / (1.0f + sycl::exp(10.0f * ((xval - sigMid) / sigwidth))));
}

/**
 * @brief Sigmoid function filtering
 * @note f(x) = min + (max-min) * \frac{1}{1 + e^{10 * (mid - x) / width}}
 */
inline float sigmoid2(float zeroVal, float endVal, float sigwidth, float sigMid, float xval)
{
    return zeroVal + (endVal - zeroVal) * (1.0f / (1.0f + sycl::exp(10.0f * ((sigMid - xval) / sigwidth))));
}

} // namespace depthMap
} // namespace aliceVision
