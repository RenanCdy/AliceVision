// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#define M_PIF 3.141592653589793238462643383279502884e+00F

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

inline sycl::float3 linePlaneIntersect(const sycl::float3& linePoint, const sycl::float3& lineVect,
                                       const sycl::float3& planePoint, const sycl::float3& planeNormal)
{
    const float k = (dot(planePoint, planeNormal) - dot(planeNormal, linePoint)) / dot(planeNormal, lineVect);
    return linePoint + k * lineVect;
}

inline sycl::float3 closestPointOnPlaneToPoint(const sycl::float3& point, const sycl::float3& planePoint,
                                               const sycl::float3& planeNormalNormalized)
{
    return point - planeNormalNormalized * dot(planeNormalNormalized, point - planePoint);
}

inline sycl::float3 closestPointToLine3D(const sycl::float3& point, const sycl::float3& linePoint,
                                         const sycl::float3& lineVectNormalized)
{
    return linePoint + lineVectNormalized * dot(lineVectNormalized, point - linePoint);
}

inline float pointLineDistance3D(const sycl::float3& point, const sycl::float3& linePoint,
                                 const sycl::float3& lineVectNormalized)
{
    return size(cross(lineVectNormalized, linePoint - point));
}

// v1,v2 dot not have to be normalized
inline float angleBetwV1andV2(const sycl::float3& iV1, const sycl::float3& iV2)
{
    sycl::float3 V1 = iV1;
    normalize(V1);

    sycl::float3 V2 = iV2;
    normalize(V2);

    return sycl::fabs(sycl::acos(V1.x() * V2.x() + V1.y() * V2.y() + V1.z() * V2.z()) / (M_PIF / 180.0f));
}

inline float angleBetwABandAC(const sycl::float3& A, const sycl::float3& B, const sycl::float3& C)
{
    sycl::float3 V1 = B - A;
    sycl::float3 V2 = C - A;

    normalize(V1);
    normalize(V2);

    const double x = double(V1.x() * V2.x() + V1.y() * V2.y() + V1.z() * V2.z());
    double a = sycl::acos((double)x);
    a = sycl::isinf(a) ? 0.0 : a;
    return float(sycl::fabs(a) / (M_PI / 180.0));
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
