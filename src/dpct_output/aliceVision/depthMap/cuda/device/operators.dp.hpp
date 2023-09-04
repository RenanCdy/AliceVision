// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace aliceVision {
namespace depthMap {

/*
DPCT1011:5: The tool detected overloaded operators for built-in vector types, which may conflict with the SYCL 2020
standard operators (see 4.14.2.1 Vec interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator*(const sycl::float4& a, const float& d)
{
    return sycl::float4(a.x() * d, a.y() * d, a.z() * d, a.w() * d);
}

inline sycl::float4 operator+(const sycl::float4& a, const sycl::float4& d)
{
    return sycl::float4(a.x() + d.x(), a.y() + d.y(), a.z() + d.z(), a.w() + d.w());
}

inline sycl::float4 operator*(const float& d, const sycl::float4& a)
{
    return sycl::float4(a.x() * d, a.y() * d, a.z() * d, a.w() * d);
}

inline sycl::float4 operator/(const sycl::float4& a, const float& d)
{
    return sycl::float4(a.x() / d, a.y() / d, a.z() / d, a.w() / d);
}

inline sycl::float3 operator*(const sycl::float3& a, const float& d)
{
    return sycl::float3(a.x() * d, a.y() * d, a.z() * d);
}

inline sycl::float3 operator/(const sycl::float3& a, const float& d)
{
    return sycl::float3(a.x() / d, a.y() / d, a.z() / d);
}

inline sycl::float3 operator+(const sycl::float3& a, const sycl::float3& b)
{
    return sycl::float3(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

inline sycl::float3 operator-(const sycl::float3& a, const sycl::float3& b)
{
    return sycl::float3(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}

inline sycl::int2 operator+(const sycl::int2& a, const sycl::int2& b)
{
    return sycl::int2(a.x() + b.x(), a.y() + b.y());
}

inline sycl::float2 operator*(const sycl::float2& a, const float& d)
{
    return sycl::float2(a.x() * d, a.y() * d);
}

inline sycl::float2 operator/(const sycl::float2& a, const float& d)
{
    return sycl::float2(a.x() / d, a.y() / d);
}

inline sycl::float2 operator+(const sycl::float2& a, const sycl::float2& b)
{
    return sycl::float2(a.x() + b.x(), a.y() + b.y());
}

inline sycl::float2 operator-(const sycl::float2& a, const sycl::float2& b)
{
    return sycl::float2(a.x() - b.x(), a.y() - b.y());
}
} // namespace dpct_operator_overloading

} // namespace depthMap
} // namespace aliceVision

