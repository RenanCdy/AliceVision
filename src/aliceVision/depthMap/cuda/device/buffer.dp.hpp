// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sycl/sycl.hpp>

namespace aliceVision {
namespace depthMap {

    /**
    * @brief
    * @param[int] ptr
    * @param[int] pitch raw length of a line in bytes
    * @param[int] x
    * @param[int] y
    * @return
    */
    template <typename T, int Dims>
    inline T get2DBufferAt(const sycl::accessor<T, Dims>& accessor,size_t x, size_t y)
    {
        return accessor[y][x];
        // return &(BufPtr<T>(ptr,pitch).at(x,y));
    }

    inline sycl::float3 make_float3(const sycl::ushort4& us)
    {
        sycl::float3 rgb;
        rgb[0] = sycl::detail::half2Float(us[0]);
        rgb[1] = sycl::detail::half2Float(us[1]);
        rgb[2] = sycl::detail::half2Float(us[2]);
        return rgb;                 
    }

    template <typename T, int Dims>
    inline void store_half4(const sycl::float3& rgb, const sycl::accessor<T, Dims>& accessor, size_t x, size_t y)
    {
        accessor[y][x].x() = sycl::detail::float2Half(rgb[0]);
        accessor[y][x].y() = sycl::detail::float2Half(rgb[1]);
        accessor[y][x].z() = sycl::detail::float2Half(rgb[2]);
    }

} // namespace depthMap
} // namespace aliceVision

