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
    template <typename Accessor>
    inline typename Accessor::value_type& get2DBufferAt(Accessor& accessor,size_t x, size_t y)
    {
        return accessor[y][x];
        // return &(BufPtr<T>(ptr,pitch).at(x,y));
    }

    /**
    * @brief
    * @param[int] ptr
    * @param[int] spitch raw length of a 2D array in bytes
    * @param[int] pitch raw length of a line in bytes
    * @param[int] x
    * @param[int] y
    * @return
    */
    template <typename Accessor>
    inline typename Accessor::value_type& get3DBufferAt(Accessor& accessor, size_t x, size_t y, size_t z)
    {
        return accessor[z][y][x];
        // return ((T*)(((char*)ptr) + z * spitch + y * pitch)) + x;
    }

    inline sycl::float3 make_float3(const sycl::ushort4& us)
    {
        sycl::float3 value;
        value[0] = sycl::detail::half2Float(us[0]);
        value[1] = sycl::detail::half2Float(us[1]);
        value[2] = sycl::detail::half2Float(us[2]);
        return value;                 
    }

    inline sycl::float4 make_float4(const sycl::ushort4& us)
    {
        sycl::float4 value;
        value[0] = sycl::detail::half2Float(us[0]);
        value[1] = sycl::detail::half2Float(us[1]);
        value[2] = sycl::detail::half2Float(us[2]);
        value[3] = sycl::detail::half2Float(us[3]);
        return value;                 
    }

    template <typename T, int Dims, sycl::access::mode Mode>
    inline void store_half4(const sycl::float3& rgb, const sycl::accessor<T, Dims, Mode>& accessor, size_t x, size_t y)
    {
        accessor[y][x].x() = sycl::detail::float2Half(rgb[0]);
        accessor[y][x].y() = sycl::detail::float2Half(rgb[1]);
        accessor[y][x].z() = sycl::detail::float2Half(rgb[2]);
    }

    template <typename T, int Dims, sycl::access::mode Mode>
    inline void store_half4(const sycl::float4& rgba, const sycl::accessor<T, Dims, Mode>& accessor, size_t x, size_t y)
    {
        accessor[y][x].x() = sycl::detail::float2Half(rgba[0]);
        accessor[y][x].y() = sycl::detail::float2Half(rgba[1]);
        accessor[y][x].z() = sycl::detail::float2Half(rgba[2]);
        accessor[y][x].w() = sycl::detail::float2Half(rgba[3]);
    }

    template <typename Accessor>
    inline typename Accessor::value_type sampleTex2DLod(Accessor tex, const sycl::sampler& sampler, float x, int width, float y, int height,
                            int level)
    {
        if (level == 0)
        {
            float one_over_width = 1.0f / (float)width;
            float one_over_height = 1.0f / (float)width;
            float u = (x + 0.5f) * one_over_width;
            float v = (y + 0.5f) * one_over_height;
            u = (u>=0.f)?( (u<=1.f)?u:1.f ):0.f;
            v = (v>=0.f)?( (v<=1.f)?v:1.f ):0.f;
            return tex.read( sycl::float2(u, v * 2.0f / 3.0f), sampler);
        }
        else
        {
            float one_over_levelScale = 1.0f / (float)(1<<level);
            float one_over_width = one_over_levelScale / (float)width;
            float one_over_height = one_over_levelScale / (float)height;

            int offset = 0;
            for (int l = level-1; l > 0; --l)
            {
                offset += (1<<l) * width * sizeof(CudaRGBA);
                // offset = ((offset + 512-1)/512)*512;
            }
            offset /= sizeof(CudaRGBA);

            float u = (x + 0.5f) * one_over_width;
            float v = (y + 0.5f) * one_over_height;
            
            u = (u>=0.f)?( (u<=one_over_levelScale)?u:(one_over_levelScale-0.5f*one_over_width) ):0.f;
            v = (v>=0.f)?( (v<=one_over_levelScale)?v:(one_over_levelScale-0.5f*one_over_height) ):0.f;
            return tex.read( sycl::float2(u + offset * one_over_width, (v + 1.0f) * 2.0f / 3.0f), sampler);
        }
    }

    template <typename Accessor>
    inline typename Accessor::value_type _tex2DLod(Accessor tex, const sycl::sampler& sampler, float x, int width, float y, int height,
                    float z)
    {
        int highLevel = (int)sycl::floor(z);
        int lowLevel  = ( (float)highLevel == z) ? highLevel : (highLevel+1);

        if (highLevel == lowLevel) 
        {
            return sampleTex2DLod(tex, sampler, x, width, y, height, highLevel);
        }
        else
        {
            auto high = sampleTex2DLod(tex, sampler, x, width, y, height, highLevel);
            auto low  = sampleTex2DLod(tex, sampler, (x+0.5)/2.0f - 0.5f, width/2, (y+0.5)/2.0f-0.5f, height/2, lowLevel);
            // TODO: This version might be better:
            // auto low  = sampleTex2DLod<T>(tex, x/2.0f, width/2, y/2.0f, height/2, lowLevel);
            float t = z - (float)highLevel;
            return (1.0f - t) * high + t * low;
        }
    }

} // namespace depthMap
} // namespace aliceVision

