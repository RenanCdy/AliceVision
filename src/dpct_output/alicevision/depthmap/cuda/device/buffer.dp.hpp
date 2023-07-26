// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "aliceVision/depthMap/BufPtr.hpp"

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
template <typename T>
inline T* get2DBufferAt(T* ptr, size_t pitch, size_t x, size_t y)
{
    return &(BufPtr<T>(ptr,pitch).at(x,y));
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
template <typename T>
inline T* get3DBufferAt(T* ptr, size_t spitch, size_t pitch, size_t x, size_t y, size_t z)
{
    return ((T*)(((char*)ptr) + z * spitch + y * pitch)) + x;
}

template <typename T>
inline const T* get3DBufferAt(const T* ptr, size_t spitch, size_t pitch, size_t x, size_t y, size_t z)
{
    return ((const T*)(((const char*)ptr) + z * spitch + y * pitch)) + x;
}

template <typename T>
inline T* get3DBufferAt(T* ptr, size_t spitch, size_t pitch, const sycl::int3& v)
{
    return get3DBufferAt(ptr, spitch, pitch, v.x, v.y, v.z);
}

template <typename T>
inline const T* get3DBufferAt(const T* ptr, size_t spitch, size_t pitch, const sycl::int3& v)
{
    return get3DBufferAt(ptr, spitch, pitch, v.x, v.y, v.z);
}

inline float multi_fminf(float a, float b, float c)
{
  return sycl::fmin(sycl::fmin(a, b), c);
}

inline float multi_fminf(float a, float b, float c, float d)
{
  return sycl::fmin(sycl::fmin(sycl::fmin(a, b), c), d);
}


#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR

__device__ inline float4 tex2D_float4(cudaTextureObject_t rc_tex, float x, float y)
{
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_INTERPOLATION
    // cudaReadNormalizedFloat
    float4 a = tex2D<float4>(rc_tex, x, y);
    return make_float4(a.x * 255.0f, a.y * 255.0f, a.z * 255.0f, a.w * 255.0f);
#else
    // cudaReadElementType
    uchar4 a = tex2D<uchar4>(rc_tex, x, y);
    return make_float4(a.x, a.y, a.z, a.w);
#endif
}

#else

inline sycl::float4 tex2D_float4(dpct::image_accessor_ext<sycl::float4, 2> rc_tex, float x, float y)
{
    return rc_tex.read(x, y);
}

#endif

template <typename T>
/*
FIXED-DPCT1050:18: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
*/
inline T sampleTex2DLod(dpct::image_accessor_ext<T, 2> tex, float x, int width,
                        float y, int height, int level)
{
    float one_over_width = 1.0f / (float)width;
    float one_over_height = 1.0f / (float)width;
    if (level == 0)
    {
        float u = (x+0.5)*one_over_width;
        float v = (2*y + 1) / 3 * one_over_height;
        return tex.read(u, v);
    }
    else
    {
        int offset = 0;
        for (int l = level-1; l > 0; --l)
        {
            offset += (1<<l) * width * sizeof(CudaRGBA);
            offset = ((offset + 512-1)/512)*512;
        }
        offset /= sizeof(CudaRGBA);

        float u = (x+0.5+offset)*one_over_width / (1<<level);
        float v = ((y+0.5)*one_over_height / (1<<level) + 1.0f) * 2.0f / 3.0f;
        return tex.read(u, v);
    }
}

template <typename T>
/*
FIXED-DPCT1050:19: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
*/
inline T _tex2DLod(dpct::image_accessor_ext<T,2> tex, float x, int width,
                   float y, int height, float z)
{
    // TODO: Handle borders, etc...

    int highLevel = (int)floor(z);
    int lowLevel  = ( (float)highLevel == z) ? highLevel : (highLevel+1);

    if (highLevel == lowLevel) 
    {
        return sampleTex2DLod<T>(tex, x, width, y, height, highLevel);
    }
    else
    {
        T high = sampleTex2DLod<T>(tex, x, width, y, height, highLevel);
        T low  = sampleTex2DLod<T>(tex, (x+0.5)/2.0 - 0.5f, width/2, (y+0.5)/2.0f-0.5f, height/2, lowLevel);
        float t = z - (float)highLevel;
        return (1.0f - t) * high + t * low;
    }
}

} // namespace depthMap
} // namespace aliceVision

