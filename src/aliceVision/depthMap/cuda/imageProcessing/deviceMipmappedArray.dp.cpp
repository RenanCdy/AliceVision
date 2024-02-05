// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "deviceMipmappedArray.hpp"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>
#include <aliceVision/depthMap/cuda/host/memory.hpp>
#include <aliceVision/depthMap/cuda/host/MemoryLocker.hpp>
#include <aliceVision/depthMap/cuda/device/buffer.dp.hpp>
#include <aliceVision/depthMap/cuda/imageProcessing/deviceGaussianFilter.dp.hpp>
#include <aliceVision/depthMap/cuda/host/DeviceMipmapImage.hpp>

#include <sycl/sycl.hpp>

namespace aliceVision {
namespace depthMap {

template <int TRadius>
void createMipmappedArrayLevel_kernel(
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
    sycl::accessor<sycl::uchar4, 2, sycl::access::mode::write, sycl::target::device> out, sycl::id<2> currentPosition,
    sycl::accessor<sycl::uchar4, 2, sycl::access::mode::read, sycl::target::device> in, sycl::id<2> previousPosition,
#else
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    sycl::accessor<sycl::ushort4, 2, sycl::access::mode::write, sycl::target::device> out, sycl::id<2> currentPosition,
    sycl::accessor<sycl::ushort4, 2, sycl::access::mode::read, sycl::target::device> in, sycl::id<2> previousPosition,
#else
    sycl::accessor<sycl::float4, 2, sycl::access::mode::write, sycl::target::device> out, sycl::id<2> currentPosition,
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::target::device> in, sycl::id<2> previousPosition,
#endif
#endif
    unsigned int width,
    unsigned int height, 
    const sycl::nd_item<2>& item_ct1, 
    int* d_gaussianArrayOffset, 
    float* d_gaussianArray)
{
    const unsigned int x = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const unsigned int y = item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0);

    if(x >= width || y >= height)
        return;

    const float px = 1.f / float(width);
    const float py = 1.f / float(height);

    sycl::float4 sumColor = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
    float sumFactor = 0.0f;

#pragma unroll
    for(int i = -TRadius; i <= TRadius; i++)
    {

#pragma unroll
        for(int j = -TRadius; j <= TRadius; j++)
        {
            // domain factor
            const float factor = __sycl::getGauss(1, i + TRadius, d_gaussianArrayOffset, d_gaussianArray) *
                                 __sycl::getGauss(1, j + TRadius, d_gaussianArrayOffset, d_gaussianArray);

            // current pixel color
            auto cy = 2*y + i;
            auto cx = 2*x + j;
            cy = (cy >= 0) ? ( (cy >= 2*height) ? (2*height-1):cy) : 0;
            cx = (cx >= 0) ? ( (cx >= 2*width) ? (2*width-1):cx) : 0;
            sycl::float4 color = make_float4( in[previousPosition[1] + cy][previousPosition[0] + cx]);

            // sum color
            sumColor = sumColor + color * factor;

            // sum factor
            sumFactor += factor;
        }
    }

    const sycl::float4 color = sumColor / sumFactor;

#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
    // TODO: Untested
    // convert color to unsigned char
    // write output color via accessor
    {
        size_t xx = x + currentPosition[0];
        size_t yy = y + currentPosition[1];
        out[yy][xx].x() = CudaColorBaseType(color[0]);
        out[yy][xx].y() = CudaColorBaseType(color[1]);
        out[yy][xx].z() = CudaColorBaseType(color[2]);
        out[yy][xx].w() = CudaColorBaseType(color[3]);
    }
#else // texture use float4 or half4
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    store_half4(color, out, x + currentPosition[0], y + currentPosition[1]);
#else // texture use float4
    // TODO: Untested
    // convert color to unsigned char
    // write output color via accessor
    {
        size_t xx = x + currentPosition[0];
        size_t yy = y + currentPosition[1];
        out[yy][xx].x() = color[0];
        out[yy][xx].y() = color[1];
        out[yy][xx].z() = color[2];
        out[yy][xx].w() = color[3];
    }
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
}

cudaError_t _cudaMallocMipmappedArray(_cudaMipmappedArray_t* out_mipmappedArrayPtr,
                                      const cudaChannelFormatDesc* desc,
                                      cudaExtent imgSize,
                                      int levels)
{
    out_mipmappedArrayPtr->allocate( {imgSize.width, imgSize.height + imgSize.height/2} );
    return cudaSuccess;
}

using _cudaArray_t = struct cudaPitchedPtr;
std::pair<sycl::buffer<sycl::ushort4,2>, sycl::id<2>> _cudaGetMipmappedArrayLevel(sycl::buffer<sycl::ushort4, 2>& mipmappedArray, int level)
{
    sycl::id<2> pos(0,0);
    if (level > 0)
    {
        pos[1] += (mipmappedArray.get_range()[0]/3)*2;
        int width = mipmappedArray.get_range()[1]/2;
        for (int l=1; l < level; ++l) {
            pos[0] += width;
            width /= 2;
        }
    }

    return {mipmappedArray, pos};
}

std::pair<sycl::buffer<sycl::uchar4,2>, sycl::id<2>> _cudaGetMipmappedArrayLevel(sycl::buffer<sycl::uchar4, 2>& mipmappedArray, int level)
{
    sycl::id<2> pos(0,0);
    if (level > 0)
    {
        pos[1] += (mipmappedArray.get_range()[0]/3)*2;
        int width = mipmappedArray.get_range()[1]/2;
        for (int l=1; l < level; ++l) {
            pos[0] += width;
            width /= 2;
        }
    }

    return {mipmappedArray, pos};
}

std::pair<sycl::buffer<sycl::float4,2>, sycl::id<2>> _cudaGetMipmappedArrayLevel(sycl::buffer<sycl::float4, 2>& mipmappedArray, int level)
{
    sycl::id<2> pos(0,0);
    if (level > 0)
    {
        pos[1] += (mipmappedArray.get_range()[0]/3)*2;
        int width = mipmappedArray.get_range()[1]/2;
        for (int l=1; l < level; ++l) {
            pos[0] += width;
            width /= 2;
        }
    }

    return {mipmappedArray, pos};
}

__host__ void cuda_createMipmappedArrayFromImage(_cudaMipmappedArray_t* out_mipmappedArrayPtr,
                                                 const CudaDeviceMemoryPitched<CudaRGBA, 2>& in_img_dmp,
                                                 const unsigned int levels,
                                                 sycl::queue& stream)
{
    const CudaSize<2>& in_imgSize = in_img_dmp.getSize();
    const cudaExtent imgSize = make_cudaExtent(in_imgSize.x(), in_imgSize.y(), 0);

#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    const cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
#else
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<CudaRGBA>();
#endif

    // allocate CUDA mipmapped array
    CHECK_CUDA_RETURN_ERROR(_cudaMallocMipmappedArray(out_mipmappedArrayPtr, &desc, imgSize, levels));

    BufferLocker in_locker( const_cast<CudaDeviceMemoryPitched<CudaRGBA, 2>&>(in_img_dmp));
    BufferLocker out_locker(*out_mipmappedArrayPtr);

    auto memsetEvent = stream.submit( [&] (sycl::handler& cgh)
    {
        auto dst = out_locker.buffer().get_access<sycl::access::mode::write>(cgh);
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
        cgh.fill(dst, sycl::uchar4(0,0,0,0));
#else
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
        cgh.fill(dst, sycl::ushort4(0,0,0,0));
#else
        cgh.fill(dst, sycl::float4(0,0,0,0));
#endif
#endif
    });

    // get mipmapped array at level 0
    auto l0 = _cudaGetMipmappedArrayLevel(out_locker.buffer(), 0);
    auto level0 = l0.first;
    auto pos = l0.second;
    size_t width  = in_imgSize.x();
    size_t height = in_imgSize.y();

    // copy input image buffer into mipmapped array at level 0
    // todo: potentially do a memcpy
    auto levelEvent = stream.submit( [&] (sycl::handler& cgh)
    {
        auto src = in_locker.buffer().get_access<sycl::access::mode::read>(cgh);
        auto dst = level0.get_access<sycl::access::mode::write>(cgh);
        cgh.depends_on(memsetEvent);

        sycl::range<2> globalSize(width, height);
        cgh.parallel_for(globalSize, [=](sycl::id<2> id)
        {
            dst[id[1]][id[0]] = src[id[1]][id[0]];
        });

    });
    
    for(size_t l = 1; l < levels; ++l)
    {
        // current level width/height
        width  /= 2;
        height /= 2;

        // previous level array (or level 0)
        auto prev = _cudaGetMipmappedArrayLevel(out_locker.buffer(), l - 1);
        auto previousLevelArray = prev.first;
        auto previousPosition = prev.second;

        // current level array
        auto curr = _cudaGetMipmappedArrayLevel(out_locker.buffer(), l);
        auto currentLevelArray = curr.first;
        auto currentPosition = curr.second;
        // downscale previous level image into the current level image
        {
            const sycl::range<2> block(16, 16);
            const sycl::range<2> grid(divUp(height, block[0]), divUp(width, block[1]));

            int* d_gaussianArrayOffset = __sycl::d_gaussianArrayOffset; 
            float* d_gaussianArray = __sycl::d_gaussianArray;

            levelEvent = stream.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.depends_on(levelEvent);
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
                    sycl::accessor<sycl::uchar4, 2, sycl::access::mode::read> previousLevel_acc(previousLevelArray, cgh);
                    sycl::accessor<sycl::uchar4, 2, sycl::access::mode::write> currentLevel_acc(currentLevelArray, cgh);

#else
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
                    sycl::accessor<sycl::ushort4, 2, sycl::access::mode::read> previousLevel_acc(previousLevelArray, cgh);
                    sycl::accessor<sycl::ushort4, 2, sycl::access::mode::write> currentLevel_acc(currentLevelArray, cgh);
#else
                    sycl::accessor<sycl::float4, 2, sycl::access::mode::read> previousLevel_acc(previousLevelArray, cgh);
                    sycl::accessor<sycl::float4, 2, sycl::access::mode::write> currentLevel_acc(currentLevelArray, cgh);
#endif
#endif
                    
                    cgh.parallel_for(sycl::nd_range<2>(grid * block, block),
                                    [=](sycl::nd_item<2> item_ct1)
                                    {
                                        createMipmappedArrayLevel_kernel<2 /* radius */>(
                                            currentLevel_acc, currentPosition,
                                            previousLevel_acc, previousPosition,
                                            (unsigned int)(width), 
                                            (unsigned int)(height),
                                            item_ct1, 
                                            d_gaussianArrayOffset, 
                                            d_gaussianArray
                                            );
                                    }
                    );
                });
        }
    }
    levelEvent.wait();
}

} // namespace depthMap
} // namespace aliceVision
