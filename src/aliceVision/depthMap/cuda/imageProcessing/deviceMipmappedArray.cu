// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "deviceMipmappedArray.hpp"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>
#include <aliceVision/depthMap/cuda/host/memory.hpp>
#include <aliceVision/depthMap/cuda/device/buffer.cuh>
#include <aliceVision/depthMap/cuda/device/operators.cuh>
#include <aliceVision/depthMap/cuda/imageProcessing/deviceGaussianFilter.hpp>
#include <aliceVision/depthMap/cuda/host/DeviceMipmapImage.hpp>

#include <cuda_runtime.h>

namespace aliceVision {
namespace depthMap {

__host__ void writeDeviceImage(const CudaDeviceMemoryPitched<CudaRGBA, 2>& in_img_dmp, const std::string& path);



__global__ void createMipmappedArrayDebugFlatImage_kernel(CudaRGBA* out_flatImage_d, int out_flatImage_p,
                                                          cudaTextureObject_t in_mipmappedArray_tex,
                                                          unsigned int levels,
                                                          unsigned int firstLevelWidth,
                                                          unsigned int firstLevelHeight)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(y >= firstLevelHeight)
        return;

    // set default color value
    float4 color = make_float4(0.f, 0.f, 0.f, 0.f);

    if(x < firstLevelWidth)
    {
        // level 0
        // corresponding texture normalized coordinates
        // const float u = (x + 0.5f) / float(firstLevelWidth);
        // const float v = (y + 0.5f) / float(firstLevelHeight);

        // set color value from mipmappedArray texture
        color = _tex2DLod<float4>(in_mipmappedArray_tex, x, firstLevelWidth, y, firstLevelHeight, 0.f);
    }
    else
    {
        // level from global y coordinate
        const unsigned int level = int(log2(1.0 / (1.0 - (y / double(firstLevelHeight))))) + 1;
        const unsigned int levelDownscale = pow(2, level);
        const unsigned int levelWidth  = firstLevelWidth  / levelDownscale;
        const unsigned int levelHeight = firstLevelHeight / levelDownscale;

        // corresponding level coordinates
        const float lx = x - firstLevelWidth;
        const float ly = y % levelHeight;

        // corresponding texture normalized coordinates
        const float u = (lx + 0.5f) / float(levelWidth);
        const float v = (ly + 0.5f) / float(levelHeight);

        if(u <= 1.f && v <= 1.f && level < levels)
        {
            // set color value from mipmappedArray texture
            color = _tex2DLod<float4>(in_mipmappedArray_tex, lx, levelWidth, ly, levelHeight, float(level));
        }
    }

    // write output color
    CudaRGBA* out_colorPtr = get2DBufferAt(out_flatImage_d, out_flatImage_p, x, y);

#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
    // convert color from (0, 1) to (0, 255)
    color.x *= 255.f;
    color.y *= 255.f;
    color.z *= 255.f;
    color.w *= 255.f;

    out_colorPtr->x = CudaColorBaseType(color.x);
    out_colorPtr->y = CudaColorBaseType(color.y);
    out_colorPtr->z = CudaColorBaseType(color.z);
    out_colorPtr->w = CudaColorBaseType(color.w);
#else // texture use float4 or half4
    // convert color from (0, 255) to (0, 1)
    color.x /= 255.f;
    color.y /= 255.f;
    color.z /= 255.f;
    color.w /= 255.f;
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    out_colorPtr->x = __float2half(color.x);
    out_colorPtr->y = __float2half(color.y);
    out_colorPtr->z = __float2half(color.z);
    out_colorPtr->w = __float2half(color.w);
#else // texture use float4
    *out_colorPtr = color;
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
}

__global__ void checkMipmapLevel(cudaTextureObject_t tex, 
                                  CudaRGBA* buffer,
                                  unsigned int buffer_pitch,
                                  unsigned int width,
                                  unsigned int height,
                                  float level)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (x >= width) || (y >= height)) {
        return;
    }

    float4 texValue = _tex2DLod<float4>(tex, (float)x-10, width, (float)y-10, height, level);

    CudaRGBA value;
    value.x = __float2half( texValue.x );
    value.y = __float2half( texValue.y );
    value.z = __float2half( texValue.z );
    value.w = __float2half( texValue.w );
    buffer[y*buffer_pitch + x] = value;
}

__host__ void cuda_createMipmappedArrayTexture(cudaTextureObject_t* out_mipmappedArray_texPtr,
                                               const _cudaMipmappedArray_t& in_mipmappedArray,
                                               const unsigned int levels)
{
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    const cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
#else
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<CudaRGBA>();
#endif

    cudaResourceDesc resDescr;
    memset(&resDescr, 0, sizeof(cudaResourceDesc));
    resDescr.resType = cudaResourceTypePitch2D;
    resDescr.res.pitch2D.devPtr = (void*)in_mipmappedArray.getBuffer();
    resDescr.res.pitch2D.desc   = desc;
    resDescr.res.pitch2D.width  = in_mipmappedArray.getSize().x();
    resDescr.res.pitch2D.height = in_mipmappedArray.getSize().y();
    resDescr.res.pitch2D.pitchInBytes = in_mipmappedArray.getPaddedBytesInRow();

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = 1; // should always be set to 1 for mipmapped array
    texDescr.filterMode = cudaFilterModeLinear;
    // texDescr.mipmapFilterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    // texDescr.maxMipmapLevelClamp = float(levels - 1);
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
    texDescr.readMode = cudaReadModeNormalizedFloat;
#else
    texDescr.readMode = cudaReadModeElementType;
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR

    CHECK_CUDA_RETURN_ERROR(cudaCreateTextureObject(out_mipmappedArray_texPtr, &resDescr, &texDescr, nullptr));
/*
    // Code to check texture sampling
    const dim3 block(16, 16, 1);
    {
        CudaSize<2> size(in_mipmappedArray.getSize().x(), (in_mipmappedArray.getSize().y()*2)/3);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff(size);

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 0);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_0_" << s_idx++ << ".png";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/2, (in_mipmappedArray.getSize().y()*2)/3/2);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff(size);

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 1);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_1_" << s_idx++ << ".png";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/2, (in_mipmappedArray.getSize().y()*2)/3/2);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff(size);

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 1.8);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_1.8_" << s_idx++ << ".png";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/4, (in_mipmappedArray.getSize().y()*2)/3/4);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff(size);

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 2);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_2_" << s_idx++ << ".png";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/8, (in_mipmappedArray.getSize().y()*2)/3/8);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff(size);

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 3);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_3_" << s_idx++ << ".png";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/16, (in_mipmappedArray.getSize().y()*2)/3/16);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff(size) );
        
        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 4);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_4_" << s_idx++ << ".png";
        writeDeviceImage(diff, ss.str());
    }
*/
}

__host__ void cuda_createMipmappedArrayDebugFlatImage(CudaDeviceMemoryPitched<CudaRGBA, 2>& out_flatImage_dmp,
                                                      const cudaTextureObject_t in_mipmappedArray_tex,
                                                      const unsigned int levels,
                                                      const int firstLevelWidth,
                                                      const int firstLevelHeight,
                                                      cudaStream_t stream)
{
    const CudaSize<2>& out_flatImageSize = out_flatImage_dmp.getSize();

    assert(out_flatImageSize.x() == size_t(firstLevelWidth * 1.5f));
    assert(out_flatImageSize.y() == size_t(firstLevelHeight));

    const dim3 block(16, 16, 1);
    const dim3 grid(divUp(out_flatImageSize.x(), block.x), divUp(out_flatImageSize.y(), block.y), 1);

    createMipmappedArrayDebugFlatImage_kernel<<<grid, block, 0, stream>>>(
        out_flatImage_dmp.getBuffer(),
        out_flatImage_dmp.getBytesPaddedUpToDim(0),
        in_mipmappedArray_tex,
        levels,
        (unsigned int)(firstLevelWidth),
        (unsigned int)(firstLevelHeight));

    CHECK_CUDA_ERROR();
}

} // namespace depthMap
} // namespace aliceVision

