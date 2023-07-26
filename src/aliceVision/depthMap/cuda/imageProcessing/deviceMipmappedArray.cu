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

template<int TRadius>
__global__ void createMipmappedArrayLevel_kernel(CudaRGBA* out_currentLevel_surf,
                                                 unsigned int pitch,
                                                 cudaTextureObject_t in_previousLevel_tex,
                                                 unsigned int width,
                                                 unsigned int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    const float px = 1.f / float(width);
    const float py = 1.f / float(height);

    float4 sumColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float sumFactor = 0.0f;

#pragma unroll
    for(int i = -TRadius; i <= TRadius; i++)
    {

#pragma unroll
        for(int j = -TRadius; j <= TRadius; j++)
        {
            // domain factor
            const float factor = getGauss(1, i + TRadius) * getGauss(1, j + TRadius);

            // normalized coordinates
            const float u = (x + j + 0.5f) * px;
            const float v = (y + i + 0.5f) * py;

            // current pixel color
            const float4 color = tex2D_float4(in_previousLevel_tex, u, v);

            // sum color
            sumColor = sumColor + color * factor;

            // sum factor
            sumFactor += factor;
        }
    }

    const float4 color = sumColor / sumFactor;

#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
    // convert color to unsigned char
    CudaRGBA out;
    out.x = CudaColorBaseType(color.x);
    out.y = CudaColorBaseType(color.y);
    out.z = CudaColorBaseType(color.z);
    out.w = CudaColorBaseType(color.w);

    // write output color
    // surf2Dwrite(out, out_currentLevel_surf, int(x * sizeof(CudaRGBA)), int(y));
    out_currentLevel_surf[y*pitch/sizeof(CudaRGBA) + x] = out;
#else // texture use float4 or half4
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    // convert color to half
    CudaRGBA out;
    out.x = __float2half(color.x);
    out.y = __float2half(color.y);
    out.z = __float2half(color.z);
    out.w = __float2half(color.w);

    // write output color
    // note: surf2Dwrite cannot write half directly
    // surf2Dwrite(*(reinterpret_cast<ushort4*>(&(out))), out_currentLevel_surf, int(x * sizeof(ushort4)), int(y));
    out_currentLevel_surf[y*pitch/sizeof(CudaRGBA) + x] = out;
#else // texture use float4
     // write output color
    // surf2Dwrite(color, out_currentLevel_surf, int(x * sizeof(float4)), int(y));
    out_currentLevel_surf[y*pitch/sizeof(CudaRGBA) + x] = color;
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
}

/*
__global__ void createMipmappedArrayLevel_kernel(cudaSurfaceObject_t out_currentLevel_surf,
                                                 cudaTextureObject_t in_previousLevel_tex,
                                                 unsigned int width,
                                                 unsigned int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    // corresponding texture normalized coordinates
    const float u = (x + 0.5f) / float(width);
    const float v = (y + 0.5f) / float(height);

    // corresponding color in previous level texture
    const float4 color = tex2D_float4(in_previousLevel_tex, u, v);

#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
    // convert color to unsigned char
    CudaRGBA out;
    out.x = CudaColorBaseType(color.x);
    out.y = CudaColorBaseType(color.y);
    out.z = CudaColorBaseType(color.z);
    out.w = CudaColorBaseType(color.w);

    // write output color
    surf2Dwrite(out, out_currentLevel_surf, int(x * sizeof(CudaRGBA)), int(y));
#else // texture use float4 or half4
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    // convert color to half
    CudaRGBA out;
    out.x = __float2half(color.x);
    out.y = __float2half(color.y);
    out.z = __float2half(color.z);
    out.w = __float2half(color.w);

    // write output color
    // note: surf2Dwrite cannot write half directly
    surf2Dwrite(*(reinterpret_cast<ushort4*>(&(out))), out_currentLevel_surf, int(x * sizeof(ushort4)), int(y));
#else // texture use float4
     // write output color
    surf2Dwrite(color, out_currentLevel_surf, int(x * sizeof(float4)), int(y));
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
}
*/

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

cudaError_t _cudaMallocMipmappedArray(_cudaMipmappedArray_t* out_mipmappedArrayPtr,
                                      const cudaChannelFormatDesc* desc,
                                      cudaExtent imgSize,
                                      int levels)
{
    out_mipmappedArrayPtr->allocate( {imgSize.width, imgSize.height + imgSize.height/2} );
    return cudaSuccess;
}

using _cudaArray_t = struct cudaPitchedPtr;
cudaError_t _cudaGetMipmappedArrayLevel(_cudaArray_t* array, const _cudaMipmappedArray_t& mipmappedArray, int level)
{
    constexpr int textureAlignment = 512; // TODO: Get it from cudaDeviceProps

    const unsigned char* ptr = mipmappedArray.getBytePtr();
    array->xsize = mipmappedArray.getUnitsInDim(0);
    array->ysize = (mipmappedArray.getUnitsInDim(1)*2)/3;
    if (level > 0)
    {
        ptr += (mipmappedArray.getUnitsInDim(1)/3)*2 * mipmappedArray.getPitch();
        auto lastLineByte = ptr + mipmappedArray.getPitch();
        int width = mipmappedArray.getUnitsInDim(0)/2;
        for (int l=1; l < level; ++l) {
            ptr += width * sizeof(CudaRGBA);

            if ( (reinterpret_cast<ptrdiff_t>(ptr) & (textureAlignment -1)) != 0) 
            {
                // Force pointer alignment
                ptr = reinterpret_cast<unsigned char*>( (reinterpret_cast<ptrdiff_t>(ptr) + (textureAlignment -1)) & ~ptrdiff_t(textureAlignment -1) );
                assert(ptr + width * sizeof(CudaRGBA) < lastLineByte);
            }
            width = width/2;
        }
        array->xsize /= (1 << level);
        array->ysize /= (1 << level);
    }
    array->ptr = (void*)ptr;
    array->pitch = mipmappedArray.getPitch();

    // TODO: error handling !
    return cudaSuccess;
}

cudaError_t _cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, _cudaArray_t array)
{
    assert(desc == nullptr);
    assert(flags == nullptr);

    if (extent)
    {
        extent->width = array.xsize;
        extent->height = array.ysize;
        extent->depth = 0;
    }
    return cudaSuccess;
}

__host__ void cuda_createMipmappedArrayFromImage(_cudaMipmappedArray_t* out_mipmappedArrayPtr,
                                                 const CudaDeviceMemoryPitched<CudaRGBA, 2>& in_img_dmp,
                                                 const unsigned int levels)
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

    // get mipmapped array at level 0
    _cudaArray_t level0;
    CHECK_CUDA_RETURN_ERROR(_cudaGetMipmappedArrayLevel(&level0, *out_mipmappedArrayPtr, 0));

    // copy input image buffer into mipmapped array at level 0
    CHECK_CUDA_RETURN_ERROR(cudaMemset2D(out_mipmappedArrayPtr->getBytePtr(), out_mipmappedArrayPtr->getPitch(), 0, out_mipmappedArrayPtr->getUnitsInDim(0), out_mipmappedArrayPtr->getUnitsInDim(1) ) );

    CHECK_CUDA_RETURN_ERROR(cudaMemcpy2D(
        level0.ptr, level0.pitch,
        (void *)in_img_dmp.getBytePtr(), in_img_dmp.getPitch(),
        in_img_dmp.getUnitsInDim(0) * sizeof(CudaRGBA), in_img_dmp.getUnitsInDim(1),
        cudaMemcpyDeviceToDevice
    ));

    // initialize each mipmapped array level from level 0
    size_t width  = in_imgSize.x();
    size_t height = in_imgSize.y();

    for(size_t l = 1; l < levels; ++l)
    {
        // current level width/height
        width  /= 2;
        height /= 2;

        // previous level array (or level 0)
        _cudaArray_t previousLevelArray;
        CHECK_CUDA_RETURN_ERROR(_cudaGetMipmappedArrayLevel(&previousLevelArray, *out_mipmappedArrayPtr, l - 1));

        // current level array
        _cudaArray_t currentLevelArray;
        CHECK_CUDA_RETURN_ERROR(_cudaGetMipmappedArrayLevel(&currentLevelArray, *out_mipmappedArrayPtr, l));

        // check current level array size
        cudaExtent currentLevelArraySize;
        CHECK_CUDA_RETURN_ERROR(_cudaArrayGetInfo(nullptr, &currentLevelArraySize, nullptr, currentLevelArray));

        assert(currentLevelArraySize.width  == width);
        assert(currentLevelArraySize.height == height);
        assert(currentLevelArraySize.depth  == 0);

        // generate texture object for previous level reading
        cudaTextureObject_t previousLevel_tex;
        {
            cudaResourceDesc texRes;
            memset(&texRes, 0, sizeof(cudaResourceDesc));
            texRes.resType = cudaResourceTypePitch2D;
            texRes.res.pitch2D.devPtr = previousLevelArray.ptr;
            texRes.res.pitch2D.desc   = desc;
            texRes.res.pitch2D.width  = previousLevelArray.xsize;
            texRes.res.pitch2D.height = previousLevelArray.ysize;
            texRes.res.pitch2D.pitchInBytes = previousLevelArray.pitch;

            cudaTextureDesc texDescr;
            memset(&texDescr, 0, sizeof(cudaTextureDesc));
            texDescr.normalizedCoords = 1;
            texDescr.filterMode = cudaFilterModeLinear;
            texDescr.addressMode[0] = cudaAddressModeClamp;
            texDescr.addressMode[1] = cudaAddressModeClamp;
            texDescr.addressMode[2] = cudaAddressModeClamp;
    #ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
            texDescr.readMode = cudaReadModeNormalizedFloat;
    #else
            texDescr.readMode = cudaReadModeElementType;
    #endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR


            CHECK_CUDA_RETURN_ERROR(cudaCreateTextureObject(&previousLevel_tex, &texRes, &texDescr, nullptr));
        }

        // downscale previous level image into the current level image
        {
            const dim3 block(16, 16, 1);
            const dim3 grid(divUp(width, block.x), divUp(height, block.y), 1);

            createMipmappedArrayLevel_kernel<2 /* radius */><<<grid, block>>>((CudaRGBA*)currentLevelArray.ptr, currentLevelArray.pitch, previousLevel_tex, (unsigned int)(width), (unsigned int)(height));
        }

        // wait for kernel completion
        // device has completed all preceding requested tasks
        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        // destroy temporary CUDA objects
        CHECK_CUDA_RETURN_ERROR(cudaDestroyTextureObject(previousLevel_tex));
        // CHECK_CUDA_RETURN_ERROR(cudaDestroySurfaceObject(currentLevel_surf));
    }
/*
    std::stringstream ss;
    static int s_idx = 0;
    ss << "mipmap_" << s_idx++ << ".exr";
    writeDeviceImage(*out_mipmappedArrayPtr, ss.str());
*/
}

__global__ void checkMipmapLevel(cudaTextureObject_t tex, 
                                  const CudaRGBA* buffer,
                                  unsigned int buffer_pitch,
                                  CudaRGBA* diff,
                                  unsigned int diff_pitch,
                                  unsigned int width,
                                  unsigned int height,
                                  float level)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (x >= width) || (y >= height)) {
        return;
    }

    float oneOverWidth  = 1.0f / (float)width;
    float oneOverHeight = 1.0f / (float)height;

    float4 texValue = _tex2DLod<float4>(tex, (float)x, width, y, height, level);
    CudaRGBA bufferValue = buffer[y*buffer_pitch + x];

    CudaRGBA diff_value;
    // diff_value.x = __float2half( abs( (float)bufferValue.x - texValue.x) );
    // diff_value.y = __float2half( abs( (float)bufferValue.y - texValue.y) );
    // diff_value.z = __float2half( abs( (float)bufferValue.z - texValue.z) );
    // diff_value.w = __float2half( abs( (float)bufferValue.w - texValue.w) );
    diff_value.x = __float2half( texValue.x );
    diff_value.y = __float2half( texValue.y );
    diff_value.z = __float2half( texValue.z );
    diff_value.w = __float2half( texValue.w );
    diff[y*diff_pitch + x] = diff_value;
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
    texDescr.mipmapFilterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.maxMipmapLevelClamp = float(levels - 1);
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
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff( size );

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, in_mipmappedArray.getBuffer(), in_mipmappedArray.getPitch() / sizeof(CudaRGBA), diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 0);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_0_" << s_idx++ << ".exr";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/2, (in_mipmappedArray.getSize().y()*2)/3/2);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff( size );

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, in_mipmappedArray.getBuffer(), in_mipmappedArray.getPitch() / sizeof(CudaRGBA), diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 1);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_1_" << s_idx++ << ".exr";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/2, (in_mipmappedArray.getSize().y()*2)/3/2);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff( size );

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, in_mipmappedArray.getBuffer(), in_mipmappedArray.getPitch() / sizeof(CudaRGBA), diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 1.8);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_1.8_" << s_idx++ << ".exr";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/4, (in_mipmappedArray.getSize().y()*2)/3/4);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff( size );

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, in_mipmappedArray.getBuffer(), in_mipmappedArray.getPitch() / sizeof(CudaRGBA), diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 2);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_2_" << s_idx++ << ".exr";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/8, (in_mipmappedArray.getSize().y()*2)/3/8);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff( size );

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, in_mipmappedArray.getBuffer(), in_mipmappedArray.getPitch() / sizeof(CudaRGBA), diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 3);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_3_" << s_idx++ << ".exr";
        writeDeviceImage(diff, ss.str());
    }

    {
        CudaSize<2> size(in_mipmappedArray.getSize().x()/16, (in_mipmappedArray.getSize().y()*2)/3/16);
        CudaDeviceMemoryPitched<CudaRGBA, 2> diff( size );

        std::cout << size.x() << ", " << size.y() << std::endl;

        const dim3 grid(divUp(size.x(), block.x), divUp(size.y(), block.y), 1);
        checkMipmapLevel<<<grid, block>>>(*out_mipmappedArray_texPtr, in_mipmappedArray.getBuffer(), in_mipmappedArray.getPitch() / sizeof(CudaRGBA), diff.getBuffer(), diff.getPitch() / sizeof(CudaRGBA), size.x(), size.y(), 4);

        CHECK_CUDA_RETURN_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR();

        std::stringstream ss;
        static int s_idx = 0;
        ss << "mipmap_4_" << s_idx++ << ".exr";
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

