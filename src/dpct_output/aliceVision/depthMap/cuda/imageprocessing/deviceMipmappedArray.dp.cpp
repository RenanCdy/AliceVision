// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "deviceMipmappedArray.hpp"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>
#include <aliceVision/depthMap/cuda/host/memory.hpp>
#include "aliceVision/depthMap/cuda/device/buffer.dp.hpp"
#include "aliceVision/depthMap/cuda/device/operators.dp.hpp"
#include <aliceVision/depthMap/cuda/imageProcessing/deviceGaussianFilter.hpp>
#include <aliceVision/depthMap/cuda/host/DeviceMipmapImage.hpp>

namespace aliceVision {
namespace depthMap {

void writeDeviceImage(const CudaDeviceMemoryPitched<CudaRGBA, 2>& in_img_dmp, const std::string& path);

template <int TRadius>
void createMipmappedArrayLevel_kernel(
    CudaRGBA* out_currentLevel_surf, unsigned int pitch,
    /*
    FIXED-DPCT1050:46: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
    */
    dpct::image_accessor_ext<sycl::float4, 2> in_previousLevel_tex, unsigned int width,
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

void createMipmappedArrayDebugFlatImage_kernel(
    CudaRGBA* out_flatImage_d, int out_flatImage_p,
    /*
    FIXED-DPCT1050:47: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
    */
    dpct::image_accessor_ext<sycl::float4, 2> in_mipmappedArray_tex, unsigned int levels,
    unsigned int firstLevelWidth, unsigned int firstLevelHeight, const sycl::nd_item<3>& item_ct1)
{
    const unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if(y >= firstLevelHeight)
        return;

    // set default color value
    sycl::float4 color = sycl::float4(0.f, 0.f, 0.f, 0.f);

    if(x < firstLevelWidth)
    {
        // level 0
        // corresponding texture normalized coordinates
        // const float u = (x + 0.5f) / float(firstLevelWidth);
        // const float v = (y + 0.5f) / float(firstLevelHeight);

        // set color value from mipmappedArray texture
        color = _tex2DLod<sycl::float4>(in_mipmappedArray_tex, x, firstLevelWidth, y, firstLevelHeight, 0.f);
    }
    else
    {
        // level from global y coordinate
        const unsigned int level = int(sycl::log2(1.0 / (1.0 - (y / double(firstLevelHeight))))) + 1;
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
            color = _tex2DLod<sycl::float4>(in_mipmappedArray_tex, lx, levelWidth, ly, levelHeight, float(level));
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
    color.x() /= 255.f;
    color.y() /= 255.f;
    color.z() /= 255.f;
    color.w() /= 255.f;
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    out_colorPtr->x = sycl::vec<float, 1>{color.x()}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    out_colorPtr->y = sycl::vec<float, 1>{color.y()}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    out_colorPtr->z = sycl::vec<float, 1>{color.z()}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    out_colorPtr->w = sycl::vec<float, 1>{color.w()}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
#else // texture use float4
    *out_colorPtr = color;
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
}

dpct::err0 _cudaMallocMipmappedArray(_cudaMipmappedArray_t* out_mipmappedArrayPtr, const dpct::image_channel* desc,
                                     sycl::range<3> imgSize, int levels)
{
    out_mipmappedArrayPtr->allocate({imgSize[0], imgSize[1] + imgSize[1] / 2});
    return 0;
}

using _cudaArray_t = dpct::pitched_data;
dpct::err0 _cudaGetMipmappedArrayLevel(_cudaArray_t* array, const _cudaMipmappedArray_t& mipmappedArray, int level)
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
    return 0;
}

dpct::err0 _cudaArrayGetInfo(dpct::image_channel* desc, sycl::range<3>* extent, unsigned int* flags, _cudaArray_t array)
{
    assert(desc == nullptr);
    assert(flags == nullptr);

    if (extent)
    {
        extent->width = array.xsize;
        extent->height = array.ysize;
        extent->depth = 0;
    }
    return 0;
}

void cuda_createMipmappedArrayFromImage(_cudaMipmappedArray_t* out_mipmappedArrayPtr,
                                                 const CudaDeviceMemoryPitched<CudaRGBA, 2>& in_img_dmp,
                                                 const unsigned int levels)
 try {
    const CudaSize<2>& in_imgSize = in_img_dmp.getSize();
    const sycl::range<3> imgSize = sycl::range<3>(in_imgSize.x(), in_imgSize.y(), 0);

#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    const dpct::image_channel desc = cudaCreateChannelDescHalf4();
#else
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<CudaRGBA>();
#endif

    // allocate CUDA mipmapped array
    CHECK_CUDA_RETURN_ERROR(_cudaMallocMipmappedArray(out_mipmappedArrayPtr, &desc, imgSize, levels));

    // get mipmapped array at level 0
    _cudaArray_t level0;
    CHECK_CUDA_RETURN_ERROR(_cudaGetMipmappedArrayLevel(&level0, *out_mipmappedArrayPtr, 0));

    // copy input image buffer into mipmapped array at level 0
    CHECK_CUDA_RETURN_ERROR(DPCT_CHECK_ERROR(
        dpct::dpct_memset(out_mipmappedArrayPtr->getBytePtr(), out_mipmappedArrayPtr->getPitch(), 0,
                          out_mipmappedArrayPtr->getUnitsInDim(0), out_mipmappedArrayPtr->getUnitsInDim(1))));

    CHECK_CUDA_RETURN_ERROR(DPCT_CHECK_ERROR(dpct::dpct_memcpy(
        level0.ptr, level0.pitch, (void*)in_img_dmp.getBytePtr(), in_img_dmp.getPitch(),
        in_img_dmp.getUnitsInDim(0) * sizeof(CudaRGBA), in_img_dmp.getUnitsInDim(1), dpct::device_to_device)));

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
        sycl::range<3> currentLevelArraySize{0, 0, 0};
        CHECK_CUDA_RETURN_ERROR(_cudaArrayGetInfo(nullptr, &currentLevelArraySize, nullptr, currentLevelArray));

        assert(currentLevelArraySize[0] == width);
        assert(currentLevelArraySize[1] == height);
        assert(currentLevelArraySize[2] == 0);

        // generate texture object for previous level reading
        dpct::image_wrapper_base_p previousLevel_tex;
        {
            dpct::image_data texRes;
            memset(&texRes, 0, sizeof(dpct::image_data));

            texRes.set_data(previousLevelArray.ptr, previousLevelArray.xsize, previousLevelArray.ysize,
                            previousLevelArray.pitch, desc);

            dpct::sampling_info texDescr;
            memset(&texDescr, 0, sizeof(dpct::sampling_info));
            texDescr.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::linear,
                         sycl::coordinate_normalization_mode::normalized);
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
            texDescr.readMode = cudaReadModeNormalizedFloat;
    #else
            /*
            IGNORED-DPCT1007:7: Migration of cudaTextureDesc::readMode is not supported.
            */
            // texDescr.readMode = cudaReadModeElementType;
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR

            CHECK_CUDA_RETURN_ERROR(DPCT_CHECK_ERROR(previousLevel_tex = dpct::create_image_wrapper(texRes, texDescr)));
        }

        // downscale previous level image into the current level image
        {
            const sycl::range<3> block(1, 16, 16);
            const sycl::range<3> grid(1, divUp(height, block[1]), divUp(width, block[2]));

            createMipmappedArrayLevel_kernel<2 /* radius */><<<grid, block>>>((CudaRGBA*)currentLevelArray.ptr, currentLevelArray.pitch, previousLevel_tex, (unsigned int)(width), (unsigned int)(height));
        }

        // wait for kernel completion
        // device has completed all preceding requested tasks
        dpct::get_current_device().queues_wait_and_throw();

        // destroy temporary CUDA objects
        delete previousLevel_tex;
    }
/*
    std::stringstream ss;
    static int s_idx = 0;
    ss << "mipmap_" << s_idx++ << ".exr";
    writeDeviceImage(*out_mipmappedArrayPtr, ss.str());
*/
}
catch(sycl::exception const& exc) {
  RETHROW_SYCL_EXCEPTION(ecx);
}

/*
FIXED-DPCT1050:48: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
*/
void checkMipmapLevel(dpct::image_accessor_ext<sycl::float4, 2> tex,
                      const CudaRGBA* buffer, unsigned int buffer_pitch, CudaRGBA* diff, unsigned int diff_pitch,
                      unsigned int width, unsigned int height, float level, const sycl::nd_item<3>& item_ct1)
{
    const unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if ( (x >= width) || (y >= height)) {
        return;
    }

    float oneOverWidth  = 1.0f / (float)width;
    float oneOverHeight = 1.0f / (float)height;

    sycl::float4 texValue = _tex2DLod<sycl::float4>(tex, (float)x, width, y, height, level);
    CudaRGBA bufferValue = buffer[y*buffer_pitch + x];

    CudaRGBA diff_value;
    // diff_value.x = __float2half( abs( (float)bufferValue.x - texValue.x) );
    // diff_value.y = __float2half( abs( (float)bufferValue.y - texValue.y) );
    // diff_value.z = __float2half( abs( (float)bufferValue.z - texValue.z) );
    // diff_value.w = __float2half( abs( (float)bufferValue.w - texValue.w) );
    diff_value.x = sycl::vec<float, 1>{texValue.x()}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    diff_value.y = sycl::vec<float, 1>{texValue.y()}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    diff_value.z = sycl::vec<float, 1>{texValue.z()}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    diff_value.w = sycl::vec<float, 1>{texValue.w()}.convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    diff[y*diff_pitch + x] = diff_value;
}

void cuda_createMipmappedArrayTexture(dpct::image_wrapper_base_p* out_mipmappedArray_texPtr,
                                      const _cudaMipmappedArray_t& in_mipmappedArray, const unsigned int levels)
 try {
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    const dpct::image_channel desc = cudaCreateChannelDescHalf4();
#else
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<CudaRGBA>();
#endif

    dpct::image_data resDescr;
    memset(&resDescr, 0, sizeof(dpct::image_data));

    resDescr.set_data((void*)in_mipmappedArray.getBuffer(), in_mipmappedArray.getSize().x(),
                      in_mipmappedArray.getSize().y(), in_mipmappedArray.getPaddedBytesInRow(), desc);

    dpct::sampling_info texDescr;
    memset(&texDescr, 0, sizeof(dpct::sampling_info));
    texDescr.set(sycl::coordinate_normalization_mode::normalized); // should always be set to 1 for mipmapped array
    texDescr.set(sycl::filtering_mode::linear);
    /*
    FIXED-DPCT1007:8: Migration of cudaTextureDesc::mipmapFilterMode is not supported.
    */
    //texDescr.mipmapFilterMode = sycl::filtering_mode::linear;
    texDescr.set(sycl::addressing_mode::clamp_to_edge);
    /*
    FIXED-DPCT1007:9: Migration of cudaTextureDesc::maxMipmapLevelClamp is not supported.
    */
    // texDescr.maxMipmapLevelClamp = float(levels - 1);
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
    texDescr.readMode = cudaReadModeNormalizedFloat;
#else
    /*
    FIXED-DPCT1007:10: Migration of cudaTextureDesc::readMode is not supported.
    */
    // texDescr.readMode = cudaReadModeElementType;
#endif // ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR

    *out_mipmappedArray_texPtr = dpct::create_image_wrapper(resDescr, texDescr);
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
catch(sycl::exception const& exc) {
  RETHROW_SYCL_EXCEPTION(exc);
}

void cuda_createMipmappedArrayDebugFlatImage(CudaDeviceMemoryPitched<CudaRGBA, 2>& out_flatImage_dmp,
                                             const dpct::image_wrapper_base_p in_mipmappedArray_tex,
                                             const unsigned int levels, const int firstLevelWidth,
                                             const int firstLevelHeight, dpct::queue_ptr stream)
try {
    const CudaSize<2>& out_flatImageSize = out_flatImage_dmp.getSize();

    assert(out_flatImageSize.x() == size_t(firstLevelWidth * 1.5f));
    assert(out_flatImageSize.y() == size_t(firstLevelHeight));

    const sycl::range<3> block(1, 16, 16);
    const sycl::range<3> grid(1, divUp(out_flatImageSize.y(), block[1]), divUp(out_flatImageSize.x(), block[2]));

    /*
    FIXED-DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    /*
    FIXED-DPCT1050:40: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
    */
    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
        stream->submit(
            [&](sycl::handler& cgh)
            {
                auto in_mipmappedArray_tex_acc =
                    static_cast<dpct::image_wrapper<sycl::float4, 2>*>(
                        in_mipmappedArray_tex)
                        ->get_access(cgh, *stream);

                auto in_mipmappedArray_tex_smpl = in_mipmappedArray_tex->get_sampler();

                auto out_flatImage_dmp_getBuffer_ct0 = out_flatImage_dmp.getBuffer();
                auto out_flatImage_dmp_getBytesPaddedUpToDim_ct1 = out_flatImage_dmp.getBytesPaddedUpToDim(0);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     createMipmappedArrayDebugFlatImage_kernel(
                                         out_flatImage_dmp_getBuffer_ct0, out_flatImage_dmp_getBytesPaddedUpToDim_ct1,
                                         dpct::image_accessor_ext<sycl::float4, 2>(
                                             in_mipmappedArray_tex_smpl, in_mipmappedArray_tex_acc),
                                         levels, (unsigned int)(firstLevelWidth), (unsigned int)(firstLevelHeight),
                                         item_ct1);
                                 });
            });
    }

} catch(sycl::exception const& e)
{
    RETHROW_SYCL_EXCEPTION(e);
}

} // namespace depthMap
} // namespace aliceVision

