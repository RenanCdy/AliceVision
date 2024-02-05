// This file is part of the AliceVision project.
// Copyright (c) 2018 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "deviceGaussianFilter.dp.hpp"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>
#include <aliceVision/depthMap/cuda/device/buffer.dp.hpp>
#include <cmath>

#include <aliceVision/depthMap/cuda/host/MemoryLocker.hpp>

namespace aliceVision {
namespace depthMap {
namespace __sycl {

/*********************************************************************************
 * global / constant data structures
 *********************************************************************************/
std::set<int> d_gaussianArrayInitialized;

// dpct::constant_memory genrate crash on exit...
// std::shared_ptr<dpct::constant_memory<int, 1>> d_gaussianArrayOffset;
// std::shared_ptr<dpct::constant_memory<float, 1>> d_gaussianArray;

// simplest way to replace cuda const memory but currentlt not freed
int* d_gaussianArrayOffset = nullptr;
float* d_gaussianArray = nullptr;


void cuda_createConstantGaussianArray(sycl::queue& stream, int scales) // float delta, int radius)
try
{
    //if (!d_gaussianArrayOffset)
    //{
    //    d_gaussianArrayOffset = std::make_shared<dpct::constant_memory<int, 1>>(MAX_CONSTANT_GAUSS_SCALES);
    //    d_gaussianArray = std::make_shared<dpct::constant_memory<float, 1>>(MAX_CONSTANT_GAUSS_MEM_SIZE);
    //}

    int cudaDeviceId = 0;

    //std::string n = stream.get_device().get_info<sycl::info::device::name>();

    if(scales >= MAX_CONSTANT_GAUSS_SCALES)
    {
        throw std::runtime_error(
            "Programming error: too few scales pre-computed for Gaussian kernels. Enlarge and recompile.");
    }

    if(d_gaussianArrayInitialized.find(cudaDeviceId) != d_gaussianArrayInitialized.end())
        return;
    d_gaussianArrayInitialized.insert(cudaDeviceId);

    int* h_gaussianArrayOffset;
    float* h_gaussianArray;

    h_gaussianArrayOffset = sycl::malloc_shared<int>(MAX_CONSTANT_GAUSS_SCALES, stream);
    h_gaussianArray = sycl::malloc_shared<float>(MAX_CONSTANT_GAUSS_MEM_SIZE, stream);

    int sumSizes = 0;

    for(int scale = 0; scale < MAX_CONSTANT_GAUSS_SCALES; ++scale)
    {
        h_gaussianArrayOffset[scale] = sumSizes;
        const int radius = scale + 1;
        const int size = 2 * radius + 1;
        sumSizes += size;
    }

    for(int scale = 0; scale < MAX_CONSTANT_GAUSS_SCALES; ++scale)
    {
        const int radius = scale + 1;
        const float delta = 1.0f;
        const int size = 2 * radius + 1;

        for(int idx = 0; idx < size; idx++)
        {
            int x = idx - radius;
            h_gaussianArray[h_gaussianArrayOffset[scale] + idx] = expf(-(x * x) / (2 * delta * delta));
        }
    }

    d_gaussianArrayOffset = h_gaussianArrayOffset;
    d_gaussianArray = h_gaussianArray;

    //d_gaussianArrayOffset->init(stream);
    //d_gaussianArray->init(stream);
    //stream.memcpy(d_gaussianArrayOffset->get_ptr(stream), h_gaussianArrayOffset, MAX_CONSTANT_GAUSS_SCALES * sizeof(int)).wait();
    //stream.memcpy(d_gaussianArray->get_ptr(stream), h_gaussianArray, sumSizes * sizeof(float)).wait();

    //sycl::free(h_gaussianArrayOffset, stream);
    //sycl::free(h_gaussianArray, stream);
}
catch(sycl::exception const& e)
{
    RETHROW_SYCL_EXCEPTION(e);
}
} 

/*
 * @note This kernel implementation is not optimized because the Gaussian filter is separable.
 */

void downscaleWithGaussianBlur_kernel(
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> in_img_tex,
    sycl::sampler sampler,
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
    sycl::accessor<sycl::uchar4, 2, sycl::access::mode::write> out_downscaledImg_d,
#else
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
    sycl::accessor<sycl::ushort4, 2, sycl::access::mode::write> out_downscaledImg_d,
#else
    sycl::accessor<sycl::float4, 2, sycl::access::mode::write> out_downscaledImg_d,
#endif
#endif
    int out_downscaledImg_p, unsigned int downscaledImgWidth, unsigned int downscaledImgHeight, int downscale,
    int gaussRadius, const sycl::nd_item<3>& item_ct1, int* d_gaussianArrayOffset, float* d_gaussianArray)
{
    const unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if((x < downscaledImgWidth) && (y < downscaledImgHeight))
    {
        const float s = float(downscale) * 0.5f;

        sycl::float4 accPix = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
        float sumFactor = 0.0f;

        for(int i = -gaussRadius; i <= gaussRadius; i++)
        {
            for(int j = -gaussRadius; j <= gaussRadius; j++)
            {
                const sycl::float4 curPix =
                    in_img_tex.read(sycl::float2{(x * downscale + j) + s, (y * downscale + i) + s}, sampler);
                const float factor =
                    __sycl::getGauss(downscale - 1, i + gaussRadius, d_gaussianArrayOffset, d_gaussianArray) *
                    __sycl::getGauss(downscale - 1, j + gaussRadius, d_gaussianArrayOffset, d_gaussianArray); // domain factor

                accPix = accPix + curPix * factor;
                sumFactor += factor;
            }
        }

        accPix /= sumFactor;

        store_half4(accPix, out_downscaledImg_d, x, y);

    }
}


void cuda_downscaleWithGaussianBlur(CudaDeviceMemoryPitched<CudaRGBA, 2>& out_downscaledImg_dmp,
                                    CudaDeviceMemoryPitched<CudaRGBA, 2>& in_img_tex, 
                                    int downscale, int gaussRadius,
                                    sycl::queue& stream)
try
{
    BufferLocker out(out_downscaledImg_dmp);
    ImageLocker in(in_img_tex);

    // kernel launch parameters
    const sycl::range<3> block(1, 2, 32);
    const sycl::range<3> grid(1, divUp(out_downscaledImg_dmp.getSize().y(), block[1]),
                              divUp(out_downscaledImg_dmp.getSize().x(), block[2])
                            );

    stream.submit(
        [&](sycl::handler& cgh)
        {

            //auto d_gaussianArrayOffset_ptr = __sycl::d_gaussianArrayOffset->get_ptr(stream);
            //auto d_gaussianArray_ptr = __sycl::d_gaussianArray->get_ptr(stream);
            
            auto d_gaussianArrayOffset_ptr = __sycl::d_gaussianArrayOffset;
            auto d_gaussianArray_ptr = __sycl::d_gaussianArray;

            sycl::accessor in_acc = in.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);
            sycl::sampler sampler(sycl::coordinate_normalization_mode::unnormalized, sycl::addressing_mode::clamp, sycl::filtering_mode::linear);

            sycl::accessor out_acc = out.buffer().get_access<sycl::access::mode::write>(cgh);
            auto pitch = out_downscaledImg_dmp.getPitch();
            auto width = (unsigned int)(out_downscaledImg_dmp.getSize().x());
            auto height = (unsigned int)(out_downscaledImg_dmp.getSize().y());

            cgh.parallel_for(sycl::nd_range(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 downscaleWithGaussianBlur_kernel(
                                     in_acc, sampler, 
                                     out_acc,
                                     pitch, width, height,
                                     downscale, gaussRadius, item_ct1,
                                     d_gaussianArrayOffset_ptr, d_gaussianArray_ptr);
                             });
        });

    stream.wait();

}
catch(sycl::exception const& e)
{
    RETHROW_SYCL_EXCEPTION(e);
}

} // namespace depthMap
} // namespace aliceVision

