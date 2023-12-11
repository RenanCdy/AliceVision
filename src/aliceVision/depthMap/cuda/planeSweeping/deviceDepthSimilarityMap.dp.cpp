// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "deviceDepthSimilarityMap.hpp"
#include "deviceDepthSimilarityMapKernels.dp.hpp"
#include <aliceVision/depthMap/cuda/host/DeviceStreamManager.hpp>
#include <aliceVision/depthMap/cuda/host/MemoryLocker.hpp>

#include <aliceVision/depthMap/cuda/host/divUp.hpp>

#include <utility>

namespace aliceVision {
namespace depthMap {

void cuda_depthSimMapCopyDepthOnly(CudaDeviceMemoryPitched<float2, 2>& out_depthSimMap_dmp,
                                   const CudaDeviceMemoryPitched<float2, 2>& in_depthSimMap_dmp, float defaultSim,
                                   DeviceStream& stream)
try {
    // get output map dimensions
    const CudaSize<2>& depthSimMapDim = out_depthSimMap_dmp.getSize();

    // kernel launch parameters
    const int blockSize = 16;
    const sycl::range<3> block(1, blockSize, blockSize);
    const sycl::range<3> grid(1, divUp(depthSimMapDim.y(), blockSize), divUp(depthSimMapDim.x(), blockSize));

    BufferLocker out_depthSimMap_dmp_locker(out_depthSimMap_dmp);
    BufferLocker in_depthSimMap_dmp_locker(in_depthSimMap_dmp);

    // kernel execution
    sycl::queue& queue = (sycl::queue&)stream;
    auto copyEvent = queue.submit(
        [&](sycl::handler& cgh)
        {
            auto out_depthSimMap_dmp_acc = out_depthSimMap_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
            auto in_depthSimMap_dmp_acc = in_depthSimMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
            // auto out_depthSimMap_dmp_getBuffer_ct0 = out_depthSimMap_dmp.getBuffer();
            // auto out_depthSimMap_dmp_getPitch_ct1 = out_depthSimMap_dmp.getPitch();
            // auto in_depthSimMap_dmp_getBuffer_ct2 = in_depthSimMap_dmp.getBuffer();
            // auto in_depthSimMap_dmp_getPitch_ct3 = in_depthSimMap_dmp.getPitch();
            auto depthSimMapDim_x_ct4 = (unsigned int)(depthSimMapDim.x());
            auto depthSimMapDim_y_ct5 = (unsigned int)(depthSimMapDim.y());

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 depthSimMapCopyDepthOnly_kernel(
                                     out_depthSimMap_dmp_acc, in_depthSimMap_dmp_acc,
                                     //out_depthSimMap_dmp_getBuffer_ct0, out_depthSimMap_dmp_getPitch_ct1,
                                     //in_depthSimMap_dmp_getBuffer_ct2, in_depthSimMap_dmp_getPitch_ct3,
                                     depthSimMapDim_x_ct4, depthSimMapDim_y_ct5, defaultSim, item_ct1);
                             });
        });
    copyEvent.wait();   

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_normalMapUpscale(CudaDeviceMemoryPitched<float3, 2>& out_upscaledMap_dmp,
                           const CudaDeviceMemoryPitched<float3, 2>& in_map_dmp, const ROI& roi,
                           DeviceStream& stream)
try {
    // compute upscale ratio
    const CudaSize<2>& out_mapDim = out_upscaledMap_dmp.getSize();
    const CudaSize<2>& in_mapDim = in_map_dmp.getSize();
    const float ratio = float(in_mapDim.x()) / float(out_mapDim.x());

    // kernel launch parameters
    const int blockSize = 16;
    const sycl::range<3> block(1, blockSize, blockSize);
    const sycl::range<3> grid(1, divUp(roi.height(), blockSize), divUp(roi.width(), blockSize));


    BufferLocker out_upscaledMap_dmp_locker(out_upscaledMap_dmp);
    BufferLocker in_map_dmp_locker(in_map_dmp);

    // kernel execution
    sycl::queue& queue = (sycl::queue&)stream;
    auto mapUpscaleEvent = queue.submit(
        [&](sycl::handler& cgh)
        {
            auto out_upscaledMap_dmp_acc = out_upscaledMap_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
            auto in_map_dmp_acc = in_map_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);

            // auto out_upscaledMap_dmp_getBuffer_ct0 = out_upscaledMap_dmp.getBuffer();
            // auto out_upscaledMap_dmp_getPitch_ct1 = out_upscaledMap_dmp.getPitch();
            // auto in_map_dmp_getBuffer_ct2 = in_map_dmp.getBuffer();
            // auto in_map_dmp_getPitch_ct3 = in_map_dmp.getPitch();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 mapUpscale_kernel<custom_sycl::custom_float3>(
                                    out_upscaledMap_dmp_acc, in_map_dmp_acc,
                                     //out_upscaledMap_dmp_getBuffer_ct0, out_upscaledMap_dmp_getPitch_ct1,
                                     //in_map_dmp_getBuffer_ct2, in_map_dmp_getPitch_ct3, 
                                     ratio, roi, item_ct1);
                             });
        });
    mapUpscaleEvent.wait();   
} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_depthThicknessSmoothThickness(CudaDeviceMemoryPitched<float2, 2>& inout_depthThicknessMap_dmp,
                                        const SgmParams& sgmParams, const RefineParams& refineParams, const ROI& roi,
                                        DeviceStream& stream)
try {
    const int sgmScaleStep = sgmParams.scale * sgmParams.stepXY;
    const int refineScaleStep = refineParams.scale * refineParams.stepXY;

    // min/max number of Refine samples in SGM thickness area
    const float minNbRefineSamples = 2.f;
    const float maxNbRefineSamples = std::max(sgmScaleStep / float(refineScaleStep), minNbRefineSamples);

    // min/max SGM thickness inflate factor
    const float minThicknessInflate = refineParams.halfNbDepths / maxNbRefineSamples;
    const float maxThicknessInflate = refineParams.halfNbDepths / minNbRefineSamples;

    // kernel launch parameters
    const int blockSize = 8;
    const sycl::range<3> block(1, blockSize, blockSize);
    const sycl::range<3> grid(1, divUp(roi.height(), blockSize), divUp(roi.width(), blockSize));

    BufferLocker inout_depthThicknessMap_dmp_locker(inout_depthThicknessMap_dmp);

    // kernel execution
    sycl::queue& queue = (sycl::queue&)stream;
    auto smoothThicknessEvent = queue.submit(
        [&](sycl::handler& cgh)
        {
            //auto inout_depthThicknessMap_dmp_getBuffer_ct0 = inout_depthThicknessMap_dmp.getBuffer();
            //auto inout_depthThicknessMap_dmp_getPitch_ct1 = inout_depthThicknessMap_dmp.getPitch();
            auto inout_depthThicknessMap_dmp_acc = inout_depthThicknessMap_dmp_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 depthThicknessMapSmoothThickness_kernel(inout_depthThicknessMap_dmp_acc,
                                                                         //inout_depthThicknessMap_dmp_getPitch_ct1,
                                                                         minThicknessInflate, maxThicknessInflate, roi,
                                                                         item_ct1);
                             });
        });
    smoothThicknessEvent.wait();

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_computeSgmUpscaledDepthPixSizeMap(CudaDeviceMemoryPitched<float2, 2>& out_upscaledDepthPixSizeMap_dmp,
                                            const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthThicknessMap_dmp,
                                            const int rcDeviceCameraParamsId,
                                            const DeviceMipmapImage& rcDeviceMipmapImage,
                                            const RefineParams& refineParams, const ROI& roi, DeviceStream& stream)
try {
    // compute upscale ratio
    const CudaSize<2>& out_mapDim = out_upscaledDepthPixSizeMap_dmp.getSize();
    const CudaSize<2>& in_mapDim = in_sgmDepthThicknessMap_dmp.getSize();
    const float ratio = float(in_mapDim.x()) / float(out_mapDim.x());

    // get R mipmap image level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(refineParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);

    // kernel launch parameters
    const int blockSize = 16;
    const sycl::range<3> block(1, blockSize, blockSize);
    const sycl::range<3> grid(1, divUp(roi.height(), blockSize), divUp(roi.width(), blockSize));

    BufferLocker out_upscaledDepthPixSizeMap_dmp_locker(out_upscaledDepthPixSizeMap_dmp);
    BufferLocker in_sgmDepthThicknessMap_dmp_locker(in_sgmDepthThicknessMap_dmp);
    ImageLocker rcDeviceMipmapImage_locker(rcDeviceMipmapImage.getMipmappedArray());


    // kernel execution
    sycl::queue& queue = (sycl::queue&)stream;
    if(refineParams.interpolateMiddleDepth)
    {
        
        auto computeEvent = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto out_upscaledDepthPixSizeMap_dmp_acc = out_upscaledDepthPixSizeMap_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
                auto in_sgmDepthThicknessMap_dmp_acc = in_sgmDepthThicknessMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                // auto out_upscaledDepthPixSizeMap_dmp_getBuffer_ct0 = out_upscaledDepthPixSizeMap_dmp.getBuffer();
                // auto out_upscaledDepthPixSizeMap_dmp_getPitch_ct1 = out_upscaledDepthPixSizeMap_dmp.getPitch();
                // auto in_sgmDepthThicknessMap_dmp_getBuffer_ct2 = in_sgmDepthThicknessMap_dmp.getBuffer();
                // auto in_sgmDepthThicknessMap_dmp_getPitch_ct3 = in_sgmDepthThicknessMap_dmp.getPitch();
                sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcDeviceMipmapImage_acc = rcDeviceMipmapImage_locker.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);
                sycl::sampler sampler(sycl::coordinate_normalization_mode::normalized, sycl::addressing_mode::clamp, sycl::filtering_mode::linear);

                //auto rcDeviceMipmapImage_getTextureObject_ct5 = rcDeviceMipmapImage.getTextureObject();
                auto rcLevelDim_x_ct6 = (unsigned int)(rcLevelDim.x());
                auto rcLevelDim_y_ct7 = (unsigned int)(rcLevelDim.y());
                const __sycl::DeviceCameraParams* cameraParametersArray_d = __sycl::cameraParametersArray_d;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1)
                    {
                        computeSgmUpscaledDepthPixSizeMap_bilinear_kernel(
                            out_upscaledDepthPixSizeMap_dmp_acc, in_sgmDepthThicknessMap_dmp_acc,
                            //out_upscaledDepthPixSizeMap_dmp_getBuffer_ct0, out_upscaledDepthPixSizeMap_dmp_getPitch_ct1,
                            //in_sgmDepthThicknessMap_dmp_getBuffer_ct2, in_sgmDepthThicknessMap_dmp_getPitch_ct3,
                            rcDeviceCameraParamsId, 
                            rcDeviceMipmapImage_acc,
                            sampler,
                            //rcDeviceMipmapImage_getTextureObject_ct5, 
                            rcLevelDim_x_ct6,
                            rcLevelDim_y_ct7, rcMipmapLevel, refineParams.stepXY, refineParams.halfNbDepths, ratio, roi,
                            item_ct1, cameraParametersArray_d);
                    });
            });
        computeEvent.wait();
    }
    else
    {
        auto computeEvent = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto out_upscaledDepthPixSizeMap_dmp_acc = out_upscaledDepthPixSizeMap_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
                auto in_sgmDepthThicknessMap_dmp_acc = in_sgmDepthThicknessMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                // auto out_upscaledDepthPixSizeMap_dmp_getBuffer_ct0 = out_upscaledDepthPixSizeMap_dmp.getBuffer();
                // auto out_upscaledDepthPixSizeMap_dmp_getPitch_ct1 = out_upscaledDepthPixSizeMap_dmp.getPitch();
                // auto in_sgmDepthThicknessMap_dmp_getBuffer_ct2 = in_sgmDepthThicknessMap_dmp.getBuffer();
                // auto in_sgmDepthThicknessMap_dmp_getPitch_ct3 = in_sgmDepthThicknessMap_dmp.getPitch();
                sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcDeviceMipmapImage_acc = rcDeviceMipmapImage_locker.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);
                sycl::sampler sampler(sycl::coordinate_normalization_mode::normalized, sycl::addressing_mode::clamp, sycl::filtering_mode::linear);
                
                //auto rcDeviceMipmapImage_getTextureObject_ct5 = rcDeviceMipmapImage.getTextureObject();
                auto rcLevelDim_x_ct6 = (unsigned int)(rcLevelDim.x());
                auto rcLevelDim_y_ct7 = (unsigned int)(rcLevelDim.y());
                const __sycl::DeviceCameraParams* cameraParametersArray_d = __sycl::cameraParametersArray_d;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1)
                    {
                        computeSgmUpscaledDepthPixSizeMap_nearestNeighbor_kernel(
                            out_upscaledDepthPixSizeMap_dmp_acc, in_sgmDepthThicknessMap_dmp_acc,
                            //out_upscaledDepthPixSizeMap_dmp_getBuffer_ct0, out_upscaledDepthPixSizeMap_dmp_getPitch_ct1,
                            //in_sgmDepthThicknessMap_dmp_getBuffer_ct2, in_sgmDepthThicknessMap_dmp_getPitch_ct3,
                            rcDeviceCameraParamsId, 
                            rcDeviceMipmapImage_acc,
                            sampler,
                            //rcDeviceMipmapImage_getTextureObject_ct5, 
                            rcLevelDim_x_ct6,
                            rcLevelDim_y_ct7, rcMipmapLevel, refineParams.stepXY, refineParams.halfNbDepths, ratio, roi,
                            item_ct1, cameraParametersArray_d);
                    });
            });
        computeEvent.wait();
    }

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_depthSimMapComputeNormal(CudaDeviceMemoryPitched<float3, 2>& out_normalMap_dmp,
                                   const CudaDeviceMemoryPitched<float2, 2>& in_depthSimMap_dmp,
                                   const int rcDeviceCameraParamsId, const int stepXY, const ROI& roi,
                                   DeviceStream& stream)
try {
    // kernel launch parameters
    const sycl::range<3> block(1, 8, 8);
    const sycl::range<3> grid(1, divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));

    BufferLocker out_normalMap_dmp_locker(out_normalMap_dmp);
    BufferLocker in_depthSimMap_dmp_locker(in_depthSimMap_dmp);

    // kernel execution
    sycl::queue& queue = (sycl::queue&)stream;
    auto computeNormalEvent = queue.submit(
        [&](sycl::handler& cgh)
        {
            auto out_normalMap_dmp_acc = out_normalMap_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
            //auto out_normalMap_dmp_getBuffer_ct0 = out_normalMap_dmp.getBuffer();
            //auto out_normalMap_dmp_getPitch_ct1 = out_normalMap_dmp.getPitch();
            auto in_depthSimMap_dmp_acc = in_depthSimMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
            //auto in_depthSimMap_dmp_getBuffer_ct2 = in_depthSimMap_dmp.getBuffer();
            //auto in_depthSimMap_dmp_getPitch_ct3 = in_depthSimMap_dmp.getPitch();

            const __sycl::DeviceCameraParams* cameraParametersArray_d = __sycl::cameraParametersArray_d;

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 depthSimMapComputeNormal_kernel<3>(
                                     //out_normalMap_dmp_getBuffer_ct0, out_normalMap_dmp_getPitch_ct1,
                                     //in_depthSimMap_dmp_getBuffer_ct2, in_depthSimMap_dmp_getPitch_ct3,
                                     out_normalMap_dmp_acc,
                                     in_depthSimMap_dmp_acc,
                                     rcDeviceCameraParamsId, stepXY, roi,
                                     item_ct1,
                                     cameraParametersArray_d);
                             });
        });
    computeNormalEvent.wait();

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_depthSimMapOptimizeGradientDescent(CudaDeviceMemoryPitched<float2, 2>& out_optimizeDepthSimMap_dmp,
                                             CudaDeviceMemoryPitched<float, 2>& inout_imgVariance_dmp,
                                             CudaDeviceMemoryPitched<float, 2>& inout_tmpOptDepthMap_dmp,
                                             const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthPixSizeMap_dmp,
                                             const CudaDeviceMemoryPitched<float2, 2>& in_refineDepthSimMap_dmp,
                                             const int rcDeviceCameraParamsId,
                                             const DeviceMipmapImage& rcDeviceMipmapImage,
                                             const RefineParams& refineParams, const ROI& roi, DeviceStream& stream)
try {
    // get R mipmap image level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(refineParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);

    // initialize depth/sim map optimized with SGM depth/pixSize map
    out_optimizeDepthSimMap_dmp.copyFrom(in_sgmDepthPixSizeMap_dmp, stream);

    {
        // kernel launch parameters
        const sycl::range<3> lblock(1, 2, 32);
        const sycl::range<3> lgrid(1, divUp(roi.height(), lblock[1]), divUp(roi.width(), lblock[2]));

        BufferLocker inout_imgVariance_dmp_locker(inout_imgVariance_dmp);
        ImageLocker rcDeviceMipmapImage_locker(rcDeviceMipmapImage.getMipmappedArray());

        // kernel execution
        sycl::queue& queue = (sycl::queue&)stream;
        auto optimizeEvent = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto inout_imgVariance_dmp_acc = inout_imgVariance_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
                //auto inout_imgVariance_dmp_getBuffer_ct0 = inout_imgVariance_dmp.getBuffer();
                //auto inout_imgVariance_dmp_getPitch_ct1 = inout_imgVariance_dmp.getPitch();
                //auto rcDeviceMipmapImage_getTextureObject_ct2 = rcDeviceMipmapImage.getTextureObject();
                auto rcLevelDim_x_ct3 = (unsigned int)(rcLevelDim.x());
                auto rcLevelDim_y_ct4 = (unsigned int)(rcLevelDim.y());

                sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcDeviceMipmapImage_acc = rcDeviceMipmapImage_locker.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);
                sycl::sampler sampler(sycl::coordinate_normalization_mode::normalized, sycl::addressing_mode::clamp, sycl::filtering_mode::linear);

                cgh.parallel_for(sycl::nd_range<3>(lgrid * lblock, lblock),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     optimize_varLofLABtoW_kernel(
                                         inout_imgVariance_dmp_acc, rcDeviceMipmapImage_acc,
                                         sampler,
                                         //inout_imgVariance_dmp_getBuffer_ct0, inout_imgVariance_dmp_getPitch_ct1,
                                         //rcDeviceMipmapImage_getTextureObject_ct2, 
                                         rcLevelDim_x_ct3, rcLevelDim_y_ct4,
                                         rcMipmapLevel, refineParams.stepXY, roi, item_ct1);
                                 });
            });
        optimizeEvent.wait();
    }

    //CudaTexture<float, false, false> imgVarianceTex(inout_imgVariance_dmp); // neighbor interpolation, without normalized coordinates
    //CudaTexture<float, false, false> depthTex(inout_tmpOptDepthMap_dmp);    // neighbor interpolation, without normalized coordinates

    // kernel launch parameters
    const int blockSize = 16;
    const sycl::range<3> block(1, blockSize, blockSize);
    const sycl::range<3> grid(1, divUp(roi.height(), blockSize), divUp(roi.width(), blockSize));

    BufferLocker inout_tmpOptDepthMap_dmp_locker(inout_tmpOptDepthMap_dmp);
    BufferLocker out_optimizeDepthSimMap_dmp_locker(out_optimizeDepthSimMap_dmp);

    for(int iter = 0; iter < refineParams.optimizationNbIterations; ++iter) // default nb iterations is 100
    {
        
        // copy depths values from out_depthSimMapOptimized_dmp to inout_tmpOptDepthMap_dmp
        sycl::queue& queue = (sycl::queue&)stream;
        auto optimizeEvent = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto inout_tmpOptDepthMap_dmp_acc = inout_tmpOptDepthMap_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
                auto out_optimizeDepthSimMap_dmp_acc = out_optimizeDepthSimMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                // auto inout_tmpOptDepthMap_dmp_getBuffer_ct0 = inout_tmpOptDepthMap_dmp.getBuffer();
                // auto inout_tmpOptDepthMap_dmp_getPitch_ct1 = inout_tmpOptDepthMap_dmp.getPitch();
                // auto out_optimizeDepthSimMap_dmp_getBuffer_ct2 = out_optimizeDepthSimMap_dmp.getBuffer();
                // auto out_optimizeDepthSimMap_dmp_getPitch_ct3 = out_optimizeDepthSimMap_dmp.getPitch();

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     optimize_getOptDeptMapFromOptDepthSimMap_kernel(
                                        inout_tmpOptDepthMap_dmp_acc, out_optimizeDepthSimMap_dmp_acc,
                                        //  inout_tmpOptDepthMap_dmp_getBuffer_ct0, inout_tmpOptDepthMap_dmp_getPitch_ct1,
                                        //  out_optimizeDepthSimMap_dmp_getBuffer_ct2,
                                        //  out_optimizeDepthSimMap_dmp_getPitch_ct3, 
                                         roi, item_ct1);
                                 });
            });
        optimizeEvent.wait();

        // adjust depth/sim by using previously computed depths
        {


            BufferLocker out_optimizeDepthSimMap_dmp_locker(out_optimizeDepthSimMap_dmp);
            BufferLocker in_sgmDepthPixSizeMap_dmp_locker(in_sgmDepthPixSizeMap_dmp);
            BufferLocker in_refineDepthSimMap_dmp_locker(in_refineDepthSimMap_dmp);
            // CudaDeviceMemoryPitched<float, 2>& inout_imgVariance_dmp
            // CudaDeviceMemoryPitched<float, 2>& inout_tmpOptDepthMap_dmp
            // TODO: Validate if we need image sampler for neighbor interpolation
            BufferLocker imgVarianceTex_locker(inout_imgVariance_dmp);
            BufferLocker depthTex_locker(inout_tmpOptDepthMap_dmp);

            sycl::queue& queue = (sycl::queue&)stream;
            auto optimizeEvent = queue.submit(
                [&](sycl::handler& cgh)
                {
                    // constantCameraParametersArray_d.init(*stream);
                    // auto constantCameraParametersArray_d_ptr_ct1 = constantCameraParametersArray_d.get_ptr();
                    const __sycl::DeviceCameraParams* cameraParametersArray_d = __sycl::cameraParametersArray_d;

                    auto out_optimizeDepthSimMap_dmp_acc = out_optimizeDepthSimMap_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
                    auto in_sgmDepthPixSizeMap_dmp_acc = in_sgmDepthPixSizeMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                    auto in_refineDepthSimMap_dmp_acc = in_refineDepthSimMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                
                    // auto out_optimizeDepthSimMap_dmp_getBuffer_ct0 = out_optimizeDepthSimMap_dmp.getBuffer();
                    // auto out_optimizeDepthSimMap_dmp_getPitch_ct1 = out_optimizeDepthSimMap_dmp.getPitch();
                    // auto in_sgmDepthPixSizeMap_dmp_getBuffer_ct2 = in_sgmDepthPixSizeMap_dmp.getBuffer();
                    // auto in_sgmDepthPixSizeMap_dmp_getPitch_ct3 = in_sgmDepthPixSizeMap_dmp.getPitch();
                    // auto in_refineDepthSimMap_dmp_getBuffer_ct4 = in_refineDepthSimMap_dmp.getBuffer();
                    // auto in_refineDepthSimMap_dmp_getPitch_ct5 = in_refineDepthSimMap_dmp.getPitch();
                    // auto imgVarianceTex_textureObj_ct7 = imgVarianceTex.textureObj;
                    // auto depthTex_textureObj_ct8 = depthTex.textureObj;

                    //sycl::accessor<float, 2, sycl::access::mode::read, sycl::access::target::image> imgVarianceTex_acc = imgVarianceTex_locker.image().get_access<float, sycl::access::mode::read>(cgh);
                    //sycl::accessor<float, 2, sycl::access::mode::read, sycl::access::target::image> depthTex_acc = depthTex_locker.image().get_access<float, sycl::access::mode::read>(cgh);
                    auto imgVarianceTex_acc = imgVarianceTex_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                    auto depthTex_acc = depthTex_locker.buffer().get_access<sycl::access::mode::read>(cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1)
                        {
                            optimize_depthSimMap_kernel(
                                out_optimizeDepthSimMap_dmp_acc, in_sgmDepthPixSizeMap_dmp_acc, in_refineDepthSimMap_dmp_acc,
                                // out_optimizeDepthSimMap_dmp_getBuffer_ct0, out_optimizeDepthSimMap_dmp_getPitch_ct1,
                                // in_sgmDepthPixSizeMap_dmp_getBuffer_ct2, in_sgmDepthPixSizeMap_dmp_getPitch_ct3,
                                // in_refineDepthSimMap_dmp_getBuffer_ct4, in_refineDepthSimMap_dmp_getPitch_ct5,
                                rcDeviceCameraParamsId, 
                                //imgVarianceTex_textureObj_ct7, depthTex_textureObj_ct8, 
                                imgVarianceTex_acc, depthTex_acc, iter,
                                roi, item_ct1, cameraParametersArray_d);
                        });
                });
        }
        optimizeEvent.wait();
    }

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

} // namespace depthMap
} // namespace aliceVision
