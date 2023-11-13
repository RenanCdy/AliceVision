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

void cuda_normalMapUpscale(CudaDeviceMemoryPitched<sycl::float3, 2>& out_upscaledMap_dmp,
                           const CudaDeviceMemoryPitched<sycl::float3, 2>& in_map_dmp, const ROI& roi,
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
                                 mapUpscale_kernel<sycl::float3>(
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

void cuda_depthThicknessSmoothThickness(CudaDeviceMemoryPitched<sycl::float2, 2>& inout_depthThicknessMap_dmp,
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

void cuda_computeSgmUpscaledDepthPixSizeMap(CudaDeviceMemoryPitched<sycl::float2, 2>& out_upscaledDepthPixSizeMap_dmp,
                                            const CudaDeviceMemoryPitched<sycl::float2, 2>& in_sgmDepthThicknessMap_dmp,
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

void cuda_depthSimMapComputeNormal(CudaDeviceMemoryPitched<sycl::float3, 2>& out_normalMap_dmp,
                                   const CudaDeviceMemoryPitched<sycl::float2, 2>& in_depthSimMap_dmp,
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

} // namespace depthMap
} // namespace aliceVision
