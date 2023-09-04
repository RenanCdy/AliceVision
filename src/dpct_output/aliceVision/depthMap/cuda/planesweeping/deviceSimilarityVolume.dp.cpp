// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "deviceSimilarityVolume.hpp"
#include "deviceSimilarityVolumeKernels.dp.hpp"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>

#include <map>

namespace aliceVision {
namespace depthMap {

/**
 * @brief Get maximum potential block size for the given kernel function.
 *        Provides optimal block size based on the capacity of the device.
 *
 * @see https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html
 *
 * @param[in] kernelFuction the given kernel function
 *
 * @return recommended or default block size for kernel execution
 */
template <class T>
sycl::range<3> getMaxPotentialBlockSize(T kernelFuction)
{
    const dim3 defaultBlock(32, 1, 1); // minimal default settings

    int recommendedMinGridSize;
    int recommendedBlockSize;

    cudaError_t err;
    err = cudaOccupancyMaxPotentialBlockSize(&recommendedMinGridSize,
                                             &recommendedBlockSize,
                                             kernelFuction,
                                             0, // dynamic shared mem size: none used
                                             0); // no block size limit, 1 thread OK

    if(err != cudaSuccess)
    {
        ALICEVISION_LOG_WARNING( "cudaOccupancyMaxPotentialBlockSize failed, using default block settings.");
        return defaultBlock;
    }

    if(recommendedBlockSize > 32)
    {
        const dim3 recommendedBlock(32, divUp(recommendedBlockSize, 32), 1);
        return recommendedBlock;
    }

    return defaultBlock;
}

void cuda_volumeInitialize(CudaDeviceMemoryPitched<TSim, 3>& inout_volume_dmp, TSim value, dpct::queue_ptr stream)
try {
    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volume_dmp.getSize();

    // kernel launch parameters
    const sycl::range<3> block(1, 4, 32);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    // kernel execution
    /*
    FIXED-DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    stream->submit(
        [&](sycl::handler& cgh)
        {
            auto inout_volume_dmp_getBuffer_ct0 = inout_volume_dmp.getBuffer();
            auto inout_volume_dmp_getBytesPaddedUpToDim_ct1 = inout_volume_dmp.getBytesPaddedUpToDim(1);
            auto inout_volume_dmp_getBytesPaddedUpToDim_ct2 = inout_volume_dmp.getBytesPaddedUpToDim(0);
            auto volDim_x_ct3 = (unsigned int)(volDim.x());
            auto volDim_y_ct4 = (unsigned int)(volDim.y());

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_init_kernel<TSim>(
                                     inout_volume_dmp_getBuffer_ct0, inout_volume_dmp_getBytesPaddedUpToDim_ct1,
                                     inout_volume_dmp_getBytesPaddedUpToDim_ct2, volDim_x_ct3, volDim_y_ct4, value);
                             });
        });

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeInitialize(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volume_dmp, TSimRefine value,
                           dpct::queue_ptr stream)
try {
    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volume_dmp.getSize();

    // kernel launch parameters
    const sycl::range<3> block(1, 4, 32);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    // kernel execution
    /*
    FIXED-DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    stream->submit(
        [&](sycl::handler& cgh)
        {
            auto inout_volume_dmp_getBuffer_ct0 = inout_volume_dmp.getBuffer();
            auto inout_volume_dmp_getBytesPaddedUpToDim_ct1 = inout_volume_dmp.getBytesPaddedUpToDim(1);
            auto inout_volume_dmp_getBytesPaddedUpToDim_ct2 = inout_volume_dmp.getBytesPaddedUpToDim(0);
            auto volDim_x_ct3 = (unsigned int)(volDim.x());
            auto volDim_y_ct4 = (unsigned int)(volDim.y());

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_init_kernel<TSimRefine>(
                                     inout_volume_dmp_getBuffer_ct0, inout_volume_dmp_getBytesPaddedUpToDim_ct1,
                                     inout_volume_dmp_getBytesPaddedUpToDim_ct2, volDim_x_ct3, volDim_y_ct4, value);
                             });
        });

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeAdd(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volume_dmp,
                    const CudaDeviceMemoryPitched<TSimRefine, 3>& in_volume_dmp, dpct::queue_ptr stream)
try {
    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volume_dmp.getSize();

    // kernel launch parameters
    const sycl::range<3> block(1, 4, 32);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    // kernel execution
    /*
    FIXED-DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->submit(
            [&](sycl::handler& cgh)
            {
                auto inout_volume_dmp_getBuffer_ct0 = inout_volume_dmp.getBuffer();
                auto inout_volume_dmp_getBytesPaddedUpToDim_ct1 = inout_volume_dmp.getBytesPaddedUpToDim(1);
                auto inout_volume_dmp_getBytesPaddedUpToDim_ct2 = inout_volume_dmp.getBytesPaddedUpToDim(0);
                auto in_volume_dmp_getBuffer_ct3 = in_volume_dmp.getBuffer();
                auto in_volume_dmp_getBytesPaddedUpToDim_ct4 = in_volume_dmp.getBytesPaddedUpToDim(1);
                auto in_volume_dmp_getBytesPaddedUpToDim_ct5 = in_volume_dmp.getBytesPaddedUpToDim(0);
                auto volDim_x_ct6 = (unsigned int)(volDim.x());
                auto volDim_y_ct7 = (unsigned int)(volDim.y());

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     volume_add_kernel(
                                         inout_volume_dmp_getBuffer_ct0, inout_volume_dmp_getBytesPaddedUpToDim_ct1,
                                         inout_volume_dmp_getBytesPaddedUpToDim_ct2, in_volume_dmp_getBuffer_ct3,
                                         in_volume_dmp_getBytesPaddedUpToDim_ct4,
                                         in_volume_dmp_getBytesPaddedUpToDim_ct5, volDim_x_ct6, volDim_y_ct7, item_ct1);
                                 });
            });
    }
} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeUpdateUninitializedSimilarity(const CudaDeviceMemoryPitched<TSim, 3>& in_volBestSim_dmp,
                                              CudaDeviceMemoryPitched<TSim, 3>& inout_volSecBestSim_dmp,
                                              dpct::queue_ptr stream)
try {
    assert(in_volBestSim_dmp.getSize() == inout_volSecBestSim_dmp.getSize());

    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volSecBestSim_dmp.getSize();

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_updateUninitialized_kernel);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    // kernel execution
    /*
    FIXED-DPCT1049:20: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->submit(
            [&](sycl::handler& cgh)
            {
                auto inout_volSecBestSim_dmp_getBuffer_ct0 = inout_volSecBestSim_dmp.getBuffer();
                auto inout_volSecBestSim_dmp_getBytesPaddedUpToDim_ct1 =
                    inout_volSecBestSim_dmp.getBytesPaddedUpToDim(1);
                auto inout_volSecBestSim_dmp_getBytesPaddedUpToDim_ct2 =
                    inout_volSecBestSim_dmp.getBytesPaddedUpToDim(0);
                auto in_volBestSim_dmp_getBuffer_ct3 = in_volBestSim_dmp.getBuffer();
                auto in_volBestSim_dmp_getBytesPaddedUpToDim_ct4 = in_volBestSim_dmp.getBytesPaddedUpToDim(1);
                auto in_volBestSim_dmp_getBytesPaddedUpToDim_ct5 = in_volBestSim_dmp.getBytesPaddedUpToDim(0);
                auto volDim_x_ct6 = (unsigned int)(volDim.x());
                auto volDim_y_ct7 = (unsigned int)(volDim.y());

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1)
                    {
                        volume_updateUninitialized_kernel(
                            inout_volSecBestSim_dmp_getBuffer_ct0, inout_volSecBestSim_dmp_getBytesPaddedUpToDim_ct1,
                            inout_volSecBestSim_dmp_getBytesPaddedUpToDim_ct2, in_volBestSim_dmp_getBuffer_ct3,
                            in_volBestSim_dmp_getBytesPaddedUpToDim_ct4, in_volBestSim_dmp_getBytesPaddedUpToDim_ct5,
                            volDim_x_ct6, volDim_y_ct7, item_ct1);
                    });
            });
    }
} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeComputeSimilarity(CudaDeviceMemoryPitched<TSim, 3>& out_volBestSim_dmp,
                                  CudaDeviceMemoryPitched<TSim, 3>& out_volSecBestSim_dmp,
                                  const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp,
                                  const int rcDeviceCameraParamsId, const int tcDeviceCameraParamsId,
                                  const DeviceMipmapImage& rcDeviceMipmapImage,
                                  const DeviceMipmapImage& tcDeviceMipmapImage, const SgmParams& sgmParams,
                                  const Range& depthRange, const ROI& roi, dpct::queue_ptr stream)
try {
    // get mipmap images level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(sgmParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(sgmParams.scale);
    const CudaSize<2> tcLevelDim = tcDeviceMipmapImage.getDimensions(sgmParams.scale);

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_computeSimilarity_kernel);
    const sycl::range<3> grid(depthRange.size(), divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));

    // kernel execution
    /*
    FIXED-DPCT1049:21: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->submit(
            [&](sycl::handler& cgh)
            {
                constantCameraParametersArray_d.init(*stream);
                constantPatchPattern_d.init(*stream);

                auto constantCameraParametersArray_d_ptr_ct1 = constantCameraParametersArray_d.get_ptr();
                auto constantPatchPattern_d_ptr_ct1 = constantPatchPattern_d.get_ptr();

                auto out_volBestSim_dmp_getBuffer_ct0 = out_volBestSim_dmp.getBuffer();
                auto out_volBestSim_dmp_getBytesPaddedUpToDim_ct1 = out_volBestSim_dmp.getBytesPaddedUpToDim(1);
                auto out_volBestSim_dmp_getBytesPaddedUpToDim_ct2 = out_volBestSim_dmp.getBytesPaddedUpToDim(0);
                auto out_volSecBestSim_dmp_getBuffer_ct3 = out_volSecBestSim_dmp.getBuffer();
                auto out_volSecBestSim_dmp_getBytesPaddedUpToDim_ct4 = out_volSecBestSim_dmp.getBytesPaddedUpToDim(1);
                auto out_volSecBestSim_dmp_getBytesPaddedUpToDim_ct5 = out_volSecBestSim_dmp.getBytesPaddedUpToDim(0);
                auto in_depths_dmp_getBuffer_ct6 = in_depths_dmp.getBuffer();
                auto in_depths_dmp_getBytesPaddedUpToDim_ct7 = in_depths_dmp.getBytesPaddedUpToDim(0);
                auto rcDeviceMipmapImage_getTextureObject_ct10 = rcDeviceMipmapImage.getTextureObject();
                auto tcDeviceMipmapImage_getTextureObject_ct11 = tcDeviceMipmapImage.getTextureObject();
                auto rcLevelDim_x_ct12 = (unsigned int)(rcLevelDim.x());
                auto rcLevelDim_y_ct13 = (unsigned int)(rcLevelDim.y());
                auto tcLevelDim_x_ct14 = (unsigned int)(tcLevelDim.x());
                auto tcLevelDim_y_ct15 = (unsigned int)(tcLevelDim.y());
                auto sgmParams_stepXY_ct17 = sgmParams.stepXY;
                auto sgmParams_wsh_ct18 = sgmParams.wsh;
                auto float_sgmParams_gammaC_ct19 = (1.f / float(sgmParams.gammaC));
                auto float_sgmParams_gammaP_ct20 = (1.f / float(sgmParams.gammaP));
                auto sgmParams_useConsistentScale_ct21 = sgmParams.useConsistentScale;
                auto sgmParams_useCustomPatchPattern_ct22 = sgmParams.useCustomPatchPattern;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1)
                    {
                        volume_computeSimilarity_kernel(
                            out_volBestSim_dmp_getBuffer_ct0, out_volBestSim_dmp_getBytesPaddedUpToDim_ct1,
                            out_volBestSim_dmp_getBytesPaddedUpToDim_ct2, out_volSecBestSim_dmp_getBuffer_ct3,
                            out_volSecBestSim_dmp_getBytesPaddedUpToDim_ct4,
                            out_volSecBestSim_dmp_getBytesPaddedUpToDim_ct5, in_depths_dmp_getBuffer_ct6,
                            in_depths_dmp_getBytesPaddedUpToDim_ct7, rcDeviceCameraParamsId, tcDeviceCameraParamsId,
                            rcDeviceMipmapImage_getTextureObject_ct10, tcDeviceMipmapImage_getTextureObject_ct11,
                            rcLevelDim_x_ct12, rcLevelDim_y_ct13, tcLevelDim_x_ct14, tcLevelDim_y_ct15, rcMipmapLevel,
                            sgmParams_stepXY_ct17, sgmParams_wsh_ct18, float_sgmParams_gammaC_ct19,
                            float_sgmParams_gammaP_ct20, sgmParams_useConsistentScale_ct21,
                            sgmParams_useCustomPatchPattern_ct22, depthRange, roi, item_ct1,
                            constantCameraParametersArray_d_ptr_ct1, *constantPatchPattern_d_ptr_ct1);
                    });
            });
    }

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

extern void cuda_volumeRefineSimilarity(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volSim_dmp,
                                        const CudaDeviceMemoryPitched<sycl::float2, 2>& in_sgmDepthPixSizeMap_dmp,
                                        const CudaDeviceMemoryPitched<sycl::float3, 2>* in_sgmNormalMap_dmpPtr,
                                        const int rcDeviceCameraParamsId, const int tcDeviceCameraParamsId,
                                        const DeviceMipmapImage& rcDeviceMipmapImage,
                                        const DeviceMipmapImage& tcDeviceMipmapImage, const RefineParams& refineParams,
                                        const Range& depthRange, const ROI& roi, dpct::queue_ptr stream)
try {
    // get mipmap images level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(refineParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);
    const CudaSize<2> tcLevelDim = tcDeviceMipmapImage.getDimensions(refineParams.scale);

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_refineSimilarity_kernel);
    const sycl::range<3> grid(depthRange.size(), divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));

    // kernel execution
    /*
    FIXED-DPCT1049:22: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->submit(
            [&](sycl::handler& cgh)
            {
                constantCameraParametersArray_d.init(*stream);
                constantPatchPattern_d.init(*stream);

                auto constantCameraParametersArray_d_ptr_ct1 = constantCameraParametersArray_d.get_ptr();
                auto constantPatchPattern_d_ptr_ct1 = constantPatchPattern_d.get_ptr();

                auto inout_volSim_dmp_getBuffer_ct0 = inout_volSim_dmp.getBuffer();
                auto inout_volSim_dmp_getBytesPaddedUpToDim_ct1 = inout_volSim_dmp.getBytesPaddedUpToDim(1);
                auto inout_volSim_dmp_getBytesPaddedUpToDim_ct2 = inout_volSim_dmp.getBytesPaddedUpToDim(0);
                auto in_sgmDepthPixSizeMap_dmp_getBuffer_ct3 = in_sgmDepthPixSizeMap_dmp.getBuffer();
                auto in_sgmDepthPixSizeMap_dmp_getBytesPaddedUpToDim_ct4 =
                    in_sgmDepthPixSizeMap_dmp.getBytesPaddedUpToDim(0);
                auto in_sgmNormalMap_dmpPtr_nullptr_nullptr_in_sgmNormalMap_dmpPtr_getBuffer_ct5 =
                    (in_sgmNormalMap_dmpPtr == nullptr) ? nullptr : in_sgmNormalMap_dmpPtr->getBuffer();
                auto in_sgmNormalMap_dmpPtr_nullptr_in_sgmNormalMap_dmpPtr_getBytesPaddedUpToDim_ct6 =
                    (in_sgmNormalMap_dmpPtr == nullptr) ? 0 : in_sgmNormalMap_dmpPtr->getBytesPaddedUpToDim(0);
                auto rcDeviceMipmapImage_getTextureObject_ct9 = rcDeviceMipmapImage.getTextureObject();
                auto tcDeviceMipmapImage_getTextureObject_ct10 = tcDeviceMipmapImage.getTextureObject();
                auto rcLevelDim_x_ct11 = (unsigned int)(rcLevelDim.x());
                auto rcLevelDim_y_ct12 = (unsigned int)(rcLevelDim.y());
                auto tcLevelDim_x_ct13 = (unsigned int)(tcLevelDim.x());
                auto tcLevelDim_y_ct14 = (unsigned int)(tcLevelDim.y());
                auto int_inout_volSim_dmp_getSize_z_ct16 = int(inout_volSim_dmp.getSize().z());

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1)
                    {
                        volume_refineSimilarity_kernel(
                            inout_volSim_dmp_getBuffer_ct0, inout_volSim_dmp_getBytesPaddedUpToDim_ct1,
                            inout_volSim_dmp_getBytesPaddedUpToDim_ct2, in_sgmDepthPixSizeMap_dmp_getBuffer_ct3,
                            in_sgmDepthPixSizeMap_dmp_getBytesPaddedUpToDim_ct4,
                            in_sgmNormalMap_dmpPtr_nullptr_nullptr_in_sgmNormalMap_dmpPtr_getBuffer_ct5,
                            in_sgmNormalMap_dmpPtr_nullptr_in_sgmNormalMap_dmpPtr_getBytesPaddedUpToDim_ct6,
                            rcDeviceCameraParamsId, tcDeviceCameraParamsId, rcDeviceMipmapImage_getTextureObject_ct9,
                            tcDeviceMipmapImage_getTextureObject_ct10, rcLevelDim_x_ct11, rcLevelDim_y_ct12,
                            tcLevelDim_x_ct13, tcLevelDim_y_ct14, rcMipmapLevel, int_inout_volSim_dmp_getSize_z_ct16,
                            refineParams.stepXY, refineParams.wsh, (1.f / float(refineParams.gammaC)),
                            (1.f / float(refineParams.gammaP)), refineParams.useConsistentScale,
                            refineParams.useCustomPatchPattern, depthRange, roi, item_ct1,
                            constantCameraParametersArray_d_ptr_ct1, *constantPatchPattern_d_ptr_ct1);
                    });
            });
    }

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}


void cuda_volumeAggregatePath(CudaDeviceMemoryPitched<TSim, 3>& out_volAgr_dmp,
                              CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volSliceAccA_dmp,
                              CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volSliceAccB_dmp,
                              CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volAxisAcc_dmp,
                              const CudaDeviceMemoryPitched<TSim, 3>& in_volSim_dmp,
                              const DeviceMipmapImage& rcDeviceMipmapImage, const CudaSize<2>& rcLevelDim,
                              const float rcMipmapLevel, const CudaSize<3>& axisT, const SgmParams& sgmParams,
                              const int lastDepthIndex, const int filteringIndex, const bool invY, const ROI& roi,
                              dpct::queue_ptr stream)
try {
    CudaSize<3> volDim = in_volSim_dmp.getSize();
    volDim[2] = lastDepthIndex; // override volume depth, use rc depth list last index

    const size_t volDimX = volDim[axisT[0]];
    const size_t volDimY = volDim[axisT[1]];
    const size_t volDimZ = volDim[axisT[2]];

    const sycl::int3 volDim_ = make_int3(volDim[0], volDim[1], volDim[2]);
    const sycl::int3 axisT_ = make_int3(axisT[0], axisT[1], axisT[2]);
    const int ySign = (invY ? -1 : 1);

    // setup block and grid
    const int blockSize = 8;
    const sycl::range<3> blockVolXZ(1, blockSize, blockSize);
    const sycl::range<3> gridVolXZ(1, divUp(volDimZ, blockVolXZ[1]), divUp(volDimX, blockVolXZ[2]));

    const int blockSizeL = 64;
    const sycl::range<3> blockColZ(1, 1, blockSizeL);
    const sycl::range<3> gridColZ(1, 1, divUp(volDimX, blockColZ[2]));

    const sycl::range<3> blockVolSlide(1, 1, blockSizeL);
    const sycl::range<3> gridVolSlide(1, volDimZ, divUp(volDimX, blockVolSlide[2]));

    CudaDeviceMemoryPitched<TSimAcc, 2>* xzSliceForY_dmpPtr   = &inout_volSliceAccA_dmp; // Y slice
    CudaDeviceMemoryPitched<TSimAcc, 2>* xzSliceForYm1_dmpPtr = &inout_volSliceAccB_dmp; // Y-1 slice
    CudaDeviceMemoryPitched<TSimAcc, 2>* bestSimInYm1_dmpPtr  = &inout_volAxisAcc_dmp;   // best sim score along the Y axis for each Z value

    // Copy the first XZ plane (at Y=0) from 'in_volSim_dmp' into 'xzSliceForYm1_dmpPtr'
    /*
    FIXED-DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);


    stream->submit(
        [&](sycl::handler& cgh)
        {
            auto xzSliceForYm1_dmpPtr_getBuffer_ct0 = xzSliceForYm1_dmpPtr->getBuffer();
            auto xzSliceForYm1_dmpPtr_getPitch_ct1 = xzSliceForYm1_dmpPtr->getPitch();
            auto in_volSim_dmp_getBuffer_ct2 = in_volSim_dmp.getBuffer();
            auto in_volSim_dmp_getBytesPaddedUpToDim_ct3 = in_volSim_dmp.getBytesPaddedUpToDim(1);
            auto in_volSim_dmp_getBytesPaddedUpToDim_ct4 = in_volSim_dmp.getBytesPaddedUpToDim(0);

            cgh.parallel_for(sycl::nd_range<3>(gridVolXZ * blockVolXZ, blockVolXZ),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_getVolumeXZSlice_kernel<TSimAcc, TSim>(
                                     xzSliceForYm1_dmpPtr_getBuffer_ct0, xzSliceForYm1_dmpPtr_getPitch_ct1,
                                     in_volSim_dmp_getBuffer_ct2, in_volSim_dmp_getBytesPaddedUpToDim_ct3,
                                     in_volSim_dmp_getBytesPaddedUpToDim_ct4, volDim_, axisT_, 0);
                             });
        });

    // Set the first Z plane from 'out_volAgr_dmp' to 255
    stream->submit(
        [&](sycl::handler& cgh)
        {
            auto out_volAgr_dmp_getBuffer_ct0 = out_volAgr_dmp.getBuffer();
            auto out_volAgr_dmp_getBytesPaddedUpToDim_ct1 = out_volAgr_dmp.getBytesPaddedUpToDim(1);
            auto out_volAgr_dmp_getBytesPaddedUpToDim_ct2 = out_volAgr_dmp.getBytesPaddedUpToDim(0);

            cgh.parallel_for(sycl::nd_range<3>(gridVolXZ * blockVolXZ, blockVolXZ),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_initVolumeYSlice_kernel<TSim>(
                                     out_volAgr_dmp_getBuffer_ct0, out_volAgr_dmp_getBytesPaddedUpToDim_ct1,
                                     out_volAgr_dmp_getBytesPaddedUpToDim_ct2, volDim_, axisT_, 0, 255);
                             });
        });

    for(int iy = 1; iy < volDimY; ++iy)
    {
        const int y = invY ? volDimY - 1 - iy : iy;

        // For each column: compute the best score
        // Foreach x:
        //   bestSimInYm1[x] = min(d_xzSliceForY[1:height])
        stream->submit(
            [&](sycl::handler& cgh)
            {
                auto xzSliceForYm1_dmpPtr_getBuffer_ct0 = xzSliceForYm1_dmpPtr->getBuffer();
                auto xzSliceForYm1_dmpPtr_getPitch_ct1 = xzSliceForYm1_dmpPtr->getPitch();
                auto bestSimInYm1_dmpPtr_getBuffer_ct2 = bestSimInYm1_dmpPtr->getBuffer();

                cgh.parallel_for(sycl::nd_range<3>(gridColZ * blockColZ, blockColZ),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     volume_computeBestZInSlice_kernel(
                                         xzSliceForYm1_dmpPtr_getBuffer_ct0, xzSliceForYm1_dmpPtr_getPitch_ct1,
                                         bestSimInYm1_dmpPtr_getBuffer_ct2, volDimX, volDimZ, item_ct1);
                                 });
            });

        // Copy the 'z' plane from 'in_volSim_dmp' into 'xzSliceForY'
        stream->submit(
            [&](sycl::handler& cgh)
            {
                auto xzSliceForY_dmpPtr_getBuffer_ct0 = xzSliceForY_dmpPtr->getBuffer();
                auto xzSliceForY_dmpPtr_getPitch_ct1 = xzSliceForY_dmpPtr->getPitch();
                auto in_volSim_dmp_getBuffer_ct2 = in_volSim_dmp.getBuffer();
                auto in_volSim_dmp_getBytesPaddedUpToDim_ct3 = in_volSim_dmp.getBytesPaddedUpToDim(1);
                auto in_volSim_dmp_getBytesPaddedUpToDim_ct4 = in_volSim_dmp.getBytesPaddedUpToDim(0);

                cgh.parallel_for(sycl::nd_range<3>(gridVolXZ * blockVolXZ, blockVolXZ),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     volume_getVolumeXZSlice_kernel<TSimAcc, TSim>(
                                         xzSliceForY_dmpPtr_getBuffer_ct0, xzSliceForY_dmpPtr_getPitch_ct1,
                                         in_volSim_dmp_getBuffer_ct2, in_volSim_dmp_getBytesPaddedUpToDim_ct3,
                                         in_volSim_dmp_getBytesPaddedUpToDim_ct4, volDim_, axisT_, y);
                                 });
            });

        {
            dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
            stream->submit(
                [&](sycl::handler& cgh)
                {
                    auto rcDeviceMipmapImage_getTextureObject_ct0 = rcDeviceMipmapImage.getTextureObject();
                    auto rcLevelDim_x_ct1 = (unsigned int)(rcLevelDim.x());
                    auto rcLevelDim_y_ct2 = (unsigned int)(rcLevelDim.y());
                    auto xzSliceForY_dmpPtr_getBuffer_ct4 = xzSliceForY_dmpPtr->getBuffer();
                    auto xzSliceForY_dmpPtr_getPitch_ct5 = xzSliceForY_dmpPtr->getPitch();
                    auto xzSliceForYm1_dmpPtr_getBuffer_ct6 = xzSliceForYm1_dmpPtr->getBuffer();
                    auto xzSliceForYm1_dmpPtr_getPitch_ct7 = xzSliceForYm1_dmpPtr->getPitch();
                    auto bestSimInYm1_dmpPtr_getBuffer_ct8 = bestSimInYm1_dmpPtr->getBuffer();
                    auto out_volAgr_dmp_getBuffer_ct9 = out_volAgr_dmp.getBuffer();
                    auto out_volAgr_dmp_getBytesPaddedUpToDim_ct10 = out_volAgr_dmp.getBytesPaddedUpToDim(1);
                    auto out_volAgr_dmp_getBytesPaddedUpToDim_ct11 = out_volAgr_dmp.getBytesPaddedUpToDim(0);
                    auto sgmParams_stepXY_ct14 = sgmParams.stepXY;
                    auto sgmParams_p1_ct16 = sgmParams.p1;
                    auto sgmParams_p2Weighting_ct17 = sgmParams.p2Weighting;

                    cgh.parallel_for(sycl::nd_range<3>(gridVolSlide * blockVolSlide, blockVolSlide),
                                     [=](sycl::nd_item<3> item_ct1)
                                     {
                                         volume_agregateCostVolumeAtXinSlices_kernel(
                                             rcDeviceMipmapImage_getTextureObject_ct0, rcLevelDim_x_ct1,
                                             rcLevelDim_y_ct2, rcMipmapLevel, xzSliceForY_dmpPtr_getBuffer_ct4,
                                             xzSliceForY_dmpPtr_getPitch_ct5, xzSliceForYm1_dmpPtr_getBuffer_ct6,
                                             xzSliceForYm1_dmpPtr_getPitch_ct7, bestSimInYm1_dmpPtr_getBuffer_ct8,
                                             out_volAgr_dmp_getBuffer_ct9, out_volAgr_dmp_getBytesPaddedUpToDim_ct10,
                                             out_volAgr_dmp_getBytesPaddedUpToDim_ct11, volDim_, axisT_,
                                             sgmParams_stepXY_ct14, y, sgmParams_p1_ct16, sgmParams_p2Weighting_ct17,
                                             ySign, filteringIndex, roi, item_ct1);
                                     });
                });
        }

        std::swap(xzSliceForYm1_dmpPtr, xzSliceForY_dmpPtr);
    }
    
} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeOptimize(CudaDeviceMemoryPitched<TSim, 3>& out_volSimFiltered_dmp,
                         CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volSliceAccA_dmp,
                         CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volSliceAccB_dmp,
                         CudaDeviceMemoryPitched<TSimAcc, 2>& inout_volAxisAcc_dmp,
                         const CudaDeviceMemoryPitched<TSim, 3>& in_volSim_dmp,
                         const DeviceMipmapImage& rcDeviceMipmapImage, const SgmParams& sgmParams,
                         const int lastDepthIndex, const ROI& roi, dpct::queue_ptr stream)
{
    // get R mipmap image level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(sgmParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(sgmParams.scale);

    // update aggregation volume
    int npaths = 0;
    const auto updateAggrVolume = [&](const CudaSize<3>& axisT, bool invX)
    {
        cuda_volumeAggregatePath(out_volSimFiltered_dmp, 
                                 inout_volSliceAccA_dmp, 
                                 inout_volSliceAccB_dmp,
                                 inout_volAxisAcc_dmp,
                                 in_volSim_dmp, 
                                 rcDeviceMipmapImage,
                                 rcLevelDim,
                                 rcMipmapLevel,
                                 axisT, 
                                 sgmParams, 
                                 lastDepthIndex,
                                 npaths,
                                 invX, 
                                 roi,
                                 stream);
        npaths++;
    };

    // filtering is done on the last axis
    const std::map<char, CudaSize<3>> mapAxes = {
        {'X', {1, 0, 2}}, // XYZ -> YXZ
        {'Y', {0, 1, 2}}, // XYZ
    };

    for(char axis : sgmParams.filteringAxes)
    {
        const CudaSize<3>& axisT = mapAxes.at(axis);
        updateAggrVolume(axisT, false); // without transpose
        updateAggrVolume(axisT, true);  // with transpose of the last axis
    }
}

void cuda_volumeRetrieveBestDepth(CudaDeviceMemoryPitched<sycl::float2, 2>& out_sgmDepthThicknessMap_dmp,
                                  CudaDeviceMemoryPitched<sycl::float2, 2>& out_sgmDepthSimMap_dmp,
                                  const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp,
                                  const CudaDeviceMemoryPitched<TSim, 3>& in_volSim_dmp,
                                  const int rcDeviceCameraParamsId, const SgmParams& sgmParams, const Range& depthRange,
                                  const ROI& roi, dpct::queue_ptr stream)
try {
    // constant kernel inputs
    const int scaleStep = sgmParams.scale * sgmParams.stepXY;
    const float thicknessMultFactor = 1.f + float(sgmParams.depthThicknessInflate);
    const float maxSimilarity = float(sgmParams.maxSimilarity) * 254.f; // convert from (0, 1) to (0, 254)

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_retrieveBestDepth_kernel);
    const sycl::range<3> grid(1, divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));

    // kernel execution
    /*
    FIXED-DPCT1049:28: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->submit(
            [&](sycl::handler& cgh)
            {
                constantCameraParametersArray_d.init(*stream);

                auto constantCameraParametersArray_d_ptr_ct1 = constantCameraParametersArray_d.get_ptr();

                auto out_sgmDepthThicknessMap_dmp_getBuffer_ct0 = out_sgmDepthThicknessMap_dmp.getBuffer();
                auto out_sgmDepthThicknessMap_dmp_getBytesPaddedUpToDim_ct1 =
                    out_sgmDepthThicknessMap_dmp.getBytesPaddedUpToDim(0);
                auto out_sgmDepthSimMap_dmp_getBuffer_ct2 = out_sgmDepthSimMap_dmp.getBuffer();
                auto out_sgmDepthSimMap_dmp_getBytesPaddedUpToDim_ct3 = out_sgmDepthSimMap_dmp.getBytesPaddedUpToDim(0);
                auto in_depths_dmp_getBuffer_ct4 = in_depths_dmp.getBuffer();
                auto in_depths_dmp_getBytesPaddedUpToDim_ct5 = in_depths_dmp.getBytesPaddedUpToDim(0);
                auto in_volSim_dmp_getBuffer_ct6 = in_volSim_dmp.getBuffer();
                auto in_volSim_dmp_getBytesPaddedUpToDim_ct7 = in_volSim_dmp.getBytesPaddedUpToDim(1);
                auto in_volSim_dmp_getBytesPaddedUpToDim_ct8 = in_volSim_dmp.getBytesPaddedUpToDim(0);
                auto int_in_volSim_dmp_getSize_z_ct10 = int(in_volSim_dmp.getSize().z());

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1)
                    {
                        volume_retrieveBestDepth_kernel(
                            out_sgmDepthThicknessMap_dmp_getBuffer_ct0,
                            out_sgmDepthThicknessMap_dmp_getBytesPaddedUpToDim_ct1,
                            out_sgmDepthSimMap_dmp_getBuffer_ct2, out_sgmDepthSimMap_dmp_getBytesPaddedUpToDim_ct3,
                            in_depths_dmp_getBuffer_ct4, in_depths_dmp_getBytesPaddedUpToDim_ct5,
                            in_volSim_dmp_getBuffer_ct6, in_volSim_dmp_getBytesPaddedUpToDim_ct7,
                            in_volSim_dmp_getBytesPaddedUpToDim_ct8, rcDeviceCameraParamsId,
                            int_in_volSim_dmp_getSize_z_ct10, scaleStep, thicknessMultFactor, maxSimilarity, depthRange,
                            roi, item_ct1, constantCameraParametersArray_d_ptr_ct1);
                    });
            });
    }

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

extern void cuda_volumeRefineBestDepth(CudaDeviceMemoryPitched<sycl::float2, 2>& out_refineDepthSimMap_dmp,
                                       const CudaDeviceMemoryPitched<sycl::float2, 2>& in_sgmDepthPixSizeMap_dmp,
                                       const CudaDeviceMemoryPitched<TSimRefine, 3>& in_volSim_dmp,
                                       const RefineParams& refineParams, const ROI& roi, dpct::queue_ptr stream)
try {
    // constant kernel inputs
    const int halfNbSamples = refineParams.nbSubsamples * refineParams.halfNbDepths;
    const float twoTimesSigmaPowerTwo = float(2.0 * refineParams.sigma * refineParams.sigma);

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_refineBestDepth_kernel);
    const sycl::range<3> grid(1, divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));

    // kernel execution
    /*
    FIXED-DPCT1049:29: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->submit(
            [&](sycl::handler& cgh)
            {
                auto out_refineDepthSimMap_dmp_getBuffer_ct0 = out_refineDepthSimMap_dmp.getBuffer();
                auto out_refineDepthSimMap_dmp_getBytesPaddedUpToDim_ct1 =
                    out_refineDepthSimMap_dmp.getBytesPaddedUpToDim(0);
                auto in_sgmDepthPixSizeMap_dmp_getBuffer_ct2 = in_sgmDepthPixSizeMap_dmp.getBuffer();
                auto in_sgmDepthPixSizeMap_dmp_getBytesPaddedUpToDim_ct3 =
                    in_sgmDepthPixSizeMap_dmp.getBytesPaddedUpToDim(0);
                auto in_volSim_dmp_getBuffer_ct4 = in_volSim_dmp.getBuffer();
                auto in_volSim_dmp_getBytesPaddedUpToDim_ct5 = in_volSim_dmp.getBytesPaddedUpToDim(1);
                auto in_volSim_dmp_getBytesPaddedUpToDim_ct6 = in_volSim_dmp.getBytesPaddedUpToDim(0);
                auto int_in_volSim_dmp_getSize_z_ct7 = int(in_volSim_dmp.getSize().z());

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     volume_refineBestDepth_kernel(
                                         out_refineDepthSimMap_dmp_getBuffer_ct0,
                                         out_refineDepthSimMap_dmp_getBytesPaddedUpToDim_ct1,
                                         in_sgmDepthPixSizeMap_dmp_getBuffer_ct2,
                                         in_sgmDepthPixSizeMap_dmp_getBytesPaddedUpToDim_ct3,
                                         in_volSim_dmp_getBuffer_ct4, in_volSim_dmp_getBytesPaddedUpToDim_ct5,
                                         in_volSim_dmp_getBytesPaddedUpToDim_ct6, int_in_volSim_dmp_getSize_z_ct7,
                                         refineParams.nbSubsamples, halfNbSamples, refineParams.halfNbDepths,
                                         twoTimesSigmaPowerTwo, roi, item_ct1);
                                 });
            });
    }
} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

} // namespace depthMap
} // namespace aliceVision
