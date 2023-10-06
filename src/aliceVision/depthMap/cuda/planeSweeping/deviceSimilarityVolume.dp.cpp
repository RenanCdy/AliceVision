#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "deviceSimilarityVolume.hpp"
#include "deviceSimilarityVolumeKernels.dp.hpp"
#include <aliceVision/depthMap/cuda/host/DeviceStreamManager.hpp>
#include <aliceVision/depthMap/cuda/host/MemoryLocker.hpp>

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
template<class T>
sycl::range<3> getMaxPotentialBlockSize(T kernelFuction)
{
    const sycl::range<3> defaultBlock(32, 1, 1); // minimal default settings

    // TODO
    /*
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
    */
    return defaultBlock;
}

void cuda_volumeInitialize(CudaDeviceMemoryPitched<TSim, 3>& inout_volume_dmp, TSim value, DeviceStream& stream)
try
{
    BufferLocker lock(inout_volume_dmp);

    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volume_dmp.getSize();

    // kernel launch parameters
    const sycl::range<3> block(1, 4, 32);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    // kernel execution
    sycl::queue& queue = stream;
    auto volumeInitializeEvent = queue.submit(
        [&](sycl::handler& cgh)
        {
            auto inout_volume_dmp_acc = lock.buffer().get_access<sycl::access::mode::read_write>(cgh);
            auto volDim_x_ct3 = (unsigned int)(volDim.x());
            auto volDim_y_ct4 = (unsigned int)(volDim.y());

            cgh.parallel_for(sycl::nd_range(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_init_kernel(inout_volume_dmp_acc,
                                                          // inout_volume_dmp_getBytesPaddedUpToDim_ct1,
                                                          // inout_volume_dmp_getBytesPaddedUpToDim_ct2, 
                                                          volDim_x_ct3,
                                                          volDim_y_ct4, value, item_ct1);
                             });
        });

    volumeInitializeEvent.wait();
}
catch(const sycl::exception& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeInitialize(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volume_dmp, TSimRefine value,
                           DeviceStream& stream)
try
{
    BufferLocker lock(inout_volume_dmp);

    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volume_dmp.getSize();

    // kernel launch parameters
    const sycl::range<3> block(1, 4, 32);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    sycl::queue& queue = stream;
    auto volumeInitializeEvent = queue.submit(
        [&](sycl::handler& cgh)
        {
            auto inout_volume_dmp_acc = lock.buffer().get_access<sycl::access::mode::read_write>(cgh);
            // auto inout_volume_dmp_getBuffer_ct0 = inout_volume_dmp.getBuffer();
            // auto inout_volume_dmp_getBytesPaddedUpToDim_ct1 = inout_volume_dmp.getBytesPaddedUpToDim(1);
            // auto inout_volume_dmp_getBytesPaddedUpToDim_ct2 = inout_volume_dmp.getBytesPaddedUpToDim(0);
            auto volDim_x_ct3 = (unsigned int)(volDim.x());
            auto volDim_y_ct4 = (unsigned int)(volDim.y());

            cgh.parallel_for(sycl::nd_range(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_init_kernel(inout_volume_dmp_acc,
                                                    volDim_x_ct3, volDim_y_ct4, value, item_ct1);
                             });
        });

        volumeInitializeEvent.wait();
}
catch(const sycl::exception& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeComputeSimilarity(CudaDeviceMemoryPitched<TSim, 3>& out_volBestSim_dmp,
                                  CudaDeviceMemoryPitched<TSim, 3>& out_volSecBestSim_dmp,
                                  const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp,
                                  const int rcDeviceCameraParamsId, const int tcDeviceCameraParamsId,
                                  const DeviceMipmapImage& rcDeviceMipmapImage,
                                  const DeviceMipmapImage& tcDeviceMipmapImage, const SgmParams& sgmParams,
                                  const Range& depthRange, const ROI& roi, DeviceStream& stream)
try
{
    // get mipmap images level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(sgmParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(sgmParams.scale);
    const CudaSize<2> tcLevelDim = tcDeviceMipmapImage.getDimensions(sgmParams.scale);

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_computeSimilarity_kernel);
    const sycl::range<3> grid(depthRange.size(), divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));

    BufferLocker out_volBestSim_dmp_locker(out_volBestSim_dmp);
    BufferLocker out_volSecBestSim_dmp_locker(out_volSecBestSim_dmp);
    BufferLocker in_depths_dmp_locker(in_depths_dmp);
    ImageLocker rcDeviceMipmapImage_locker(rcDeviceMipmapImage.getMipmappedArray());
    ImageLocker tcDeviceMipmapImage_locker(tcDeviceMipmapImage.getMipmappedArray());

    // kernel execution
    sycl::queue& queue = (sycl::queue&)stream;
    auto volume_computeSimilarity_event = queue.submit(
        [&](sycl::handler& cgh)
        {
            auto out_volBestSim_dmp_acc = out_volBestSim_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
            // auto out_volBestSim_dmp_getBuffer_ct0 = out_volBestSim_dmp.getBuffer();
            // auto out_volBestSim_dmp_getBytesPaddedUpToDim_ct1 = out_volBestSim_dmp.getBytesPaddedUpToDim(1);
            // auto out_volBestSim_dmp_getBytesPaddedUpToDim_ct2 = out_volBestSim_dmp.getBytesPaddedUpToDim(0);
            auto out_volSecBestSim_dmp_acc = out_volSecBestSim_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
            // auto out_volSecBestSim_dmp_getBuffer_ct3 = out_volSecBestSim_dmp.getBuffer();
            // auto out_volSecBestSim_dmp_getBytesPaddedUpToDim_ct4 = out_volSecBestSim_dmp.getBytesPaddedUpToDim(1);
            // auto out_volSecBestSim_dmp_getBytesPaddedUpToDim_ct5 = out_volSecBestSim_dmp.getBytesPaddedUpToDim(0);
            auto in_depths_dmp_acc = in_depths_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
            // auto in_depths_dmp_getBuffer_ct6 = in_depths_dmp.getBuffer();
            // auto in_depths_dmp_getBytesPaddedUpToDim_ct7 = in_depths_dmp.getBytesPaddedUpToDim(0);

            sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcDeviceMipmapImage_acc = rcDeviceMipmapImage_locker.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);
            sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> tcDeviceMipmapImage_acc = tcDeviceMipmapImage_locker.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);
            sycl::sampler sampler(sycl::coordinate_normalization_mode::normalized, sycl::addressing_mode::clamp, sycl::filtering_mode::linear);

            // auto rcDeviceMipmapImage_getTextureObject_ct10 = rcDeviceMipmapImage.getTextureObject();
            // auto tcDeviceMipmapImage_getTextureObject_ct11 = tcDeviceMipmapImage.getTextureObject();
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

            const __sycl::DeviceCameraParams* cameraParametersArray_d = __sycl::cameraParametersArray_d;
            const __sycl::DevicePatchPattern* patchPattern_d = __sycl::patchPattern_d;

            cgh.parallel_for(sycl::nd_range(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_computeSimilarity_kernel(
                                     out_volBestSim_dmp_acc, 
                                     out_volSecBestSim_dmp_acc, 
                                     in_depths_dmp_acc,
                                     rcDeviceCameraParamsId,
                                     tcDeviceCameraParamsId, 
                                     cameraParametersArray_d,
                                     rcDeviceMipmapImage_acc,
                                     tcDeviceMipmapImage_acc, 
                                     sampler,
                                     rcLevelDim_x_ct12, rcLevelDim_y_ct13,
                                     tcLevelDim_x_ct14, tcLevelDim_y_ct15, 
                                     rcMipmapLevel, sgmParams_stepXY_ct17,
                                     sgmParams_wsh_ct18, float_sgmParams_gammaC_ct19, float_sgmParams_gammaP_ct20,
                                     sgmParams_useConsistentScale_ct21, sgmParams_useCustomPatchPattern_ct22,
                                     patchPattern_d,
                                     depthRange, roi, item_ct1);
                             });
        });

        volume_computeSimilarity_event.wait();
}
catch(const sycl::exception& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

}
}