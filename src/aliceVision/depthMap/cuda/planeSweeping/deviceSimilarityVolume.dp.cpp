#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "deviceSimilarityVolume.hpp"
#include "deviceSimilarityVolumeKernels.dp.hpp"
#include <aliceVision/depthMap/cuda/host/DeviceStreamManager.hpp>
#include <aliceVision/depthMap/cuda/host/MemoryLocker.hpp>

#include <aliceVision/depthMap/cuda/host/divUp.hpp>
#include <aliceVision/depthMap/cuda/device/customDataType.dp.hpp>
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
    const sycl::range<3> defaultBlock(1, 12, 32); // minimal default settings

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
    //const sycl::range<3> grid(divUp(roi.width(), block[2]), divUp(roi.height(), block[1]), depthRange.size());

    BufferLocker out_volBestSim_dmp_locker(out_volBestSim_dmp);
    BufferLocker out_volSecBestSim_dmp_locker(out_volSecBestSim_dmp);
    BufferLocker in_depths_dmp_locker(in_depths_dmp);
    ImageLocker rcDeviceMipmapImage_locker(rcDeviceMipmapImage.getMipmappedArray());
    ImageLocker tcDeviceMipmapImage_locker(tcDeviceMipmapImage.getMipmappedArray());

    // size_t glob_size = (grid * block).size();
    // std::vector<sycl::float3> patches_data_rc(glob_size);
    // std::vector<sycl::float3> patches_data_tc(glob_size);
    // std::vector<sycl::float3> patches_data_p(glob_size);
    // std::vector<sycl::float3> patches_data_n(glob_size);
    // std::vector<sycl::float3> patches_data_x(glob_size);
    // std::vector<sycl::float3> patches_data_y(glob_size);
    // std::vector<float> patches_data_d(glob_size);
    // std::vector<sycl::uint3> patches_data_roi(glob_size);
    // std::vector<float> patches_data_depth(glob_size);
    // std::vector<sycl::float2> patches_data_xy(glob_size);

    // sycl::buffer<sycl::float3, 1> patch_buf_rc(patches_data_rc.data(), patches_data_rc.size());
    // sycl::buffer<sycl::float3, 1> patch_buf_tc(patches_data_tc.data(), patches_data_tc.size());
    // sycl::buffer<sycl::float3, 1> patch_buf_p(patches_data_p.data(), patches_data_p.size());
    // sycl::buffer<sycl::float3, 1> patch_buf_n(patches_data_n.data(), patches_data_n.size());
    // sycl::buffer<sycl::float3, 1> patch_buf_x(patches_data_x.data(), patches_data_x.size());
    // sycl::buffer<sycl::float3, 1> patch_buf_y(patches_data_y.data(), patches_data_y.size());
    // sycl::buffer<float, 1> patch_buf_d(patches_data_d.data(), patches_data_d.size());

    // sycl::buffer<sycl::uint3, 1> patch_buf_roi(patches_data_roi.data(), patches_data_roi.size());
    // sycl::buffer<float, 1> patch_buf_depth(patches_data_depth.data(), patches_data_depth.size());
    // sycl::buffer<sycl::float2, 1> patch_buf_xy(patches_data_xy.data(), patches_data_xy.size());

    printf("ROI : {%u, %u, %u}.\n",
        roi.width(), roi.height(), depthRange.size());
    printf("Grid : {%zu, %zu, %zu} blocks. Blocks : {%zu, %zu, %zu} threads.\n",
        grid[0], grid[1], grid[2], block[0], block[1], block[2]);

    // kernel execution
    sycl::queue& queue = (sycl::queue&)stream;
    auto volume_computeSimilarity_event = queue.submit(
        [&](sycl::handler& cgh)
        {
            // sycl::accessor patch_acc_rc(patch_buf_rc, cgh, sycl::write_only);
            // sycl::accessor patch_acc_tc(patch_buf_tc, cgh, sycl::write_only);
            // sycl::accessor patch_acc_p(patch_buf_p, cgh, sycl::write_only);
            // sycl::accessor patch_acc_n(patch_buf_n, cgh, sycl::write_only);
            // sycl::accessor patch_acc_x(patch_buf_x, cgh, sycl::write_only);
            // sycl::accessor patch_acc_y(patch_buf_y, cgh, sycl::write_only);
            // sycl::accessor patch_acc_d(patch_buf_d, cgh, sycl::write_only);
            // sycl::accessor roi_acc(patch_buf_roi, cgh, sycl::write_only);
            // sycl::accessor depth_acc(patch_buf_depth, cgh, sycl::write_only);
            // sycl::accessor xy_acc(patch_buf_xy, cgh, sycl::write_only);
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
                                //out << "volume_computeSimilarity_kernel! :" << item_ct1 << sycl::endl;
                                 volume_computeSimilarity_kernel(
                                    // patch_acc_rc,
                                    // patch_acc_tc,
                                    // patch_acc_p,
                                    // patch_acc_n,
                                    // patch_acc_x,
                                    // patch_acc_y,
                                    // patch_acc_d,
                                    // roi_acc,
                                    // depth_acc,
                                    // xy_acc,
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

        const __sycl::DeviceCameraParams& params = __sycl::cameraParametersArray_d[rcDeviceCameraParamsId];

    //     printf("rcCam %d---\n", rcDeviceCameraParamsId);
    //     printf("P: ");
    // for (int i = 0; i < 12; i++) {
    //     printf("%.6f ", params.P[i]);
    // }
    // printf("\n");

    // printf("iP: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params.iP[i]);
    // }
    // printf("\n");

    // printf("R: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params.R[i]);
    // }
    // printf("\n");

    // printf("iR: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params.iR[i]);
    // }
    // printf("\n");

    // printf("K: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params.K[i]);
    // }
    // printf("\n");

    // printf("iK: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params.iK[i]);
    // }
    // printf("\n");

    // printf("C: ");
    // printf("(%f, %f, %f)\n", params.C[0], params.C[1], params.C[2]);

    // printf("XVect: ");
    // printf("(%f, %f, %f)\n", params.XVect[0], params.XVect[1], params.XVect[2]);

    // printf("YVect: ");
    // printf("(%f, %f, %f)\n", params.YVect[0], params.YVect[1], params.YVect[2]);

    // printf("ZVect: ");
    // printf("(%f, %f, %f)\n", params.ZVect[0], params.ZVect[1], params.ZVect[2]);

    //     const __sycl::DeviceCameraParams& params2 = __sycl::cameraParametersArray_d[tcDeviceCameraParamsId];
    //     printf("tcCam %d---\n", tcDeviceCameraParamsId);

    //     printf("P: ");
    // for (int i = 0; i < 12; i++) {
    //     printf("%.6f ", params2.P[i]);
    // }
    // printf("\n");

    // printf("iP: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params2.iP[i]);
    // }
    // printf("\n");

    // printf("R: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params2.R[i]);
    // }
    // printf("\n");

    // printf("iR: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params2.iR[i]);
    // }
    // printf("\n");

    // printf("K: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params2.K[i]);
    // }
    // printf("\n");

    // printf("iK: ");
    // for (int i = 0; i < 9; i++) {
    //     printf("%.6f ", params2.iK[i]);
    // }
    // printf("\n");

    // printf("C: ");
    // printf("(%f, %f, %f)\n", params2.C[0], params2.C[1], params2.C[2]);

    // printf("XVect: ");
    // printf("(%f, %f, %f)\n", params2.XVect[0], params2.XVect[1], params2.XVect[2]);

    // printf("YVect: ");
    // printf("(%f, %f, %f)\n", params2.YVect[0], params2.YVect[1], params2.YVect[2]);

    // printf("ZVect: ");
    // printf("(%f, %f, %f)\n", params2.ZVect[0], params2.ZVect[1], params2.ZVect[2]);

    //     std::cout << "Patches is size: "<<glob_size<<"\n";

    //     // Create an ofstream object
    //     std::ofstream outFile;
    //     std::string outputFileName = "output_" + std::to_string(rcDeviceCameraParamsId) + "_" + std::to_string(tcDeviceCameraParamsId) + ".txt";
    //     outFile.open(outputFileName);
    //     // Write the vectors to the file
    //     for (size_t i = 0; i < glob_size; ++i) {
    //     //if(patches_data_x[i].x() && patches_data_x[i].y() && patches_data_x[i].z() && patches_data_y[i])
    //     outFile << i << " " << patches_data_rc[i].x() << " " << patches_data_rc[i].y() << " " << patches_data_rc[i].z() <<
    //     " " << patches_data_tc[i].x() << " " << patches_data_tc[i].y() << " " << patches_data_tc[i].z() <<
    //     " " << patches_data_p[i].x() << " " << patches_data_p[i].y() << " " << patches_data_p[i].z() <<
    //     " " << patches_data_n[i].x() << " " << patches_data_n[i].y() << " " << patches_data_n[i].z() <<
    //     " " << patches_data_x[i].x() << " " << patches_data_x[i].y() << " " << patches_data_x[i].z() << 
    //     " " << patches_data_y[i].x() << " " << patches_data_y[i].y() << " " << patches_data_y[i].z() <<
    //     " " << patches_data_d[i] << 
    //     " " << patches_data_depth[i] << " " << patches_data_roi[i].x() << " " << patches_data_roi[i].y() << " " << patches_data_roi[i].z() << " " << patches_data_xy[i].x() << " " << patches_data_xy[i].y() << "\n";
    // }
    //     outFile.close();
}
catch(const sycl::exception& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeUpdateUninitializedSimilarity(const CudaDeviceMemoryPitched<TSim, 3>& in_volBestSim_dmp,
                                              CudaDeviceMemoryPitched<TSim, 3>& inout_volSecBestSim_dmp,
                                              DeviceStream& stream)
try {
    assert(in_volBestSim_dmp.getSize() == inout_volSecBestSim_dmp.getSize());

    // get input/output volume dimensions
    const CudaSize<3>& volDim = inout_volSecBestSim_dmp.getSize();

    // kernel launch parameters
    const sycl::range<3> block(1,18,32);
    //TODO: const sycl::range<3> block = getMaxPotentialBlockSize(volume_updateUninitialized_kernel);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    // kernel execution
    BufferLocker inout_volSecBestSim_dmp_locker(inout_volSecBestSim_dmp);
    BufferLocker in_volBestSim_dmp_locker(in_volBestSim_dmp);

    {
        // kernel execution
        sycl::queue& queue = (sycl::queue&)stream;
        auto volume_computeUninitialized_event = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto inout_volSecBestSim_dmp_getBuffer_acc = inout_volSecBestSim_dmp_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                //auto inout_volSecBestSim_dmp_getBuffer_ct0 = inout_volSecBestSim_dmp.getBuffer();
                //auto inout_volSecBestSim_dmp_getBytesPaddedUpToDim_ct1 =
                //    inout_volSecBestSim_dmp.getBytesPaddedUpToDim(1);
                //auto inout_volSecBestSim_dmp_getBytesPaddedUpToDim_ct2 =
                //    inout_volSecBestSim_dmp.getBytesPaddedUpToDim(0);
                auto in_volBestSim_dmp_getBuffer_acc = in_volBestSim_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                //auto in_volBestSim_dmp_getBuffer_ct3 = in_volBestSim_dmp.getBuffer();
                //auto in_volBestSim_dmp_getBytesPaddedUpToDim_ct4 = in_volBestSim_dmp.getBytesPaddedUpToDim(1);
                //auto in_volBestSim_dmp_getBytesPaddedUpToDim_ct5 = in_volBestSim_dmp.getBytesPaddedUpToDim(0);
                auto volDim_x_ct6 = (unsigned int)(volDim.x());
                auto volDim_y_ct7 = (unsigned int)(volDim.y());

                cgh.parallel_for(sycl::nd_range(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1)
                    {
                        volume_updateUninitialized_kernel(
                            
                            inout_volSecBestSim_dmp_getBuffer_acc, 
                            //inout_volSecBestSim_dmp_getBytesPaddedUpToDim_ct1,
                            //inout_volSecBestSim_dmp_getBytesPaddedUpToDim_ct2,
                            in_volBestSim_dmp_getBuffer_acc,
                            //in_volBestSim_dmp_getBytesPaddedUpToDim_ct4,
                            //in_volBestSim_dmp_getBytesPaddedUpToDim_ct5,
                            volDim_x_ct6, volDim_y_ct7, item_ct1);
                    });
            });
            volume_computeUninitialized_event.wait();

    }
} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_volumeRefineSimilarity(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volSim_dmp, 
                                        const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthPixSizeMap_dmp,
                                        const CudaDeviceMemoryPitched<float3, 2>* in_sgmNormalMap_dmpPtr,
                                        const int rcDeviceCameraParamsId,
                                        const int tcDeviceCameraParamsId,
                                        const DeviceMipmapImage& rcDeviceMipmapImage,
                                        const DeviceMipmapImage& tcDeviceMipmapImage,
                                        const RefineParams& refineParams, 
                                        const Range& depthRange,
                                        const ROI& roi, DeviceStream& stream)
try {
    // get mipmap images level and dimensions
    const float rcMipmapLevel = rcDeviceMipmapImage.getLevel(refineParams.scale);
    const CudaSize<2> rcLevelDim = rcDeviceMipmapImage.getDimensions(refineParams.scale);
    const CudaSize<2> tcLevelDim = tcDeviceMipmapImage.getDimensions(refineParams.scale);

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_refineSimilarity_kernel);
    const sycl::range<3> grid(depthRange.size(), divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));

    BufferLocker inout_volSim_dmp_locker(inout_volSim_dmp);
    BufferLocker in_sgmDepthPixSizeMap_dmp_locker(in_sgmDepthPixSizeMap_dmp);
    BufferLocker in_sgmNormalMap_dmpPtr_locker(in_sgmNormalMap_dmpPtr);
    ImageLocker rcDeviceMipmapImage_locker(rcDeviceMipmapImage.getMipmappedArray());
    ImageLocker tcDeviceMipmapImage_locker(tcDeviceMipmapImage.getMipmappedArray());

    sycl::sampler sampler(sycl::coordinate_normalization_mode::normalized, sycl::addressing_mode::clamp, sycl::filtering_mode::linear);

    {
        sycl::queue& queue = (sycl::queue&)stream;
        auto volume_getSlice_event = queue.submit(
            [&](sycl::handler& cgh)
            {
                // constantCameraParametersArray_d.init(*stream);
                // constantPatchPattern_d.init(*stream);
                // auto constantCameraParametersArray_d_ptr_ct1 = constantCameraParametersArray_d.get_ptr();
                // auto constantPatchPattern_d_ptr_ct1 = constantPatchPattern_d.get_ptr();
                const __sycl::DeviceCameraParams* cameraParametersArray_d = __sycl::cameraParametersArray_d;
                const __sycl::DevicePatchPattern* patchPattern_d = __sycl::patchPattern_d;

                auto inout_volSim_dmp_acc = inout_volSim_dmp_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                auto in_sgmDepthPixSizeMap_dmp_acc = in_sgmDepthPixSizeMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                auto in_sgmNormalMap_dmpPtr_acc = in_sgmNormalMap_dmpPtr_locker.buffer().get_access<sycl::access::mode::read>(cgh);

                sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcDeviceMipmapImage_acc = rcDeviceMipmapImage_locker.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);
                sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> tcDeviceMipmapImage_acc = tcDeviceMipmapImage_locker.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);
                sycl::sampler sampler(sycl::coordinate_normalization_mode::normalized, sycl::addressing_mode::clamp, sycl::filtering_mode::linear);

                // auto inout_volSim_dmp_getBuffer_ct0 = inout_volSim_dmp.getBuffer();
                // auto inout_volSim_dmp_getBytesPaddedUpToDim_ct1 = inout_volSim_dmp.getBytesPaddedUpToDim(1);
                // auto inout_volSim_dmp_getBytesPaddedUpToDim_ct2 = inout_volSim_dmp.getBytesPaddedUpToDim(0);
                // auto in_sgmDepthPixSizeMap_dmp_getBuffer_ct3 = in_sgmDepthPixSizeMap_dmp.getBuffer();
                // auto in_sgmDepthPixSizeMap_dmp_getBytesPaddedUpToDim_ct4 =
                //     in_sgmDepthPixSizeMap_dmp.getBytesPaddedUpToDim(0);
                // auto in_sgmNormalMap_dmpPtr_nullptr_nullptr_in_sgmNormalMap_dmpPtr_getBuffer_ct5 =
                //     (in_sgmNormalMap_dmpPtr == nullptr) ? nullptr : in_sgmNormalMap_dmpPtr->getBuffer();
                // auto in_sgmNormalMap_dmpPtr_nullptr_in_sgmNormalMap_dmpPtr_getBytesPaddedUpToDim_ct6 =
                //     (in_sgmNormalMap_dmpPtr == nullptr) ? 0 : in_sgmNormalMap_dmpPtr->getBytesPaddedUpToDim(0);
                //auto rcDeviceMipmapImage_getTextureObject_ct9 = rcDeviceMipmapImage.getTextureObject();
                //auto tcDeviceMipmapImage_getTextureObject_ct10 = tcDeviceMipmapImage.getTextureObject();
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
                            inout_volSim_dmp_acc,
                            in_sgmDepthPixSizeMap_dmp_acc,
                            in_sgmNormalMap_dmpPtr_acc, // check for nullptr
                            // inout_volSim_dmp_getBuffer_ct0, inout_volSim_dmp_getBytesPaddedUpToDim_ct1,
                            // inout_volSim_dmp_getBytesPaddedUpToDim_ct2, 
                            // in_sgmDepthPixSizeMap_dmp_getBuffer_ct3,
                            // in_sgmDepthPixSizeMap_dmp_getBytesPaddedUpToDim_ct4,
                            // in_sgmNormalMap_dmpPtr_nullptr_nullptr_in_sgmNormalMap_dmpPtr_getBuffer_ct5,
                            // in_sgmNormalMap_dmpPtr_nullptr_in_sgmNormalMap_dmpPtr_getBytesPaddedUpToDim_ct6,
                            rcDeviceCameraParamsId, tcDeviceCameraParamsId, 
                            rcDeviceMipmapImage_acc,
                            tcDeviceMipmapImage_acc,
                            sampler,
                            // rcDeviceMipmapImage_getTextureObject_ct9,
                            // tcDeviceMipmapImage_getTextureObject_ct10, 
                            rcLevelDim_x_ct11, rcLevelDim_y_ct12,
                            tcLevelDim_x_ct13, tcLevelDim_y_ct14, rcMipmapLevel, int_inout_volSim_dmp_getSize_z_ct16,
                            refineParams.stepXY, refineParams.wsh, (1.f / float(refineParams.gammaC)),
                            (1.f / float(refineParams.gammaP)), refineParams.useConsistentScale,
                            refineParams.useCustomPatchPattern, depthRange, roi, item_ct1,
                            cameraParametersArray_d, patchPattern_d);
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
                              DeviceStream& stream)
try {
    CudaSize<3> volDim = in_volSim_dmp.getSize();
    volDim[2] = lastDepthIndex; // override volume depth, use rc depth list last index

    const size_t volDimX = volDim[axisT.x()];
    const size_t volDimY = volDim[axisT.y()];
    const size_t volDimZ = volDim[axisT.z()];

    const sycl::int3 volDim_ (volDim.x(), volDim.y(), volDim.z());
    const sycl::int3 axisT_ (axisT.x(), axisT.y(), axisT.z());
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

    BufferLocker xzSliceForYm1_dmpPtr_locker(*xzSliceForYm1_dmpPtr);
    BufferLocker in_volSim_dmp_locker(in_volSim_dmp);
    
    sycl::sampler sampler(sycl::coordinate_normalization_mode::normalized, sycl::addressing_mode::clamp, sycl::filtering_mode::linear);

    // kernel execution
    // Copy the first XZ plane (at Y=0) from 'in_volSim_dmp' into 'xzSliceForYm1_dmpPtr'
    sycl::queue& queue = (sycl::queue&)stream;
    auto volume_getSlice_event = queue.submit(
        [&](sycl::handler& cgh)
        {
            auto xzSliceForYm1_dmpPtr_acc = xzSliceForYm1_dmpPtr_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
            //auto xzSliceForYm1_dmpPtr_getBuffer_ct0 = xzSliceForYm1_dmpPtr->getBuffer();
            //auto xzSliceForYm1_dmpPtr_getPitch_ct1 = xzSliceForYm1_dmpPtr->getPitch();
            auto in_volSim_dmp_acc = in_volSim_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
            //auto in_volSim_dmp_getBuffer_ct2 = in_volSim_dmp.getBuffer();
            //auto in_volSim_dmp_getBytesPaddedUpToDim_ct3 = in_volSim_dmp.getBytesPaddedUpToDim(1);
            //auto in_volSim_dmp_getBytesPaddedUpToDim_ct4 = in_volSim_dmp.getBytesPaddedUpToDim(0);

            cgh.parallel_for(sycl::nd_range<3>(gridVolXZ * blockVolXZ, blockVolXZ),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_getVolumeXZSlice_kernel<TSimAcc, TSim>(
                                     xzSliceForYm1_dmpPtr_acc, //xzSliceForYm1_dmpPtr_getPitch_ct1,
                                     in_volSim_dmp_acc, 
                                     //in_volSim_dmp_getBytesPaddedUpToDim_ct3,
                                     //in_volSim_dmp_getBytesPaddedUpToDim_ct4, 
                                     volDim_, axisT_, 0, item_ct1);
                             });
        });
    volume_getSlice_event.wait();

    BufferLocker out_volAgr_dmp_locker(out_volAgr_dmp);
    
    // Set the first Z plane from 'out_volAgr_dmp' to 255
    auto volume_initSlice_event = queue.submit(
        [&](sycl::handler& cgh)
        {
            auto out_volAgr_dmp_acc = out_volAgr_dmp_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
            //auto out_volAgr_dmp_getBuffer_ct0 = out_volAgr_dmp.getBuffer();
            //auto out_volAgr_dmp_getBytesPaddedUpToDim_ct1 = out_volAgr_dmp.getBytesPaddedUpToDim(1);
            //auto out_volAgr_dmp_getBytesPaddedUpToDim_ct2 = out_volAgr_dmp.getBytesPaddedUpToDim(0);

            cgh.parallel_for(sycl::nd_range<3>(gridVolXZ * blockVolXZ, blockVolXZ),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 volume_initVolumeYSlice_kernel<TSim>(
                                     out_volAgr_dmp_acc, 
                                     //out_volAgr_dmp_getBytesPaddedUpToDim_ct1,
                                     //out_volAgr_dmp_getBytesPaddedUpToDim_ct2, 
                                     volDim_, axisT_, 0, 255, item_ct1);
                             });
        });
    volume_initSlice_event.wait();


    BufferLocker bestSimInYm1_dmpPtr_locker(*bestSimInYm1_dmpPtr);
    
    for(int iy = 1; iy < volDimY; ++iy)
    {
        const int y = invY ? volDimY - 1 - iy : iy;

        // For each column: compute the best score
        // Foreach x:
        //   bestSimInYm1[x] = min(d_xzSliceForY[1:height])
        auto volume_computeSlice_event = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto xzSliceForYm1_dmpPtr_acc = xzSliceForYm1_dmpPtr_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                auto bestSimInYm1_dmpPtr_acc = bestSimInYm1_dmpPtr_locker.buffer().get_access<sycl::access::mode::write>(cgh);

                //auto xzSliceForYm1_dmpPtr_getBuffer_ct0 = xzSliceForYm1_dmpPtr->getBuffer();
                //auto xzSliceForYm1_dmpPtr_getPitch_ct1 = xzSliceForYm1_dmpPtr->getPitch();
                //auto bestSimInYm1_dmpPtr_getBuffer_ct2 = bestSimInYm1_dmpPtr->getBuffer();

                cgh.parallel_for(sycl::nd_range<3>(gridColZ * blockColZ, blockColZ),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     volume_computeBestZInSlice_kernel(
                                         xzSliceForYm1_dmpPtr_acc, //xzSliceForYm1_dmpPtr_getPitch_ct1,
                                         bestSimInYm1_dmpPtr_acc, volDimX, volDimZ, item_ct1);
                                 });
            });
        volume_computeSlice_event.wait();

        BufferLocker xzSliceForY_dmpPtr_locker(*xzSliceForY_dmpPtr);
        
        // Copy the 'z' plane from 'in_volSim_dmp' into 'xzSliceForY'
        auto volume_getSlice2_event = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto xzSliceForY_dmpPtr_acc = xzSliceForY_dmpPtr_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                auto in_volSim_dmp_acc = in_volSim_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);

                //auto xzSliceForY_dmpPtr_getBuffer_ct0 = xzSliceForY_dmpPtr->getBuffer();
                //auto xzSliceForY_dmpPtr_getPitch_ct1 = xzSliceForY_dmpPtr->getPitch();
                //auto in_volSim_dmp_getBuffer_ct2 = in_volSim_dmp.getBuffer();
                //auto in_volSim_dmp_getBytesPaddedUpToDim_ct3 = in_volSim_dmp.getBytesPaddedUpToDim(1);
                //auto in_volSim_dmp_getBytesPaddedUpToDim_ct4 = in_volSim_dmp.getBytesPaddedUpToDim(0);

                cgh.parallel_for(sycl::nd_range<3>(gridVolXZ * blockVolXZ, blockVolXZ),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     volume_getVolumeXZSlice_kernel<TSimAcc, TSim>(
                                         xzSliceForY_dmpPtr_acc, //xzSliceForY_dmpPtr_getPitch_ct1,
                                         in_volSim_dmp_acc, 
                                        //  in_volSim_dmp_getBytesPaddedUpToDim_ct3,
                                        //  in_volSim_dmp_getBytesPaddedUpToDim_ct4, 
                                         volDim_, axisT_, y, item_ct1);
                                 });
            });
        volume_getSlice2_event.wait();

        BufferLocker xzSliceForYm1_dmpPtr_locker(*xzSliceForYm1_dmpPtr);
        ImageLocker rcDeviceMipmapImage_locker(rcDeviceMipmapImage.getMipmappedArray());
        {
            auto volume_aggregateSlice_event = queue.submit(
                [&](sycl::handler& cgh)
                {

                    auto xzSliceForY_dmpPtr_acc = xzSliceForY_dmpPtr_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                    auto xzSliceForYm1_dmpPtr_acc = xzSliceForYm1_dmpPtr_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                    auto bestSimInYm1_dmpPtr_acc = bestSimInYm1_dmpPtr_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                    auto out_volAgr_dmp_acc = out_volAgr_dmp_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                    sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcDeviceMipmapImage_acc = rcDeviceMipmapImage_locker.image().get_access<sycl::float4, sycl::access::mode::read>(cgh);

                    //auto rcDeviceMipmapImage_getTextureObject_ct0 = rcDeviceMipmapImage.getTextureObject();
                    auto rcLevelDim_x_ct1 = (unsigned int)(rcLevelDim.x());
                    auto rcLevelDim_y_ct2 = (unsigned int)(rcLevelDim.y());
                    //auto xzSliceForY_dmpPtr_getBuffer_ct4 = xzSliceForY_dmpPtr->getBuffer();
                    //auto xzSliceForY_dmpPtr_getPitch_ct5 = xzSliceForY_dmpPtr->getPitch();
                    //auto xzSliceForYm1_dmpPtr_getBuffer_ct6 = xzSliceForYm1_dmpPtr->getBuffer();
                    //auto xzSliceForYm1_dmpPtr_getPitch_ct7 = xzSliceForYm1_dmpPtr->getPitch();
                    //auto bestSimInYm1_dmpPtr_getBuffer_ct8 = bestSimInYm1_dmpPtr->getBuffer();
                    //auto out_volAgr_dmp_getBuffer_ct9 = out_volAgr_dmp.getBuffer();
                    //auto out_volAgr_dmp_getBytesPaddedUpToDim_ct10 = out_volAgr_dmp.getBytesPaddedUpToDim(1);
                    //auto out_volAgr_dmp_getBytesPaddedUpToDim_ct11 = out_volAgr_dmp.getBytesPaddedUpToDim(0);
                    auto sgmParams_stepXY_ct14 = sgmParams.stepXY;
                    auto sgmParams_p1_ct16 = sgmParams.p1;
                    auto sgmParams_p2Weighting_ct17 = sgmParams.p2Weighting;

                    cgh.parallel_for(sycl::nd_range<3>(gridVolSlide * blockVolSlide, blockVolSlide),
                                     [=](sycl::nd_item<3> item_ct1)
                                     {
                                         volume_agregateCostVolumeAtXinSlices_kernel(
                                             rcDeviceMipmapImage_acc, 
                                             rcLevelDim_x_ct1,
                                             rcLevelDim_y_ct2, rcMipmapLevel, xzSliceForY_dmpPtr_acc,
                                             //xzSliceForY_dmpPtr_getPitch_ct5, 
                                             xzSliceForYm1_dmpPtr_acc,
                                             //xzSliceForYm1_dmpPtr_getPitch_ct7, 
                                             sampler,
                                             bestSimInYm1_dmpPtr_acc,
                                             out_volAgr_dmp_acc, 
                                             //out_volAgr_dmp_getBytesPaddedUpToDim_ct10,
                                             //out_volAgr_dmp_getBytesPaddedUpToDim_ct11, 
                                             volDim_, axisT_,
                                             sgmParams_stepXY_ct14, y, sgmParams_p1_ct16, sgmParams_p2Weighting_ct17,
                                             ySign, filteringIndex, roi, item_ct1);
                                     });
                });
            volume_aggregateSlice_event.wait();
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
                         const int lastDepthIndex, const ROI& roi, DeviceStream& stream)
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


void cuda_volumeRetrieveBestDepth(CudaDeviceMemoryPitched<float2, 2>& out_sgmDepthThicknessMap_dmp,
                                  CudaDeviceMemoryPitched<float2, 2>& out_sgmDepthSimMap_dmp,
                                  const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp,
                                  const CudaDeviceMemoryPitched<TSim, 3>& in_volSim_dmp,
                                  const int rcDeviceCameraParamsId, const SgmParams& sgmParams, const Range& depthRange,
                                  const ROI& roi, DeviceStream& stream)
try {
    // constant kernel inputs
    const int scaleStep = sgmParams.scale * sgmParams.stepXY;
    const float thicknessMultFactor = 1.f + float(sgmParams.depthThicknessInflate);
    const float maxSimilarity = float(sgmParams.maxSimilarity) * 254.f; // convert from (0, 1) to (0, 254)

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_retrieveBestDepth_kernel);
    const sycl::range<3> grid(1, divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));


    BufferLocker out_sgmDepthThicknessMap_dmp_locker(out_sgmDepthThicknessMap_dmp);
    BufferLocker out_sgmDepthSimMap_dmp_locker(out_sgmDepthSimMap_dmp);
    BufferLocker in_depths_dmp_locker(in_depths_dmp);
    BufferLocker in_volSim_dmp_locker(in_volSim_dmp);

    // kernel execution
    {
        sycl::queue& queue = (sycl::queue&)stream;
        auto retrieveBestDepth_event = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto out_sgmDepthThicknessMap_dmp_acc = out_sgmDepthThicknessMap_dmp_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                auto out_sgmDepthSimMap_dmp_acc = out_sgmDepthSimMap_dmp_locker.buffer().get_access<sycl::access::mode::read_write>(cgh);
                auto in_depths_dmp_acc = in_depths_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                auto in_volSim_dmp_acc = in_volSim_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);

                //__sycl::cameraParametersArray_d->init(*stream);
                //auto constantCameraParametersArray_d_ptr_ct1 = __sycl::cameraParametersArray_d->get_ptr();

                const __sycl::DeviceCameraParams* cameraParametersArray_d = __sycl::cameraParametersArray_d;
                //const __sycl::DevicePatchPattern* patchPattern_d = __sycl::patchPattern_d;

                //auto out_sgmDepthThicknessMap_dmp_getBuffer_ct0 = out_sgmDepthThicknessMap_dmp.getBuffer();
                //auto out_sgmDepthThicknessMap_dmp_getBytesPaddedUpToDim_ct1 =
                //    out_sgmDepthThicknessMap_dmp.getBytesPaddedUpToDim(0);
                //auto out_sgmDepthSimMap_dmp_getBuffer_ct2 = out_sgmDepthSimMap_dmp.getBuffer();
                //auto out_sgmDepthSimMap_dmp_getBytesPaddedUpToDim_ct3 = out_sgmDepthSimMap_dmp.getBytesPaddedUpToDim(0);
                //auto in_depths_dmp_getBuffer_ct4 = in_depths_dmp.getBuffer();
                //auto in_depths_dmp_getBytesPaddedUpToDim_ct5 = in_depths_dmp.getBytesPaddedUpToDim(0);
                //auto in_volSim_dmp_getBuffer_ct6 = in_volSim_dmp.getBuffer();
                //auto in_volSim_dmp_getBytesPaddedUpToDim_ct7 = in_volSim_dmp.getBytesPaddedUpToDim(1);
                //auto in_volSim_dmp_getBytesPaddedUpToDim_ct8 = in_volSim_dmp.getBytesPaddedUpToDim(0);
                auto int_in_volSim_dmp_getSize_z_ct10 = int(in_volSim_dmp.getSize().z());

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1)
                    {
                        volume_retrieveBestDepth_kernel(
                            //out_sgmDepthThicknessMap_dmp_getBuffer_ct0,
                            //out_sgmDepthThicknessMap_dmp_getBytesPaddedUpToDim_ct1,
                            //out_sgmDepthSimMap_dmp_getBuffer_ct2, out_sgmDepthSimMap_dmp_getBytesPaddedUpToDim_ct3,
                            //in_depths_dmp_getBuffer_ct4, in_depths_dmp_getBytesPaddedUpToDim_ct5,
                            //in_volSim_dmp_getBuffer_ct6, in_volSim_dmp_getBytesPaddedUpToDim_ct7,
                            //in_volSim_dmp_getBytesPaddedUpToDim_ct8,
                            out_sgmDepthThicknessMap_dmp_acc,
                            out_sgmDepthSimMap_dmp_acc,
                            in_depths_dmp_acc, in_volSim_dmp_acc,
                            rcDeviceCameraParamsId,
                            int_in_volSim_dmp_getSize_z_ct10, scaleStep, thicknessMultFactor, maxSimilarity, depthRange,
                            roi, item_ct1, cameraParametersArray_d);
                    });
            });
            retrieveBestDepth_event.wait();

    }

} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

extern void cuda_volumeRefineBestDepth(CudaDeviceMemoryPitched<float2, 2>& out_refineDepthSimMap_dmp,
                                       const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthPixSizeMap_dmp,
                                       const CudaDeviceMemoryPitched<TSimRefine, 3>& in_volSim_dmp,
                                       const RefineParams& refineParams, const ROI& roi, DeviceStream& stream)
try {
    // constant kernel inputs
    const int halfNbSamples = refineParams.nbSubsamples * refineParams.halfNbDepths;
    const float twoTimesSigmaPowerTwo = float(2.0 * refineParams.sigma * refineParams.sigma);

    // kernel launch parameters
    const sycl::range<3> block = getMaxPotentialBlockSize(volume_refineBestDepth_kernel);
    const sycl::range<3> grid(1, divUp(roi.height(), block[1]), divUp(roi.width(), block[2]));

    BufferLocker out_refineDepthSimMap_dmp_locker(out_refineDepthSimMap_dmp);
    BufferLocker in_sgmDepthPixSizeMap_dmp_locker(in_sgmDepthPixSizeMap_dmp);
    BufferLocker in_volSim_dmp_locker(in_volSim_dmp);

    // kernel execution
    {
        sycl::queue& queue = (sycl::queue&)stream;
        auto refineBestDepth_event = queue.submit(
            [&](sycl::handler& cgh)
            {
                auto out_refineDepthSimMap_dmp_acc = out_refineDepthSimMap_dmp_locker.buffer().get_access<sycl::access::mode::write>(cgh);
                auto in_sgmDepthPixSizeMap_dmp_acc = in_sgmDepthPixSizeMap_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);
                auto in_volSim_dmp_acc = in_volSim_dmp_locker.buffer().get_access<sycl::access::mode::read>(cgh);

                // auto out_refineDepthSimMap_dmp_getBuffer_ct0 = out_refineDepthSimMap_dmp.getBuffer();
                // auto out_refineDepthSimMap_dmp_getBytesPaddedUpToDim_ct1 =
                //     out_refineDepthSimMap_dmp.getBytesPaddedUpToDim(0);
                // auto in_sgmDepthPixSizeMap_dmp_getBuffer_ct2 = in_sgmDepthPixSizeMap_dmp.getBuffer();
                // auto in_sgmDepthPixSizeMap_dmp_getBytesPaddedUpToDim_ct3 =
                //     in_sgmDepthPixSizeMap_dmp.getBytesPaddedUpToDim(0);
                // auto in_volSim_dmp_getBuffer_ct4 = in_volSim_dmp.getBuffer();
                // auto in_volSim_dmp_getBytesPaddedUpToDim_ct5 = in_volSim_dmp.getBytesPaddedUpToDim(1);
                // auto in_volSim_dmp_getBytesPaddedUpToDim_ct6 = in_volSim_dmp.getBytesPaddedUpToDim(0);
                auto int_in_volSim_dmp_getSize_z_ct7 = int(in_volSim_dmp.getSize().z());

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1)
                                 {
                                     volume_refineBestDepth_kernel(
                                        out_refineDepthSimMap_dmp_acc,
                                        in_sgmDepthPixSizeMap_dmp_acc,
                                        in_volSim_dmp_acc,
                                        //  out_refineDepthSimMap_dmp_getBuffer_ct0,
                                        //  out_refineDepthSimMap_dmp_getBytesPaddedUpToDim_ct1,
                                        //  in_sgmDepthPixSizeMap_dmp_getBuffer_ct2,
                                        //  in_sgmDepthPixSizeMap_dmp_getBytesPaddedUpToDim_ct3,
                                        //  in_volSim_dmp_getBuffer_ct4, in_volSim_dmp_getBytesPaddedUpToDim_ct5,
                                        //  in_volSim_dmp_getBytesPaddedUpToDim_ct6, 
                                         int_in_volSim_dmp_getSize_z_ct7,
                                         refineParams.nbSubsamples, halfNbSamples, refineParams.halfNbDepths,
                                         twoTimesSigmaPowerTwo, roi, item_ct1);
                                 });
            });
            refineBestDepth_event.wait();
    }
} catch(sycl::exception const & e) {
    RETHROW_SYCL_EXCEPTION(e);
}

}
}
