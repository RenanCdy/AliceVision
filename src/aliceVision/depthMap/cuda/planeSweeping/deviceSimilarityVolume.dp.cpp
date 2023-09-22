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

}
}