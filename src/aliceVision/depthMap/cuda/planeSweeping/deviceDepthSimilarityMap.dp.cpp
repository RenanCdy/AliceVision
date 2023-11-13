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
