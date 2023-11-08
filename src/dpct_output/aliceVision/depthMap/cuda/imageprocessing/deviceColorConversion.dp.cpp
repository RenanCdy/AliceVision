// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "deviceColorConversion.hpp"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>
#include "aliceVision/depthMap/cuda/device/buffer.dp.hpp"
#include "aliceVision/depthMap/cuda/device/color.dp.hpp"

namespace aliceVision {
namespace depthMap {

void rgb2lab_kernel(CudaRGBA* inout_img_d,
                               unsigned int inout_img_p,
                               unsigned int width,
                               unsigned int height,
                               const sycl::nd_item<3> &item_ct1)
{
    const unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if((x >= width) || (y >= height))
        return;

    // corresponding input CudaRGBA
    CudaRGBA* rgb = get2DBufferAt(inout_img_d, inout_img_p, x, y);

    // CudaRGBA (uchar4 or float4) in range (0, 255)
    // rgb2xyz needs RGB in range (0, 1)
    constexpr float d = 1 / 255.f;

    // compute output CIELAB
    // RGB(0, 255) to XYZ(0, 1) to CIELAB(0, 255)
    sycl::float3 flab = xyz2lab(rgb2xyz(sycl::float3(float(rgb->x) * d, float(rgb->y) * d, float(rgb->z) * d)));

    // write output CIELAB
    rgb->x = flab.x();
    rgb->y = flab.y();
    rgb->z = flab.z();
}

void cuda_rgb2lab(CudaDeviceMemoryPitched<CudaRGBA, 2>& inout_img_dmp, dpct::queue_ptr stream)
try
{
    // kernel launch parameters
    const sycl::range<3> block(1, 2, 32);
    const sycl::range<3> grid(divUp(inout_img_dmp.getSize().x(), block[2]),
                              divUp(inout_img_dmp.getSize().y(), block[1]), 1);

    // in-place color conversion from RGB to CIELAB
    /*
    FIXED-DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);
    stream->submit(
        [&](sycl::handler& cgh)
        {
            auto inout_img_dmp_getBuffer_ct0 = inout_img_dmp.getBuffer();
            auto inout_img_dmp_getPitch_ct1 = (unsigned int)inout_img_dmp.getPitch();
            auto inout_img_dmp_getSize_x_ct2 = (unsigned int)inout_img_dmp.getSize().x();
            auto inout_img_dmp_getSize_y_ct3 = (unsigned int)inout_img_dmp.getSize().y();

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 rgb2lab_kernel(inout_img_dmp_getBuffer_ct0, inout_img_dmp_getPitch_ct1,
                                                inout_img_dmp_getSize_x_ct2, inout_img_dmp_getSize_y_ct3, item_ct1);
                             });
        });
} catch(sycl::exception const& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

} // namespace depthMap
} // namespace aliceVision
