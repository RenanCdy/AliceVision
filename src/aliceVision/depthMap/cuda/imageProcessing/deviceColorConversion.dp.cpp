// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "deviceColorConversion.dp.hpp"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>
#include <aliceVision/depthMap/cuda/device/buffer.dp.hpp>
#include <aliceVision/depthMap/cuda/device/color.dp.hpp>

#include <aliceVision/depthMap/cuda/host/MemoryLocker.hpp>

namespace aliceVision {
namespace depthMap {

void rgb2lab_kernel(sycl::accessor<sycl::ushort4, 2> inout_img_d,
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
    sycl::ushort4 rgb_us = get2DBufferAt(inout_img_d, x, y);

    sycl::float3 rgb = make_float3(rgb_us);

    // CudaRGBA (uchar4 or float4) in range (0, 255)
    // rgb2xyz needs RGB in range (0, 1)
    constexpr float d = 1 / 255.f;

    // compute output CIELAB
    // RGB(0, 255) to XYZ(0, 1) to CIELAB(0, 255)
    sycl::float3 flab = xyz2lab(rgb2xyz(rgb * d));

    // write output CIELAB
    store_half4(flab, inout_img_d, x, y);
}

void cuda_rgb2lab(CudaDeviceMemoryPitched<CudaRGBA, 2>& inout_img_dmp, sycl::queue& stream)
try
{
    BufferLocker lock(inout_img_dmp);

    // kernel launch parameters
    const sycl::range<3> block(1, 2, 32);
    const sycl::range<3> grid(1,
                              divUp(inout_img_dmp.getSize().y(), block[1]),
                              divUp(inout_img_dmp.getSize().x(), block[2])
                            );

    // in-place color conversion from RGB to CIELAB
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream.submit(
        [&](sycl::handler& cgh)
        {
            sycl::accessor inout = lock.buffer().get_access<sycl::access::mode::read_write>(cgh);
            auto pitch = (unsigned int)inout_img_dmp.getPitch();
            auto width = (unsigned int)inout_img_dmp.getSize().x();
            auto height = (unsigned int)inout_img_dmp.getSize().y();

            cgh.parallel_for(sycl::nd_range(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 rgb2lab_kernel(inout, pitch, width, height, item_ct1);
                             });
        }).wait();
} catch(sycl::exception const& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

} // namespace depthMap
} // namespace aliceVision
