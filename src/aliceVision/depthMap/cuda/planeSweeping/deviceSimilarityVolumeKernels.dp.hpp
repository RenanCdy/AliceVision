// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/mvsData/ROI.hpp>
#include <aliceVision/depthMap/cuda/device/buffer.dp.hpp>
#include <aliceVision/depthMap/cuda/planeSweeping/similarity.hpp>

namespace aliceVision {
namespace depthMap {

template <typename Accessor, typename T>
void volume_init_kernel(Accessor& inout_volume_d_acc,
                                   const unsigned int volDimX,
                                   const unsigned int volDimY,
                                   const T value, const sycl::nd_item<3> &item_ct1)
{
    const unsigned int vx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int vy = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const unsigned int vz = item_ct1.get_group(0);

    if(vx >= volDimX || vy >= volDimY)
        return;

    get3DBufferAt(inout_volume_d_acc, vx, vy, vz) = value;
}

}
}
