// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/mvsData/ROI.hpp>
#include <aliceVision/depthMap/cuda/device/buffer.dp.hpp>
#include <aliceVision/depthMap/cuda/device/Patch.dp.hpp>
#include <aliceVision/depthMap/cuda/planeSweeping/similarity.hpp>
#include <aliceVision/depthMap/cuda/device/DeviceCameraParams.dp.hpp>

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

/*
DPCT1110:0: The total declared local variable size in device function volume_computeSimilarity_kernel exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void volume_computeSimilarity_kernel(
    sycl::accessor<TSim, 3, sycl::access::mode::write> out_volume1st,
    sycl::accessor<TSim, 3, sycl::access::mode::write> out_volume2nd_d,
    sycl::accessor<float, 2, sycl::access::mode::read> in_depths_d,
    // TSim* out_volume1st_d, int out_volume1st_s, int out_volume1st_p, 
    // TSim* out_volume2nd_d, int out_volume2nd_s, int out_volume2nd_p,  
    // const float* in_depths_d, const int in_depths_p, 
    const int rcDeviceCameraParamsId,
    const int tcDeviceCameraParamsId,
    const __sycl::DeviceCameraParams* cameraParametersArray_d,
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read> rcMipmapImage_tex,
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read> tcMipmapImage_tex,
    // const unsigned int rcSgmLevelWidth, const unsigned int rcSgmLevelHeight, 
    // const unsigned int tcSgmLevelWidth, const unsigned int tcSgmLevelHeight, 
    const float rcMipmapLevel, const int stepXY, const int wsh,
    const float invGammaC, const float invGammaP, const bool useConsistentScale, const bool useCustomPatchPattern,
    const Range depthRange, const ROI roi, const sycl::nd_item<3>& item_ct1)
{
    const unsigned int roiX = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int roiY = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const unsigned int roiZ = item_ct1.get_group(0);

    if(roiX >= roi.width() || roiY >= roi.height()) // no need to check roiZ
        return;

    // R and T camera parameters
    const __sycl::DeviceCameraParams& rcDeviceCamParams = cameraParametersArray_d[rcDeviceCameraParamsId];
    const __sycl::DeviceCameraParams& tcDeviceCamParams = cameraParametersArray_d[tcDeviceCameraParamsId];

    // corresponding volume coordinates
    const unsigned int vx = roiX;
    const unsigned int vy = roiY;
    const unsigned int vz = depthRange.begin + roiZ;

    // corresponding image coordinates
    const float x = float(roi.x.begin + vx) * float(stepXY);
    const float y = float(roi.y.begin + vy) * float(stepXY);

    // corresponding depth plane
    const float depthPlane = get2DBufferAt(in_depths_d, size_t(vz), 0);

    // compute patch
    Patch patch;
    volume_computePatch(patch, rcDeviceCamParams, tcDeviceCamParams, depthPlane, sycl::float2(x, y));

    // we do not need positive and filtered similarity values
    constexpr bool invertAndFilter = false;

    float fsim = sycl::bit_cast<float, int>(0x7f800000U);

    // compute patch similarity
    if(useCustomPatchPattern)
    {
        fsim = compNCCby3DptsYK_customPatchPattern<invertAndFilter>(rcDeviceCamParams,
                                                                    tcDeviceCamParams,
                                                                    rcMipmapImage_tex,
                                                                    tcMipmapImage_tex,
                                                                    rcSgmLevelWidth,
                                                                    rcSgmLevelHeight,
                                                                    tcSgmLevelWidth,
                                                                    tcSgmLevelHeight,
                                                                    rcMipmapLevel,
                                                                    invGammaC,
                                                                    invGammaP,
                                                                    useConsistentScale,
                                                                    patch);
    }
    else
    {
        fsim = compNCCby3DptsYK<invertAndFilter>(rcDeviceCamParams,
                                                 tcDeviceCamParams,
                                                 rcMipmapImage_tex,
                                                 tcMipmapImage_tex,
                                                 rcSgmLevelWidth,
                                                 rcSgmLevelHeight,
                                                 tcSgmLevelWidth,
                                                 tcSgmLevelHeight,
                                                 rcMipmapLevel,
                                                 wsh,
                                                 invGammaC,
                                                 invGammaP,
                                                 useConsistentScale,
                                                 patch);
    }

    if(fsim == sycl::bit_cast<float, int>(0x7f800000U)) // invalid similarity
    {
      fsim = 255.0f; // 255 is the invalid similarity value
    }
    else // valid similarity
    {
      // remap similarity value
      constexpr const float fminVal = -1.0f;
      constexpr const float fmaxVal = 1.0f;
      constexpr const float fmultiplier = 1.0f / (fmaxVal - fminVal);

      fsim = (fsim - fminVal) * fmultiplier;

#ifdef TSIM_USE_FLOAT
      // no clamp
#else
      fsim = sycl::fmin(1.0f, sycl::fmax(0.0f, fsim));
#endif
      // convert from (0, 1) to (0, 254)
      // needed to store in the volume in uchar
      // 255 is reserved for the similarity initialization, i.e. undefined values
      fsim *= 254.0f;
    }

    TSim* fsim_1st = get3DBufferAt(out_volume1st_d, out_volume1st_s, out_volume1st_p, size_t(vx), size_t(vy), size_t(vz));
    TSim* fsim_2nd = get3DBufferAt(out_volume2nd_d, out_volume2nd_s, out_volume2nd_p, size_t(vx), size_t(vy), size_t(vz));

    if(fsim < *fsim_1st)
    {
        *fsim_2nd = *fsim_1st;
        *fsim_1st = TSim(fsim);
    }
    else if(fsim < *fsim_2nd)
    {
        *fsim_2nd = TSim(fsim);
    }
}

}
}
