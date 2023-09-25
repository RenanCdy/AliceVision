// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "aliceVision/depthMap/cuda/device/buffer.dp.hpp"
#include "aliceVision/depthMap/cuda/device/color.dp.hpp"
#include "aliceVision/depthMap/cuda/device/matrix.dp.hpp"
#include "aliceVision/depthMap/cuda/device/SimStat.dp.hpp"
#include <aliceVision/depthMap/cuda/device/DeviceCameraParams.hpp>
#include <aliceVision/depthMap/cuda/device/DevicePatchPattern.hpp>

namespace aliceVision {
namespace depthMap {

struct Patch
{
    sycl::float3 p; //< 3d point
    sycl::float3 n; //< normal
    sycl::float3 x; //< x axis
    sycl::float3 y; //< y axis
    float d;  //< pixel size
};

/**
 * @brief Compute Normalized Cross-Correlation of a full square patch at given half-width.
 *
 * @tparam TInvertAndFilter invert and filter output similarity value
 *
 * @param[in] rcDeviceCameraParamsId the R camera parameters in device constant memory array
 * @param[in] tcDeviceCameraParamsId the T camera parameters in device constant memory array
 * @param[in] rcMipmapImage_tex the R camera mipmap image texture
 * @param[in] tcMipmapImage_tex the T camera mipmap image texture
 * @param[in] rcLevelWidth the R camera image width at given mipmapLevel
 * @param[in] rcLevelHeight the R camera image height at given mipmapLevel
 * @param[in] tcLevelWidth the T camera image width at given mipmapLevel
 * @param[in] tcLevelHeight the T camera image height at given mipmapLevel
 * @param[in] mipmapLevel the workflow current mipmap level (e.g. SGM=1.f, Refine=0.f)
 * @param[in] wsh the half-width of the patch
 * @param[in] invGammaC the inverted strength of grouping by color similarity
 * @param[in] invGammaP the inverted strength of grouping by proximity
 * @param[in] useConsistentScale enable consistent scale patch comparison
 * @param[in] tcLevelWidth the T camera image width at given mipmapLevel
 * @param[in] patch the input patch struct
 *
 * @return similarity value in range (-1.f, 0.f) or (0.f, 1.f) if TinvertAndFilter enabled
 *         special cases:
 *          -> infinite similarity value: 1
 *          -> invalid/uninitialized/masked similarity: CUDART_INF_F
 */
template <bool TInvertAndFilter>
/*
DPCT1110:2: The total declared local variable size in device function compNCCby3DptsYK exceeds 128 bytes and may cause
high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
inline float compNCCby3DptsYK(const DeviceCameraParams& rcDeviceCamParams, const DeviceCameraParams& tcDeviceCamParams,
                              const dpct::image_accessor_ext << dependent type >, 2 > rcMipmapImage_tex,
                              const dpct::image_accessor_ext << dependent type >, 2 > tcMipmapImage_tex,
                              const unsigned int rcLevelWidth, const unsigned int rcLevelHeight,
                              const unsigned int tcLevelWidth, const unsigned int tcLevelHeight,
                              const float mipmapLevel, const int wsh, const float invGammaC, const float invGammaP,
                              const bool useConsistentScale, const Patch& patch)
{
    // get R and T image 2d coordinates from patch center 3d point
    const sycl::float2 rp = project3DPoint(rcDeviceCamParams.P, patch.p);
    const sycl::float2 tp = project3DPoint(tcDeviceCamParams.P, patch.p);

    // image 2d coordinates margin
    const float dd = wsh + 2.0f; // TODO: FACA

    // check R and T image 2d coordinates
    if((rp.x() < dd) || (rp.x() > float(rcLevelWidth - 1) - dd) || (tp.x() < dd) ||
       (tp.x() > float(tcLevelWidth - 1) - dd) || (rp.y() < dd) || (rp.y() > float(rcLevelHeight - 1) - dd) ||
       (tp.y() < dd) || (tp.y() > float(tcLevelHeight - 1) - dd))
    {
        return sycl::bit_cast<float, int>(0x7f800000U); // uninitialized
    }

    // compute inverse width / height
    // note: useful to compute normalized coordinates
    const float rcInvLevelWidth  = 1.f / float(rcLevelWidth);
    const float rcInvLevelHeight = 1.f / float(rcLevelHeight);
    const float tcInvLevelWidth  = 1.f / float(tcLevelWidth);
    const float tcInvLevelHeight = 1.f / float(tcLevelHeight);

    // initialize R and T mipmap image level at the given mipmap image level
    float rcMipmapLevel = mipmapLevel;
    float tcMipmapLevel = mipmapLevel;

    // update R and T mipmap image level in order to get consistent scale patch comparison
    if(useConsistentScale)
    {
        computeRcTcMipmapLevels(rcMipmapLevel, tcMipmapLevel, mipmapLevel, rcDeviceCamParams, tcDeviceCamParams, rp, tp, patch.p);
    }

    // create and initialize SimStat struct
    simStat sst;

    // compute patch center color (CIELAB) at R and T mipmap image level
    const sycl::float4 rcCenterColor =
        _tex2DLod<sycl::float4>(rcMipmapImage_tex, rp.x(), rcLevelWidth, rp.y(), rcLevelHeight, rcMipmapLevel);
    const sycl::float4 tcCenterColor =
        _tex2DLod<sycl::float4>(tcMipmapImage_tex, tp.x(), tcLevelWidth, tp.y(), tcLevelHeight, tcMipmapLevel);

    // check the alpha values of the patch pixel center of the R and T cameras
    if(rcCenterColor.w() < ALICEVISION_DEPTHMAP_RC_MIN_ALPHA || tcCenterColor.w() < ALICEVISION_DEPTHMAP_TC_MIN_ALPHA)
    {
        return sycl::bit_cast<float, int>(0x7f800000U); // masked
    }

    // compute patch (wsh*2+1)x(wsh*2+1)
    for(int yp = -wsh; yp <= wsh; ++yp)
    {
        for(int xp = -wsh; xp <= wsh; ++xp)
        {
            // get 3d point
            const sycl::float3 p = dpct_operator_overloading::operator+(
                dpct_operator_overloading::operator+(
                    patch.p, dpct_operator_overloading::operator*(patch.x, float(patch.d * float(xp)))),
                dpct_operator_overloading::operator*(patch.y, float(patch.d * float(yp))));

            // get R and T image 2d coordinates from 3d point
            const sycl::float2 rpc = project3DPoint(rcDeviceCamParams.P, p);
            const sycl::float2 tpc = project3DPoint(tcDeviceCamParams.P, p);

            // get R and T image color (CIELAB) from 2d coordinates
            const sycl::float4 rcPatchCoordColor = _tex2DLod<sycl::float4>(rcMipmapImage_tex, rpc.x(), rcLevelWidth,
                                                                           rpc.y(), rcLevelHeight, rcMipmapLevel);
            const sycl::float4 tcPatchCoordColor = _tex2DLod<sycl::float4>(tcMipmapImage_tex, tpc.x(), tcLevelWidth,
                                                                           tpc.y(), tcLevelHeight, tcMipmapLevel);

            // compute weighting based on:
            // - color difference to the center pixel of the patch:
            //    - low value (close to 0) means that the color is different from the center pixel (ie. strongly supported surface)
            //    - high value (close to 1) means that the color is close the center pixel (ie. uniform color)
            // - distance in image to the center pixel of the patch:
            //    - low value (close to 0) means that the pixel is close to the center of the patch
            //    - high value (close to 1) means that the pixel is far from the center of the patch
            const float w = CostYKfromLab(xp, yp, rcCenterColor, rcPatchCoordColor, invGammaC, invGammaP) * CostYKfromLab(xp, yp, tcCenterColor, tcPatchCoordColor, invGammaC, invGammaP);

            // update simStat
            sst.update(rcPatchCoordColor.x(), tcPatchCoordColor.x(), w);
        }
    }

    if(TInvertAndFilter)
    {
        // compute patch similarity
        const float fsim = sst.computeWSim();

        // invert and filter similarity
        // apply sigmoid see: https://www.desmos.com/calculator/skmhf1gpyf
        // best similarity value was -1, worst was 0
        // best similarity value is 1, worst is still 0
        return sigmoid(0.0f, 1.0f, 0.7f, -0.7f, fsim);
    }

    // compute output patch similarity
    return sst.computeWSim();
}

/**
 * @brief Compute Normalized Cross-Correlation of a patch with an user custom patch pattern.
 *
 * @tparam TInvertAndFilter invert and filter output similarity value
 *
 * @param[in] rcDeviceCameraParamsId the R camera parameters in device constant memory array
 * @param[in] tcDeviceCameraParamsId the T camera parameters in device constant memory array
 * @param[in] rcMipmapImage_tex the R camera mipmap image texture
 * @param[in] tcMipmapImage_tex the T camera mipmap image texture
 * @param[in] rcLevelWidth the R camera image width at given mipmapLevel
 * @param[in] rcLevelHeight the R camera image height at given mipmapLevel
 * @param[in] tcLevelWidth the T camera image width at given mipmapLevel
 * @param[in] tcLevelHeight the T camera image height at given mipmapLevel
 * @param[in] mipmapLevel the workflow current mipmap level (e.g. SGM=1.f, Refine=0.f)
 * @param[in] invGammaC the inverted strength of grouping by color similarity
 * @param[in] invGammaP the inverted strength of grouping by proximity
 * @param[in] useConsistentScale enable consistent scale patch comparison
 * @param[in] patch the input patch struct
 *
 * @return similarity value in range (-1.f, 0.f) or (0.f, 1.f) if TinvertAndFilter enabled
 *         special cases:
 *          -> infinite similarity value: 1
 *          -> invalid/uninitialized/masked similarity: CUDART_INF_F
 */
template <bool TInvertAndFilter>
/*
DPCT1110:3: The total declared local variable size in device function compNCCby3DptsYK_customPatchPattern exceeds 128
bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available
and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
inline float compNCCby3DptsYK_customPatchPattern(
    const DeviceCameraParams& rcDeviceCamParams, const DeviceCameraParams& tcDeviceCamParams,
    const dpct::image_accessor_ext << dependent type >, 2 > rcMipmapImage_tex,
    const dpct::image_accessor_ext << dependent type >, 2 > tcMipmapImage_tex, const unsigned int rcLevelWidth,
    const unsigned int rcLevelHeight, const unsigned int tcLevelWidth, const unsigned int tcLevelHeight,
    const float mipmapLevel, const float invGammaC, const float invGammaP, const bool useConsistentScale,
    const Patch& patch, DevicePatchPattern constantPatchPattern_d)
{
    // get R and T image 2d coordinates from patch center 3d point
    const sycl::float2 rp = project3DPoint(rcDeviceCamParams.P, patch.p);
    const sycl::float2 tp = project3DPoint(tcDeviceCamParams.P, patch.p);

    // image 2d coordinates margin
    const float dd = 2.f; // TODO: proper wsh handling

    // check R and T image 2d coordinates
    if((rp.x() < dd) || (rp.x() > float(rcLevelWidth - 1) - dd) || (tp.x() < dd) ||
       (tp.x() > float(tcLevelWidth - 1) - dd) || (rp.y() < dd) || (rp.y() > float(rcLevelHeight - 1) - dd) ||
       (tp.y() < dd) || (tp.y() > float(tcLevelHeight - 1) - dd))
    {
        return sycl::bit_cast<float, int>(0x7f800000U); // uninitialized
    }

    // compute inverse width / height
    // note: useful to compute normalized coordinates
    const float rcInvLevelWidth  = 1.f / float(rcLevelWidth);
    const float rcInvLevelHeight = 1.f / float(rcLevelHeight);
    const float tcInvLevelWidth  = 1.f / float(tcLevelWidth);
    const float tcInvLevelHeight = 1.f / float(tcLevelHeight);

    // get patch center pixel alpha at the given mipmap image level
    const float rcAlpha = _tex2DLod<float4>(rcMipmapImage_tex, rp.x(), rcLevelWidth, rp.y(), rcLevelHeight, mipmapLevel)
                              .w(); // alpha only
    const float tcAlpha = _tex2DLod<float4>(tcMipmapImage_tex, tp.x(), tcLevelWidth, tp.y(), tcLevelHeight, mipmapLevel)
                              .w(); // alpha only

    // check the alpha values of the patch pixel center of the R and T cameras
    if(rcAlpha < ALICEVISION_DEPTHMAP_RC_MIN_ALPHA || tcAlpha < ALICEVISION_DEPTHMAP_TC_MIN_ALPHA)
    {
        return sycl::bit_cast<float, int>(0x7f800000U); // masked
    }

    // initialize R and T mipmap image level at the given mipmap image level
    float rcMipmapLevel = mipmapLevel;
    float tcMipmapLevel = mipmapLevel;

    // update R and T mipmap image level in order to get consistent scale patch comparison
    if(useConsistentScale)
    {
        computeRcTcMipmapLevels(rcMipmapLevel, tcMipmapLevel, mipmapLevel, rcDeviceCamParams, tcDeviceCamParams, rp, tp, patch.p);
    }

    // output similarity initialization
    float fsim = 0.f;
    float wsum = 0.f;

    for(int s = 0; s < constantPatchPattern_d.nbSubparts; ++s)
    {
        // create and initialize patch subpart SimStat
        simStat sst;

        // get patch pattern subpart
        const DevicePatchPatternSubpart& subpart = constantPatchPattern_d.subparts[s];

        // compute patch center color (CIELAB) at subpart level resolution
        const sycl::float4 rcCenterColor = _tex2DLod<sycl::float4>(rcMipmapImage_tex, rp.x(), rcLevelWidth, rp.y(),
                                                                   rcLevelHeight, rcMipmapLevel + subpart.level);
        const sycl::float4 tcCenterColor = _tex2DLod<sycl::float4>(tcMipmapImage_tex, tp.x(), tcLevelWidth, tp.y(),
                                                                   tcLevelHeight, tcMipmapLevel + subpart.level);

        if(subpart.isCircle)
        {
            for(int c = 0; c < subpart.nbCoordinates; ++c)
            {
                // get patch relative coordinates
                const sycl::float2& relativeCoord = subpart.coordinates[c];

                // get 3d point from relative coordinates
                const sycl::float3 p = dpct_operator_overloading::operator+(
                    dpct_operator_overloading::operator+(
                        patch.p, dpct_operator_overloading::operator*(patch.x, float(patch.d * relativeCoord.x()))),
                    dpct_operator_overloading::operator*(patch.y, float(patch.d * relativeCoord.y())));

                // get R and T image 2d coordinates from 3d point
                const sycl::float2 rpc = project3DPoint(rcDeviceCamParams.P, p);
                const sycl::float2 tpc = project3DPoint(tcDeviceCamParams.P, p);

                // get R and T image color (CIELAB) from 2d coordinates
                const sycl::float4 rcPatchCoordColor = _tex2DLod<sycl::float4>(
                    rcMipmapImage_tex, rpc.x(), rcLevelWidth, rpc.y(), rcLevelHeight, rcMipmapLevel + subpart.level);
                const sycl::float4 tcPatchCoordColor = _tex2DLod<sycl::float4>(
                    tcMipmapImage_tex, tpc.x(), tcLevelWidth, tpc.y(), tcLevelHeight, tcMipmapLevel + subpart.level);

                // compute weighting based on color difference to the center pixel of the patch:
                // - low value (close to 0) means that the color is different from the center pixel (ie. strongly supported surface)
                // - high value (close to 1) means that the color is close the center pixel (ie. uniform color)
                const float w = CostYKfromLab(rcCenterColor, rcPatchCoordColor, invGammaC) * CostYKfromLab(tcCenterColor, tcPatchCoordColor, invGammaC);

                // update simStat
                sst.update(rcPatchCoordColor.x(), tcPatchCoordColor.x(), w);
            }
        }
        else // full patch pattern
        {
            for(int yp = -subpart.wsh; yp <= subpart.wsh; ++yp)
            {
                for(int xp = -subpart.wsh; xp <= subpart.wsh; ++xp)
                {
                    // get 3d point
                    const sycl::float3 p = dpct_operator_overloading::operator+(
                        dpct_operator_overloading::operator+(
                            patch.p, dpct_operator_overloading::operator*(
                                         patch.x, float(patch.d * float(xp) * subpart.downscale))),
                        dpct_operator_overloading::operator*(patch.y, float(patch.d * float(yp) * subpart.downscale)));

                    // get R and T image 2d coordinates from 3d point
                    const sycl::float2 rpc = project3DPoint(rcDeviceCamParams.P, p);
                    const sycl::float2 tpc = project3DPoint(tcDeviceCamParams.P, p);

                    // get R and T image color (CIELAB) from 2d coordinates
                    const sycl::float4 rcPatchCoordColor =
                        _tex2DLod<sycl::float4>(rcMipmapImage_tex, rpc.x(), rcLevelWidth, rpc.y(), rcLevelHeight,
                                                rcMipmapLevel + subpart.level);
                    const sycl::float4 tcPatchCoordColor =
                        _tex2DLod<sycl::float4>(tcMipmapImage_tex, tpc.x(), tcLevelWidth, tpc.y(), tcLevelHeight,
                                                tcMipmapLevel + subpart.level);

                    // compute weighting based on:
                    // - color difference to the center pixel of the patch:
                    //    - low value (close to 0) means that the color is different from the center pixel (ie. strongly supported surface)
                    //    - high value (close to 1) means that the color is close the center pixel (ie. uniform color)
                    // - distance in image to the center pixel of the patch:
                    //    - low value (close to 0) means that the pixel is close to the center of the patch
                    //    - high value (close to 1) means that the pixel is far from the center of the patch
                    const float w = CostYKfromLab(xp, yp, rcCenterColor, rcPatchCoordColor, invGammaC, invGammaP) * CostYKfromLab(xp, yp, tcCenterColor, tcPatchCoordColor, invGammaC, invGammaP);

                    // update simStat
                    sst.update(rcPatchCoordColor.x(), tcPatchCoordColor.x(), w);
                }
            }
        }

        // compute patch subpart similarity
        const float fsimSubpart = sst.computeWSim();

        // similarity value in range (-1.f, 0.f) or invalid
        if(fsimSubpart < 0.f)
        {
            // add patch pattern subpart similarity to patch similarity
            if(TInvertAndFilter)
            {
                // invert and filter similarity
                // apply sigmoid see: https://www.desmos.com/calculator/skmhf1gpyf
                // best similarity value was -1, worst was 0
                // best similarity value is 1, worst is still 0
                const float fsimInverted = sigmoid(0.0f, 1.0f, 0.7f, -0.7f, fsimSubpart);
                fsim += fsimInverted * subpart.weight;

            }
            else
            {
                // weight and add similarity
                fsim += fsimSubpart * subpart.weight;
            }

            // sum subpart weight
            wsum += subpart.weight;
        }
    }

    // invalid patch similarity
    if(wsum == 0.f)
    {
        return sycl::bit_cast<float, int>(0x7f800000U);
    }

    if(TInvertAndFilter)
    {
        // for now, we do not average
        return fsim;
    }

    // output average similarity
    return (fsim / wsum);
}

} // namespace depthMap
} // namespace aliceVision
