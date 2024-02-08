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
#include <aliceVision/depthMap/cuda/device/customDataType.dp.hpp>

namespace aliceVision {
namespace depthMap {

inline void move3DPointByRcPixSize(sycl::float3& p, const __sycl::DeviceCameraParams& rcDeviceCamParams, const float rcPixSize)
{
    sycl::float3 rpv = p - rcDeviceCamParams.C;
    normalize(rpv);
    p = p + rpv * rcPixSize;
}

inline float depthPlaneToDepth(const __sycl::DeviceCameraParams& deviceCamParams, const float fpPlaneDepth,
                               const sycl::float2& pix)
{
    const sycl::float3 planep = deviceCamParams.C + deviceCamParams.ZVect * fpPlaneDepth;
    sycl::float3 v = M3x3mulV2(deviceCamParams.iP, pix);
    normalize(v);
    sycl::float3 p = linePlaneIntersect(deviceCamParams.C, v, planep, deviceCamParams.ZVect);
    return size(deviceCamParams.C - p);
}

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

inline void volume_computePatch(Patch& patch, const __sycl::DeviceCameraParams& rcDeviceCamParams,
                                const __sycl::DeviceCameraParams& tcDeviceCamParams, const float fpPlaneDepth,
                                const sycl::float2& pix)
{
    patch.p = get3DPointForPixelAndFrontoParellePlaneRC(rcDeviceCamParams, pix, fpPlaneDepth);
    patch.d = computePixSize(rcDeviceCamParams, patch.p);
    computeRotCSEpip(patch, rcDeviceCamParams, tcDeviceCamParams);
}

void volume_updateUninitialized_kernel(
                                                sycl::accessor<TSim, 3, sycl::access::mode::read_write> inout_volume2nd_d,
                                                sycl::accessor<TSim, 3, sycl::access::mode::read> in_volume1st_d,
                                                  const unsigned int volDimX,
                                                  const unsigned int volDimY, const sycl::nd_item<3> &item_ct1)
{
    const unsigned int vx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int vy = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const unsigned int vz = item_ct1.get_group(0);

    if(vx >= volDimX || vy >= volDimY)
        return;

    // input/output second best similarity value
    TSim& inout_sim = get3DBufferAt(inout_volume2nd_d, size_t(vx), size_t(vy), size_t(vz));

    if(inout_sim >= 255.f) // invalid or uninitialized similarity value
    {
        // update second best similarity value with first best similarity value
        inout_sim = get3DBufferAt(in_volume1st_d, size_t(vx), size_t(vy), size_t(vz));
    }
}

/*
DPCT1110:0: The total declared local variable size in device function volume_computeSimilarity_kernel exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void volume_computeSimilarity_kernel(
    // sycl::accessor<sycl::float3, 1, sycl::access::mode::write> out_patch_rc,
    // sycl::accessor<sycl::float3, 1, sycl::access::mode::write> out_patch_tc,
    // sycl::accessor<sycl::float3, 1, sycl::access::mode::write> out_patch_p,
    // sycl::accessor<sycl::float3, 1, sycl::access::mode::write> out_patch_n,
    // sycl::accessor<sycl::float3, 1, sycl::access::mode::write> out_patch_x,
    // sycl::accessor<sycl::float3, 1, sycl::access::mode::write> out_patch_y,
    // sycl::accessor<float, 1, sycl::access::mode::write> out_patch_d,
    // sycl::accessor<sycl::uint3, 1, sycl::access::mode::write> out_patch_roi,
    // sycl::accessor<float, 1, sycl::access::mode::write> out_patch_depth,
    // sycl::accessor<sycl::float2, 1, sycl::access::mode::write> out_patch_xy,

    sycl::accessor<TSim, 3, sycl::access::mode::write> out_volume1st_d,
    sycl::accessor<TSim, 3, sycl::access::mode::write> out_volume2nd_d,
    sycl::accessor<float, 2, sycl::access::mode::read> in_depths_d,
    // TSim* out_volume1st_d, int out_volume1st_s, int out_volume1st_p, 
    // TSim* out_volume2nd_d, int out_volume2nd_s, int out_volume2nd_p,  
    // const float* in_depths_d, const int in_depths_p, 
    const int rcDeviceCameraParamsId,
    const int tcDeviceCameraParamsId,
    const __sycl::DeviceCameraParams* cameraParametersArray_d,
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcMipmapImage_tex,
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> tcMipmapImage_tex,
    sycl::sampler sampler,
    const unsigned int rcSgmLevelWidth, const unsigned int rcSgmLevelHeight, 
    const unsigned int tcSgmLevelWidth, const unsigned int tcSgmLevelHeight, 
    const float rcMipmapLevel, const int stepXY, const int wsh,
    const float invGammaC, const float invGammaP, const bool useConsistentScale, const bool useCustomPatchPattern,
    const __sycl::DevicePatchPattern* const patchPattern_d,
    const Range depthRange, const ROI roi, const sycl::nd_item<3>& item_ct1)
{
    const unsigned int roiX = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int roiY = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const unsigned int roiZ = item_ct1.get_group(0);
    
    // out_patch_rc[item_ct1.get_global_linear_id()] = sycl::float3(-42.0);
    // out_patch_tc[item_ct1.get_global_linear_id()] = sycl::float3(-42.0);
    // out_patch_roi[item_ct1.get_global_linear_id()] = sycl::uint3(roiX,roiY,roiZ);
    // out_patch_x[item_ct1.get_global_linear_id()] = sycl::float3(-42.0);
    // out_patch_y[item_ct1.get_global_linear_id()] = -42.0;
    // out_patch_depth[item_ct1.get_global_linear_id()] = -42.0;
    // out_patch_xy[item_ct1.get_global_linear_id()] = sycl::float2(-42.0, -42.0);

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
    
    
    // out_patch_rc[item_ct1.get_global_linear_id()] = rcDeviceCamParams.C;
    // out_patch_tc[item_ct1.get_global_linear_id()] = tcDeviceCamParams.C;
    // out_patch_p[item_ct1.get_global_linear_id()] = patch.p;
    // out_patch_n[item_ct1.get_global_linear_id()] = patch.n;
    // out_patch_x[item_ct1.get_global_linear_id()] = patch.x;
    // out_patch_y[item_ct1.get_global_linear_id()] = patch.y;
    // out_patch_d[item_ct1.get_global_linear_id()] = patch.d;
    // out_patch_roi[item_ct1.get_global_linear_id()] = sycl::uint3(roiX,roiY,roiZ);
    // out_patch_depth[item_ct1.get_global_linear_id()] = depthPlane;
    // out_patch_xy[item_ct1.get_global_linear_id()] = sycl::float2(x, y);
    // we do not need positive and filtered similarity values
    constexpr bool invertAndFilter = false;

    std::pair<float,bool> fsim = std::make_pair(sycl::bit_cast<float, int>(0x7f800000U),true);
    
    // compute patch similarity
    if(useCustomPatchPattern)
    {
        fsim = compNCCby3DptsYK_customPatchPattern<invertAndFilter>(rcDeviceCamParams,
                                                                    tcDeviceCamParams,
                                                                    rcMipmapImage_tex,
                                                                    tcMipmapImage_tex,
                                                                    sampler,
                                                                    rcSgmLevelWidth,
                                                                    rcSgmLevelHeight,
                                                                    tcSgmLevelWidth,
                                                                    tcSgmLevelHeight,
                                                                    rcMipmapLevel,
                                                                    invGammaC,
                                                                    invGammaP,
                                                                    useConsistentScale,
                                                                    patch,
                                                                    *patchPattern_d);
    }
    else
    {
        fsim = compNCCby3DptsYK<invertAndFilter>(rcDeviceCamParams,
                                                 tcDeviceCamParams,
                                                 rcMipmapImage_tex,
                                                 tcMipmapImage_tex,
                                                 sampler,
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

    if(fsim.second) // invalid similarity
    {
      fsim.first = 255.0f; // 255 is the invalid similarity value
    }
    else // valid similarity
    {
      // remap similarity value
      constexpr const float fminVal = -1.0f;
      constexpr const float fmaxVal = 1.0f;
      constexpr const float fmultiplier = 1.0f / (fmaxVal - fminVal);

      fsim.first = (fsim.first - fminVal) * fmultiplier;

#ifdef TSIM_USE_FLOAT
      // no clamp
#else
      fsim.first = sycl::fmin(1.0f, sycl::fmax(0.0f, fsim.first));
#endif
      // convert from (0, 1) to (0, 254)
      // needed to store in the volume in uchar
      // 255 is reserved for the similarity initialization, i.e. undefined values
      fsim.first *= 254.0f;
    }

    TSim& fsim_1st = get3DBufferAt(out_volume1st_d, size_t(vx), size_t(vy), size_t(vz));
    TSim& fsim_2nd = get3DBufferAt(out_volume2nd_d, size_t(vx), size_t(vy), size_t(vz));

    if(fsim.first < fsim_1st)
    {
        fsim_2nd = fsim_1st;
        fsim_1st = TSim(fsim.first);
    }
    else if(fsim.first < fsim_2nd)
    {
        fsim_2nd = TSim(fsim.first);
    }
}

void volume_refineBestDepth_kernel(
    sycl::accessor<sycl::float2, 2, sycl::access::mode::write> out_refineDepthSimMap_d,
    sycl::accessor<sycl::float2, 2, sycl::access::mode::read> in_sgmDepthPixSizeMap_d,
    sycl::accessor<TSimRefine, 3, sycl::access::mode::read> in_volSim_d,
    // sycl::float2* out_refineDepthSimMap_d, int out_refineDepthSimMap_p, 
    // const sycl::float2* in_sgmDepthPixSizeMap_d, int in_sgmDepthPixSizeMap_p, 
    // const TSimRefine* in_volSim_d, int in_volSim_s, int in_volSim_p, 
    int volDimZ,
    int samplesPerPixSize, // number of subsamples (samples between two depths)
    int halfNbSamples,     // number of samples (in front and behind mid depth)
    int halfNbDepths,      // number of depths  (in front and behind mid depth) should be equal to (volDimZ - 1) / 2
    float twoTimesSigmaPowerTwo, const ROI roi, const sycl::nd_item<3>& item_ct1)
{
    const unsigned int vx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int vy = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if(vx >= roi.width() || vy >= roi.height())
        return;

    // corresponding input sgm depth/pixSize (middle depth)
    const sycl::float2& in_sgmDepthPixSize = get2DBufferAt(in_sgmDepthPixSizeMap_d, vx, vy);

    // corresponding output depth/sim pointer
    sycl::float2& out_bestDepthSimPtr = get2DBufferAt(out_refineDepthSimMap_d, vx, vy);

    // sgm depth (middle depth) invalid or masked
    if(in_sgmDepthPixSize.x() <= 0.0f)
    {
        out_bestDepthSimPtr.x() = in_sgmDepthPixSize.x(); // -1 (invalid) or -2 (masked)
        out_bestDepthSimPtr.y() = 1.0f;                   // similarity between (-1, +1)
        return;
    }

    // find best z sample per pixel
    float bestSampleSim = 0.f;      // all sample sim <= 0.f
    int bestSampleOffsetIndex = 0;  // default is middle depth (SGM)

    // sliding gaussian window
    for(int sample = -halfNbSamples; sample <= halfNbSamples; ++sample)
    {
        float sampleSim = 0.f; 

        for(int vz = 0; vz < volDimZ; ++vz)
        {
            const int rz = (vz - halfNbDepths);    // relative depth index offset
            const int zs = rz * samplesPerPixSize; // relative sample offset

            // get the inverted similarity sum value
            // best value is the HIGHEST
            // worst value is 0
            const float& invSimSum = get3DBufferAt(in_volSim_d, vx, vy, vz);

            // reverse the inverted similarity sum value
            // best value is the LOWEST
            // worst value is 0
            const float simSum = -invSimSum;

            // apply gaussian
            // see: https://www.desmos.com/calculator/ribalnoawq
            sampleSim += simSum * sycl::exp(-((zs - sample) * (zs - sample)) / twoTimesSigmaPowerTwo);
        }

        if(sampleSim < bestSampleSim)
        {
            bestSampleOffsetIndex = sample;
            bestSampleSim = sampleSim;
        }
    }

    // compute sample size
    const float sampleSize = in_sgmDepthPixSize.y() / samplesPerPixSize; // input sgm pixSize / samplesPerPixSize

    // compute sample size offset from z center
    const float sampleSizeOffset = bestSampleOffsetIndex * sampleSize;

    // compute best depth
    // input sgm depth (middle depth) + sample size offset from z center
    const float bestDepth = in_sgmDepthPixSize.x() + sampleSizeOffset;

    // write output best depth/sim
    out_bestDepthSimPtr.x() = bestDepth;
    out_bestDepthSimPtr.y() = bestSampleSim;
}

template <typename T>
void volume_initVolumeYSlice_kernel(//T* volume_d, int volume_s, int volume_p, 
                                    sycl::accessor<T, 3, sycl::access::mode::read_write> volume_d,
                                    const sycl::int3 volDim,
                                    const sycl::int3 axisT, int y, T cst,
                                       const sycl::nd_item<3> &item_ct1)
{
    const int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int z = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    sycl::int3 v;
    (&v.x())[axisT.x()] = x;
    (&v.x())[axisT.y()] = y;
    (&v.x())[axisT.z()] = z;

    if ((x >= 0) && (x < (&volDim.x())[axisT.x()]) && (z >= 0) && (z < (&volDim.x())[axisT.z()]))
    {
        T& volume_zyx = get3DBufferAt(volume_d, v.x(), v.y(), v.z());
        volume_zyx = cst;
    }
}

template <typename T1, typename T2>
void volume_getVolumeXZSlice_kernel(//T1* slice_d, int slice_p, 
                                    sycl::accessor<T1, 2, sycl::access::mode::read_write> slice_d,
                                    sycl::accessor<T2, 3, sycl::access::mode::read> volume_d,
                                    //const T2* volume_d, int volume_s, int volume_p,
                                    const sycl::int3 volDim, const sycl::int3 axisT, int y,
                                       const sycl::nd_item<3> &item_ct1)
{
    const int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int z = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    sycl::int3 v;
    (&v.x())[axisT.x()] = x;
    (&v.x())[axisT.y()] = y;
    (&v.x())[axisT.z()] = z;

    if (x >= (&volDim.x())[axisT.x()] || z >= (&volDim.x())[axisT.z()])
      return;

    const T2& volume_xyz = get3DBufferAt(volume_d, v);
    T1& slice_xz = get2DBufferAt(slice_d, x, z);
    slice_xz = (T1)(volume_xyz);
}

void volume_computeBestZInSlice_kernel(//TSimAcc* xzSlice_d, int xzSlice_p, 
                                    sycl::accessor<TSimAcc, 2, sycl::access::mode::read_write> xzSlice_d,
                                    sycl::accessor<TSimAcc, 2, sycl::access::mode::write> ySliceBestInColCst_d,
                                    //TSimAcc* ySliceBestInColCst_d, 
                                    int volDimX, int volDimZ,
                                    
                                    //sycl::accessor<T2, 3, sycl::access::mode::read> ySliceBestInColCst_d,
                                       const sycl::nd_item<3> &item_ct1)
{
    const int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if(x >= volDimX)
        return;

    TSimAcc bestCst = get2DBufferAt(xzSlice_d, x, 0);

    for(int z = 1; z < volDimZ; ++z)
    {
        const TSimAcc cst = get2DBufferAt(xzSlice_d, x, z);
        bestCst = cst < bestCst ? cst : bestCst;  // min(cst, bestCst);
    }
    ySliceBestInColCst_d[x][0] = bestCst;
}

/**
 * @param[inout] xySliceForZ input similarity plane
 * @param[in] xySliceForZM1
 * @param[in] xSliceBestInColCst
 * @param[out] volSimT output similarity volume
 */
void volume_agregateCostVolumeAtXinSlices_kernel(
    const sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcMipmapImage_tex, const unsigned int rcSgmLevelWidth,
    const unsigned int rcSgmLevelHeight, const float rcMipmapLevel, 
    //TSimAcc* xzSliceForY_d, int xzSliceForY_p,
    sycl::accessor<TSimAcc, 2, sycl::access::mode::read_write> xzSliceForY_d,
    //const TSimAcc* xzSliceForYm1_d, const int xzSliceForYm1_p, 
    sycl::accessor<TSimAcc, 2, sycl::access::mode::read> xzSliceForYm1_d,
    const sycl::sampler sampler,
    sycl::accessor<TSimAcc, 2, sycl::access::mode::read> bestSimInYm1_d,
    //const TSimAcc* bestSimInYm1_d, 
    sycl::accessor<TSim, 3, sycl::access::mode::read_write> volAgr_d,
    //TSim* volAgr_d, const int volAgr_s, const int volAgr_p, 
    const sycl::int3 volDim, const sycl::int3 axisT, const float step,
    const int y, const float P1, const float _P2, const int ySign, const int filteringIndex, const ROI roi,
    const sycl::nd_item<3>& item_ct1)
{
    const int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int z = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    sycl::int3 v;
    (&v.x())[axisT.x()] = x;
    (&v.x())[axisT.y()] = y;
    (&v.x())[axisT.z()] = z;

    if(x >= (&volDim.x())[axisT.x()] || z >= volDim.z())
        return;

    // find texture offset
    const int beginX = (axisT.x() == 0) ? roi.x.begin : roi.y.begin;
    const int beginY = (axisT.x() == 0) ? roi.y.begin : roi.x.begin;

    TSimAcc& sim_xz = get2DBufferAt(xzSliceForY_d, x, z);
    float pathCost = 255.0f;

    if((z >= 1) && (z < volDim.z() - 1))
    {
        float P2 = 0;

        if(_P2 < 0)
        {
          // _P2 convention: use negative value to skip the use of deltaC.
          P2 = sycl::fabs(_P2);
        }
        else
        {
          const int imX0 = (beginX + v.x()) * step; // current
          const int imY0 = (beginY + v.y()) * step;

          const int imX1 = imX0 - ySign * step * (axisT.y() == 0); // M1
          const int imY1 = imY0 - ySign * step * (axisT.y() == 1);

          const sycl::float4 gcr0 = _tex2DLod(rcMipmapImage_tex, sampler, float(imX0), rcSgmLevelWidth,
                                                            float(imY0), rcSgmLevelHeight, rcMipmapLevel);
          const sycl::float4 gcr1 = _tex2DLod(rcMipmapImage_tex, sampler, float(imX1), rcSgmLevelWidth,
                                                            float(imY1), rcSgmLevelHeight, rcMipmapLevel);
          const float deltaC = euclideanDist3(gcr0, gcr1);

          // sigmoid f(x) = i + (a - i) * (1 / ( 1 + e^(10 * (x - P2) / w)))
          // see: https://www.desmos.com/calculator/1qvampwbyx
          // best values found from tests: i = 80, a = 255, w = 80, P2 = 100
          // historical values: i = 15, a = 255, w = 80, P2 = 20
          P2 = sigmoid(80.f, 255.f, 80.f, _P2, deltaC);
        }

        const TSimAcc bestCostInColM1 = bestSimInYm1_d[x][0];
        const TSimAcc pathCostMDM1 = get2DBufferAt(xzSliceForYm1_d, x, z - 1); // M1: minus 1 over depths
        const TSimAcc pathCostMD   = get2DBufferAt(xzSliceForYm1_d, x, z);
        const TSimAcc pathCostMDP1 = get2DBufferAt(xzSliceForYm1_d, x, z + 1); // P1: plus 1 over depths
        const float minCost = multi_fminf(pathCostMD, pathCostMDM1 + P1, pathCostMDP1 + P1, bestCostInColM1 + P2);

        // if 'pathCostMD' is the minimal value of the depth
        pathCost = (sim_xz) + minCost - bestCostInColM1;
    }

    // fill the current slice with the new similarity score
    sim_xz = TSimAcc(pathCost);

#ifndef TSIM_USE_FLOAT
    // clamp if TSim = uchar (TSimAcc = unsigned int)
    pathCost = sycl::fmin(255.0f, sycl::fmax(0.0f, pathCost));
#endif

    // aggregate into the final output
    TSim& volume_xyz = get3DBufferAt(volAgr_d, v.x(), v.y(), v.z());
    const float val = (float(volume_xyz) * float(filteringIndex) + pathCost) / float(filteringIndex + 1);
    volume_xyz = TSim(val);
}


/*
DPCT1110:16: The total declared local variable size in device function volume_refineSimilarity_kernel exceeds 128 bytes
and may cause high register pressure. Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void volume_refineSimilarity_kernel(
    sycl::accessor<TSimRefine, 3, sycl::access::mode::read_write> inout_volSim_d,
    const sycl::accessor<sycl::float2, 2, sycl::access::mode::read> in_sgmDepthPixSizeMap_d,
    sycl::accessor<custom_sycl::custom_float3, 2, sycl::access::mode::read> in_sgmNormalMap_d,
    //TSimRefine* inout_volSim_d, int inout_volSim_s, int inout_volSim_p, 
    //const sycl::float2* in_sgmDepthPixSizeMap_d, const int in_sgmDepthPixSizeMap_p, 
    //const sycl::float3* in_sgmNormalMap_d, const int in_sgmNormalMap_p,
    const int rcDeviceCameraParamsId, const int tcDeviceCameraParamsId,
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> rcMipmapImage_tex,
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read, sycl::access::target::image> tcMipmapImage_tex,
    sycl::sampler sampler,
    // const dpct::image_accessor_ext<sycl::float4, 2> rcMipmapImage_tex,
    // const dpct::image_accessor_ext<sycl::float4, 2> tcMipmapImage_tex, 
    const unsigned int rcRefineLevelWidth,
    const unsigned int rcRefineLevelHeight, const unsigned int tcRefineLevelWidth,
    const unsigned int tcRefineLevelHeight, const float rcMipmapLevel, const int volDimZ, const int stepXY,
    const int wsh, const float invGammaC, const float invGammaP, const bool useConsistentScale,
    const bool useCustomPatchPattern, const Range depthRange, const ROI roi, const sycl::nd_item<3>& item_ct1,
    const __sycl::DeviceCameraParams* cameraParametersArray_d, const __sycl::DevicePatchPattern* const patchPattern_d)
{
    const unsigned int roiX = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int roiY = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const unsigned int roiZ = item_ct1.get_group(0);

    if(roiX >= roi.width() || roiY >= roi.height()) // no need to check roiZ
        return;

    // R and T camera parameters
    const __sycl::DeviceCameraParams& rcDeviceCamParams = cameraParametersArray_d[rcDeviceCameraParamsId];
    const __sycl::DeviceCameraParams& tcDeviceCamParams = cameraParametersArray_d[tcDeviceCameraParamsId];

    // corresponding volume and depth/sim map coordinates
    const unsigned int vx = roiX;
    const unsigned int vy = roiY;
    const unsigned int vz = depthRange.begin + roiZ;

    // corresponding image coordinates
    const float x = float(roi.x.begin + vx) * float(stepXY);
    const float y = float(roi.y.begin + vy) * float(stepXY);

    // corresponding input sgm depth/pixSize (middle depth)
    const sycl::float2& in_sgmDepthPixSize = get2DBufferAt(in_sgmDepthPixSizeMap_d, vx, vy);

    // sgm depth (middle depth) invalid or masked
    if(in_sgmDepthPixSize.x() <= 0.0f)
        return; 

    // initialize rc 3d point at sgm depth (middle depth)
    sycl::float3 p = get3DPointForPixelAndDepthFromRC(rcDeviceCamParams, sycl::float2(x, y), in_sgmDepthPixSize.x());

    // compute relative depth index offset from z center
    const int relativeDepthIndexOffset = vz - ((volDimZ - 1) / 2);

    if(relativeDepthIndexOffset != 0)
    {
        // not z center
        // move rc 3d point by relative depth index offset * sgm pixSize
        const float pixSizeOffset = relativeDepthIndexOffset * in_sgmDepthPixSize.y(); // input sgm pixSize
        move3DPointByRcPixSize(p, rcDeviceCamParams, pixSizeOffset);
    }

    // compute patch
    Patch patch;
    patch.p = p;
    patch.d = computePixSize(rcDeviceCamParams, p);

    // computeRotCSEpip
    {
      // vector from the reference camera to the 3d point
      sycl::float3 v1 = rcDeviceCamParams.C - patch.p;
      // vector from the target camera to the 3d point
      sycl::float3 v2 = tcDeviceCamParams.C - patch.p;
      normalize(v1);
      normalize(v2);

      // y has to be ortogonal to the epipolar plane
      // n has to be on the epipolar plane
      // x has to be on the epipolar plane

      patch.y = cross(v1, v2);
      normalize(patch.y);

        // corresponding to nullptr check
      bool in_sgmNormalMapPtr_is_null = (in_sgmNormalMap_d.get_size() == 0);

      if(!in_sgmNormalMapPtr_is_null) // initialize patch normal from input normal map
      {
        // TODO: not optimized to float3
        auto& temp = get2DBufferAt(in_sgmNormalMap_d, vx, vy);
        patch.n = sycl::vec<float, 3>(temp[0], temp[1], temp[2]);
        //patch.n = get2DBufferAt(in_sgmNormalMap_d, vx, vy);
      }
      else // initialize patch normal from v1 & v2
      {
        patch.n = (v1 + v2) / 2.0f;
        normalize(patch.n);
      }

      patch.x = cross(patch.y, patch.n);
      normalize(patch.x);
    }

    // we need positive and filtered similarity values
    constexpr bool invertAndFilter = true;

    std::pair<float,bool> fsimInvertedFiltered = std::make_pair(sycl::bit_cast<float, int>(0x7f800000U),true);

    // compute similarity
    if(useCustomPatchPattern)
    {
        fsimInvertedFiltered = compNCCby3DptsYK_customPatchPattern<invertAndFilter>(
            rcDeviceCamParams, tcDeviceCamParams, rcMipmapImage_tex, tcMipmapImage_tex, sampler, rcRefineLevelWidth,
            rcRefineLevelHeight, tcRefineLevelWidth, tcRefineLevelHeight, rcMipmapLevel, invGammaC, invGammaP,
            useConsistentScale, patch, *patchPattern_d);
    }
    else
    {
        fsimInvertedFiltered = compNCCby3DptsYK<invertAndFilter>(rcDeviceCamParams,
                                                                 tcDeviceCamParams,
                                                                 rcMipmapImage_tex,
                                                                 tcMipmapImage_tex,
                                                                 sampler,
                                                                 rcRefineLevelWidth,
                                                                 rcRefineLevelHeight,
                                                                 tcRefineLevelWidth,
                                                                 tcRefineLevelHeight,
                                                                 rcMipmapLevel,
                                                                 wsh,
                                                                 invGammaC,
                                                                 invGammaP,
                                                                 useConsistentScale,
                                                                 patch);
    }

    if(fsimInvertedFiltered.second) // invalid similarity
    {
        // do nothing
        return;
    }

    // get output similarity pointer
    TSimRefine& outSimPtr = get3DBufferAt(inout_volSim_d, vx, vy, vz);

    // add the output similarity value
#ifdef TSIM_REFINE_USE_HALF
    // note: using built-in half addition can give bad results on some gpus
    //*outSimPtr = __hadd(*outSimPtr, TSimRefine(fsimInvertedFiltered));
    //*outSimPtr = __hadd(*outSimPtr, __float2half(fsimInvertedFiltered));
    outSimPtr =
        sycl::vec<float, 1>{
            (sycl::vec<sycl::half, 1>{static_cast<sycl::half>(outSimPtr)}.convert<float, sycl::rounding_mode::automatic>()[0] +
             fsimInvertedFiltered.first)}
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0]; // perform the addition in float
#else
   outSimPtr += TSimRefine(fsimInvertedFiltered.first);
#endif
}

void volume_retrieveBestDepth_kernel(sycl::accessor<sycl::float2, 2, sycl::access::mode::read_write> out_sgmDepthThicknessMap_d,
                                     sycl::accessor<sycl::float2, 2, sycl::access::mode::read_write> out_sgmDepthSimMap_d,
                                     //sycl::float2* out_sgmDepthThicknessMap_d, int out_sgmDepthThicknessMap_p,
                                     //sycl::float2* out_sgmDepthSimMap_d,
                                     //int out_sgmDepthSimMap_p, // output depth/sim map is optional (nullptr)
                                     sycl::accessor<float, 2, sycl::access::mode::read> in_depths_d,
                                     sycl::accessor<TSim, 3, sycl::access::mode::read> in_volSim_d,
                                     //const float* in_depths_d, const int in_depths_p, const TSim* in_volSim_d,
                                     //const int in_volSim_s, const int in_volSim_p, 
                                     const int rcDeviceCameraParamsId,
                                     const int volDimZ, // useful for depth/sim interpolation
                                     const int scaleStep,
                                     const float thicknessMultFactor, // default 1
                                     const float maxSimilarity, const Range depthRange, const ROI roi,
                                     const sycl::nd_item<3>& item_ct1,
                                     const __sycl::DeviceCameraParams* cameraParametersArray_d)
{
    const unsigned int vx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int vy = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if(vx >= roi.width() || vy >= roi.height())
        return;

    // R camera parameters
    const __sycl::DeviceCameraParams& rcDeviceCamParams = cameraParametersArray_d[rcDeviceCameraParamsId];

    // corresponding image coordinates
    const sycl::float2 pix{float((roi.x.begin + vx) * scaleStep), float((roi.y.begin + vy) * scaleStep)};

    // corresponding output depth/thickness pointer
    sycl::float2& out_bestDepthThicknessPtr =
        get2DBufferAt(out_sgmDepthThicknessMap_d, vx, vy);

    // corresponding output depth/sim pointer or nullptr
    bool out_bestDepthSimPtr_is_null = (out_sgmDepthSimMap_d.get_size() == 0);
    sycl::float2 placeholderfloat;
    sycl::float2& out_bestDepthSimPtr =
        (out_bestDepthSimPtr_is_null) ? placeholderfloat : get2DBufferAt(out_sgmDepthSimMap_d, vx, vy);

    // find the best depth plane index for the current pixel
    // the best depth plane has the best similarity value
    // - best possible similarity value is 0
    // - worst possible similarity value is 254
    // - invalid similarity value is 255
    float bestSim = 255.f;
    int bestZIdx = -1;

    for(int vz = depthRange.begin; vz < depthRange.end; ++vz)
    {
      const float& simAtZ = get3DBufferAt(in_volSim_d, vx, vy, vz);

      if(simAtZ < bestSim)
      {
        bestSim = simAtZ;
        bestZIdx = vz;
      }
    }

    // filtering out invalid values and values with a too bad score (above the user maximum similarity threshold)
    // note: this helps to reduce following calculations and also the storage volume of the depth maps.
    if((bestZIdx == -1) || (bestSim > maxSimilarity))
    {
        out_bestDepthThicknessPtr.x() = -1.f; // invalid depth
        out_bestDepthThicknessPtr.y() = -1.f; // invalid thickness

        if(!out_bestDepthSimPtr_is_null)
        {
            out_bestDepthSimPtr.x() = -1.f; // invalid depth
            out_bestDepthSimPtr.y() = 1.f;  // worst similarity value
        }
        return;
    }

    // find best depth plane previous and next indexes
    const int bestZIdx_m1 = sycl::max(0, bestZIdx - 1);           // best depth plane previous index
    const int bestZIdx_p1 = sycl::min(volDimZ - 1, bestZIdx + 1); // best depth plane next index

    // get best best depth current, previous and next plane depth values
    // note: float3 struct is useful for depth interpolation
    sycl::float3 depthPlanes;
    depthPlanes.x() = get2DBufferAt(in_depths_d, bestZIdx_m1, 0); // best depth previous plane
    depthPlanes.y() = get2DBufferAt(in_depths_d, bestZIdx, 0);    // best depth plane
    depthPlanes.z() = get2DBufferAt(in_depths_d, bestZIdx_p1, 0); // best depth next plane

    const float bestDepth = depthPlaneToDepth(rcDeviceCamParams, depthPlanes.y(), pix);    // best depth
    const float bestDepth_m1 = depthPlaneToDepth(rcDeviceCamParams, depthPlanes.x(), pix); // previous best depth
    const float bestDepth_p1 = depthPlaneToDepth(rcDeviceCamParams, depthPlanes.z(), pix); // next best depth

#ifdef ALICEVISION_DEPTHMAP_RETRIEVE_BEST_Z_INTERPOLATION
    // with depth/sim interpolation
    // note: disable by default

    float3 sims;
    sims.x = *get3DBufferAt(in_volSim_d, in_volSim_s, in_volSim_p, vx, vy, bestZIdx_m1);
    sims.y = bestSim;
    sims.z = *get3DBufferAt(in_volSim_d, in_volSim_s, in_volSim_p, vx, vy, bestZIdx_p1);

    // convert sims from (0, 255) to (-1, +1)
    sims.x = (sims.x / 255.0f) * 2.0f - 1.0f;
    sims.y = (sims.y / 255.0f) * 2.0f - 1.0f;
    sims.z = (sims.z / 255.0f) * 2.0f - 1.0f;

    // interpolation between the 3 depth planes candidates
    const float refinedDepthPlane = refineDepthSubPixel(depthPlanes, sims);

    const float out_bestDepth = depthPlaneToDepth(rcDeviceCamParams, refinedDepthPlane, pix);
    const float out_bestSim = sims.y;
#else
    // without depth interpolation
    const float out_bestDepth = bestDepth;
    const float out_bestSim = (bestSim / 255.0f) * 2.0f - 1.0f; // convert from (0, 255) to (-1, +1)
#endif

    // compute output best depth thickness
    // thickness is the maximum distance between output best depth and previous or next depth
    // thickness can be inflate with thicknessMultFactor
    const float out_bestDepthThickness =
        sycl::max(bestDepth_p1 - out_bestDepth, out_bestDepth - bestDepth_m1) * thicknessMultFactor;

    // write output depth/thickness
    out_bestDepthThicknessPtr.x() = out_bestDepth;
    out_bestDepthThicknessPtr.y() = out_bestDepthThickness;

    if(!out_bestDepthSimPtr_is_null)
    {
        // write output depth/sim
        out_bestDepthSimPtr.x() = out_bestDepth;
        out_bestDepthSimPtr.y() = out_bestSim;
    }
}

}
}
