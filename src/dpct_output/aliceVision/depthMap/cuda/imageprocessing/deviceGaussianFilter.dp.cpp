// This file is part of the AliceVision project.
// Copyright (c) 2018 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "deviceGaussianFilter.hpp"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>
#include <aliceVision/depthMap/cuda/host/memory.hpp>
#include "aliceVision/depthMap/cuda/device/buffer.dp.hpp"
#include "aliceVision/depthMap/cuda/device/operators.dp.hpp"
#include <cmath>

namespace aliceVision {
namespace depthMap {

/*********************************************************************************
* global / constant data structures
*********************************************************************************/
std::set<int>                 d_gaussianArrayInitialized;
static dpct::constant_memory<int, 1> d_gaussianArrayOffset(MAX_CONSTANT_GAUSS_SCALES);
static dpct::constant_memory<float, 1> d_gaussianArray(MAX_CONSTANT_GAUSS_MEM_SIZE);

/*********************************************************************************
 * device functions definitions
 *********************************************************************************/

void cuda_swap_float(float& a, float& b)
{
    float temp = a;
    a = b;
    b = temp;
}

/*********************************************************************************
 * kernel definitions
 *********************************************************************************/

/*
 * @note This kernel implementation is not optimized because the Gaussian filter is separable.
 */
/*
FIXED-DPCT1050:41: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
*/
void downscaleWithGaussianBlur_kernel(
    dpct::image_accessor_ext<sycl::float4, 2> in_img_tex, CudaRGBA* out_downscaledImg_d,
    int out_downscaledImg_p, unsigned int downscaledImgWidth, unsigned int downscaledImgHeight, int downscale,
    int gaussRadius, const sycl::nd_item<3>& item_ct1, int* d_gaussianArrayOffset, float* d_gaussianArray)
{
    const unsigned int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if((x < downscaledImgWidth) && (y < downscaledImgHeight))
    {
        const float s = float(downscale) * 0.5f;

        sycl::float4 accPix = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
        float sumFactor = 0.0f;

        for(int i = -gaussRadius; i <= gaussRadius; i++)
        {
            for(int j = -gaussRadius; j <= gaussRadius; j++)
            {
                const sycl::float4 curPix =
                    tex2D_float4(in_img_tex, float(x * downscale + j) + s, float(y * downscale + i) + s);
                const float factor =
                    getGauss(downscale - 1, i + gaussRadius, d_gaussianArrayOffset, d_gaussianArray) *
                    getGauss(downscale - 1, j + gaussRadius, d_gaussianArrayOffset, d_gaussianArray); // domain factor

                accPix =
                    dpct_operator_overloading::operator+(accPix, dpct_operator_overloading::operator*(curPix, factor));
                sumFactor += factor;
            }
        }

        CudaRGBA& out = BufPtr<CudaRGBA>(out_downscaledImg_d, out_downscaledImg_p).at(size_t(x), size_t(y));
        out.x = accPix.x() / sumFactor;
        out.y = accPix.y() / sumFactor;
        out.z = accPix.z() / sumFactor;
        out.w = accPix.w() / sumFactor;
    }
}

void gaussianBlurVolumeZ_kernel(float* out_volume_d, int out_volume_s, int out_volume_p, 
                                           const float* in_volume_d, int in_volume_s, int in_volume_p, 
                                           int volDimX, int volDimY, int volDimZ, int gaussRadius,
                                           const sycl::nd_item<3> &item_ct1, int *d_gaussianArrayOffset,
                                           float *d_gaussianArray)
{
    const int vx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int vy = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const int vz = item_ct1.get_group(0);

    const int gaussScale = gaussRadius - 1;

    if(vx >= volDimX || vy >= volDimY)
        return;

    float sum = 0.0f;
    float sumFactor = 0.0f;

    for(int rz = -gaussRadius; rz <= gaussRadius; rz++)
    {
        const int iz = vz + rz;
        if((iz < volDimZ) && (iz > 0))
        {
            const float value = float(*get3DBufferAt(in_volume_d, in_volume_s, in_volume_p, vx, vy, iz));
            const float factor = getGauss(gaussScale, rz + gaussRadius, d_gaussianArrayOffset, d_gaussianArray);
            sum += value * factor;
            sumFactor += factor;
        }
    }

    *get3DBufferAt(out_volume_d, out_volume_s, out_volume_p, vx, vy, vz) = float(sum / sumFactor);
}

void gaussianBlurVolumeXYZ_kernel(float* out_volume_d, int out_volume_s, int out_volume_p,
                                             const float* in_volume_d, int in_volume_s, int in_volume_p,
                                             int volDimX, int volDimY, int volDimZ, int gaussRadius,
                                             const sycl::nd_item<3> &item_ct1, int *d_gaussianArrayOffset,
                                             float *d_gaussianArray)
{
    const int vx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int vy = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const int vz = item_ct1.get_group(0);

    const int gaussScale = gaussRadius - 1;

    if(vx >= volDimX || vy >= volDimY)
        return;

    const int xMinRadius = sycl::max(-gaussRadius, -vx);
    const int yMinRadius = sycl::max(-gaussRadius, -vy);
    const int zMinRadius = sycl::max(-gaussRadius, -vz);

    const int xMaxRadius = sycl::min(gaussRadius, volDimX - vx - 1);
    const int yMaxRadius = sycl::min(gaussRadius, volDimY - vy - 1);
    const int zMaxRadius = sycl::min(gaussRadius, volDimZ - vz - 1);

    float sum = 0.0f;
    float sumFactor = 0.0f;

    for(int rx = xMinRadius; rx <= xMaxRadius; rx++)
    {
        const int ix = vx + rx;

        for(int ry = yMinRadius; ry <= yMaxRadius; ry++)
        {
            const int iy = vy + ry;

            for(int rz = zMinRadius; rz <= zMaxRadius; rz++)
            {
                const int iz = vz + rz;
   
                const float value = float(*get3DBufferAt(in_volume_d, in_volume_s, in_volume_p, ix, iy, iz));
                const float factor = getGauss(gaussScale, rx + gaussRadius, d_gaussianArrayOffset, d_gaussianArray) *
                                     getGauss(gaussScale, ry + gaussRadius, d_gaussianArrayOffset, d_gaussianArray) *
                                     getGauss(gaussScale, rz + gaussRadius, d_gaussianArrayOffset, d_gaussianArray);
                sum += value * factor;
                sumFactor += factor;
            }
        }
    }

    *get3DBufferAt(out_volume_d, out_volume_s, out_volume_p, vx, vy, vz) = float(sum / sumFactor);
}

/**
 * @warning: use an hardcoded buffer size, so max radius value is 3.
 */
/*
DPCT1110:2: The total declared local variable size in device function medianFilter3_kernel exceeds 128 bytes and may
cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
/*
FIXED-DPCT1050:42: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
*/
void medianFilter3_kernel(dpct::image_accessor_ext<sycl::float4,2> tex, float* texLab_d,
                          int texLab_p, int width, int height, int scale, const sycl::nd_item<3>& item_ct1)
{
    const int radius = 3;
    const int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if((x >= width - radius) || (y >= height - radius) || (x < radius) || (y < radius))
        return;

    const int filterWidth = radius * 2 + 1;
    const int filterNbPixels = filterWidth * filterWidth;

    float buf[filterNbPixels]; // filterNbPixels

    // Assign masked values to buf
    for(int yi = 0; yi < filterWidth; ++yi)
    {
        for(int xi = 0; xi < filterWidth; ++xi)
        {
            float pix = tex2D<float>(tex, x + xi - radius, y + yi - radius);
            buf[yi * filterWidth + xi] = pix;
        }
    }

    // Calculate until we get the median value
    for(int k = 0; k < filterNbPixels; ++k) // (filterNbPixels + 1) / 2
        for(int l = 0; l < filterNbPixels; ++l)
            if(buf[k] < buf[l])
                cuda_swap_float(buf[k], buf[l]);

    BufPtr<float>(texLab_d, texLab_p).at(x, y) = buf[radius * filterWidth + radius];
}

/*********************************************************************************
 * exported host function
 *********************************************************************************/
void cuda_createConstantGaussianArray(int cudaDeviceId, int scales) // float delta, int radius)
 try {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
    if(scales >= MAX_CONSTANT_GAUSS_SCALES)
    {
        throw std::runtime_error( "Programming error: too few scales pre-computed for Gaussian kernels. Enlarge and recompile." );
    }

    dpct::err0 err;

    if(d_gaussianArrayInitialized.find(cudaDeviceId) != d_gaussianArrayInitialized.end())
        return;

    d_gaussianArrayInitialized.insert(cudaDeviceId);

    int*   h_gaussianArrayOffset;
    float* h_gaussianArray;

    err = cudaMallocHost(&h_gaussianArrayOffset, MAX_CONSTANT_GAUSS_SCALES * sizeof(int));
    THROW_ON_CUDA_ERROR(err, "Failed to allocate " << MAX_CONSTANT_GAUSS_SCALES * sizeof(int) << " of CUDA host memory."); 

    err = cudaMallocHost(&h_gaussianArray, MAX_CONSTANT_GAUSS_MEM_SIZE * sizeof(float));
    THROW_ON_CUDA_ERROR(err, "Failed to allocate " << MAX_CONSTANT_GAUSS_MEM_SIZE * sizeof(float) << " of CUDA host memory.");

    int sumSizes = 0;

    for(int scale = 0; scale < MAX_CONSTANT_GAUSS_SCALES; ++scale)
    {
        h_gaussianArrayOffset[scale] = sumSizes;
        const int radius = scale + 1;
        const int size = 2 * radius + 1;
        sumSizes += size;
    }

    if(sumSizes >= MAX_CONSTANT_GAUSS_MEM_SIZE)
    {
        throw std::runtime_error( "Programming error: too little memory allocated for " 
            + std::to_string(MAX_CONSTANT_GAUSS_SCALES) + " Gaussian kernels. Enlarge and recompile." );
    }

    for(int scale = 0; scale < MAX_CONSTANT_GAUSS_SCALES; ++scale)
    {
        const int radius = scale + 1;
        const float delta  = 1.0f;
        const int size   = 2 * radius + 1;

        for(int idx = 0; idx < size; idx++)
        {
            int x = idx - radius;
            h_gaussianArray[h_gaussianArrayOffset[scale]+idx] = expf(-(x * x) / (2 * delta * delta));
        }
    }

    // create cuda array
    err = DPCT_CHECK_ERROR(
        q_ct1.memcpy(d_gaussianArrayOffset.get_ptr(), h_gaussianArrayOffset, MAX_CONSTANT_GAUSS_SCALES * sizeof(int))
            .wait());

    THROW_ON_CUDA_ERROR(err, "Failed to move Gaussian filter to symbol.");

    err = DPCT_CHECK_ERROR(q_ct1.memcpy(d_gaussianArray.get_ptr(), h_gaussianArray, sumSizes * sizeof(float)).wait());

    THROW_ON_CUDA_ERROR(err, "Failed to move Gaussian filter to symbol." );

    sycl::free(h_gaussianArrayOffset, q_ct1);
    sycl::free(h_gaussianArray, q_ct1);
}
catch(sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void cuda_downscaleWithGaussianBlur(CudaDeviceMemoryPitched<CudaRGBA, 2>& out_downscaledImg_dmp,
                                    dpct::image_wrapper_base_p in_img_tex, int downscale, int gaussRadius,
                                    dpct::queue_ptr stream)
try {
    const sycl::range<3> block(1, 2, 32);
    const sycl::range<3> grid(divUp(out_downscaledImg_dmp.getSize().x(), block[2]),
                              divUp(out_downscaledImg_dmp.getSize().y(), block[1]), 1);

    /*
    FIXED-DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);
    /*
    FIXED-DPCT1050:38: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
    */
    stream->submit(
        [&](sycl::handler& cgh)
        {
            d_gaussianArrayOffset.init(*stream);
            d_gaussianArray.init(*stream);

            auto d_gaussianArrayOffset_ptr_ct1 = d_gaussianArrayOffset.get_ptr();
            auto d_gaussianArray_ptr_ct1 = d_gaussianArray.get_ptr();

            auto in_img_tex_acc = static_cast<dpct::image_wrapper<sycl::float4, 2>*>(in_img_tex)->get_access(cgh, *stream);

            auto in_img_tex_smpl = in_img_tex->get_sampler();

            auto out_downscaledImg_dmp_getBuffer_ct1 = out_downscaledImg_dmp.getBuffer();
            auto out_downscaledImg_dmp_getPitch_ct2 = out_downscaledImg_dmp.getPitch();
            auto out_downscaledImg_dmp_getSize_x_ct3 = (unsigned int)(out_downscaledImg_dmp.getSize().x());
            auto out_downscaledImg_dmp_getSize_y_ct4 = (unsigned int)(out_downscaledImg_dmp.getSize().y());

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 downscaleWithGaussianBlur_kernel(
                                     dpct::image_accessor_ext<sycl::float4, 2>(in_img_tex_smpl, in_img_tex_acc),
                                     out_downscaledImg_dmp_getBuffer_ct1, out_downscaledImg_dmp_getPitch_ct2,
                                     out_downscaledImg_dmp_getSize_x_ct3, out_downscaledImg_dmp_getSize_y_ct4,
                                     downscale, gaussRadius, item_ct1, d_gaussianArrayOffset_ptr_ct1,
                                     d_gaussianArray_ptr_ct1);
                             });
        });

} catch(sycl::exception const& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_gaussianBlurVolumeZ(CudaDeviceMemoryPitched<float, 3>& inout_volume_dmp, int gaussRadius,
                              dpct::queue_ptr stream)
try {
    const CudaSize<3>& volDim = inout_volume_dmp.getSize();
    CudaDeviceMemoryPitched<float, 3> volSmoothZ_dmp(volDim);

    const sycl::range<3> block(1, 1, 32);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    /*
    FIXED-DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    stream->submit(
        [&](sycl::handler& cgh)
        {
            d_gaussianArrayOffset.init(*stream);
            d_gaussianArray.init(*stream);

            auto d_gaussianArrayOffset_ptr_ct1 = d_gaussianArrayOffset.get_ptr();
            auto d_gaussianArray_ptr_ct1 = d_gaussianArray.get_ptr();

            auto volSmoothZ_dmp_getBuffer_ct0 = volSmoothZ_dmp.getBuffer();
            auto volSmoothZ_dmp_getBytesPaddedUpToDim_ct1 = volSmoothZ_dmp.getBytesPaddedUpToDim(1);
            auto volSmoothZ_dmp_getBytesPaddedUpToDim_ct2 = volSmoothZ_dmp.getBytesPaddedUpToDim(0);
            auto inout_volume_dmp_getBuffer_ct3 = inout_volume_dmp.getBuffer();
            auto inout_volume_dmp_getBytesPaddedUpToDim_ct4 = inout_volume_dmp.getBytesPaddedUpToDim(1);
            auto inout_volume_dmp_getBytesPaddedUpToDim_ct5 = inout_volume_dmp.getBytesPaddedUpToDim(0);
            auto volDim_x_ct6 = int(volDim.x());
            auto volDim_y_ct7 = int(volDim.y());
            auto volDim_z_ct8 = int(volDim.z());

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 gaussianBlurVolumeZ_kernel(
                                     volSmoothZ_dmp_getBuffer_ct0, volSmoothZ_dmp_getBytesPaddedUpToDim_ct1,
                                     volSmoothZ_dmp_getBytesPaddedUpToDim_ct2, inout_volume_dmp_getBuffer_ct3,
                                     inout_volume_dmp_getBytesPaddedUpToDim_ct4,
                                     inout_volume_dmp_getBytesPaddedUpToDim_ct5, volDim_x_ct6, volDim_y_ct7,
                                     volDim_z_ct8, gaussRadius, item_ct1, d_gaussianArrayOffset_ptr_ct1,
                                     d_gaussianArray_ptr_ct1);
                             });
        });

    inout_volume_dmp.copyFrom(volSmoothZ_dmp);
} catch(sycl::exception const& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_gaussianBlurVolumeXYZ(CudaDeviceMemoryPitched<float, 3>& inout_volume_dmp, int gaussRadius,
                                dpct::queue_ptr stream)
try {
    const CudaSize<3>& volDim = inout_volume_dmp.getSize();
    CudaDeviceMemoryPitched<float, 3> volSmoothXYZ_dmp(volDim);

    const sycl::range<3> block(1, 1, 32);
    const sycl::range<3> grid(volDim.z(), divUp(volDim.y(), block[1]), divUp(volDim.x(), block[2]));

    /*
    FIXED-DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    stream->submit(
        [&](sycl::handler& cgh)
        {
            d_gaussianArrayOffset.init(*stream);
            d_gaussianArray.init(*stream);

            auto d_gaussianArrayOffset_ptr_ct1 = d_gaussianArrayOffset.get_ptr();
            auto d_gaussianArray_ptr_ct1 = d_gaussianArray.get_ptr();

            auto volSmoothXYZ_dmp_getBuffer_ct0 = volSmoothXYZ_dmp.getBuffer();
            auto volSmoothXYZ_dmp_getBytesPaddedUpToDim_ct1 = volSmoothXYZ_dmp.getBytesPaddedUpToDim(1);
            auto volSmoothXYZ_dmp_getBytesPaddedUpToDim_ct2 = volSmoothXYZ_dmp.getBytesPaddedUpToDim(0);
            auto inout_volume_dmp_getBuffer_ct3 = inout_volume_dmp.getBuffer();
            auto inout_volume_dmp_getBytesPaddedUpToDim_ct4 = inout_volume_dmp.getBytesPaddedUpToDim(1);
            auto inout_volume_dmp_getBytesPaddedUpToDim_ct5 = inout_volume_dmp.getBytesPaddedUpToDim(0);
            auto volDim_x_ct6 = int(volDim.x());
            auto volDim_y_ct7 = int(volDim.y());
            auto volDim_z_ct8 = int(volDim.z());

            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1)
                             {
                                 gaussianBlurVolumeXYZ_kernel(
                                     volSmoothXYZ_dmp_getBuffer_ct0, volSmoothXYZ_dmp_getBytesPaddedUpToDim_ct1,
                                     volSmoothXYZ_dmp_getBytesPaddedUpToDim_ct2, inout_volume_dmp_getBuffer_ct3,
                                     inout_volume_dmp_getBytesPaddedUpToDim_ct4,
                                     inout_volume_dmp_getBytesPaddedUpToDim_ct5, volDim_x_ct6, volDim_y_ct7,
                                     volDim_z_ct8, gaussRadius, item_ct1, d_gaussianArrayOffset_ptr_ct1,
                                     d_gaussianArray_ptr_ct1);
                             });
        });

    inout_volume_dmp.copyFrom(volSmoothXYZ_dmp);
} catch(sycl::exception const& e) {
    RETHROW_SYCL_EXCEPTION(e);
}

void cuda_medianFilter3(dpct::image_wrapper_base_p tex, CudaDeviceMemoryPitched<float, 2>& img)
try {
    int scale = 1;
    const sycl::range<3> block(1, 2, 32);
    const sycl::range<3> grid(divUp(img.getSize()[0], block[2]), divUp(img.getSize()[1], block[1]), 1);

    /*
    FIXED-DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    size_t max_work_group_size = stream->get_device().get_info<info::device::max_work_group_size>();
    assert(block[0]*block[1]*block[2] <= max_work_group_size);

    /*
    FIXED-DPCT1050:39: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
    */
    dpct::get_default_queue().submit(
        [&](sycl::handler& cgh)
        {
            auto tex_acc =
                static_cast<dpct::image_wrapper<sycl::float4, 2>*>(tex)->get_access(cgh);

            auto tex_smpl = tex->get_sampler();

            auto img_getBuffer_ct1 = img.getBuffer();
            auto img_getPitch_ct2 = img.getPitch();
            auto img_getSize_ct3 = img.getSize()[0];
            auto img_getSize_ct4 = img.getSize()[1];

            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item_ct1)
                {
                    medianFilter3_kernel(
                        dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>(tex_smpl, tex_acc),
                        img_getBuffer_ct1, img_getPitch_ct2, img_getSize_ct3, img_getSize_ct4, scale, item_ct1);
                });
        });
} catch(sycl::exception const& e) {
    RETHROW_SYCL_EXCEPTION(e);
}


} // namespace depthMap
} // namespace aliceVision

