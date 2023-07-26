// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "utils.hpp"

#include <aliceVision/system/Logger.hpp>

namespace aliceVision {
namespace depthMap {

int listCudaDevices()
 try {
    int nbDevices = 0; // number of CUDA GPUs

    // determine the number of CUDA capable GPUs
    nbDevices = dpct::dev_mgr::instance().device_count();

    if(nbDevices < 1)
    {
        ALICEVISION_LOG_ERROR("No CUDA capable devices detected.");
        return 0;
    }


    // display CPU and GPU configuration
    std::stringstream s; 
    for(int i = 0; i < nbDevices; ++i)
    {
        dpct::device_info dprop;
        dpct::dev_mgr::instance().get_device(i).get_device_info(dprop);
        s << "\t- Device " << i << ": " << dprop.get_name() << std::endl;
    }
    ALICEVISION_LOG_DEBUG(nbDevices << " CUDA devices found:" << std::endl << s.str());

    return nbDevices;
}
catch(sycl::exception const& exc) {
  RETHROW_SYCL_EXCEPTION(exc);
}

int getCudaDeviceId()
 try {
    int currentCudaDeviceId;

    if(currentCudaDeviceId = dpct::dev_mgr::instance().current_device_id() != 0)
    {
        ALICEVISION_LOG_ERROR("Cannot get current CUDA device id.");
    }

    return currentCudaDeviceId;
}
catch(sycl::exception const& exc) {
  RETHROW_SYCL_EXCEPTION(exc);
}

void setCudaDeviceId(int cudaDeviceId)
 try {
    /*
    IGNORED-DPCT1093:16: The "cudaDeviceId" device may be not the one intended for use. Adjust the selected device if needed.
    */
    dpct::select_device(cudaDeviceId));
    {
        ALICEVISION_LOG_ERROR("Cannot set device id " << cudaDeviceId << " as current CUDA device.");
    }
}
catch(sycl::exception const& exc) {
  ALICEVISION_LOG_ERROR("Cannot set device id " << cudaDeviceId << " as current CUDA device.");
  RETHROW_SYCL_EXCEPTION(exc);
}

bool testCudaDeviceId(int cudaDeviceId)
{
  int currentCudaDeviceId;
  currentCudaDeviceId = dpct::dev_mgr::instance().current_device_id();
  if(currentCudaDeviceId != cudaDeviceId)
  {
      ALICEVISION_LOG_WARNING("CUDA device id should be: " << cudaDeviceId << ", program curently use device id: " << currentCudaDeviceId << ".");
      return false;
  }
  return true;
}

void logDeviceMemoryInfo()
{
    size_t iavail;
    size_t itotal;

    /*
    DPCT1106:18: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be
    supported by all compilers or runtimes. You may need to adjust the code.
    */
    dpct::get_current_device().get_memory_info(iavail, itotal);

    const double availableMB = double(iavail) / (1024.0 * 1024.0);
    const double totalMB = double(itotal) / (1024.0 * 1024.0);
    const double usedMB = double(itotal - iavail) / (1024.0 * 1024.0);

    int cudaDeviceId;
    cudaDeviceId = dpct::dev_mgr::instance().current_device_id();

    ALICEVISION_LOG_INFO("Device memory (device id: "<< cudaDeviceId <<"):" << std::endl
                      << "\t- used: " << usedMB << " MB" << std::endl
                      << "\t- available: " << availableMB << " MB" << std::endl
                      << "\t- total: " << totalMB << " MB");
}

void getDeviceMemoryInfo(double& availableMB, double& usedMB, double& totalMB)
{
    size_t iavail;
    size_t itotal;

    /*
    DPCT1106:19: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be
    supported by all compilers or runtimes. You may need to adjust the code.
    */
    dpct::get_current_device().get_memory_info(iavail, itotal);

    availableMB = double(iavail) / (1024.0 * 1024.0);
    totalMB = double(itotal) / (1024.0 * 1024.0);
    usedMB = double(itotal - iavail) / (1024.0 * 1024.0);
}

} // namespace depthMap
} // namespace aliceVision
