// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "DeviceStreamManager.hpp"

#include<aliceVision/system/Logger.hpp>

namespace aliceVision {
namespace depthMap {

DeviceStreamManager::DeviceStreamManager(int nbStreams)
    : _nbStreams(nbStreams)
{
    assert(nbStreams > 0);

    _streams.resize(nbStreams);

    for(int i = 0; i < nbStreams; ++i)
    {
        try {
            _streams.at(i) = dpct::get_current_device().create_queue();
        }
        catch(sycl::exception const& exc)
        {
            ALICEVISION_LOG_WARNING("DeviceStreamManager: Failed to create a CUDA stream object " << i << "/" << nbStreams << ", " << exc.what());
            _streams.at(i) = &dpct::get_default_queue();
        }
    }
}

DeviceStreamManager::~DeviceStreamManager() 
{
    for(cudaStream_t& stream : _streams)
    {
        cudaStreamSynchronize(stream);

        if(stream != 0) 
        {
            cudaStreamDestroy(stream);
        }
    }
}

dpct::queue_ptr DeviceStreamManager::getStream(int streamIndex)
{
    return _streams.at(streamIndex % _nbStreams);
}

void DeviceStreamManager::waitStream(int streamIndex)
{
    getStream(streamIndex)->wait();
}

} // namespace depthMap
} // namespace aliceVision
