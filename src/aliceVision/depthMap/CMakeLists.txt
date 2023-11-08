# Headers
set(depthMap_files_headers
  BufPtr.hpp
  computeOnMultiGPUs.hpp
  CustomPatchPatternParams.hpp
  DepthMapEstimator.hpp
  DepthMapParams.hpp
  depthMapUtils.hpp
  NormalMapEstimator.hpp
  Refine.hpp
  RefineParams.hpp
  Sgm.hpp
  SgmDepthList.hpp
  SgmParams.hpp
  Tile.hpp
  volumeIO.hpp
  depthMapApi.hpp
)

# Sources
set(depthMap_files_sources
  computeOnMultiGPUs.cpp
  CustomPatchPatternParams.cpp
  DepthMapEstimator.cpp
  depthMapUtils.cpp
  NormalMapEstimator.cpp
  Refine.cpp
  Sgm.cpp
  SgmDepthList.cpp
  volumeIO.cpp
)

# Cuda Host Headers Only
set(depthMap_cuda_host_headers
  cuda/host/LRUCameraCache.hpp
  cuda/host/LRUCache.hpp
  cuda/host/divUp.hpp
  cuda/host/memory.hpp
  cuda/host/MemoryLocker.hpp
)

# Cuda Host Sources
set(depthMap_cuda_host_sources
  cuda/host/DeviceCache.hpp
  cuda/host/DeviceCache.cpp
  cuda/host/DeviceMipmapImage.hpp
  cuda/host/DeviceMipmapImage.cpp
  cuda/host/DeviceStreamManager.hpp
  cuda/host/DeviceStreamManager.cpp
  cuda/host/patchPattern.hpp
  cuda/host/patchPattern.cpp
  cuda/host/utils.hpp
  cuda/host/utils.cpp
)

# device CUDA Headers Only
set(depthMap_cuda_device_headers
  cuda/device/buffer.cuh
  cuda/device/color.cuh
  cuda/device/eig33.cuh
  cuda/device/matrix.cuh
  cuda/device/operators.cuh
  cuda/device/Patch.cuh
  cuda/device/SimStat.cuh
)

set(depthMap_sycl_headers
  cuda/device/buffer.dp.hpp
  cuda/device/color.dp.hpp
  cuda/imageProcessing/deviceColorConversion.dp.hpp
  cuda/imageProcessing/deviceGaussianFilter.dp.hpp
  cuda/device/DeviceCameraParams.dp.hpp
  cuda/device/DevicePatchPattern.dp.hpp
)

# device CUDA Sources
set(depthMap_cuda_device_sources
  cuda/device/DeviceCameraParams.hpp
  cuda/device/DeviceCameraParams.cu
  cuda/device/DevicePatchPattern.hpp
  cuda/device/DevicePatchPattern.cu
)

# imageProcessing CUDA Sources
set(depthMap_cuda_imageProcessing_sources
  cuda/imageProcessing/deviceGaussianFilter.hpp
  cuda/imageProcessing/deviceGaussianFilter.cu
  cuda/imageProcessing/deviceMipmappedArray.hpp
  cuda/imageProcessing/deviceMipmappedArray.cu
)

set(depthMap_sycl_imageProcessing_sources
  cuda/imageProcessing/deviceColorConversion.dp.cpp
  cuda/imageProcessing/deviceGaussianFilter.dp.cpp
  cuda/imageProcessing/deviceMipmappedArray.dp.cpp
  cuda/planeSweeping/deviceSimilarityVolume.dp.cpp
)

# planeSweeping CUDA Headers Only
set(depthMap_cuda_planeSweeping_headers
  cuda/planeSweeping/deviceDepthSimilarityMapKernels.cuh
  cuda/planeSweeping/deviceSimilarityVolumeKernels.cuh
)

# planeSweeping CUDA Sources
set(depthMap_cuda_planeSweeping_sources
  cuda/device/DeviceCameraParams.dp.cpp
  cuda/device/DevicePatchPattern.dp.cpp
  cuda/planeSweeping/similarity.hpp
  cuda/planeSweeping/deviceDepthSimilarityMap.hpp
  cuda/planeSweeping/deviceDepthSimilarityMap.cu
  cuda/planeSweeping/deviceSimilarityVolume.hpp
  cuda/planeSweeping/deviceSimilarityVolume.cu
)

set_source_files_properties(${depthMap_cuda_host_headers}
			    ${depthMap_cuda_device_headers} 
          ${depthMap_sycl_headers}
			    ${depthMap_cuda_planeSweeping_headers}

  PROPERTIES HEADER_FILE_ONLY true
)

source_group("aliceVision_depthMap_cuda_host" FILES ${depthMap_cuda_host_headers} ${depthMap_cuda_host_sources})
source_group("aliceVision_depthMap_cuda_device" FILES ${depthMap_cuda_device_headers} ${depthMap_cuda_device_sources})
source_group("aliceVision_depthMap_cuda_imageProcessing" FILES ${depthMap_cuda_imageProcessing_sources} ${depthMap_sycl_imageProcessing_sources})
source_group("aliceVision_depthMap_cuda_planeSweeping" FILES ${depthMap_cuda_planeSweeping_headers} ${depthMap_cuda_planeSweeping_sources})

# Cuda Sources
set(depthMap_cuda_files_sources
  ${depthMap_cuda_host_headers} 
  ${depthMap_cuda_host_sources}
  ${depthMap_cuda_device_headers} 
  ${depthMap_cuda_device_sources}
  ${depthMap_cuda_imageProcessing_sources}
  ${depthMap_cuda_planeSweeping_headers} 
  ${depthMap_cuda_planeSweeping_sources}
)
set(depthMap_sycl_files_sources
  ${depthMap_sycl_imageProcessing_sources}
)

# with DCP++ .def file for exporting symbols does not work
set(AutoExport ${CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS})
if(BUILD_SHARED_LIBS)
  if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  endif()
endif()

alicevision_add_library(aliceVision_depthMap
	USE_CUDA
  SOURCES
    ${depthMap_files_headers}
    ${depthMap_files_sources}
    ${depthMap_cuda_files_sources}
    ${depthMap_sycl_files_sources}
  PUBLIC_LINKS
    aliceVision_mvsData
    aliceVision_mvsUtils
    aliceVision_system
    Boost::filesystem
    assimp::assimp
    ${CUDA_CUDADEVRT_LIBRARY}
    ${CUDA_CUBLAS_LIBRARIES} #TODO shouldn't be here, but required to build on some machines
  PRIVATE_LINKS
    aliceVision_gpu
    aliceVision_sfmData
    aliceVision_sfmDataIO
  PUBLIC_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    "/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/include/"

)


set(USE_DEFAULT_FLAGS ON)
if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    message(STATUS "Using DEFAULT compilation flags")
else()
    message(STATUS "Overriding DEFAULT compilation flags")
    set(USE_DEFAULT_FLAGS OFF)
endif()

option(USE_NVIDIA_BACKEND      "Build for NVIDIA backend"     ON)
#set(NVIDIA_GPU_CXX_FLAGS " -cxx=clang++ -O3 -std=c++17 -fsycl ")
#string(APPEND CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}") # Default flags for NVIDIA backend

if(USE_NVIDIA_BACKEND)
    message(STATUS "Enabling NVIDIA backend")
    if(USE_DEFAULT_FLAGS)
	    #set(CMAKE_CXX_FLAGS "${NVIDIA_GPU_CXX_FLAGS}") # Default flags for NVIDIA backend
    endif()
    if(USE_SM)
        message("-- Building for SM_${USE_SM} compatibility")
	#string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_${USE_SM} ")
    else()
        message("-- Building for SM_80 compatibility (DEFAULT)")
	#string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 ")
    endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
	message("Compiling Depthmap with intel compiler")
	#string(APPEND CMAKE_CXX_FLAGS " -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend -qopenmp ")
	#target_include_directories(aliceVision_depthMap PRIVATE "/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/include/")
	include_directories("/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/include/")
	set(CMAKE_CUDA_HOST_COMPILER icpx)
  add_compile_options(
    	$<$<COMPILE_LANGUAGE:C>:-fiopenmp>
    	$<$<COMPILE_LANGUAGE:CXX>:-fiopenmp -fsycl>
    	$<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend>
	)

  string(APPEND CMAKE_CXX_FLAGS " -Wno-error -fsycl")
  string(APPEND CMAKE_C_FLAGS " -allow-unsupported-compiler -Wno-error")
  string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend ")
  SET( CUDA_NVCC_FLAGS " ;-allow-unsupported-compiler;-fsycl;-fsycl-targets=nvptx64-nvidia-cuda;-Xsycl-target-backend;")

  set(APPEND CMAKE_CUDA_FLAGS_RELEASE " -allow-unsupported-compiler -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend ")
message(STATUS "Include Directories: ${CMAKE_INCLUDE_PATH}")
message(STATUS "CXX Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CUDA Flags: ${CUDA_NVCC_FLAGS}")



	#target_compile_options(aliceVision_depthMap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler)
endif()
message(STATUS "Compilation flags set to: ${CMAKE_CXX_FLAGS}")
# enable DPC++ compiler for windows
if(WIN32)
  target_compile_definitions(aliceVision_depthMap PUBLIC EIGEN_DONT_VECTORIZE)    
  target_include_directories(aliceVision_depthMap PUBLIC "$ENV{ONEAPI_ROOT}/dpcpp-ct/latest/include")
  set_target_properties( aliceVision_depthMap 
    PROPERTIES
    VS_PLATFORM_TOOLSET $(DefaultIntelDPCppPlatformToolset) 
  )
  if(BUILD_SHARED_LIBS)
    set(aliceVision_depthMap
      PROPERTIES
      CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ${AutoExport})
  endif()
endif()
unset(AutoExport)



# target_compile_definitions(aliceVision_depthMap PUBLIC TSIM_USE_FLOAT)

