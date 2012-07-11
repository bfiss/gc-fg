PROJECT(firstCudaExample)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#set(CMAKE_BUILD_TYPE Release)

#Link with libcutil (expected to be in ${CUDA_SDK_ROOT_DIR}/C/lib)
INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/../shared/inc)
INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)
INCLUDE_DIRECTORIES(/usr/local/include)

LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib)
LINK_DIRECTORIES(/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/common/lib/linux)
LINK_DIRECTORIES(/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/lib)
LINK_DIRECTORIES(/usr/local/cuda/NVIDIA_GPU_Computing_SDK/shared/lib)
LINK_DIRECTORIES(/usr/local/lib)

#LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/lib -lcutil")

LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/C/lib -lcutil_x86_64")
LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/shared/lib -lshrutil_x86_64")
LINK_LIBRARIES("opencv_core")
LINK_LIBRARIES("opencv_highgui")


#MESSAGE("${CUDA_SDK_ROOT_DIR}")
#Include FindCUDA script

set(CUDA_NVCC_FLAGS "-arch=sm_20")

CUDA_ADD_EXECUTABLE(multiplyMRF ./multiplyMRF.cpp)
CUDA_ADD_EXECUTABLE(solveMRF ./solveMRF.cu)

CUDA_ADD_EXECUTABLE(app ./Application.cu)
CUDA_ADD_EXECUTABLE(dbg ./Debug.cu)

CUDA_ADD_EXECUTABLE(presentation ./firstCudaExample.cu)
