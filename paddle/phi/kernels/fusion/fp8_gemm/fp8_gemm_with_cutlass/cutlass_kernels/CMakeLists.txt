cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED COMPUTE_CAPABILITY)
  message(
    FATAL_ERROR
      "please set COMPUTE_CAPABILITY with -DCOMPUTE_CAPABILITY=your gpu compute capability"
  )
endif()

set(PADDLE_LIB
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../build/paddle_inference_install_dir
)

add_definitions("-DPADDLE_WITH_CUDA")
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../build/paddle_inference_install_dir/paddle/include"
)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include")

link_directories("${PADDLE_LIB}/paddle/lib")

find_package(CUDA)
# you can append -std=c++17 in CUDA_NVCC_FLAGS for compiling cutlass 3.0
set(CUDA_NVCC_FLAGS
    -gencode arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++17")

#set(CMAKE_CXX_FLAGS -fvisibility=hidden)
set(CMAKE_BUILD_TYPE "Release")
file(GLOB cutlass_fp8_fp8_gemm_fused "${CMAKE_CURRENT_BINARY_DIR}/../*.cu")

message(STATUS "gemm kernel files:\n" ${cutlass_fp8_fp8_gemm_fused})
message(STATUS "gemm kernel files:\n" ${CMAKE_CURRENT_SOURCE_DIR})

#list(APPEND cutlass_gemms_cu "${CMAKE_CURRENT_SOURCE_DIR}/.cu")

cuda_add_library(cutlassFP8FP8GemmFused SHARED ${cutlass_fp8_fp8_gemm_fused})

message(STATUS "PADDLE_LIB:\n" ${PADDLE_LIB})

target_link_libraries(cutlassFP8FP8GemmFused
                      "${PADDLE_LIB}/paddle/lib/libphi_core.so")
