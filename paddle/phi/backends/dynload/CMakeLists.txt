set(DYNLOAD_COMMON_SRCS dynamic_loader.cc warpctc.cc warprnnt.cc lapack.cc)
if(WITH_ASCEND_CL)
  list(REMOVE_ITEM DYNLOAD_COMMON_SRCS warprnnt.cc)
endif()
list(
  APPEND
  CUDA_SRCS
  cublas.cc
  cublasLt.cc
  cudnn.cc
  curand.cc
  cusolver.cc
  cusparse.cc
  nvtx.cc
  cufft.cc
  cutlass_conv2d.cc
  cutlass_gemm_epilogue.cc)

if(NOT WITH_NV_JETSON)
  list(APPEND CUDA_SRCS nvjpeg.cc)
endif()

if(WITH_ROCM)
  list(
    APPEND
    HIP_SRCS
    rocblas.cc
    hipblasLt.cc
    miopen.cc
    hiprand.cc
    hipfft.cc
    rocsparse.cc
    rocsolver.cc)
endif()

# There is no macOS version of NCCL.
# Disable nvrtc and cuda_driver api on macOS, and only do an early test on Linux and Windows.
if(NOT APPLE)
  list(APPEND CUDA_SRCS nvrtc.cc cuda_driver.cc)
  if(WITH_NCCL)
    list(APPEND CUDA_SRCS nccl.cc)
  endif()
  if(WITH_ROCM)
    list(APPEND HIP_SRCS hiprtc.cc rocm_driver.cc)
    if(WITH_RCCL)
      list(APPEND HIP_SRCS rccl.cc)
    endif()
    if(CUPTI_FOUND)
      list(APPEND HIP_SRCS cupti.cc)
    endif()
  endif()
endif()

if(TENSORRT_FOUND)
  list(APPEND CUDA_SRCS tensorrt.cc)
endif()

if(CUSPARSELT_FOUND)
  list(APPEND CUDA_SRCS cusparseLt.cc)
endif()

configure_file(cupti_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/cupti_lib_path.h)
if(CUPTI_FOUND)
  list(APPEND CUDA_SRCS cupti.cc)
endif()

if(WITH_MKLML)
  # Only deps libmklml.so, not link
  add_library(dynload_mklml STATIC mklml.cc)
  add_dependencies(dynload_mklml mklml)
  if(WIN32)
    target_link_libraries(dynload_mklml ${MKLML_IOMP_LIB})
  else()
    target_link_libraries(dynload_mklml
                          "-L${MKLML_LIB_DIR} -liomp5 -Wl,--as-needed")
  endif()
endif()

if(WITH_XPU)
  collect_srcs(backends_srcs SRCS xpti.cc)
endif()

if(WITH_FLASHATTN)
  list(APPEND DYNLOAD_COMMON_SRCS flashattn.cc)
endif()

if(WITH_FLASHATTN_V3)
  list(APPEND DYNLOAD_COMMON_SRCS flashattnv3.cc)
endif()

if(WITH_PSCORE)
  list(APPEND DYNLOAD_COMMON_SRCS afs_api.cc)
endif()

if(MKL_FOUND AND WITH_ONEMKL)
  message("ONEMKL INCLUDE directory is ${MKL_INCLUDE}")
  list(APPEND DYNLOAD_COMMON_SRCS mklrt.cc)
endif()

if(WITH_ROCM)
  collect_srcs(backends_srcs SRCS ${DYNLOAD_COMMON_SRCS} ${HIP_SRCS})
elseif(WITH_GPU)
  collect_srcs(backends_srcs SRCS ${DYNLOAD_COMMON_SRCS} ${CUDA_SRCS})
else()
  collect_srcs(backends_srcs SRCS ${DYNLOAD_COMMON_SRCS})
endif()

if(WITH_CUDNN_FRONTEND)
  nv_test(
    cudnn_frontend_test
    SRCS cudnn_frontend_test.cc
    DEPS phi common cudnn-frontend)
endif()
