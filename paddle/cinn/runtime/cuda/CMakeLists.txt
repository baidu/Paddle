if(NOT WITH_CUDA)
  return()
endif()

core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  cuda_module.cc
  cuda_util.cc
  cuda_intrinsics.cc
  cuda_intrinsics_reduce.cc
  cuda_intrinsics_float16.cc
  cuda_intrinsics_bfloat16.cc)

cinn_nv_library(cuda_runtime SRCS cinn_cuda_runtime_source.cuh)
