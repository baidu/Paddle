if(NOT WITH_CUDA)
  return()
endif()

cinn_nv_test(test_cuda_module SRCS cuda_module_test.cc DEPS cinncore)
