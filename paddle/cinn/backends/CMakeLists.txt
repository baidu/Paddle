core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  outputs.cc
  codegen_c.cc
  codegen_c_x86.cc
  codegen_cuda_host.cc
  codegen_invoke_module.cc
  extern_func_emitter.cc
  extern_func_emitter_builtin.cc
  function_prototype.cc
  extern_func_protos.cc
  extern_func_jit_register.cc
  compiler.cc
  codegen_device_util.cc
  codegen_gpu_dev.cc)

if(WITH_CUDA)
  add_subdirectory(nvrtc)
  list(APPEND srcs cuda_util.cc codegen_cuda_dev.cc)
endif()

if(WITH_ROCM)
  add_subdirectory(hip)
endif()

if(CINN_WITH_SYCL)
  add_subdirectory(sycl)
endif()

if(WITH_OPENMP)
  cinn_cc_library(__x86_source_fake_lib SRCS _x86_builtin_source.cc)
endif()
add_subdirectory(llvm)

include_directories(${CMAKE_SOURCE_DIR}/paddle/cinn/runtime)

foreach(cpp ${srcs})
  set(cinnapi_src
      "${cinnapi_src};paddle/cinn/backends/${cpp}"
      CACHE INTERNAL "")
endforeach()

file(
  GLOB includes
  LIST_DIRECTORIES false
  RELATIVE ${CMAKE_SOURCE_DIR}
  *.h)

foreach(header ${includes})
  set(core_includes
      "${core_includes};${header}"
      CACHE INTERNAL "")
endforeach()
