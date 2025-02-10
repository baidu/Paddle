configure_file(config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)

# phi auto cmake utils
include(phi)

set(common_srcs CACHE INTERNAL "" FORCE)
set(api_srcs CACHE INTERNAL "" FORCE)
set(capi_srcs CACHE INTERNAL "" FORCE)
set(core_srcs CACHE INTERNAL "" FORCE)
set(backends_srcs CACHE INTERNAL "" FORCE)
set(kernels_srcs CACHE INTERNAL "" FORCE)
set(infermeta_srcs CACHE INTERNAL "" FORCE)
#set(excluded_srcs CACHE INTERNAL "" FORCE)

# paddle experimental common components
add_subdirectory(common)
# phi (low level) api headers: include
# phi (high level) api
add_subdirectory(api)
# phi core components
add_subdirectory(core)
# phi components of specific backends
add_subdirectory(backends)
# phi kernels for diff device
add_subdirectory(kernels)
# phi infermeta
add_subdirectory(infermeta)
# phi tools
add_subdirectory(tools)
# phi capi
if(WITH_CUSTOM_DEVICE)
  add_subdirectory(capi)
endif()

if(WIN32)
  file(GLOB_RECURSE YAML_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ops/yaml/*.yaml")
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${YAML_FILE})
endif()

set(PHI_DEPS
    phi_profiler_proto
    auto_parallel_proto
    glog
    warpctc
    warprnnt
    eigen3
    xxhash
    cblas
    utf8proc
    common)

list(
  APPEND
  PHI_DEPS
  framework_proto
  pass_desc_proto
  data_feed_proto
  trainer_desc_proto
  heter_service_proto)

set(INFERENCE_DEPS phi_profiler_proto auto_parallel_proto)

if(WITH_GPU)
  list(APPEND PHI_DEPS external_error_proto)
endif()

if(WITH_ASCEND_CL)
  list(APPEND PHI_DEPS npu_hccl)
endif()

if(WITH_FLASHATTN)
  list(APPEND PHI_DEPS flashattn)
endif()

if(WITH_XBYAK)
  list(APPEND PHI_DEPS xbyak)
endif()

if(WITH_ONEDNN)
  list(APPEND PHI_DEPS onednn)
endif()

if(WITH_GLOO)
  list(APPEND PHI_DEPS gloo)
endif()

if(WITH_CUDNN_FRONTEND)
  list(APPEND PHI_DEPS cudnn-frontend)
endif()

if(WITH_POCKETFFT)
  list(APPEND PHI_DEPS pocketfft)
endif()

if(WITH_MKLML)
  list(APPEND PHI_DEPS pocketfft dynload_mklml)
  list(APPEND INFERENCE_DEPS dynload_mklml)
endif()

if(WITH_XPU)
  list(APPEND PHI_DEPS xpulib)
  if(WITH_XPU_PLUGIN)
    add_subdirectory(kernels/xpu/plugin)
    list(APPEND PHI_DEPS xpuplugin)
  endif()
endif()

if(WITH_UCC)
  list(APPEND PHI_DEPS ucclib)
endif()

if(WITH_DGC)
  list(APPEND PHI_DEPS dgc)
endif()

set(PHI_CORE_SRCS
    ${common_srcs}
    ${api_srcs}
    ${core_srcs}
    ${backends_srcs}
    ${kernels_srcs}
    ${infermeta_srcs}
    ${capi_srcs})

set(PHI_GPU_SRCS ${kernels_gpu_srcs})

if(WITH_SHARED_PHI)
  set(PHI_BUILD_TYPE
      SHARED
      CACHE INTERNAL "" FORCE)
else()
  set(PHI_BUILD_TYPE
      STATIC
      CACHE INTERNAL "" FORCE)
endif()

if(WITH_AVX
   AND AVX512F_FOUND
   AND AVX512F_FLAG
   AND WITH_MKL)
  set_source_files_properties(
    kernels/fusion/cpu/fused_layer_norm_avx_kernel.cc
    kernels/fusion/cpu/self_dp_attention_kernel.cc
    kernels/fusion/cpu/rms_norm_avx_kernel.cc
    PROPERTIES
      COMPILE_FLAGS
      "${Wno_Maybe_Uninitialized} ${FMA_FLAG} ${AVX512F_FLAG} ${NO_INLINE}")
endif()

if(WITH_GPU)
  set_source_files_properties(
    backends/gpu/gpu_resources.cc
    PROPERTIES COMPILE_FLAGS
               "-DCUDA_REAL_ARCHS=\"${NVCC_FLAGS_EXTRA_real_archs}\"")
  nv_library(
    phi_core ${PHI_BUILD_TYPE}
    SRCS ${PHI_CORE_SRCS}
    DEPS ${PHI_DEPS})
  nv_library(
    phi_gpu ${PHI_BUILD_TYPE}
    SRCS ${PHI_GPU_SRCS}
    DEPS ${PHI_DEPS} phi_core)
elseif(WITH_ROCM)
  hip_library(
    phi_core ${PHI_BUILD_TYPE}
    SRCS ${PHI_CORE_SRCS}
    DEPS ${PHI_DEPS})
  hip_library(
    phi_gpu ${PHI_BUILD_TYPE}
    SRCS ${PHI_GPU_SRCS}
    DEPS ${PHI_DEPS} phi_core)
elseif(WITH_XPU_KP)
  xpu_library(
    phi_core ${PHI_BUILD_TYPE}
    SRCS ${PHI_CORE_SRCS}
    DEPS ${PHI_DEPS})
else()
  cc_library(
    phi_core ${PHI_BUILD_TYPE}
    SRCS ${PHI_CORE_SRCS}
    DEPS ${PHI_DEPS})
endif()

# core/memory/allocation uses shm_unlink and requires the rt library
if(UNIX AND NOT APPLE)
  target_link_libraries(phi_core rt)
endif()

set(PHI_DUMMY_FILE ${CMAKE_CURRENT_BINARY_DIR}/phi_dummy.cpp)
if(MSVC)
  set(PHI_DUMMY_FILE_CONTENT
      "__declspec(dllexport) int phi_dummy_placeholder_function(){ return 0\\; }"
  )
else()
  set(PHI_DUMMY_FILE_CONTENT "")
endif()
file(WRITE ${PHI_DUMMY_FILE} ${PHI_DUMMY_FILE_CONTENT})

add_library(phi ${PHI_BUILD_TYPE} ${PHI_DUMMY_FILE})
target_link_libraries(phi phi_core)
if(WITH_GPU OR WITH_ROCM)
  target_link_libraries(phi phi_gpu)
  target_link_libraries(phi_gpu ${ROCM_HIPRTC_LIB})
  target_link_libraries(phi_core ${ROCM_HIPRTC_LIB})
endif()

# Note(silverling): some functions in phi_core depend on phi_gpu,
# when phi is built to dynamic library, it's fine. But when phi
# is built to static library, phi_gpu should be linked to phi_core.
# By the way, cyclic dependency is allowed in static library.
if((WITH_GPU OR WITH_ROCM) AND NOT WITH_SHARED_PHI)
  target_link_libraries(phi_core phi_gpu)
endif()

target_compile_definitions(phi_core PUBLIC PHI_INNER)

if(WIN32)
  target_link_libraries(phi_core shlwapi.lib)
endif()

if(WIN32)
  if(WITH_SHARED_PHI)
    set_property(TARGET phi_core PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set(PHI_NAME
        phi.dll
        CACHE INTERNAL "" FORCE)
  else()
    set(PHI_NAME
        phi.lib
        CACHE INTERNAL "" FORCE)
  endif()
elseif(APPLE)
  if(WITH_SHARED_PHI)
    set(PHI_NAME
        libphi.dylib
        CACHE INTERNAL "" FORCE)
  else()
    set(PHI_NAME
        libphi.a
        CACHE INTERNAL "" FORCE)
  endif()
else()
  if(WITH_SHARED_PHI)
    set(PHI_NAME
        libphi.so
        CACHE INTERNAL "" FORCE)
  else()
    set(PHI_NAME
        libphi.a
        CACHE INTERNAL "" FORCE)
  endif()
endif()

set(PHI_LIB
    "${CMAKE_CURRENT_BINARY_DIR}/${PHI_NAME}"
    CACHE FILEPATH "PHI Dummy Library" FORCE)

string(REPLACE "phi" "phi_core" PHI_CORE_NAME ${PHI_NAME})
set(PHI_CORE_NAME
    ${PHI_CORE_NAME}
    CACHE INTERNAL "" FORCE)
set(PHI_CORE_LIB
    "${CMAKE_CURRENT_BINARY_DIR}/${PHI_CORE_NAME}"
    CACHE FILEPATH "PHI CPU Library" FORCE)

if(WITH_GPU OR WITH_ROCM)
  string(REPLACE "phi" "phi_gpu" PHI_GPU_NAME ${PHI_NAME})
  set(PHI_GPU_NAME
      ${PHI_GPU_NAME}
      CACHE INTERNAL "" FORCE)
  set(PHI_GPU_LIB
      "${CMAKE_CURRENT_BINARY_DIR}/${PHI_GPU_NAME}"
      CACHE FILEPATH "PHI GPU Library" FORCE)
endif()

if(MKL_FOUND AND WITH_ONEMKL)
  target_include_directories(phi_core PRIVATE ${MKL_INCLUDE})
endif()

add_dependencies(phi extern_lapack)
if(WITH_CUTLASS)
  add_dependencies(phi cutlass_codegen)
  add_definitions("-DPADDLE_WITH_MEMORY_EFFICIENT_ATTENTION"
  )# for memory_efficient_attention.h
endif()
if(WITH_FLASHATTN)
  add_dependencies(phi flashattn)
endif()

# for inference static library
if(NOT WITH_SHARED_PHI)
  get_property(phi_modules GLOBAL PROPERTY PHI_MODULES)
  set(phi_modules ${phi_modules} ${INFERENCE_DEPS} phi)
  set_property(GLOBAL PROPERTY PHI_MODULES "${phi_modules}")
endif()

set(phi_extension_header_file
    ${CMAKE_CURRENT_SOURCE_DIR}/extension.h
    CACHE INTERNAL "phi/extension.h file")
file(
  WRITE ${phi_extension_header_file}
  "// Header file generated by paddle/phi/CMakeLists.txt for external users,\n// DO NOT edit or include it within paddle.\n\n#pragma once\n\n"
)

file(APPEND ${phi_extension_header_file} "#include \"paddle/phi/config.h\"\n\n")
# generate inner headers include dir for users
generate_unify_header(backends EXCLUDES context_pool_utils.h)
generate_unify_header(core EXCLUDES cuda_stream.h)
generate_unify_header(infermeta)
generate_unify_header(kernels SKIP_SUFFIX grad_kernel)
