set(CINN_THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DOWNLOAD_MODEL_DIR "${CINN_THIRD_PARTY_PATH}/model")

string(REGEX MATCH "-std=(c\\+\\+[^ ]+)" STD_FLAG "${CMAKE_CXX_FLAGS}")
if(NOT STD_FLAG)
  if(NOT CMAKE_CXX_STANDARD)
    message(
      STATUS
        "STD_FLAG and CMAKE_CXX_STANDARD not found, using default flag: -std=c++17"
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    set(CMAKE_CXX_STANDARD 17)
  else()
    message(
      STATUS
        "Got CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}, append -std=c++${CMAKE_CXX_STANDARD} to CMAKE_CXX_FLAGS"
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
  endif()
else()
  string(REGEX MATCH "([0-9]+)" STD_VALUE "${STD_FLAG}")
  message(
    STATUS "Got STD_FLAG=${STD_FLAG}, set CMAKE_CXX_STANDARD=${STD_VALUE}")
  set(CMAKE_CXX_STANDARD ${STD_VALUE})
endif()

if(NOT DEFINED ENV{runtime_include_dir})
  message(
    STATUS
      "set runtime_include_dir: ${CMAKE_SOURCE_DIR}/paddle/cinn/runtime/cuda")
  set(ENV{runtime_include_dir} "${CMAKE_SOURCE_DIR}/paddle/cinn/runtime/cuda")
  add_definitions(
    -DRUNTIME_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/paddle/cinn/runtime/cuda")
endif()

if(WITH_TESTING)
  add_definitions(-DCINN_WITH_TEST)
endif()
if(WITH_DEBUG)
  add_definitions(-DCINN_WITH_DEBUG)
endif()

# TODO(zhhsplendid): CINN has lots of warnings during early development.
# They will be treated as errors under paddle. We set no-error now and we will
# clean the code in the future.
add_definitions(-w)

include(cmake/cinn/version.cmake)
if(NOT EXISTS ${CMAKE_BINARY_DIR}/cmake/cinn/config.cmake)
  file(COPY ${PROJECT_SOURCE_DIR}/cmake/cinn/config.cmake
       DESTINATION ${CMAKE_BINARY_DIR}/cmake/cinn)
endif()
include(${CMAKE_BINARY_DIR}/cmake/cinn/config.cmake)

if(WITH_MKL)
  generate_dummy_static_lib(LIB_NAME "cinn_mklml" GENERATOR "mklml.cmake")
  target_link_libraries(cinn_mklml ${MKLML_LIB} ${MKLML_IOMP_LIB})
  add_dependencies(cinn_mklml ${MKLML_PROJECT})
  add_definitions(-DCINN_WITH_MKL_CBLAS)
endif()
if(WITH_ONEDNN)
  add_definitions(-DCINN_WITH_DNNL)
endif()

if(WITH_GPU)
  message(STATUS "Enable CINN CUDA")
  add_definitions(-DCINN_WITH_CUDA)
  if(WITH_CUDNN)
    message(STATUS "Enable CINN CUDNN")
    add_definitions(-DCINN_WITH_CUDNN)
  endif()
  if(WITH_CUTLASS)
    message(STATUS "Enable CINN CUTLASS")
    add_definitions(-DCINN_WITH_CUTLASS)
  endif()
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  include_directories(${CMAKE_SOURCE_DIR}/paddle/cinn/runtime/cuda)
  include_directories(/usr/lib/x86_64-linux-gnu)
  set(CUDA_SEPARABLE_COMPILATION ON)

  cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

  message(
    STATUS
      "copy paddle/cinn/common/float16.h paddle/cinn/common/bfloat16.h to $ENV{runtime_include_dir}"
  )
  file(COPY paddle/cinn/common/float16.h paddle/cinn/common/bfloat16.h
       DESTINATION $ENV{runtime_include_dir})

  find_library(CUDASTUB libcuda.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/
                                         REQUIRED)
  find_library(CUBLAS libcublas.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                                         /usr/lib /usr/lib64 REQUIRED)
  find_library(CUDNN libcudnn.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/lib
                                       /usr/lib64 REQUIRED)
  find_library(CURAND libcurand.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                                         /usr/lib /usr/lib64 REQUIRED)
  find_library(CUSOLVER libcusolver.so HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                                             /usr/lib /usr/lib64 REQUIRED)
endif()

if(WITH_ROCM)
  message(STATUS "CINN Compile with ROCM support")
  add_definitions(-DCINN_WITH_HIP)
endif()

if(CINN_WITH_SYCL)
  message(STATUS "CINN Compile with SYCL support")
  set(DPCPP_DIR ${PROJECT_SOURCE_DIR}/cmake/cinn)
  find_package(DPCPP REQUIRED CONFIG)
  add_definitions(-DCINN_WITH_SYCL)
endif()

set(cinnapi_src CACHE INTERNAL "" FORCE)
set(core_src CACHE INTERNAL "" FORCE)
set(core_includes CACHE INTERNAL "" FORCE)
set(core_proto_includes CACHE INTERNAL "" FORCE)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

include(cmake/generic.cmake)
include(cmake/cinn/system.cmake)
include(cmake/cinn/core.cmake)
include(cmake/cinn/nvrtc.cmake)
include(cmake/cinn/nvtx.cmake)

set(LINK_FLAGS
    "-Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cinn/export.map"
    CACHE INTERNAL "")
set(global_test_args
    "--cinn_x86_builtin_code_root=${CMAKE_SOURCE_DIR}/paddle/cinn/backends")

set(Python_VIRTUALENV FIRST)

if(NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
endif()

if(NOT PYTHON_LIBRARIES)
  find_package(PythonLibs ${PY_VERSION} REQUIRED)
endif()

message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")

include_directories(${PYTHON_INCLUDE_DIR})

set(core_deps CACHE INTERNAL "" FORCE)
set(hlir_src CACHE INTERNAL "" FORCE)

# TODO(chenweihang): The logic later depends adding cinn subdirectory here,
# but better to move to paddle/CMakeLists.txt
add_subdirectory(paddle/cinn)

set(core_src "${cinnapi_src}")

cinn_cc_library(
  cinnapi
  SHARED
  SRCS
  ${cinnapi_src}
  DEPS
  glog
  ${llvm_libs}
  param_proto
  schedule_desc_proto
  tile_config_proto
  absl
  isl
  ginac
  op_fusion
  cinn_op_dialect
  ${jitify_deps})
add_dependencies(cinnapi GEN_LLVM_RUNTIME_IR_HEADER ZLIB::ZLIB)
add_dependencies(cinnapi GEN_LLVM_RUNTIME_IR_HEADER ${core_deps})
target_link_libraries(cinnapi op_dialect pir phi)
add_dependencies(cinnapi op_dialect pir phi)

add_dependencies(cinnapi python)
if(WITH_MKL)
  target_link_libraries(cinnapi cinn_mklml)
  add_dependencies(cinnapi cinn_mklml)
  if(WITH_ONEDNN)
    target_link_libraries(cinnapi ${ONEDNN_LIB})
    add_dependencies(cinnapi ${ONEDNN_PROJECT})
  endif()
endif()

if(WITH_GPU)
  target_link_libraries(
    cinnapi
    ${CUDA_NVRTC_LIB}
    ${CUDA_LIBRARIES}
    ${CUDASTUB}
    ${CUBLAS}
    ${CUDNN}
    ${CURAND}
    ${CUSOLVER})
  if(NVTX_FOUND)
    target_link_libraries(cinnapi ${CUDA_NVTX_LIB})
  endif()
endif()

if(WITH_CUTLASS)
  target_link_libraries(cinnapi cutlass)
  add_dependencies(cinnapi cutlass)
endif()

function(gen_cinncore LINKTYPE)
  set(CINNCORE_TARGET cinncore)
  if(${LINKTYPE} STREQUAL "STATIC")
    set(CINNCORE_TARGET cinncore_static)
  endif()
  cinn_cc_library(
    ${CINNCORE_TARGET}
    ${LINKTYPE}
    SRCS
    ${core_src}
    DEPS
    glog
    ${llvm_libs}
    param_proto
    schedule_desc_proto
    tile_config_proto
    absl
    isl
    ginac
    pybind
    op_fusion
    cinn_op_dialect
    ${jitify_deps})
  add_dependencies(${CINNCORE_TARGET} GEN_LLVM_RUNTIME_IR_HEADER ZLIB::ZLIB)
  add_dependencies(${CINNCORE_TARGET} GEN_LLVM_RUNTIME_IR_HEADER ${core_deps})
  target_link_libraries(${CINNCORE_TARGET} op_dialect pir phi)
  add_dependencies(${CINNCORE_TARGET} op_dialect pir phi)

  # add_dependencies(${CINNCORE_TARGET} pybind)
  target_link_libraries(${CINNCORE_TARGET} ${PYTHON_LIBRARIES})

  if(WITH_MKL)
    target_link_libraries(${CINNCORE_TARGET} cinn_mklml)
    add_dependencies(${CINNCORE_TARGET} cinn_mklml)
    if(WITH_ONEDNN)
      target_link_libraries(${CINNCORE_TARGET} ${ONEDNN_LIB})
      add_dependencies(${CINNCORE_TARGET} ${ONEDNN_PROJECT})
    endif()
  endif()

  if(WITH_GPU)
    target_link_libraries(
      ${CINNCORE_TARGET}
      ${CUDA_NVRTC_LIB}
      ${CUDA_LIBRARIES}
      ${CUDASTUB}
      ${CUBLAS}
      ${CUDNN}
      ${CURAND}
      ${CUSOLVER})
    # ${jitify_deps})
    if(NVTX_FOUND)
      target_link_libraries(${CINNCORE_TARGET} ${CUDA_NVTX_LIB})
    endif()
  endif()

  if(WITH_CUTLASS)
    target_link_libraries(${CINNCORE_TARGET} cutlass)
    add_dependencies(${CINNCORE_TARGET} cutlass)
  endif()
endfunction()

gen_cinncore(STATIC)
gen_cinncore(SHARED)

# --------distribute cinncore lib and include begin--------
set(PUBLISH_LIBS ON)
if(PUBLISH_LIBS)
  set(core_includes
      "${core_includes};paddle/cinn/runtime/cuda/cinn_cuda_runtime_source.cuh")
  set(core_includes
      "${core_includes};paddle/cinn/runtime/hip/cinn_hip_runtime_source.h")
  set(core_includes
      "${core_includes};paddle/common/flags.h;paddle/utils/test_macros.h")
  foreach(header ${core_includes})
    get_filename_component(prefix ${header} DIRECTORY)
    file(COPY ${header}
         DESTINATION ${CMAKE_BINARY_DIR}/dist/cinn/include/${prefix})
  endforeach()

  foreach(proto_header ${core_proto_includes})
    string(REPLACE ${CMAKE_BINARY_DIR}/ "" relname ${proto_header})
    get_filename_component(prefix ${relname} DIRECTORY)
    set(target_name ${CMAKE_BINARY_DIR}/dist/cinn/include/${relname})
    add_custom_command(
      TARGET cinnapi
      POST_BUILD
      COMMENT "copy generated proto header '${relname}' to dist"
      COMMAND cmake -E copy ${proto_header} ${target_name} DEPENDS cinnapi)
  endforeach()

  add_custom_command(
    TARGET cinnapi
    POST_BUILD
    COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/libcinnapi.so
            ${CMAKE_BINARY_DIR}/dist/cinn/lib/libcinnapi.so
    COMMAND cmake -E copy_directory ${CINN_THIRD_PARTY_PATH}/install
            ${CMAKE_BINARY_DIR}/dist/third_party DEPENDS cinnapi)
  add_custom_command(
    TARGET cinncore_static
    POST_BUILD
    COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/libcinncore_static.a
            ${CMAKE_BINARY_DIR}/dist/cinn/lib/libcinncore_static.a
    COMMAND
      cmake -E copy ${CMAKE_BINARY_DIR}/paddle/cinn/hlir/pe/libparam_proto.a
      ${CMAKE_BINARY_DIR}/dist/cinn/lib/libparam_proto.a
    COMMAND
      cmake -E copy
      ${CMAKE_BINARY_DIR}/paddle/cinn/ir/schedule/libschedule_desc_proto.a
      ${CMAKE_BINARY_DIR}/dist/cinn/lib/libschedule_desc_proto.a
    COMMENT "distribute libcinncore_static.a and related header files." DEPENDS
            cinncore_static)
endif()
# --------distribute cinncore lib and include end--------

set(CINN_LIB_NAME "libcinnapi.so")
set(CINN_LIB_LOCATION "${CMAKE_BINARY_DIR}/dist/cinn/lib")
set(CINN_LIB "${CINN_LIB_LOCATION}/${CINN_LIB_NAME}")

######################################
# Add CINN's dependencies header files
######################################

# Add isl
set(ISL_INCLUDE_DIR "${CMAKE_BINARY_DIR}/third_party/install/isl/include")
include_directories(${ISL_INCLUDE_DIR})

# Add LLVM
set(LLVM_INCLUDE_DIR "${CMAKE_BINARY_DIR}/dist/third_party/llvm/include")
include_directories(${LLVM_INCLUDE_DIR})

######################################################
# Put external_cinn and dependencies together as a lib
######################################################

set(CINN_INCLUDE_DIR "${CMAKE_BINARY_DIR}/dist/cinn/include")
include_directories(${CINN_INCLUDE_DIR})
