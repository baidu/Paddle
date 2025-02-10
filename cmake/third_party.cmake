# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)
# Create a target named "third_party", which can compile external dependencies on all platform(windows/linux/mac)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

set(THIRD_PARTY_PATH
    "${CMAKE_BINARY_DIR}/third_party"
    CACHE STRING
          "A path setting third party libraries download & build directories.")
set(THIRD_PARTY_CACHE_PATH
    "${CMAKE_SOURCE_DIR}"
    CACHE STRING
          "A path cache third party source code to avoid repeated download.")

set(THIRD_PARTY_BUILD_TYPE Release)
set(third_party_deps)

include(ProcessorCount)
ProcessorCount(NPROC)
if(NOT WITH_SETUP_INSTALL)
  #NOTE(risemeup1):Initialize any submodules.
  message(
    STATUS
      "Check submodules of paddle, and run 'git submodule sync --recursive && git submodule update --init --recursive'"
  )

  # execute_process does not support sequential commands, so we execute echo command separately
  execute_process(
    COMMAND git submodule sync --recursive
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
    RESULT_VARIABLE result_var)
  if(NOT result_var EQUAL 0)
    message(FATAL_ERROR "Failed to sync submodule, please check your network !")
  endif()

  if(WITH_OPENVINO)
    execute_process(
      COMMAND git submodule update --init --depth=1 third_party/openvino
      WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
      RESULT_VARIABLE result_var)
    # List of modules to be deleted
    set(delete_module
        "thirdparty/zlib/zlib"
        "thirdparty/gflags/gflags"
        "thirdparty/gtest/gtest"
        "thirdparty/ocl/icd_loader"
        "thirdparty/ocl/cl_headers"
        "thirdparty/ocl/clhpp_headers"
        "thirdparty/onnx/onnx"
        "src/bindings/python/thirdparty/pybind11"
        "thirdparty/ittapi/ittapi"
        "cmake/developer_package/ncc_naming_style/ncc"
        "src/plugins/intel_gpu/thirdparty/onednn_gpu"
        "thirdparty/open_model_zoo"
        "thirdparty/json/nlohmann_json"
        "thirdparty/flatbuffers/flatbuffers"
        "thirdparty/snappy"
        "thirdparty/level_zero/level-zero"
        "src/plugins/intel_npu/thirdparty/level-zero-ext"
        "src/plugins/intel_npu/thirdparty/yaml-cpp")
    # Iterate over each module and perform actions
    foreach(module IN LISTS delete_module)
      # Remove the module from git cache
      execute_process(
        COMMAND git rm --cached ${module}
        WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/third_party/openvino
        RESULT_VARIABLE git_rm_result)
    endforeach()
    execute_process(
      COMMAND git submodule update --init --recursive
      WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
      RESULT_VARIABLE result_var)
  else()
    execute_process(
      COMMAND git submodule status
      WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
      OUTPUT_VARIABLE submodule_list
      RESULT_VARIABLE result_var)
    string(REGEX MATCHALL "third_party/[^ )\n]+" submodule_paths
                 "${submodule_list}")
    foreach(submodule IN LISTS submodule_paths)
      if(NOT submodule STREQUAL "third_party/openvino")
        execute_process(
          COMMAND git submodule update --init --recursive ${submodule}
          WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
          RESULT_VARIABLE result_var)
      endif()
    endforeach()
  endif()
  if(NOT result_var EQUAL 0)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      set(THIRD_PARTY_TAR_URL
          https://xly-devops.bj.bcebos.com/PR/build_whl/0/third_party.tar.gz
          CACHE STRING "third_party.tar.gz url")
      execute_process(
        COMMAND wget -q ${THIRD_PARTY_TAR_URL}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE wget_result)
      if(NOT wget_result EQUAL 0)
        message(
          FATAL_ERROR
            "Failed to download third_party.tar.gz, please check your network !"
        )
      else()
        execute_process(
          COMMAND tar -xzf third_party.tar.gz -C ${CMAKE_SOURCE_DIR}/
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
          RESULT_VARIABLE tar_result)
        if(NOT tar_result EQUAL 0)
          message(
            FATAL_ERROR
              "Failed to extract third_party.tar.gz, please make sure tar.gz file is not corrupted !"
          )
        endif()
      endif()
    else()
      message(
        FATAL_ERROR "Failed to update submodule, please check your network !")
    endif()
  endif()

endif()
# cache function to avoid repeat download code of third_party.
# This function has 4 parameters, URL / REPOSITORY / TAG / DIR:
# 1. URL:           specify download url of 3rd party
# 2. REPOSITORY:    specify git REPOSITORY of 3rd party
# 3. TAG:           specify git tag/branch/commitID of 3rd party
# 4. DIR:           overwrite the original SOURCE_DIR when cache directory
#
# The function Return 1 PARENT_SCOPE variables:
#  - ${TARGET}_DOWNLOAD_CMD: Simply place "${TARGET}_DOWNLOAD_CMD" in ExternalProject_Add,
#                            and you no longer need to set any download steps in ExternalProject_Add.
# For example:
#    Cache_third_party(${TARGET}
#            REPOSITORY ${TARGET_REPOSITORY}
#            TAG        ${TARGET_TAG}
#            DIR        ${TARGET_SOURCE_DIR})

function(cache_third_party TARGET)
  set(options "")
  set(oneValueArgs URL REPOSITORY TAG DIR)
  set(multiValueArgs "")
  cmake_parse_arguments(cache_third_party "${optionps}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  string(REPLACE "extern_" "" TARGET_NAME ${TARGET})
  string(REGEX REPLACE "[0-9]+" "" TARGET_NAME ${TARGET_NAME})
  string(TOUPPER ${TARGET_NAME} TARGET_NAME)
  if(cache_third_party_REPOSITORY)
    set(${TARGET_NAME}_DOWNLOAD_CMD GIT_REPOSITORY
                                    ${cache_third_party_REPOSITORY})
    if(cache_third_party_TAG)
      list(APPEND ${TARGET_NAME}_DOWNLOAD_CMD GIT_TAG ${cache_third_party_TAG})
    endif()
  elseif(cache_third_party_URL)
    set(${TARGET_NAME}_DOWNLOAD_CMD URL ${cache_third_party_URL})
  else()
    message(
      FATAL_ERROR "Download link (Git repo or URL) must be specified for cache!"
    )
  endif()
  if(WITH_TP_CACHE)
    if(NOT cache_third_party_DIR)
      message(
        FATAL_ERROR
          "Please input the ${TARGET_NAME}_SOURCE_DIR for overwriting when -DWITH_TP_CACHE=ON"
      )
    endif()
    # Generate and verify cache dir for third_party source code
    set(cache_third_party_REPOSITORY ${cache_third_party_REPOSITORY}
                                     ${cache_third_party_URL})
    if(cache_third_party_REPOSITORY AND cache_third_party_TAG)
      string(MD5 HASH_REPO ${cache_third_party_REPOSITORY})
      string(MD5 HASH_GIT ${cache_third_party_TAG})
      string(SUBSTRING ${HASH_REPO} 0 8 HASH_REPO)
      string(SUBSTRING ${HASH_GIT} 0 8 HASH_GIT)
      string(CONCAT HASH ${HASH_REPO} ${HASH_GIT})
      # overwrite the original SOURCE_DIR when cache directory
      set(${cache_third_party_DIR}
          ${THIRD_PARTY_CACHE_PATH}/third_party/${TARGET}_${HASH})
    elseif(cache_third_party_REPOSITORY)
      string(MD5 HASH_REPO ${cache_third_party_REPOSITORY})
      string(SUBSTRING ${HASH_REPO} 0 16 HASH)
      # overwrite the original SOURCE_DIR when cache directory
      set(${cache_third_party_DIR}
          ${THIRD_PARTY_CACHE_PATH}/third_party/${TARGET}_${HASH})
    endif()

    if(EXISTS ${${cache_third_party_DIR}})
      # judge whether the cache dir is empty
      file(GLOB files ${${cache_third_party_DIR}}/*)
      list(LENGTH files files_len)
      if(files_len GREATER 0)
        list(APPEND ${TARGET_NAME}_DOWNLOAD_CMD DOWNLOAD_COMMAND "")
      endif()
    endif()
    set(${cache_third_party_DIR}
        ${${cache_third_party_DIR}}
        PARENT_SCOPE)
  endif()

  # Pass ${TARGET_NAME}_DOWNLOAD_CMD to parent scope, the double quotation marks can't be removed
  set(${TARGET_NAME}_DOWNLOAD_CMD
      "${${TARGET_NAME}_DOWNLOAD_CMD}"
      PARENT_SCOPE)
endfunction()

macro(UNSET_VAR VAR_NAME)
  unset(${VAR_NAME} CACHE)
  unset(${VAR_NAME})
endmacro()

# Function to Download the dependencies during compilation
# This function has 2 parameters, URL / DIRNAME:
# 1. URL:           The download url of 3rd dependencies
# 2. NAME:          The name of file, that determine the dirname
#
function(file_download_and_uncompress URL NAME)
  set(options "")
  set(oneValueArgs MD5)
  set(multiValueArgs "")
  cmake_parse_arguments(URL "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  message(STATUS "Download dependence[${NAME}] from ${URL}, MD5: ${URL_MD5}")
  set(${NAME}_INCLUDE_DIR
      ${THIRD_PARTY_PATH}/${NAME}/data
      PARENT_SCOPE)
  ExternalProject_Add(
    download_${NAME}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${THIRD_PARTY_PATH}/${NAME}
    URL ${URL}
    URL_MD5 ${URL_MD5}
    TIMEOUT 120
    DOWNLOAD_DIR ${THIRD_PARTY_PATH}/${NAME}/data/
    SOURCE_DIR ${THIRD_PARTY_PATH}/${NAME}/data/
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND "")
  set(third_party_deps
      ${third_party_deps} download_${NAME}
      PARENT_SCOPE)
endfunction()

# Correction of flags on different Platform(WIN/MAC) and Print Warning Message
if(APPLE)
  if(WITH_MKL)
    message(
      WARNING "Mac is not supported with MKL in Paddle yet. Force WITH_MKL=OFF."
    )
    set(WITH_MKL
        OFF
        CACHE STRING "Disable MKL for building on mac" FORCE)
  endif()
endif()

if(WIN32 OR APPLE)
  message(STATUS "Disable XBYAK in Windows and MacOS")
  set(WITH_XBYAK
      OFF
      CACHE STRING "Disable XBYAK in Windows and MacOS" FORCE)

  if(WITH_LIBXSMM)
    message(WARNING "Windows, Mac are not supported with libxsmm in Paddle yet."
                    "Force WITH_LIBXSMM=OFF")
    set(WITH_LIBXSMM
        OFF
        CACHE STRING "Disable LIBXSMM in Windows and MacOS" FORCE)
  endif()

  if(WITH_BOX_PS)
    message(WARNING "Windows or Mac is not supported with BOX_PS in Paddle yet."
                    "Force WITH_BOX_PS=OFF")
    set(WITH_BOX_PS
        OFF
        CACHE STRING "Disable BOX_PS package in Windows and MacOS" FORCE)
  endif()

  if(WITH_PSLIB)
    message(WARNING "Windows or Mac is not supported with PSLIB in Paddle yet."
                    "Force WITH_PSLIB=OFF")
    set(WITH_PSLIB
        OFF
        CACHE STRING "Disable PSLIB package in Windows and MacOS" FORCE)
  endif()

  if(WITH_ARM_BRPC)
    message(
      WARNING "Windows or Mac is not supported with ARM_BRPC in Paddle yet."
              "Force WITH_ARM_BRPC=OFF")
    set(WITH_ARM_BRPC
        OFF
        CACHE STRING "Disable ARM_BRPC package in Windows and MacOS" FORCE)
  endif()

  if(WITH_LIBMCT)
    message(WARNING "Windows or Mac is not supported with LIBMCT in Paddle yet."
                    "Force WITH_LIBMCT=OFF")
    set(WITH_LIBMCT
        OFF
        CACHE STRING "Disable LIBMCT package in Windows and MacOS" FORCE)
  endif()

  if(WITH_PSLIB_BRPC)
    message(
      WARNING "Windows or Mac is not supported with PSLIB_BRPC in Paddle yet."
              "Force WITH_PSLIB_BRPC=OFF")
    set(WITH_PSLIB_BRPC
        OFF
        CACHE STRING "Disable PSLIB_BRPC package in Windows and MacOS" FORCE)
  endif()
endif()

set(WITH_MKLML ${WITH_MKL})
if(NOT DEFINED WITH_ONEDNN)
  if(WITH_MKL AND AVX2_FOUND)
    set(WITH_ONEDNN ON)
  else()
    message(STATUS "Do not have AVX2 intrinsics and disabled MKL-DNN.")
    set(WITH_ONEDNN OFF)
  endif()
endif()

if(WIN32)
  if(MSVC)
    if(MSVC_VERSION LESS 1920)
      set(WITH_ONEDNN OFF)
    endif()
  endif()
endif()

if(WIN32
   OR APPLE
   OR NOT WITH_GPU
   OR (ON_INFER AND NOT WITH_PYTHON))
  set(WITH_DGC OFF)
endif()

if(${CMAKE_VERSION} VERSION_GREATER "3.5.2")
  set(SHALLOW_CLONE "GIT_SHALLOW TRUE"
  )# adds --depth=1 arg to git clone of External_Projects
endif()

include(external/zlib) # download, build, install zlib
include(external/gflags) # download, build, install gflags
include(external/glog) # download, build, install glog

########################### include third_party according to flags ###############################
if(WITH_GPU
   AND NOT WITH_ARM
   AND NOT WIN32
   AND NOT APPLE)
  if(${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 11.0)
    include(external/cutlass) # download, build, install cusparselt
    list(APPEND third_party_deps extern_cutlass)
    set(WITH_CUTLASS ON)
  endif()
endif()

if(WITH_CINN)
  if(WITH_MKL)
    add_definitions(-DCINN_WITH_MKL_CBLAS)
  endif()
  if(WITH_ONEDNN)
    add_definitions(-DCINN_WITH_DNNL)
  endif()
  include(cmake/cinn/version.cmake)
  if(NOT EXISTS ${CMAKE_BINARY_DIR}/cmake/cinn/config.cmake)
    file(COPY ${PROJECT_SOURCE_DIR}/cmake/cinn/config.cmake
         DESTINATION ${CMAKE_BINARY_DIR}/cmake/cinn)
  endif()
  include(${CMAKE_BINARY_DIR}/cmake/cinn/config.cmake)
  include(cmake/cinn/external/absl.cmake)
  include(cmake/cinn/external/llvm.cmake)
  include(cmake/cinn/external/isl.cmake)
  include(cmake/cinn/external/ginac.cmake)
  include(cmake/cinn/external/openmp.cmake)
  include(cmake/cinn/external/jitify.cmake)
endif()

include(external/eigen) # download eigen3
include(external/threadpool) # download threadpool
include(external/dlpack) # download dlpack
include(external/xxhash) # download, build, install xxhash
include(external/warpctc) # download, build, install warpctc
include(external/warprnnt) # download, build, install warprnnt
include(external/utf8proc) # download, build, install utf8proc

list(APPEND third_party_deps extern_eigen3 extern_gflags extern_glog
     extern_xxhash)
list(
  APPEND
  third_party_deps
  extern_zlib
  extern_dlpack
  extern_warpctc
  extern_warprnnt
  extern_threadpool
  extern_utf8proc)
include(external/lapack) # download, build, install lapack

list(APPEND third_party_deps extern_eigen3 extern_gflags extern_glog
     extern_xxhash)
list(
  APPEND
  third_party_deps
  extern_zlib
  extern_dlpack
  extern_warpctc
  extern_warprnnt
  extern_threadpool
  extern_lapack)

include(cblas) # find first, then download, build, install openblas

message(STATUS "CBLAS_PROVIDER: ${CBLAS_PROVIDER}")
if(${CBLAS_PROVIDER} STREQUAL MKLML)
  list(APPEND third_party_deps extern_mklml)
elseif(${CBLAS_PROVIDER} STREQUAL EXTERN_OPENBLAS)
  list(APPEND third_party_deps extern_openblas)
endif()

if(WITH_ONEDNN)
  include(external/onednn) # download, build, install onednn
  list(APPEND third_party_deps extern_onednn)
endif()

include(external/protobuf) # find first, then download, build, install protobuf
if(TARGET extern_protobuf)
  list(APPEND third_party_deps extern_protobuf)
endif()

include(external/json) # find first, then build json
if(TARGET extern_json)
  list(APPEND third_party_deps extern_json)
endif()

include(external/yaml) # find first, then build yaml
if(TARGET extern_yaml)
  list(APPEND third_party_deps extern_yaml)
endif()

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  include(external/python) # find python and python_module
  include(external/pybind11) # prepare submodule pybind11
  list(APPEND third_party_deps extern_pybind)
endif()

if(WITH_TESTING OR WITH_DISTRIBUTE)
  include(external/gtest) # download, build, install gtest
  list(APPEND third_party_deps extern_gtest)
endif()

if(WITH_ONNXRUNTIME)
  include(external/onnxruntime
  )# download, build, install onnxruntime、paddle2onnx
  include(external/paddle2onnx)
  list(APPEND third_party_deps extern_onnxruntime extern_paddle2onnx)
endif()

if(WITH_GPU)
  if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.0)
    include(external/cub) # download cub
    list(APPEND third_party_deps extern_cub)
  elseif(${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 12.0 AND WITH_SHARED_PHI)
    include(external/cccl)
    add_definitions(-DPADDLE_WITH_CCCL)
  endif()
  set(URL
      "https://paddlepaddledeps.bj.bcebos.com/externalErrorMsg_20210928.tar.gz"
      CACHE STRING "" FORCE)
  file_download_and_uncompress(
    ${URL} "externalError" MD5 a712a49384e77ca216ad866712f7cafa
  )# download file externalErrorMsg.tar.gz
  if(WITH_TESTING)
    # copy externalErrorMsg.pb for UnitTest
    set(SRC_DIR ${THIRD_PARTY_PATH}/externalError/data)
    # for python UT 'test_exception.py'
    set(DST_DIR1
        ${CMAKE_BINARY_DIR}/python/paddle/include/third_party/externalError/data
    )
    # for C++ UT 'enforce_test'
    set(DST_DIR2 ${CMAKE_BINARY_DIR}/paddle/third_party/externalError/data)
    add_custom_command(
      TARGET download_externalError
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${SRC_DIR} ${DST_DIR1}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${SRC_DIR} ${DST_DIR2}
      COMMENT "copy_directory from ${SRC_DIR} to ${DST_DIR1}"
      COMMENT "copy_directory from ${SRC_DIR} to ${DST_DIR2}")
  endif()
endif()

if(WITH_XPU)
  include(external/xpu) # download, build, install xpu
  list(APPEND third_party_deps extern_xpu)
endif()

if(WITH_PSLIB)
  include(external/pslib) # download, build, install pslib
  list(APPEND third_party_deps extern_pslib)
  if(WITH_LIBMCT)
    include(external/libmct) # download, build, install libmct
    list(APPEND third_party_deps extern_libxsmm)
  endif()
  if(WITH_PSLIB_BRPC)
    include(external/pslib_brpc) # download, build, install pslib_brpc
    list(APPEND third_party_deps extern_pslib_brpc)
  else()
    include(external/snappy)
    list(APPEND third_party_deps extern_snappy)

    include(external/leveldb)
    list(APPEND third_party_deps extern_leveldb)
    if(NOT WITH_HETERPS)
      include(external/brpc)
      list(APPEND third_party_deps extern_brpc)
    endif()
  endif()
endif()

if(NOT WIN32 AND NOT APPLE)
  include(external/gloo)
  list(APPEND third_party_deps extern_gloo)
endif()

if(WITH_BOX_PS)
  include(external/box_ps)
  list(APPEND third_party_deps extern_box_ps)
endif()

if(WITH_PSCORE)
  include(external/snappy)
  list(APPEND third_party_deps extern_snappy)

  include(external/leveldb)
  list(APPEND third_party_deps extern_leveldb)

  if(WITH_ARM_BRPC)
    include(external/arm_brpc)
    list(APPEND third_party_deps extern_arm_brpc)
  else()
    include(external/brpc)
    list(APPEND third_party_deps extern_brpc)
  endif()

  include(external/libmct) # download, build, install libmct
  list(APPEND third_party_deps extern_libmct)

  include(external/rocksdb) # download, build, install rocksdb
  list(APPEND third_party_deps extern_rocksdb)

  include(external/jemalloc) # download, build, install jemalloc
  list(APPEND third_party_deps extern_jemalloc)

  include(external/afs_api)
  list(APPEND third_party_deps extern_afs_api)
endif()

if(WITH_RPC
   AND NOT WITH_PSCORE
   AND NOT WITH_PSLIB)
  include(external/snappy)
  list(APPEND third_party_deps extern_snappy)

  include(external/leveldb)
  list(APPEND third_party_deps extern_leveldb)

  include(external/brpc)
  list(APPEND third_party_deps extern_brpc)
endif()

if(WITH_DISTRIBUTE
   AND NOT WITH_PSLIB
   AND NOT WITH_PSCORE
   AND NOT WITH_RPC)
  include(external/snappy)
  list(APPEND third_party_deps extern_snappy)

  include(external/leveldb)
  list(APPEND third_party_deps extern_leveldb)
  include(external/brpc)
  list(APPEND third_party_deps extern_brpc)
endif()

if(WITH_XBYAK)
  include(external/xbyak) # prepare submodule xbyak
  list(APPEND third_party_deps extern_xbyak)
endif()

if(WITH_LIBXSMM)
  include(external/libxsmm) # download, build, install libxsmm
  list(APPEND third_party_deps extern_libxsmm)
endif()

if(WITH_DGC)
  message(STATUS "add dgc lib.")
  include(external/dgc) # download, build, install dgc
  add_definitions(-DPADDLE_WITH_DGC)
  list(APPEND third_party_deps extern_dgc)
endif()

if(WITH_CRYPTO)
  include(external/cryptopp) # download, build, install cryptopp
  list(APPEND third_party_deps extern_cryptopp)
  add_definitions(-DPADDLE_WITH_CRYPTO)
endif()

if(WITH_POCKETFFT)
  include(external/pocketfft)
  list(APPEND third_party_deps extern_pocketfft)
  add_definitions(-DPADDLE_WITH_POCKETFFT)
endif()

if(WIN32)
  include(external/dirent)
  list(APPEND third_party_deps extern_dirent)
endif()

if(WITH_IPU)
  include(external/poplar)
  list(APPEND third_party_deps extern_poplar)
endif()

if(WITH_CUSPARSELT)
  include(external/cusparselt) # download, build, install cusparselt
  list(APPEND third_party_deps extern_cusparselt)
endif()

if(WITH_ROCM AND NOT ROCM_6)
  include(external/flashattn)
  list(APPEND third_party_deps extern_flashattn)
  set(WITH_FLASHATTN ON)
endif()

if(WITH_GPU
   AND NOT WITH_ARM
   AND NOT WIN32
   AND NOT APPLE)
  if(${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 12.3)
    foreach(arch ${NVCC_ARCH_BIN})
      if(${arch} GREATER_EQUAL 90)
        set(WITH_FLASHATTN_V3 ON)
        break()
      endif()
    endforeach()
    foreach(arch ${NVCC_ARCH_BIN})
      if(${arch} GREATER_EQUAL 80)
        include(external/flashattn)
        list(APPEND third_party_deps extern_flashattn)
        set(WITH_FLASHATTN ON)
        break()
      endif()
    endforeach()
  elseif(${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 11.4)
    foreach(arch ${NVCC_ARCH_BIN})
      if(${arch} GREATER_EQUAL 80)
        include(external/flashattn)
        list(APPEND third_party_deps extern_flashattn)
        set(WITH_FLASHATTN ON)
        break()
      endif()
    endforeach()
  endif()
endif()

if(WITH_CUDNN_FRONTEND)
  include(external/cudnn-frontend) # download cudnn-frontend
  list(APPEND third_party_deps extern_cudnn_frontend)
endif()

if(WITH_OPENVINO)
  include(external/openvino)
  list(APPEND third_party_deps extern_openvino)
endif()

add_custom_target(third_party ALL DEPENDS ${third_party_deps})
