# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

if(WITH_ROCM)
  add_definitions(-DWARPRNNT_WITH_HIP)
endif()

set(WARPRNNT_PREFIX_DIR ${THIRD_PARTY_PATH}/warprnnt)
set(WARPRNNT_INSTALL_DIR ${THIRD_PARTY_PATH}/install/warprnnt)
set(WARPRNNT_TAG 7ea6bfe748779c245a0fcaa5dd9383826273eff2)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/warprnnt)
set(WARPRNNT_PATCH_COMMAND "")
set(WARPRNNT_CCBIN_OPTION "")
if(WIN32)
  set(WARPCTC_PATCH_CUDA_COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      ${PADDLE_SOURCE_DIR}/patches/warprnnt/CMakeLists.txt.cuda.patch
      "<SOURCE_DIR>/")
else()
  set(WARPCTC_PATCH_CUDA_COMMAND
      git checkout -- . && git checkout ${WARPRNNT_TAG} && patch -Nd
      ${SOURCE_DIR} <
      ${PADDLE_SOURCE_DIR}/patches/warprnnt/CMakeLists.txt.cuda.patch)
endif()
if(WITH_ROCM)
  set(WARPRNNT_PATCH_ROCM_COMMAND
      patch -p1 <
      ${PADDLE_SOURCE_DIR}/patches/warprnnt/CMakeLists.txt.rocm.patch)
  if(ROCM_6)
    set(WARPRNNT_PATCH_ROCM_COMMAND
        patch -p1 < ${PADDLE_SOURCE_DIR}/patches/warprnnt/hip.cmake.patch)
  endif()
endif()
if(NOT WIN32 AND WITH_GPU)
  if(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0 AND ${CMAKE_CXX_COMPILER_VERSION}
                                                  VERSION_GREATER 12.0)
    file(TO_NATIVE_PATH
         ${PADDLE_SOURCE_DIR}/patches/warprnnt/CMakeLists.txt.patch native_src)
    set(WARPRNNT_PATCH_COMMAND
        git checkout -- . && git checkout ${WARPRNNT_TAG} && patch -Nd
        ${SOURCE_DIR} < ${native_src})
    set(WARPRNNT_CCBIN_OPTION -DCCBIN_COMPILER=${CCBIN_COMPILER})
  endif()
endif()

set(WARPRNNT_INCLUDE_DIR
    "${WARPRNNT_INSTALL_DIR}/include"
    CACHE PATH "Warp-rnnt Directory" FORCE)
# Used in unit test test_WarpCTCLayer
set(WARPRNNT_LIB_DIR
    "${WARPRNNT_INSTALL_DIR}/lib"
    CACHE PATH "Warp-rnnt Library Directory" FORCE)

if(WIN32)
  set(WARPRNNT_LIBRARIES
      "${WARPRNNT_INSTALL_DIR}/bin/warprnnt${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "Warp-rnnt Library" FORCE)
else()
  set(WARPRNNT_LIBRARIES
      "${WARPRNNT_INSTALL_DIR}/lib/libwarprnnt${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "Warp-rnnt Library" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
   OR WIN32)
  set(USE_OMP OFF)
else()
  set(USE_OMP ON)
endif()

if(WIN32)
  set(WARPRNNT_C_FLAGS $<FILTER:${CMAKE_C_FLAGS},EXCLUDE,/Zc:inline>)
  set(WARPRNNT_C_FLAGS_DEBUG
      $<FILTER:${CMAKE_C_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
  set(WARPRNNT_C_FLAGS_RELEASE
      $<FILTER:${CMAKE_C_FLAGS_RELEASE},EXCLUDE,/Zc:inline>)
  set(WARPRNNT_CXX_FLAGS $<FILTER:${CMAKE_CXX_FLAGS},EXCLUDE,/Zc:inline>)
  set(WARPRNNT_CXX_FLAGS_RELEASE
      $<FILTER:${CMAKE_CXX_FLAGS_RELEASE},EXCLUDE,/Zc:inline>)
  set(WARPRNNT_CXX_FLAGS_DEBUG
      $<FILTER:${CMAKE_CXX_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
else()
  set(WARPRNNT_C_FLAGS ${CMAKE_C_FLAGS})
  set(WARPRNNT_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  set(WARPRNNT_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  set(WARPRNNT_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(WARPRNNT_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(WARPRNNT_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()
ExternalProject_Add(
  extern_warprnnt
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${WARPRNNT_PREFIX_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND
  COMMAND ${WARPCTC_PATCH_CUDA_COMMAND}
  COMMAND ${WARPRNNT_PATCH_ROCM_COMMAND}
  #BUILD_ALWAYS    1
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${WARPRNNT_C_FLAGS}
             -DCMAKE_C_FLAGS_DEBUG=${WARPRNNT_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${WARPRNNT_C_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS=${WARPRNNT_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_RELEASE=${WARPRNNT_CXX_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${WARPRNNT_CXX_FLAGS_DEBUG}
             -DCMAKE_INSTALL_PREFIX=${WARPRNNT_INSTALL_DIR}
             -DWITH_GPU=${WITH_GPU}
             -DWITH_ROCM=${WITH_ROCM}
             -DWITH_OMP=${USE_OMP}
             -DNVCC_FLAGS_EXTRA=${NVCC_FLAGS_EXTRA}
             -DBUILD_SHARED=ON
             -DBUILD_TESTS=OFF
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
             ${WARPCTC_CCBIN_OPTION}
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${WARPRNNT_INSTALL_DIR}
  BUILD_BYPRODUCTS ${WARPRNNT_LIBRARIES})

message(STATUS "warp-rnnt library: ${WARPRNNT_LIBRARIES}")
get_filename_component(WARPRNNT_LIBRARY_PATH ${WARPRNNT_LIBRARIES} DIRECTORY)
include_directories(${WARPRNNT_INCLUDE_DIR}
)# For warprnnt code to include its headers.

add_library(warprnnt INTERFACE)
# set_property(TARGET warprnnt PROPERTY IMPORTED_LOCATION ${WARPRNNT_LIBRARIES})
add_dependencies(warprnnt extern_warprnnt)
