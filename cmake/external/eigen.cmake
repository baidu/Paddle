# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
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

# update eigen to the commit id f612df27 on 03/16/2021
set(EIGEN_PREFIX_DIR ${THIRD_PARTY_PATH}/eigen3)
set(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3/src/extern_eigen3)
set(EIGEN_TAG f612df273689a19d25b45ca4f8269463207c4fee)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/eigen3)

if(WIN32)
  add_definitions(-DEIGEN_STRONG_INLINE=inline)
elseif(LINUX)
  if(WITH_ROCM)
    # For HIPCC Eigen::internal::device::numeric_limits is not EIGEN_DEVICE_FUNC
    # which will cause compiler error of using __host__ function
    # in __host__ __device__
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/Meta.h native_src)
    file(TO_NATIVE_PATH ${SOURCE_DIR}/Eigen/src/Core/util/Meta.h native_dst)
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/TensorReductionGpu.h
         native_src1)
    file(TO_NATIVE_PATH
         ${SOURCE_DIR}/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h
         native_dst1)
    set(EIGEN_PATCH_COMMAND cp ${native_src} ${native_dst} && cp ${native_src1}
                            ${native_dst1})
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/TensorRandom.h.patch
       tensor_random_header)
  # See: [Why calling some `git` commands before `patch`?]
  set(EIGEN_PATCH_COMMAND
      git checkout -- . && git checkout ${EIGEN_TAG} && patch -Nd
      ${SOURCE_DIR}/unsupported/Eigen/CXX11/src/Tensor <
      ${tensor_random_header})
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/Complex.h.patch
       complex_header)
  set(EIGEN_PATCH_COMMAND
      ${EIGEN_PATCH_COMMAND} && patch -Nd
      ${SOURCE_DIR}/Eigen/src/Core/arch/SSE/ < ${complex_header})
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/Half.h.patch
       half_header)
  set(EIGEN_PATCH_COMMAND
      ${EIGEN_PATCH_COMMAND} && patch -Nd
      ${SOURCE_DIR}/Eigen/src/Core/arch/Default/ < ${half_header})
endif()

set(EIGEN_INCLUDE_DIR ${SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})
if(NOT WIN32)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=maybe-uninitialized")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=maybe-uninitialized")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=uninitialized")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=uninitialized")
  endif()
endif()
if(NOT WIN32
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  message(STATUS "GCC version is >= 13.0, adding -Wno-error=stringop-overflow.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=stringop-overflow ")
endif()
ExternalProject_Add(
  extern_eigen3
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${EIGEN_PREFIX_DIR}
  CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  UPDATE_COMMAND ""
  PATCH_COMMAND ${EIGEN_PATCH_COMMAND}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(eigen3 INTERFACE)

add_dependencies(eigen3 extern_eigen3)

# sw not support thread_local semantic
if(WITH_SW)
  add_definitions(-DEIGEN_AVOID_THREAD_LOCAL)
endif()
