# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT WITH_XPU)
  return()
endif()

include(ExternalProject)
set(XPU_PROJECT "extern_xpu")
set(XPU_API_LIB_NAME "libxpuapi.so")
set(XPU_RT_LIB_NAME "libxpurt.so")
set(XPU_CUDA_LIB_NAME "libxpucuda.so")
set(XPU_CUDA_RT_LIB_NAME "libcudart.so")
set(XPU_ML_LIB_NAME "libxpuml.so")
set(XPU_XFT_LIB_NAME "libxft.so")
set(XPU_XPTI_LIB_NAME "libxpti.so")
set(XPU_XBLAS_LIB_NAME "libxpu_blas.so")
set(XPU_XFA_LIB_NAME "libxpu_flash_attention.so")
set(XPU_XPUDNN_LIB_NAME "libxpu_dnn.so")

if(NOT DEFINED XPU_XHPC_BASE_DATE)
  set(XPU_XHPC_BASE_DATE "dev/20250204")
endif()
set(XPU_XCCL_BASE_VERSION "3.0.2.1") # For XRE5
if(NOT DEFINED XPU_XFT_BASE_VERSION)
  set(XPU_XFT_BASE_VERSION "20230602")
endif()

if(NOT DEFINED XPU_XRE_BASE_VERSION)
  if(WITH_XPU_XRE5)
    set(XPU_XRE_BASE_VERSION "5.0.21.15")
  else()
    set(XPU_XRE_BASE_VERSION "4.32.0.1")
  endif()
endif()

set(XPU_XPTI_BASE_VERSION "0.0.1")

set(XPU_XRE_BASE_URL
    "https://klx-sdk-release-public.su.bcebos.com/xre/release/${XPU_XRE_BASE_VERSION}"
)

set(XPU_XCCL_BASE_URL
    "https://klx-sdk-release-public.su.bcebos.com/xccl/release/2.0.0.1" # 2.0.0.1 is the final version for R200
)

if(NOT XPU_XFT_BASE_URL)
  set(XPU_XFT_BASE_URL
      "https://klx-sdk-release-public.su.bcebos.com/xft/dev/${XPU_XFT_BASE_VERSION}"
  )
endif()

if(WITH_XPTI)
  set(XPU_XPTI_BASE_URL
      "https://klx-sdk-release-public.su.bcebos.com/xpti/dev/${XPU_XPTI_BASE_VERSION}"
  )
  set(XPU_XPTI_DIR_NAME "xpti")
endif()

if(WITH_XPU_XRE5)
  set(XPU_XRE_BASE_URL
      "https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/${XPU_XRE_BASE_VERSION}"
  )
  set(XPU_XCCL_BASE_URL
      "https://klx-sdk-release-public.su.bcebos.com/xccl/release/${XPU_XCCL_BASE_VERSION}"
  )
endif()

if(WITH_AARCH64)
  set(XPU_XRE_DIR_NAME "xre-kylin_aarch64")
  set(XPU_XCCL_DIR_NAME "") # TODO: xccl has no kylin output now.
  set(XPU_XFT_DIR_NAME "") # TODO: xft has no kylin output at now.
elseif(WITH_SUNWAY)
  set(XPU_XRE_DIR_NAME "xre-deepin_sw6_64")
  set(XPU_XCCL_DIR_NAME "") # TODO: xccl has no deepin output at now.
  set(XPU_XFT_DIR_NAME "") # TODO: xft has no deepin output at now.
elseif(WITH_BDCENTOS)
  set(XPU_XHPC_DIR_NAME "xhpc-bdcentos7_x86_64")
  if(WITH_XPU_XRE5)
    set(XPU_XRE_DIR_NAME "xre-bdcentos-x86_64-${XPU_XRE_BASE_VERSION}")
  else()
    set(XPU_XRE_DIR_NAME "xre-bdcentos_x86_64")
  endif()
  set(XPU_XCCL_DIR_NAME "xccl_bdcentos_x86_64")
  set(XPU_XFT_DIR_NAME "xft_bdcentos6u3_x86_64_gcc82")
elseif(WITH_CENTOS)
  set(XPU_XRE_DIR_NAME "xre-centos7_x86_64")
  set(XPU_XCCL_DIR_NAME "xccl_Linux_x86_64")
  set(XPU_XFT_DIR_NAME "xft_bdcentos6u3_x86_64_gcc82")
else()
  # Ubuntu as default
  if(WITH_XPU_XRE5)
    set(XPU_XRE_DIR_NAME "xre-ubuntu_2004-x86_64-${XPU_XRE_BASE_VERSION}")
    set(XPU_XHPC_DIR_NAME "xhpc-ubuntu2004_x86_64")
  else()
    set(XPU_XRE_DIR_NAME "xre-ubuntu_1604_x86_64")
    set(XPU_XHPC_DIR_NAME "xhpc-ubuntu1604_x86_64")
  endif()
  set(XPU_XCCL_DIR_NAME "xccl_Linux_x86_64")
  set(XPU_XFT_DIR_NAME "xft_ubuntu1604_x86_64")
endif()

set(XPU_XRE_URL
    "${XPU_XRE_BASE_URL}/${XPU_XRE_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)
set(XPU_XCCL_URL
    "${XPU_XCCL_BASE_URL}/${XPU_XCCL_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)
set(XPU_XFT_URL "${XPU_XFT_BASE_URL}/${XPU_XFT_DIR_NAME}.tar.gz")
set(XPU_XFT_GET_DEPENCE_URL
    "https://baidu-kunlun-public.su.bcebos.com/paddle_depence/get_xft_dependence.sh"
    CACHE STRING "" FORCE)

if(WITH_XPTI)
  set(XPU_XPTI_URL "${XPU_XPTI_BASE_URL}/${XPU_XPTI_DIR_NAME}.tar.gz")
endif()

set(XPU_XHPC_URL
    "https://klx-sdk-release-public.su.bcebos.com/xhpc/${XPU_XHPC_BASE_DATE}/${XPU_XHPC_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)

if(DEFINED XPU_BASE_URL)
  set(XPU_XRE_URL "${XPU_BASE_URL}/${XPU_XRE_DIR_NAME}.tar.gz")
  set(XPU_XHPC_URL "${XPU_BASE_URL}/${XPU_XHPC_DIR_NAME}.tar.gz")
  set(XPU_XCCL_URL "${XPU_BASE_URL}/${XPU_XCCL_DIR_NAME}.tar.gz")
endif()

set(SNAPPY_PREFIX_DIR "${THIRD_PARTY_PATH}/xpu")
set(XPU_DOWNLOAD_DIR "${SNAPPY_PREFIX_DIR}/src/${XPU_PROJECT}")
set(XPU_INSTALL_DIR "${THIRD_PARTY_PATH}/install/xpu")
set(XPU_INC_DIR "${THIRD_PARTY_PATH}/install/xpu/include")
set(XPU_LIB_DIR "${THIRD_PARTY_PATH}/install/xpu/lib")

set(XPU_API_LIB "${XPU_LIB_DIR}/${XPU_API_LIB_NAME}")
set(XPU_XBLAS_LIB "${XPU_LIB_DIR}/${XPU_XBLAS_LIB_NAME}")
set(XPU_RT_LIB "${XPU_LIB_DIR}/${XPU_RT_LIB_NAME}")
set(XPU_CUDA_LIB "${XPU_LIB_DIR}/${XPU_CUDA_LIB_NAME}")
set(XPU_CUDA_RT_LIB "${XPU_LIB_DIR}/${XPU_CUDA_RT_LIB_NAME}")
set(XPU_ML_LIB "${XPU_LIB_DIR}/${XPU_ML_LIB_NAME}")
set(XPU_XFA_LIB "${XPU_LIB_DIR}/${XPU_XFA_LIB_NAME}")
set(XPU_XPUDNN_LIB "${XPU_LIB_DIR}/${XPU_XPUDNN_LIB_NAME}")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${XPU_INSTALL_DIR}/lib")

file(
  WRITE ${XPU_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(XPU)\n" "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY xpu/include xpu/lib \n"
  "        DESTINATION ${XPU_INSTALL_DIR})\n")

if(WITH_XPU_BKCL)
  message(STATUS "Compile with XPU BKCL!")
  add_definitions(-DPADDLE_WITH_XPU_BKCL)

  set(XPU_BKCL_LIB_NAME "libbkcl.so")
  set(XPU_BKCL_LIB "${XPU_LIB_DIR}/${XPU_BKCL_LIB_NAME}")
  set(XPU_BKCL_INC_DIR "${THIRD_PARTY_PATH}/install/xpu/include")
  include_directories(${XPU_BKCL_INC_DIR})
endif()

set(XFT_COMMAND "get_xft_dependence.sh")
# If the corresponding path exists locally, use it directly; otherwise, download it.
# The current implementation only supports XPU-related libraries being located in the same directory.
if(DEFINED ENV{XPU_LIB_ROOT})
  message(STATUS "Compile with LOCAL XPU LIBS!")
  set(XPU_LIB_ROOT "$ENV{XPU_LIB_ROOT}")

  # XRE
  if(DEFINED ENV{XPU_XRE_DIR_NAME})
    set(XPU_XRE_URL "${XPU_LIB_ROOT}/$ENV{XPU_XRE_DIR_NAME}")
    set(XPU_XRE_DIR_NAME "$ENV{XPU_XRE_DIR_NAME}")
  endif()

  # XCCL
  if(DEFINED ENV{XCCL_DIR_NAME})
    set(XPU_XCCL_URL "${XPU_LIB_ROOT}/$ENV{XCCL_DIR_NAME}")
    set(XCCL_DIR_NAME "$ENV{XCCL_DIR_NAME}")
  endif()

  # XHPC
  if(DEFINED ENV{XPU_XHPC_DIR_NAME})
    set(XPU_XHPC_URL "${XPU_LIB_ROOT}/$ENV{XPU_XHPC_DIR_NAME}")
    set(XPU_XHPC_DIR_NAME "$ENV{XPU_XHPC_DIR_NAME}")
  endif()

  # XFT
  if(DEFINED ENV{XPU_XFT_DIR_NAME})
    set(XPU_XFT_DIR_NAME "$ENV{XPU_XFT_DIR_NAME}")
    set(XPU_XFT_URL "${XPU_LIB_ROOT}/${XPU_XFT_DIR_NAME}")
    set(XFT_COMMAND
        "${CMAKE_SOURCE_DIR}/tools/xpu/get_xft_dependence_from_custom_path.sh")
  endif()
endif()

if(WITH_XPU_XRE5)
  ExternalProject_Add(
    ${XPU_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${SNAPPY_PREFIX_DIR}
    DOWNLOAD_DIR ${XPU_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND
      bash ${CMAKE_SOURCE_DIR}/tools/xpu/pack_paddle_dependence.sh
      ${XPU_XRE_URL} ${XPU_XRE_DIR_NAME} ${XPU_XHPC_URL} ${XPU_XHPC_DIR_NAME}
      ${XPU_XCCL_URL} ${XPU_XCCL_DIR_NAME} 1 && wget ${XPU_XFT_GET_DEPENCE_URL}
      && bash ${XFT_COMMAND} ${XPU_XFT_URL} ${XPU_XFT_DIR_NAME} && bash
      ${CMAKE_SOURCE_DIR}/tools/xpu/get_xpti_dependence.sh ${XPU_XPTI_URL}
      ${XPU_XPTI_DIR_NAME}
    DOWNLOAD_NO_PROGRESS 1
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${XPU_INSTALL_ROOT}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${XPU_INSTALL_ROOT}
    BUILD_BYPRODUCTS ${XPU_API_LIB}
    BUILD_BYPRODUCTS ${XPU_XBLAS_LIB}
    BUILD_BYPRODUCTS ${XPU_XPUDNN_LIB}
    BUILD_BYPRODUCTS ${XPU_XFA_LIB}
    BUILD_BYPRODUCTS ${XPU_RT_LIB}
    BUILD_BYPRODUCTS ${XPU_CUDA_RT_LIB}
    BUILD_BYPRODUCTS ${XPU_ML_LIB}
    BUILD_BYPRODUCTS ${XPU_BKCL_LIB})
else()
  ExternalProject_Add(
    ${XPU_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${SNAPPY_PREFIX_DIR}
    DOWNLOAD_DIR ${XPU_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND
      bash ${CMAKE_SOURCE_DIR}/tools/xpu/pack_paddle_dependence.sh
      ${XPU_XRE_URL} ${XPU_XRE_DIR_NAME} ${XPU_XHPC_URL} ${XPU_XHPC_DIR_NAME}
      ${XPU_XCCL_URL} ${XPU_XCCL_DIR_NAME} 0 && wget ${XPU_XFT_GET_DEPENCE_URL}
      && bash get_xft_dependence.sh ${XPU_XFT_URL} ${XPU_XFT_DIR_NAME} && bash
      ${CMAKE_SOURCE_DIR}/tools/xpu/get_xpti_dependence.sh ${XPU_XPTI_URL}
      ${XPU_XPTI_DIR_NAME}
    DOWNLOAD_NO_PROGRESS 1
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${XPU_INSTALL_ROOT}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${XPU_INSTALL_ROOT}
    BUILD_BYPRODUCTS ${XPU_API_LIB}
    BUILD_BYPRODUCTS ${XPU_RT_LIB}
    BUILD_BYPRODUCTS ${XPU_BKCL_LIB})
endif()

include_directories(${XPU_INC_DIR})
add_library(shared_xpuapi SHARED IMPORTED GLOBAL)
set_property(TARGET shared_xpuapi PROPERTY IMPORTED_LOCATION "${XPU_API_LIB}")

# generate a static dummy target to track xpulib dependencies
# for cc_library(xxx SRCS xxx.c DEPS xpulib)
generate_dummy_static_lib(LIB_NAME "xpulib" GENERATOR "xpu.cmake")

if(WITH_XPU_XFT)
  message(STATUS "Compile with XPU XFT!")
  add_definitions(-DPADDLE_WITH_XPU_XFT)

  set(XPU_XFT_INC_DIR "${XPU_INC_DIR}/xft")
  include_directories(${XPU_XFT_INC_DIR})
  set(XPU_XFT_LIB "${XPU_LIB_DIR}/${XPU_XFT_LIB_NAME}")
  target_link_libraries(xpulib ${XPU_XFT_LIB})
endif()

set(XPU_XHPC_INC_DIR "${XPU_INC_DIR}/xhpc")
include_directories(${XPU_XHPC_INC_DIR})
set(XPU_XRE_INC_DIR "${XPU_INC_DIR}/xre")
include_directories(${XPU_XRE_INC_DIR})

if(WITH_XPU_XRE5)
  add_definitions(-DPADDLE_WITH_XPU_XRE5)
  set(XPU_XBLAS_INC_DIR "${XPU_INC_DIR}/xhpc/xblas")
  include_directories(${XPU_XBLAS_INC_DIR})
  set(XPU_XFA_INC_DIR "${XPU_INC_DIR}/xhpc/xfa")
  include_directories(${XPU_XFA_INC_DIR})
  set(XPU_XPUDNN_INC_DIR "${XPU_INC_DIR}/xhpc/xpudnn")
  include_directories(${XPU_XPUDNN_INC_DIR})
endif()

if(WITH_XPTI)
  message(STATUS "Compile with XPU XPTI!")
  add_definitions(-DPADDLE_WITH_XPTI)
  set(XPU_XPTI_LIB "${XPU_LIB_DIR}/${XPU_XPTI_LIB_NAME}")
endif()

if(WITH_XPU_PLUGIN)
  message(STATUS "Compile with XPU PLUGIN!")
  add_definitions(-DPADDLE_WITH_XPU_PLUGIN)
  include_directories(${CMAKE_SOURCE_DIR}/paddle/phi/kernels/xpu/plugin/include)
endif()

if(WITH_XPTI)
  target_link_libraries(xpulib ${XPU_XPTI_LIB})
endif()

if(WITH_XPU_XRE5)
  target_link_libraries(
    xpulib
    ${XPU_RT_LIB}
    ${XPU_CUDA_RT_LIB}
    ${XPU_XBLAS_LIB}
    ${XPU_API_LIB}
    ${XPU_XFA_LIB}
    ${XPU_XPUDNN_LIB})
else()
  target_link_libraries(xpulib ${XPU_RT_LIB} ${XPU_API_LIB})
endif()

if(WITH_XPU_BKCL)
  if(WITH_XPU_XRE5)
    target_link_libraries(xpulib ${XPU_ML_LIB} ${XPU_BKCL_LIB})
  else()
    target_link_libraries(xpulib ${XPU_BKCL_LIB})
  endif()
endif()

add_dependencies(xpulib ${XPU_PROJECT})

# Ensure that xpu/api.h can be included without dependency errors.
file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc
  CONTENT "")
add_library(xpu_headers_dummy STATIC
            ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc)
add_dependencies(xpu_headers_dummy extern_xpu)
link_libraries(xpu_headers_dummy)
