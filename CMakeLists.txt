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
# limitations under the License

if(APPLE AND WITH_ARM)
  # cmake 3.19.2 version starts to support M1
  cmake_minimum_required(VERSION 3.19.2)
  cmake_policy(VERSION 3.19.2)
else()
  cmake_minimum_required(VERSION 3.15)
  cmake_policy(VERSION 3.10)
endif()
# use to get_property location of static lib
# https://cmake.org/cmake/help/v3.0/policy/CMP0026.html?highlight=cmp0026
cmake_policy(SET CMP0026 OLD)
cmake_policy(SET CMP0079 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PADDLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PADDLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(system)

# Note(zhouwei): Ninja Generator will set CMAKE_BUILD_TYPE to Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

project(paddle CXX C)

# enable language CUDA
# TODO(Shibo Tao): remove find_package(CUDA) completely.
find_package(CUDA QUIET)
find_package(MKL CONFIG QUIET)
option(WITH_ONEMKL "Compile PaddlePaddle with oneMKL" OFF)
option(WITH_GPU "Compile PaddlePaddle with NVIDIA GPU" ${CUDA_FOUND})
option(WITH_MPI "Compile PaddlePaddle with MPI" OFF)
option(WITH_TENSORRT "Compile PaddlePaddle with NVIDIA TensorRT" OFF)
option(WITH_OPENVINO "Compile PaddlePaddle with Intel OpenVINO" OFF)
option(WITH_XPU "Compile PaddlePaddle with BAIDU KUNLUN XPU" OFF)
option(WITH_XPU_KP "Compile PaddlePaddle with BAIDU XPU compiler " OFF)
option(WITH_XPU_XFT "Compile PaddlePaddle with BAIDU XPU-XFT" OFF)
option(WITH_XPU_PLUGIN "Compile PaddlePaddle with BAIDU XPU plugin" OFF)
option(WITH_XPU_XRE5 "Compile PaddlePaddle with BAIDU XPU XRE 5" OFF)
option(WITH_WIN_DUMP_DBG "Compile with windows core dump debug mode" OFF)
option(WITH_ROCM "Compile PaddlePaddle with ROCM platform" OFF)
option(WITH_IPU "Compile PaddlePaddle with Graphcore IPU" OFF)
option(WITH_ONNXRUNTIME "Compile PaddlePaddle with ONNXRUNTIME" OFF)
option(WITH_CUSPARSELT "Compile PaddlePaddle with CUSPARSELT" OFF)
option(WITH_SETUP_INSTALL "Compile PaddlePaddle with setup.py" OFF)
option(WITH_SHARED_PHI "Compile PaddlePaddle with SHARED LIB of PHI" ON)
option(CINN_WITH_CUDNN "Compile CINN with CUDNN support" ON)
option(WITH_PIP_CUDA_LIBRARIES
       "Paddle uses the CUDA library provided by NVIDIA" OFF)
option(WITH_PIP_TENSORRT "Paddle uses the tensorrt provided by NVIDIA" OFF)
option(WITH_NIGHTLY_BUILD
       "Compile nightly paddle whl package of the develop branch" OFF)
option(WITH_CPP_TEST "Compile PaddlePaddle skip cpp test" ON)
find_package(Git REQUIRED)

# config GIT_URL with github mirrors to speed up dependent repos clone
option(GIT_URL "Git URL to clone dependent repos" ${GIT_URL})
if(NOT GIT_URL)
  set(GIT_URL "https://github.com")
endif()

# Note(zhouwei): It use option above, so put here
include(init)
include(generic) # simplify cmake module
include(experimental) # experimental build options

if(WITH_GPU AND WITH_XPU)
  message(FATAL_ERROR "Error when compile GPU and XPU at the same time")
endif()
if(WITH_GPU AND WITH_XPU_KP)
  message(FATAL_ERROR "Error when compile GPU and XPU2 at the same time")
endif()
if(WITH_GPU AND WITH_XPU_XFT)
  message(FATAL_ERROR "Error when compile GPU and XPU-XFT at the same time")
endif()
if(WITH_GPU AND WITH_XPU_XRE5)
  message(FATAL_ERROR "Error when compile GPU and XPU-XRE5 at the same time")
endif()
if(WITH_GPU AND WITH_ROCM)
  message(FATAL_ERROR "Error when compile CUDA and ROCM at the same time")
endif()

if(WITH_GPU AND NOT APPLE)
  if(WITH_PIP_CUDA_LIBRARIES AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_definitions(-DPADDLE_WITH_PIP_CUDA_LIBRARIES)
  endif()
  #(Note risemeup1): The cudart dynamic library libcudart.so is used by set CUDA_USE_STATIC_CUDA_RUNTIME and CMAKE_CUDA_FLAGS
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL
                                            "x86_64")
    set(CUDA_USE_STATIC_CUDA_RUNTIME
        OFF
        CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_FLAGS "--cudart shared")
    if(WITH_PIP_CUDA_LIBRARIES)
      #(Note risemeup1): Flag 'PADDLE_WITH_PIP_CUDA_LIBRARIES' will be used in dynamic_loader.cc to search for CUDA-related .so files through the Python libraries provided by NVIDIA.
      add_definitions(-DPADDLE_WITH_PIP_CUDA_LIBRARIES)
    endif()
  endif()
  enable_language(CUDA)
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}, version: "
                 "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND CMAKE_CXX_COMPILER_VERSION
                                              VERSION_GREATER 10.4)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=uninitialized")
endif()

# MUSL build turn off warnings

if(WITH_MUSL)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations -Wno-error=pessimizing-move -Wno-error=deprecated-copy"
  )
endif()

if(APPLE AND WITH_ARM)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target arm64-apple-darwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -target arm64-apple-darwin")
endif()

if(WIN32)
  option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
  message("Build static library of PHI")
  # (Note xuxinyi04): If CMAKE_SUPPRESS_REGENERATION is OFF, which is default, then CMake adds a
  # special target on which all other targets depend that checks the build system and optionally
  # re-runs CMake to regenerate the build system when the target specification source changes.
  set(CMAKE_SUPPRESS_REGENERATION OFF)
  set(CMAKE_STATIC_LIBRARY_PREFIX lib)
  set(WITH_SHARED_PHI
      OFF
      CACHE BOOL "Disable WITH_SHARED_PHI when compiling PADDLE ON WIN32" FORCE)

  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /bigobj")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /bigobj")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")

  if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Zc:inline")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zc:inline")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zc:inline")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zc:inline")
  endif()

  if(MSVC_STATIC_CRT)
    message(
      STATUS
        "Use static C runtime time, refer to https://docs.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=vs-2019"
    )
    foreach(
      flag_var
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
      CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
  endif()

  # msvc max/min macro conflict with std::min/max, define NOMINMAX globally
  add_definitions("-DNOMINMAX")

  # 1. windows.h define 'small' cause CUDA11.6/11.7/11.8 's cub compile error,
  # see https://github.com/microsoft/onnxruntime/issues/11227
  # 2. WIN32_LEAN_AND_MEAN minimize the windows include files, avoid define 'small'
  add_definitions(-DWIN32_LEAN_AND_MEAN)

  # windows build turn off warnings, use parallel compiling.
  foreach(
    flag_var
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "/W[1-4]" " /W0 " ${flag_var} "${${flag_var}}")

    # NOTE(zhouwei25): GPU compile have too high memory utilization when parallel compiling,
    # For Visual Studio generators, /MP should be added.
    # For other generators like Ninja, it is not need to add /MP.
    if(CMAKE_GENERATOR MATCHES "Visual Studio" AND NOT WITH_GPU)
      math(EXPR PROCESS_MAX "${CPU_CORES} * 2 / 3")
      set(${flag_var} "${${flag_var}} /MP${PROCESS_MAX}")
    endif()
  endforeach()
  foreach(flag_var CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
    set(${flag_var} "${${flag_var}} /w")
  endforeach()

  # Windows Remove /Zi, /ZI for Release, MinSizeRel builds
  foreach(flag_var
          CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_MINSIZEREL
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL)
    if(${flag_var} MATCHES "/Z[iI]")
      string(REGEX REPLACE "/Z[iI]" "" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()

  set(CMAKE_C_FLAGS
      "${CMAKE_C_FLAGS} /wd4068 /wd4129 /wd4244 /wd4267 /wd4297 /wd4530 /wd4577 /wd4819 /wd4838"
  )
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} /wd4068 /wd4129 /wd4244 /wd4267 /wd4297 /wd4530 /wd4577 /wd4819 /wd4838"
  )

  foreach(flag_var CMAKE_SHARED_LINKER_FLAGS CMAKE_STATIC_LINKER_FLAGS
                   CMAKE_EXE_LINKER_FLAGS CMAKE_LINKER_FLAGS)
    set(${flag_var}
        "${${flag_var}} /ignore:4049 /ignore:4217 /ignore:4006 /ignore:4221")
    if(MSVC_STATIC_CRT)
      set(${flag_var} "${${flag_var}} /NODEFAULTLIB:MSVCRT.LIB")
    else()
      set(${flag_var} "${${flag_var}} /NODEFAULTLIB:LIBCMT.LIB")
    endif()
  endforeach()

  if(WITH_WIN_DUMP_DBG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Zi")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")

    foreach(flag_var CMAKE_SHARED_LINKER_FLAGS CMAKE_STATIC_LINKER_FLAGS
                     CMAKE_EXE_LINKER_FLAGS CMAKE_LINKER_FLAGS)
      set(${flag_var} "${${flag_var}} /DEBUG /OPT:REF /OPT:ICF")
    endforeach()

    add_definitions("-DWITH_WIN_DUMP_DBG")
  endif()

else()
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations"
  )
endif()

find_package(Threads REQUIRED)

include(simd)

################################ Exposed Configurations #######################################
option(WITH_AVX "Compile PaddlePaddle with AVX intrinsics" ${AVX_FOUND})
option(WITH_PYTHON "Compile PaddlePaddle with python interpreter" ON)
option(WITH_TESTING "Compile PaddlePaddle with unit testing" OFF)
option(WITH_MULTINODE_TESTING "Test multinode apis and ops" OFF)
option(WITH_MKL "Compile PaddlePaddle with MKL support." ${AVX_FOUND})
option(WITH_SYSTEM_BLAS "Use system blas library" OFF)
option(WITH_DISTRIBUTE "Compile with distributed support" OFF)
option(WITH_BRPC_RDMA "Use brpc rdma as the rpc protocal" OFF)
option(ON_INFER "Turn on inference optimization and inference-lib generation"
       ON)
option(WITH_CPP_DIST "Install PaddlePaddle C++ distribution" OFF)
option(WITH_GFLAGS "Compile PaddlePaddle with gflags support" OFF)
################################ Internal Configurations #######################################
option(WITH_NV_JETSON "Compile PaddlePaddle with NV JETSON" OFF)
option(WITH_PROFILER "Compile PaddlePaddle with GPU profiler and gperftools"
       OFF)
option(WITH_COVERAGE "Compile PaddlePaddle with code coverage" OFF)
option(WITH_INCREMENTAL_COVERAGE
       "Generate coverage reports only for incremental code" OFF)
option(WITH_LIBXSMM "Compile with libxsmm" OFF)
option(COVERALLS_UPLOAD "Package code coverage data to coveralls" OFF)
option(WITH_PSLIB "Compile with pslib support" OFF)
option(WITH_BOX_PS "Compile with box_ps support" OFF)
option(WITH_XBYAK "Compile with xbyak support" ON)
option(WITH_PSCORE "Compile with parameter server support" ${WITH_DISTRIBUTE})
option(WITH_HETERPS "Compile with heterps" OFF)
option(WITH_INFERENCE_API_TEST
       "Test fluid inference C++ high-level api interface" OFF)
option(WITH_NVTX "Paddle with nvtx for profiler" OFF)
option(PY_VERSION "Compile PaddlePaddle with python3 support" ${PY_VERSION})
option(WITH_DGC "Use DGC(Deep Gradient Compression) or not" ${WITH_DISTRIBUTE})
option(
  SANITIZER_TYPE
  "Choose the type of sanitizer, options are: Address, Leak, Memory, Thread, Undefined"
  OFF)
option(WITH_CINN "Compile PaddlePaddle with CINN" OFF)
option(WITH_NCCL "Compile PaddlePaddle with NCCL support" ON)
option(WITH_RCCL "Compile PaddlePaddle with RCCL support" ON)
option(WITH_XPU_BKCL "Compile PaddlePaddle with BAIDU KUNLUN XPU BKCL" OFF)
option(WITH_UCC "Compile PaddlePaddle with UCC support" OFF)
option(WITH_CRYPTO "Compile PaddlePaddle with crypto support" ON)
option(WITH_ARM "Compile PaddlePaddle with arm support" OFF)
option(WITH_SW "Compile PaddlePaddle with sw support" OFF)
option(WITH_MIPS "Compile PaddlePaddle with mips support" OFF)
option(WITH_LOONGARCH "Compile PaddlePaddle with loongarch support" OFF)
option(WITH_MUSL "Compile with musl libc instead of gblic" OFF)
option(WITH_UNITY_BUILD "Compile with UnityBuild mode" OFF)
option(WITH_STRIP "Strip so files of Whl packages" OFF)
option(NEW_RELEASE_PYPI
       "PaddlePaddle next-level release strategy for pypi cubin package" OFF)
option(NEW_RELEASE_ALL
       "PaddlePaddle next-level release strategy for all arches cubin package"
       OFF)
option(NEW_RELEASE_JIT
       "PaddlePaddle next-level release strategy for backup jit package" OFF)
option(WITH_POCKETFFT "Compile with pocketfft support" ON)
option(WITH_RECORD_BUILDTIME
       "Compile PaddlePaddle with record all targets build time" OFF)
option(WITH_CUSTOM_DEVICE "Compile with custom device support" OFF)
option(WITH_ARM_BRPC "Supprot Brpc in Arm" OFF)
option(WITH_FLPS "FL PS mode" OFF)
option(WITH_RPC "Compile with rpc support" ${WITH_DISTRIBUTE})
option(WITH_CUDNN_FRONTEND
       "Compile with CUDNN Frontend API support (experimental)" OFF)
option(WITH_SHARED_IR "Compile PaddlePaddle with SHARED LIB of IR" ON)
option(WITH_NVCC_LAZY
       "Compile PaddlePaddle with nvcc lazy mode, used for CI-Inference only."
       ON)
option(BUILD_WHL_PACKAGE "Build paddle whl package after compilation" ON)

if(WITH_RECORD_BUILDTIME)
  set_property(
    GLOBAL
    PROPERTY
      RULE_LAUNCH_COMPILE
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/get_build_time.sh ${CMAKE_CURRENT_BINARY_DIR}"
  )
  set_property(
    GLOBAL
    PROPERTY
      RULE_LAUNCH_LINK
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/get_build_time.sh ${CMAKE_CURRENT_BINARY_DIR}"
  )
else()
  include(ccache
  )# set ccache for compilation ; if WITH_RECORD_BUILDTIME=ON can't use ccache
endif()
unset(WITH_RECORD_BUILDTIME CACHE)

# PY_VERSION
if(NOT PY_VERSION)
  set(PY_VERSION 3.8)
elseif(${PY_VERSION} VERSION_LESS 3.8)
  message(FATAL_ERROR "Paddle only support Python version>=3.8 now")
endif()
set(PYBIND11_PYTHON_VERSION ${PY_VERSION})

# the type of sanitizer, options are: Address, Leak, Memory, Thread, Undefined. Default: OFF
if(SANITIZER_TYPE AND NOT "${SANITIZER_TYPE}" MATCHES
                      "^(Address|Leak|Memory|Thread|Undefined)$")
  message("Choose the correct type of sanitizer")
  return()
endif()

if(LINUX
   AND NOT WITH_CUSTOM_DEVICE
   AND NOT WITH_GPU
   AND NOT WITH_ROCM
   AND NOT WITH_XPU
   AND NOT WITH_XPU_KP
   AND NOT WITH_XPU_XFT
   AND WITH_PYTHON)
  set(WITH_CUSTOM_DEVICE
      ON
      CACHE BOOL "Enable Custom Device when compiling for Linux" FORCE)
  message(
    "Enable Custom Device when compiling for Linux. Force WITH_CUSTOM_DEVICE=ON."
  )
endif()

if(WIN32)
  if(WITH_DISTRIBUTE)
    message(
      WARNING
        "Disable DISTRIBUTE when compiling for Windows. Force WITH_DISTRIBUTE=OFF."
    )
    set(WITH_DISTRIBUTE
        OFF
        CACHE STRING "Disable DISTRIBUTE when compiling for Windows" FORCE)
  endif()
  if(WITH_NCCL)
    message(
      WARNING "Disable NCCL when compiling for Windows. Force WITH_NCCL=OFF.")
    set(WITH_NCCL
        OFF
        CACHE STRING "Disable NCCL when compiling for Windows" FORCE)
  endif()
endif()

if(NOT WITH_TESTING AND WITH_MULTINODE_TESTING)
  message(
    WARNING
      "Disable WITH_MULTINODE_TESTING when compiling without TESTING. Force WITH_MULTINODE_TESTING=OFF."
  )
  set(WITH_MULTINODE_TESTING
      OFF
      CACHE STRING
            "Disable WITH_MULTINODE_TESTING when compiling without TESTING"
            FORCE)
endif()

if(NOT WITH_GPU AND WITH_NCCL)
  message(
    WARNING "Disable NCCL when compiling without GPU. Force WITH_NCCL=OFF.")
  set(WITH_NCCL
      OFF
      CACHE STRING "Disable NCCL when compiling without GPU" FORCE)
endif()

# force WITH_XPU on when WITH_XPU_KP
if(WITH_XPU_KP AND NOT WITH_XPU)
  message(
    WARNING
      "Enable WITH_XPU when compiling with WITH_XPU_KP. Force WITH_XPU=ON.")
  set(WITH_XPU
      ON
      CACHE STRING "Enable WITH_XPU when compiling with WITH_XPU_KP" FORCE)
endif()

if(NOT WITH_XPU AND WITH_XPU_XFT)
  message(
    WARNING
      "Enable WITH_XPU when compiling with WITH_XPU_XFT. Force WITH_XPU=ON.")
  set(WITH_XPU
      ON
      CACHE STRING "Enable WITH_XPU when compiling with WITH_XPU_XFT" FORCE)
endif()

if(NOT WITH_XPU AND WITH_XPTI)
  message(
    WARNING "Disable XPTI when compiling without XPU. Force WITH_XPTI=OFF.")
  set(WITH_XPTI
      OFF
      CACHE STRING "Disable XPTI when compiling without XPU" FORCE)
endif()

if(NOT WITH_XPU AND WITH_XPU_BKCL)
  message(
    WARNING "Disable BKCL when compiling without XPU. Force WITH_XPU_BKCL=OFF.")
  set(WITH_XPU_BKCL
      OFF
      CACHE STRING "Disable BKCL when compiling without XPU" FORCE)
endif()

if(NOT WITH_XPU AND WITH_XPU_XRE5)
  message((WARNING
           "Disable XRE5 when compiling without XPU. Force WITH_XPU_XRE5=OFF"))
  set(WITH_XPU_XRE5
      OFF
      CACHE STRING "Disable XRE5 when compiling without XPU" FORCE)
endif()

if(WITH_NCCL)
  add_definitions("-DPADDLE_WITH_NCCL")
  include(nccl)
else()
  if(WITH_GPU)
    message(
      WARNING
        "If the environment is multi-card, the WITH_NCCL option needs to be turned on, otherwise only a single card can be used."
    )
  endif()
endif()

if(WITH_BRPC_RDMA)
  message(STATUS "Use brpc with rdma.")
  if(NOT WITH_DISTRIBUTE)
    message(FATAL_ERROR "Can't use brpc rdma in no distribute env.")
  endif()
endif()

if(WITH_GPU)
  include(cuda)
  # lite subgraph compilation depends on CUDNN_ROOT,
  # so include(cudnn) needs to be in front of include(third_party/lite)
  include(cudnn) # set cudnn libraries, must before configure
  include(tensorrt)
  # there is no official support of nccl, cupti in windows
  if(NOT WIN32)
    include(cupti)
  endif()
endif()

if(WITH_ROCM)
  include(hip)
  include(miopen) # set miopen libraries, must before configure
  include(cupti)
endif()

if(WITH_XPU_KP)
  include(xpu_kp)
endif()

if(NOT WITH_ROCM AND WITH_RCCL)
  message(
    WARNING "Disable RCCL when compiling without ROCM. Force WITH_RCCL=OFF.")
  set(WITH_RCCL
      OFF
      CACHE STRING "Disable RCCL when compiling without ROCM" FORCE)
endif()

if(WITH_RCCL)
  add_definitions("-DPADDLE_WITH_RCCL")
  include(rccl)
else()
  if(WITH_ROCM)
    message(
      WARNING
        "If the environment is multi-card, the WITH_RCCL option needs to be turned on, otherwise only a single card can be used."
    )
  endif()
endif()

if(WITH_UCC)
  add_definitions("-DPADDLE_WITH_UCC")
  include(ucc)
endif()

if(WITH_HETERPS AND WITH_PSLIB)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(WITH_DISTRIBUTE)
  if(LINUX)
    set(WITH_GLOO
        ON
        CACHE STRING "Enable GLOO when compiling WITH_DISTRIBUTE=ON." FORCE)
  endif()
  if(WITH_ROCM AND HIP_VERSION LESS_EQUAL 40020496)
    # TODO(qili93): third-party rocksdb throw Illegal instruction with HIP version 40020496
    message(
      WARNING
        "Disable WITH_PSCORE when HIP_VERSION is less than or equal 40020496. Force WITH_PSCORE=OFF."
    )
    set(WITH_PSCORE
        OFF
        CACHE
          BOOL
          "Disable WITH_PSCORE when HIP_VERSION is less than or equal 40020496"
          FORCE)
  endif()
endif()

if(WITH_RPC)
  if(NOT LINUX)
    message(
      WARNING "Disable WITH_RPC when not compiled on Linux. Force WITH_RPC=OFF."
    )
    set(WITH_RPC
        OFF
        CACHE BOOL "Disable WITH_RPC when not compiled on Linux" FORCE)
  endif()
  if(NOT WITH_DISTRIBUTE AND WITH_RPC)
    message(
      WARNING
        "Disable WITH_RPC when not compiled with distribute. Force WITH_RPC=OFF."
    )
    set(WITH_RPC
        OFF
        CACHE BOOL "Disable WITH_RPC when not compiled with distribute" FORCE)
  endif()
  if(WITH_ROCM AND WITH_RPC)
    message(
      WARNING "Disable WITH_RPC when compiling with ROCM. Force WITH_RPC=OFF.")
    set(WITH_RPC
        OFF
        CACHE BOOL "Disable WITH_RPC when compiling with ROCM" FORCE)
  endif()
  if(WITH_XPU AND WITH_RPC)
    message(
      WARNING "Disable WITH_RPC when compiling with XPU. Force WITH_RPC=OFF.")
    set(WITH_RPC
        OFF
        CACHE BOOL "Disable WITH_RPC when compiling with XPU" FORCE)
  endif()
endif()

if(WITH_MPI)
  include(mpi)
endif()

include(third_party
)# download, build, install third_party, Contains about 20+ dependencies

include(flags) # set paddle compile flags
include(util) # set unittest and link libs
include(version) # set PADDLE_VERSION
include(coveralls) # set code coverage
include(configure) # add paddle env configuration
#------------- cinn cmake config start --------------

if(WITH_CINN)
  message(STATUS "Compile Paddle with CINN.")
  # TODO(6clc): Use CINN_WITH_CUDNN to completely replace WITH_CUDNN in CINN.
  #             Use WITH_GPU to completely replace WITH_CUDA in CINN.
  set(WITH_MKL_CBLAS ${WITH_MKL})
  if(WITH_GPU)
    set(WITH_CUDA ${WITH_GPU})
    add_definitions(-DCINN_WITH_CUDA)
    set(WITH_CUDNN ${CINN_WITH_CUDNN})
    if(WITH_CUDNN)
      add_definitions(-DCINN_WITH_CUDNN)
    endif()
  endif()

  include(cmake/cinn.cmake)
  add_definitions(-DPADDLE_WITH_CINN)
endif()

#------------- cinn cmake config end --------------

if(WITH_PROFILER)
  find_package(Gperftools REQUIRED)
  include_directories(${GPERFTOOLS_INCLUDE_DIR})
  add_definitions(-DWITH_GPERFTOOLS)
endif()

include_directories("${PADDLE_SOURCE_DIR}")

if(WITH_NV_JETSON)
  set(WITH_ARM
      ON
      CACHE STRING "Set WITH_ARM=ON when compiling WITH_NV_JETSON=ON." FORCE)
endif()

if(WITH_ARM)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(WITH_XBYAK
      OFF
      CACHE STRING "Disable XBYAK when compiling WITH_ARM=ON." FORCE)
  set(WITH_MKL
      OFF
      CACHE STRING "Disable MKL when compiling WITH_ARM=ON." FORCE)
  set(WITH_AVX
      OFF
      CACHE STRING "Disable AVX when compiling WITH_AVX=OFF." FORCE)
  add_definitions(-DPADDLE_WITH_ARM)
endif()

if(WITH_SW)
  # mieee flag solves floating-point exceptions under sw and ALPHA architectures
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -mieee")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -mieee")
  set(WITH_XBYAK
      OFF
      CACHE STRING "Disable XBYAK when compiling WITH_SW=ON" FORCE)
  set(WITH_MKL
      OFF
      CACHE STRING "Disable MKL when compiling WITH_SW=ON." FORCE)
  add_definitions(-DPADDLE_WITH_SW)
endif()

if(WITH_MIPS)
  set(WITH_XBYAK
      OFF
      CACHE STRING "Disable XBYAK when compiling WITH_MIPS=ON" FORCE)
  add_definitions(-DPADDLE_WITH_MIPS)
endif()

if(WITH_NVTX AND NOT WIN32)
  add_definitions(-DPADDLE_WITH_NVTX)
endif()

if(WITH_LOONGARCH)
  set(WITH_XBYAK
      OFF
      CACHE STRING "Disable XBYAK when compiling WITH_LOONGARCH=ON" FORCE)
  set(WITH_MKL
      OFF
      CACHE STRING "Disable MKL when compiling WITH_LOONGARCH=ON." FORCE)
  add_definitions(-DPADDLE_WITH_LOONGARCH)
endif()

if(WITH_ONEMKL)
  add_definitions(-DPADDLE_WITH_ONEMKL)
endif()

if(WITH_HETERPS)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
  endif()
endif()
set(PADDLE_PYTHON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/build")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

add_definitions(-DPADDLE_DLL_EXPORT)

if(ON_INFER)
  # you can trun off the paddle fluid and inference lib by set ON_INFER=OFF
  message(
    STATUS "On inference mode, will take place some specific optimization.")
  include(inference_lib)
  add_definitions(-DPADDLE_ON_INFERENCE)
  set(WITH_SHARED_IR
      OFF
      CACHE BOOL "Only paddle_inference.so is allowed in inference." FORCE)
else()
  #TODO(luotao), combine this warning with `make inference_lib_dist` command.
  message(
    WARNING
      "On inference mode, will take place some specific optimization. Turn on the ON_INFER flag when building inference_lib only."
  )
endif()

if(NOT WITH_SHARED_IR)
  add_definitions(-DSTATIC_IR)
endif()

if(WITH_STRIP)
  find_program(STRIP_PATH strip)
  if(NOT STRIP_PATH OR NOT LINUX)
    set(WITH_STRIP
        OFF
        CACHE STRING "Command strip is only used on Linux when it exists."
              FORCE)
  endif()
endif()

if(WITH_CPP_DIST)
  # TODO(huangjiyi): Separate installing C++ distribution from python package
  # installation and support for installing C++ distribution on more platforms.
  if(NOT LINUX OR NOT WITH_PYTHON)
    set(WITH_CPP_DIST
        OFF
        CACHE
          STRING
          "Currently C++ Distribution Generation is only available on Linux and compiling WITH_PYTHON=ON."
          FORCE)
  else()
    include(paddle_lib)
  endif()
endif()

add_subdirectory(paddle)
if(WITH_PYTHON)
  add_subdirectory(python)
endif()
add_subdirectory(test)

get_directory_property(all_inc_dirs INCLUDE_DIRECTORIES)
list(JOIN all_inc_dirs "\r\n" all_inc_dirs)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/includes.txt" ${all_inc_dirs})
