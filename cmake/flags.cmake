# Setting Paddle Compile Flags
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(CheckCXXSymbolExists)
include(CheckTypeSize)

function(check_compiler_cxx14_flag)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 5.4)
      message(FATAL_ERROR "Unsupported GCC version. GCC >= 5.4 required.")
    elseif(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 8.2)
      message(
        WARNING
          "Found GCC ${CMAKE_CXX_COMPILER_VERSION} which is too high, recommended to use GCC 8.2"
      )
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    # cmake >= 3.0 compiler id "AppleClang" on Mac OS X, otherwise "Clang"
    # Apple Clang is a different compiler than upstream Clang which has different version numbers.
    # https://gist.github.com/yamaya/2924292
    if(APPLE) # cmake < 3.0 compiler id "Clang" on Mac OS X
      if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 5.1)
        message(
          FATAL_ERROR
            "Unsupported AppleClang version. AppleClang >= 5.1 required.")
      endif()
    else()
      if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.4)
        message(FATAL_ERROR "Unsupported Clang version. Clang >= 3.4 required.")
      endif()
    endif()
  endif()
endfunction()

check_compiler_cxx14_flag()

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
  set(CMAKE_CXX_STANDARD 17)
endif()

# safe_set_flag
#
# Set a compile flag only if compiler is support
# is_c: is C flag or C++ flag, bool type.
# src_list: The list name which the flag name will be append to.
# flag_name: the flag name for compiler, such as '-Werror' '-Wall' etc
# rest arguments: not used.
function(safe_set_flag is_c src_list flag_name)
  string(REPLACE "-" "_" safe_name ${flag_name})
  string(REPLACE "=" "_" safe_name ${safe_name})

  if(${flag_name} MATCHES "fsanitize")
    set(CMAKE_REQUIRED_FLAGS_RETAINED ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS ${flag_name})
  endif()

  if(is_c)
    check_c_compiler_flag(${flag_name} C_COMPILER_SUPPORT_FLAG_${safe_name})
    set(safe_name C_COMPILER_SUPPORT_FLAG_${safe_name})
  else()
    check_cxx_compiler_flag(${flag_name} CXX_COMPILER_SUPPORT_FLAG_${safe_name})
    set(safe_name CXX_COMPILER_SUPPORT_FLAG_${safe_name})
  endif()
  if(${safe_name})
    set(${src_list}
        "${${src_list}} ${flag_name}"
        PARENT_SCOPE)
  endif()

  if(${flag_name} MATCHES "fsanitize")
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_RETAINED})
  endif()
endfunction()

# helper macro to set cflag
macro(safe_set_cflag src_list flag_name)
  safe_set_flag(ON ${src_list} ${flag_name})
endmacro()

# helper macro to set cxxflag
macro(safe_set_cxxflag src_list flag_name)
  safe_set_flag(OFF ${src_list} ${flag_name})
endmacro()

# helper macro to set nvcc flag
macro(safe_set_nvflag flag_name)
  string(REPLACE "-" "_" safe_name ${flag_name})
  string(REPLACE "=" "_" safe_name ${safe_name})
  check_c_compiler_flag(${flag_name} C_COMPILER_SUPPORT_FLAG_${safe_name})
  set(safe_name C_COMPILER_SUPPORT_FLAG_${safe_name})
  if(${safe_name})
    if(WITH_ROCM)
      set(SAFE_GPU_COMMON_FLAGS
          "${SAFE_GPU_COMMON_FLAGS} -Xcompiler \"${flag_name}\"")
    else()
      set(SAFE_GPU_COMMON_FLAGS
          "${SAFE_GPU_COMMON_FLAGS} -Xcompiler=\"${flag_name}\"")
    endif()
  endif()
endmacro()

check_cxx_symbol_exists(UINT64_MAX "stdint.h" UINT64_MAX_EXISTS)
if(NOT UINT64_MAX_EXISTS)
  set(CMAKE_REQUIRED_DEFINITIONS -D__STDC_LIMIT_MACROS)
  check_cxx_symbol_exists(UINT64_MAX "stdint.h" UINT64_MAX_EXISTS_HERE)
  if(UINT64_MAX_EXISTS_HERE)
    set(CMAKE_REQUIRED_DEFINITIONS)
    add_definitions(-D__STDC_LIMIT_MACROS)
  else()
    message(FATAL_ERROR "Cannot find symbol UINT64_MAX")
  endif()
endif()

set(CMAKE_EXTRA_INCLUDE_FILES "pthread.h")
check_type_size(pthread_spinlock_t SPINLOCK_FOUND)
check_type_size(pthread_barrier_t BARRIER_FOUND)
if(SPINLOCK_FOUND)
  add_definitions(-DPADDLE_USE_PTHREAD_SPINLOCK)
endif()
if(BARRIER_FOUND)
  add_definitions(-DPADDLE_USE_PTHREAD_BARRIER)
endif()
set(CMAKE_EXTRA_INCLUDE_FILES "")

# Only one sanitizer is allowed in compile time
string(TOLOWER "${SANITIZER_TYPE}" sanitizer_type)
if(sanitizer_type STREQUAL "address")
  set(fsanitize "-fsanitize=address")
elseif(sanitizer_type STREQUAL "leak")
  set(fsanitize "-fsanitize=leak")
elseif(sanitizer_type STREQUAL "memory")
  set(fsanitize "-fsanitize=memory")
elseif(sanitizer_type STREQUAL "thread")
  set(fsanitize "-fsanitize=thread")
elseif(sanitizer_type STREQUAL "undefined")
  set(fsanitize "-fsanitize=undefined")
endif()

# Common flags. the compiler flag used for C/C++ sources whenever release or debug
# Do not care if this flag is support for gcc.

# https://github.com/PaddlePaddle/Paddle/issues/12773
if(NOT WIN32)
  set(COMMON_FLAGS
      -fPIC
      -fno-omit-frame-pointer
      -pipe
      -ffunction-sections
      -fdata-sections
      -Wl
      -gc-sections
      -Werror
      -Wall
      -Wextra
      -Wno-unused-parameter
      -Wno-unused-function
      -Wno-error=array-bounds #Warning in Eigen, gcc 12.2
      -Wno-error=ignored-attributes # Warnings in Eigen, gcc 6.3
      -Wno-error=int-in-bool-context # Warning in Eigen gcc 7.2
      -Wimplicit-fallthrough=0 # Warning in tinyformat.h
      ${fsanitize})

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
      set(COMMON_FLAGS ${COMMON_FLAGS} -Wno-error=deprecated-copy)
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(COMMON_FLAGS
        ${COMMON_FLAGS}
        -Wno-error=unknown-warning-option # For some unknown warning options in lower version clang
        -Wno-error=unused-private-field
        -Wno-error=unused-const-variable
        -Wno-error=deprecated-copy-with-user-provided-copy # For three/five/zeros rule, clang
        -Wno-error=deprecated-copy # Same above
        -Wno-error=inconsistent-missing-override # For lots of warnings when not using override for virtual functions, clang
        -Wno-error=bitwise-instead-of-logical # Warning in "unsupported/Eigen/CXX11/Tensor"
        -Wno-error=overloaded-virtual # For some inconsistent virtual function signature, clang
        -Wno-error=defaulted-function-deleted # header file from GLOO, clang
    )
  endif()

  if(WITH_IPU)
    set(COMMON_FLAGS ${COMMON_FLAGS} -Wno-sign-compare # Warnings in Popart
                     -Wno-non-virtual-dtor # Warnings in Popart
    )
  endif()

  if(WITH_CUDNN_FRONTEND)
    # flags from https://github.com/NVIDIA/cudnn-frontend/blob/v0.7.1/CMakeLists.txt
    set(COMMON_FLAGS ${COMMON_FLAGS} -Wno-sign-compare -Wno-non-virtual-dtor)
  endif()

  if(NOT APPLE)
    if((${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 8.0) OR (WITH_ROCM))
      set(COMMON_FLAGS
          ${COMMON_FLAGS} -Wno-ignored-qualifiers # Warning in Paddle-Lite
          -Wno-ignored-attributes # Warning in Eigen gcc 8.3
          -Wno-parentheses # Warning in Eigen gcc 8.3
      )
    endif()
  endif()

  set(GPU_COMMON_FLAGS
      -fPIC
      -fno-omit-frame-pointer
      -Wnon-virtual-dtor
      -Wdelete-non-virtual-dtor
      -Wno-unused-parameter
      -Wno-unused-function
      -Wno-error=literal-suffix
      -Wno-error=unused-local-typedefs
      -Wno-error=unused-function # Warnings in Numpy Header.
      -Wno-error=array-bounds # Warnings in Eigen::array
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(GPU_COMMON_FLAGS -ccbin=${CMAKE_CXX_COMPILER} ${GPU_COMMON_FLAGS})
  endif()

  if(NOT WITH_NV_JETSON
     AND NOT WITH_ARM
     AND NOT WITH_SW
     AND NOT WITH_MIPS
     AND NOT WITH_LOONGARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
  endif()
endif()

if(APPLE)
  if(WITH_ARM)
    set(CMAKE_OSX_ARCHITECTURES
        "arm64"
        CACHE STRING "Build architectures for OSX" FORCE)
  else()
    set(CMAKE_OSX_ARCHITECTURES
        "x86_64"
        CACHE STRING "Build architectures for OSX" FORCE)
  endif()
  # On Mac OS X register class specifier is deprecated and will cause warning error on latest clang 10.0
  set(COMMON_FLAGS
      -Wno-deprecated-register
      -Werror=format
      -Werror=inconsistent-missing-override
      -Werror=braced-scalar-init
      -Werror=uninitialized
      -Werror=tautological-constant-out-of-range-compare
      -Werror=literal-conversion
      -Werror=pragma-pack
      -Werror=c++17-extensions)
endif()

if(WITH_HETERPS AND WITH_PSLIB)
  set(COMMON_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0 ${COMMON_FLAGS})

  set(GPU_COMMON_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0 ${GPU_COMMON_FLAGS})
endif()

if(LINUX)
  set(GPU_COMMON_FLAGS -Wall -Wextra -Werror ${GPU_COMMON_FLAGS})
endif()

foreach(flag ${COMMON_FLAGS})
  safe_set_cflag(CMAKE_C_FLAGS ${flag})
  safe_set_cxxflag(CMAKE_CXX_FLAGS ${flag})
endforeach()

set(SAFE_GPU_COMMON_FLAGS "")
foreach(flag ${GPU_COMMON_FLAGS})
  safe_set_nvflag(${flag})
endforeach()

if(WITH_GPU)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${SAFE_GPU_COMMON_FLAGS}")
endif()

if(WITH_ROCM)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} ${SAFE_GPU_COMMON_FLAGS}")
endif()

# Disable -Werror, otherwise the compile will fail for rocblas_gemm_ex
if(WITH_ROCM)
  string(REPLACE "-Werror" "-Wno-error" HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS})
  string(REPLACE "-Werror" "-Wno-error" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  string(REPLACE "-Werror" "-Wno-error" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
  # disable warnings
  string(APPEND HIP_HIPCC_FLAGS " -Wno-dev ")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-result")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-aliasing")
endif()

if(WITH_PSCORE
   OR WITH_PSLIB
   OR WITH_TENSORRT)
  string(REPLACE "-Wnon-virtual-dtor" "-Wno-non-virtual-dtor" CMAKE_CXX_FLAGS
                 ${CMAKE_CXX_FLAGS})
  string(REPLACE "-Wnon-virtual-dtor" "-Wno-non-virtual-dtor" CMAKE_C_FLAGS
                 ${CMAKE_C_FLAGS})
endif()
