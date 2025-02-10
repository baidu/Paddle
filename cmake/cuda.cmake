if(NOT WITH_GPU)
  return()
endif()

#(risemeup1) note: CMake 3.18 needs to specify the value of CMAKE_CUDA_ARCHITECTURES，otherwise a large number of warnings may appear in cmake
set(CMAKE_CUDA_ARCHITECTURES OFF)

if(WITH_NV_JETSON)
  add_definitions(-DWITH_NV_JETSON)
  set(paddle_known_gpu_archs "53 62 72")
  set(paddle_known_gpu_archs10 "53 62 72")
  set(paddle_known_gpu_archs11 "53 62 72 87")
  set(paddle_known_gpu_archs12 "53 62 72 87 90")
elseif(NEW_RELEASE_ALL)
  message("Using New Release Strategy - All Arches Package")
  add_definitions(-DNEW_RELEASE_ALL)
  set(paddle_known_gpu_archs "50 52 60 61 70 75 80 86 90")
  set(paddle_known_gpu_archs10 "50 52 60 61 70 75")
  set(paddle_known_gpu_archs11 "50 60 61 70 75 80")
  set(paddle_known_gpu_archs12 "50 60 61 70 75 80 90")
elseif(NEW_RELEASE_PYPI)
  message("Using New Release Strategy - Cubin Package")
  add_definitions(-DNEW_RELEASE_PYPI)
  set(paddle_known_gpu_archs "50 52 60 61 70 75 80 86 90")
  set(paddle_known_gpu_archs10 "")
  set(paddle_known_gpu_archs11 "61 70 75 80")
  set(paddle_known_gpu_archs12 "61 70 75 80 90")
elseif(NEW_RELEASE_JIT)
  message("Using New Release Strategy - JIT Package")
  add_definitions(-DNEW_RELEASE_JIT)
  set(paddle_known_gpu_archs "50 52 60 61 70 75 80 86 90")
  set(paddle_known_gpu_archs10 "50 60 70 75")
  set(paddle_known_gpu_archs11 "50 60 70 75 80")
  set(paddle_known_gpu_archs12 "50 60 70 75 80 90")
else()
  set(paddle_known_gpu_archs "50 52 60 61 70 75 80 90")
  set(paddle_known_gpu_archs10 "50 52 60 61 70 75")
  set(paddle_known_gpu_archs11 "52 60 61 70 75 80")
  set(paddle_known_gpu_archs12 "52 60 61 70 75 80 90")
endif()

######################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   detect_installed_gpus(out_variable)
function(detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(
      WRITE ${cufile}
      ""
      "#include \"stdio.h\"\n"
      "#include \"cuda.h\"\n"
      "#include \"cuda_runtime.h\"\n"
      "int main() {\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device) {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(
      COMMAND "${CUDA_NVCC_EXECUTABLE}" "-ccbin=${CMAKE_C_COMPILER}" "--run"
              "${cufile}"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
      RESULT_VARIABLE nvcc_res
      OUTPUT_VARIABLE nvcc_out
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(nvcc_res EQUAL 0)
      # only keep the last line of nvcc_out
      string(REGEX REPLACE ";" "\\\\;" nvcc_out "${nvcc_out}")
      string(REGEX REPLACE "\n" ";" nvcc_out "${nvcc_out}")
      list(GET nvcc_out -1 nvcc_out)
      string(REPLACE "2.1" "2.1(2.0)" nvcc_out "${nvcc_out}")
      set(CUDA_gpu_detect_output
          ${nvcc_out}
          CACHE INTERNAL
                "Returned GPU architectures from detect_installed_gpus tool"
                FORCE)
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(
      STATUS
        "Automatic GPU detection failed. Building for all known architectures.")
    set(${out_variable}
        ${paddle_known_gpu_archs}
        PARENT_SCOPE)
  else()
    set(${out_variable}
        ${CUDA_gpu_detect_output}
        PARENT_SCOPE)
  endif()
endfunction()

########################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   select_nvcc_arch_flags(out_variable)
function(select_nvcc_arch_flags out_variable out_arch_bin)
  # List of arch names
  set(archs_names
      "Maxwell"
      "Pascal"
      "Volta"
      "Turing"
      "Ampere"
      "Hopper"
      "All"
      "Manual")
  set(archs_name_default "Auto")
  list(APPEND archs_names "Auto")

  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME
      ${archs_name_default}
      CACHE STRING "Select target NVIDIA GPU architecture.")
  set_property(CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${archs_names})
  mark_as_advanced(CUDA_ARCH_NAME)

  # verify CUDA_ARCH_NAME value
  if(NOT ";${archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " archs_names "${archs_names}")
    message(
      FATAL_ERROR "Only ${archs_names} architectures names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN
        ${paddle_known_gpu_archs}
        CACHE
          STRING
          "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported"
    )
    set(CUDA_ARCH_PTX
        ""
        CACHE
          STRING
          "Specify 'virtual' PTX architectures to build PTX intermediate code for"
    )
    mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
  else()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    if(WITH_NV_JETSON)
      set(cuda_arch_bin "53")
    else()
      set(cuda_arch_bin "50")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    if(WITH_NV_JETSON)
      set(cuda_arch_bin "62")
    else()
      set(cuda_arch_bin "60 61")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
    if(WITH_NV_JETSON)
      set(cuda_arch_bin "72")
    else()
      set(cuda_arch_bin "70")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Turing")
    set(cuda_arch_bin "75")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Ampere")
    if(WITH_NV_JETSON)
      set(cuda_arch_bin "87")
    else()
      if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.1) # CUDA 11.0
        set(cuda_arch_bin "80")
      else()
        set(cuda_arch_bin "80 86")
      endif()
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Hopper")
    set(cuda_arch_bin "90")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(cuda_arch_bin ${paddle_known_gpu_archs})
  elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
    message(
      STATUS
        "WARNING: This is just a warning for publishing release.
      You are building GPU version without supporting different architectures.
      So the wheel package may fail on other GPU architectures.
      You can add -DCUDA_ARCH_NAME=All in cmake command
      to get a full wheel package to resolve this warning.
      While, this version will still work on local GPU architecture.")
    detect_installed_gpus(cuda_arch_bin)
  else() # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  if(NEW_RELEASE_JIT)
    set(cuda_arch_ptx "${cuda_arch_ptx}${cuda_arch_bin}")
    set(cuda_arch_bin "")
  endif()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" cuda_arch_ptx "${cuda_arch_ptx}")
  string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+" cuda_arch_ptx "${cuda_arch_ptx}")

  list(REMOVE_DUPLICATES cuda_arch_bin)
  list(REMOVE_DUPLICATES cuda_arch_ptx)

  set(nvcc_flags "")
  set(nvcc_archs_readable "")
  set(nvcc_archs_bin_list "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(arch ${cuda_arch_bin})
    if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      string(APPEND nvcc_flags
             " -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1}")
      string(APPEND nvcc_archs_readable " sm_${CMAKE_MATCH_1}")
      string(APPEND nvcc_archs_bin_list " ${CMAKE_MATCH_1}")
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      string(APPEND nvcc_flags " -gencode arch=compute_${arch},code=sm_${arch}")
      string(APPEND nvcc_archs_readable " sm_${arch}")
      string(APPEND nvcc_archs_bin_list " ${arch}")
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(arch ${cuda_arch_ptx})
    string(APPEND nvcc_flags
           " -gencode arch=compute_${arch},code=compute_${arch}")
    string(APPEND nvcc_archs_readable " compute_${arch}")
  endforeach()

  string(REPLACE ";" " " nvcc_archs_readable "${nvcc_archs_readable}")
  string(REGEX MATCHALL "[0-9()]+" nvcc_archs_bin_list "${nvcc_archs_bin_list}")
  string(JOIN "," nvcc_real_archs ${nvcc_archs_bin_list})

  set(COMPILED_CUDA_ARCHS
      "${nvcc_real_archs}"
      CACHE STRING "Specify compiled cuda archs.")

  set(${out_variable}
      ${nvcc_flags}
      PARENT_SCOPE)
  set(${out_variable}_readable
      ${nvcc_archs_readable}
      PARENT_SCOPE)
  set(${out_variable}_real_archs
      ${nvcc_real_archs}
      PARENT_SCOPE)
  set(${out_arch_bin}
      ${cuda_arch_bin}
      PARENT_SCOPE)
endfunction()

message(STATUS "CUDA detected: " ${CMAKE_CUDA_COMPILER_VERSION})
if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.0) # CUDA 10.x
  set(paddle_known_gpu_archs ${paddle_known_gpu_archs10})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STRICT_ANSI__")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
elseif(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.2) # CUDA 11.0/11.1
  set(paddle_known_gpu_archs ${paddle_known_gpu_archs11})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STRICT_ANSI__")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
elseif(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0) # CUDA 11.2+
  set(paddle_known_gpu_archs "${paddle_known_gpu_archs11} 86")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STRICT_ANSI__")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
elseif(${CMAKE_CUDA_COMPILER_VERSION} LESS 13.0) # CUDA 12.0+
  set(paddle_known_gpu_archs "${paddle_known_gpu_archs12} 86")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STRICT_ANSI__")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
endif()

if(NOT ${CMAKE_CUDA_COMPILER_VERSION} LESS 10.0)
  add_definitions("-DTRT_PLUGIN_FP16_AVAILABLE")
endif()

add_definitions("-DCUDA_VERSION_MAJOR=\"${CUDA_VERSION_MAJOR}\"")
add_definitions("-DCUDA_VERSION_MINOR=\"${CUDA_VERSION_MINOR}\"")
add_definitions("-DCUDA_TOOLKIT_ROOT_DIR=\"${CUDA_TOOLKIT_ROOT_DIR}\"")

# setting nvcc arch flags
select_nvcc_arch_flags(NVCC_FLAGS_EXTRA NVCC_ARCH_BIN)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS_EXTRA}")
message(STATUS "NVCC_FLAGS_EXTRA: ${NVCC_FLAGS_EXTRA}")

# Set C++17 support
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
# Release/Debug flags set by cmake. Such as -O3 -g -DNDEBUG etc.
# So, don't set these flags here.
set(CMAKE_CUDA_STANDARD 17)

# (Note) For windows, if delete /W[1-4], /W1 will be added defaultly and conflict with -w
# So replace /W[1-4] with /W0
if(WIN32)
  string(REGEX REPLACE "/W[1-4]" " /W0 " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()
# in cuda9, suppress cuda warning on eigen
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
# Set :expt-relaxed-constexpr to suppress Eigen warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# Set :expt-extended-lambda to enable HOSTDEVICE annotation on lambdas
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

if(WIN32)
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd4244 /wd4267 /wd4819 \"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /bigobj")
  if(MSVC_STATIC_CRT)
    foreach(flag_var
            CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE
            CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "-MD")
        string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
  endif()
endif()

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)

include(thrust)
