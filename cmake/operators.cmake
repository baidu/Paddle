# CMake file `unity_build` is used to handle Unity Build compilation.
include(unity_build)
set(PART_CUDA_KERNEL_FILES)

function(find_register FILENAME PATTERN OUTPUT)
  # find the op_name of REGISTER_OPERATOR(op_name, ...), REGISTER_OP_CPU_KERNEL(op_name, ...) , etc.
  # set op_name to OUTPUT
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs "")
  file(READ ${FILENAME} CONTENT)
  # message ("number of arguments sent to function: ${ARGC}")
  # message ("all function arguments:               ${ARGV}")
  # message("PATTERN ${PATTERN}")
  string(REGEX MATCH "${PATTERN}\\([ \t\r\n]*[a-z0-9_]*," register "${CONTENT}")
  if(NOT register STREQUAL "")
    string(REPLACE "${PATTERN}(" "" register "${register}")
    string(REPLACE "," "" register "${register}")
    # [ \t\r\n]+ is used for blank characters.
    # Here we use '+' instead of '*' since it is a REPLACE operation.
    string(REGEX REPLACE "[ \t\r\n]+" "" register "${register}")
  endif()

  set(${OUTPUT}
      ${register}
      PARENT_SCOPE)
endfunction()

function(find_phi_register FILENAME ADD_PATH PATTERN)
  # set op_name to OUTPUT
  file(READ ${FILENAME} CONTENT)
  string(
    REGEX
      MATCH
      "${PATTERN}\\([ \t\r\n]*[a-z0-9_]*,[[ \\\t\r\n\/]*[a-z0-9_]*]?[ \\\t\r\n]*[a-zA-Z]*,[ \\\t\r\n]*[A-Z_]*"
      register
      "${CONTENT}")
  if(NOT register STREQUAL "")
    string(REPLACE "${PATTERN}(" "" register "${register}")
    string(REPLACE "," ";" register "${register}")
    string(REGEX REPLACE "[ \\\t\r\n]+" "" register "${register}")
    string(REGEX REPLACE "//cuda_only" "" register "${register}")
    list(GET register 0 kernel_name)
    list(GET register 1 kernel_backend)
    list(GET register 2 kernel_layout)

    file(
      APPEND ${ADD_PATH}
      "PD_DECLARE_KERNEL(${kernel_name}, ${kernel_backend}, ${kernel_layout});\n"
    )
  endif()
endfunction()

# Just for those gpu kernels locating at "fluid/operators/", such as 'class_center_sample_op.cu'.
# Add other file modes if need in the future.
function(register_cu_kernel TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(register_cu_kernel "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(cu_srcs)
  set(op_common_deps operator op_registry layer)
  foreach(cu_src ${register_cu_kernel_SRCS})
    if(${cu_src} MATCHES ".*\\.cu$")
      list(APPEND cu_srcs ${cu_src})
    endif()
  endforeach()
  list(LENGTH cu_srcs cu_srcs_len)
  if(${cu_srcs_len} EQUAL 0)
    message(
      FATAL_ERROR
        "The GPU kernel file of ${TARGET} should contains at least one .cu file"
    )
  endif()
  if(WITH_GPU)
    nv_library(
      ${TARGET}
      SRCS ${cu_srcs}
      DEPS ${op_library_DEPS} ${op_common_deps})
  elseif(WITH_ROCM)
    hip_library(
      ${TARGET}
      SRCS ${cu_srcs}
      DEPS ${op_library_DEPS} ${op_common_deps})
  endif()
  set(OP_LIBRARY
      ${TARGET} ${OP_LIBRARY}
      CACHE INTERNAL "op libs")
  foreach(cu_src ${cu_srcs})
    set(op_name "")
    # Add PHI Kernel Registry Message
    find_phi_register(${cu_src} ${pybind_file} "PD_REGISTER_KERNEL")
    find_phi_register(${cu_src} ${pybind_file} "PD_REGISTER_STRUCT_KERNEL")
    find_phi_register(${cu_src} ${pybind_file}
                      "PD_REGISTER_KERNEL_FOR_ALL_DTYPE")
    find_register(${cu_src} "REGISTER_OP_CUDA_KERNEL" op_name)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, CUDA);\n")
    endif()
  endforeach()
endfunction()

# Just for those onednn kernels locating at "fluid/operators/onednn/", such as 'layer_norm_onednn_op.cc'.
# Add other file modes if need in the future.
function(register_onednn_kernel TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(register_onednn_kernel "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(onednn_cc_srcs)
  set(op_common_deps operator op_registry phi layer)
  foreach(onednn_src ${register_onednn_kernel_SRCS})
    if(${onednn_src} MATCHES ".*_onednn_op.cc$")
      list(APPEND onednn_cc_srcs onednn/${onednn_src})
    endif()
  endforeach()
  list(LENGTH onednn_cc_srcs onednn_cc_srcs_len)
  if(${onednn_cc_srcs_len} EQUAL 0)
    message(
      FATAL_ERROR
        "The MKLDNN kernel file of ${TARGET} should contains at least one *.*_onednn_op.cc file"
    )
  endif()
  if(WITH_ONEDNN)
    cc_library(
      ${TARGET}
      SRCS ${onednn_cc_srcs}
      DEPS ${op_library_DEPS} ${op_common_deps})
  endif()
  set(OP_LIBRARY
      ${TARGET} ${OP_LIBRARY}
      CACHE INTERNAL "op libs")
  foreach(onednn_src ${onednn_cc_srcs})
    set(op_name "")
    find_register(${onednn_src} "REGISTER_OP_KERNEL" op_name)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, MKLDNN);\n")
    endif()
  endforeach()
endfunction()

function(op_library TARGET)
  # op_library is a function to create op library. The interface is same as
  # cc_library. But it handle split GPU/CPU code and link some common library
  # for ops.
  set(cc_srcs)
  set(cu_srcs)
  set(hip_srcs)
  set(cu_cc_srcs)
  set(hip_cc_srcs)
  set(xpu_cc_srcs)
  set(xpu_kp_cc_srcs)
  set(cudnn_cu_cc_srcs)
  set(miopen_cu_cc_srcs)
  set(cudnn_cu_srcs)
  set(miopen_cu_srcs)
  set(CUDNN_FILE)
  set(MIOPEN_FILE)
  set(onednn_cc_srcs)
  set(ONEDNN_FILE)
  set(op_common_deps operator op_registry phi layer)

  # Option `UNITY` is used to specify that operator `TARGET` will compiles with Unity Build.
  set(options UNITY)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  set(pybind_flag 0)
  cmake_parse_arguments(op_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  list(LENGTH op_library_SRCS op_library_SRCS_len)
  if(${op_library_SRCS_len} EQUAL 0)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
      list(APPEND cc_srcs ${TARGET}.cc)
    endif()
    if(WITH_GPU)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu.cc)
        list(APPEND cu_cc_srcs ${TARGET}.cu.cc)
      endif()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu)
        list(APPEND cu_srcs ${TARGET}.cu)
      endif()
      # rename in KP: .kps -> .cu
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.kps)
        file(COPY ${TARGET}.kps DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.kps
             ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.cu)
        list(APPEND cu_srcs ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.cu)
      endif()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu)
        set(PART_CUDA_KERNEL_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu
            ${PART_CUDA_KERNEL_FILES}
            PARENT_SCOPE)
        list(APPEND cu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu)
      endif()
      string(REPLACE "_op" "_cudnn_op" CUDNN_FILE "${TARGET}")
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUDNN_FILE}.cu.cc)
        list(APPEND cudnn_cu_cc_srcs ${CUDNN_FILE}.cu.cc)
      endif()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${CUDNN_FILE}.cu)
        list(APPEND cudnn_cu_srcs ${CUDNN_FILE}.cu)
      endif()
    endif()
    if(WITH_ROCM)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu.cc)
        list(APPEND hip_cc_srcs ${TARGET}.cu.cc)
      endif()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu)
        list(APPEND hip_srcs ${TARGET}.cu)
      endif()
      # rename in KP: .kps -> .cu
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.kps)
        file(COPY ${TARGET}.kps DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.kps
             ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.cu)
        list(APPEND hip_srcs ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.cu)
      endif()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu)
        set(PART_CUDA_KERNEL_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu
            ${PART_CUDA_KERNEL_FILES}
            PARENT_SCOPE)
        list(APPEND hip_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.part.cu)
      endif()
      string(REPLACE "_op" "_cudnn_op" MIOPEN_FILE "${TARGET}")
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${MIOPEN_FILE}.cu.cc)
        list(APPEND miopen_cu_cc_srcs ${MIOPEN_FILE}.cu.cc)
      endif()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${MIOPEN_FILE}.cu)
        list(APPEND miopen_cu_srcs ${MIOPEN_FILE}.cu)
      endif()
    endif()
    if(WITH_ONEDNN)
      string(REPLACE "_op" "_onednn_op" ONEDNN_FILE "${TARGET}")
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/onednn/${ONEDNN_FILE}.cc)
        list(APPEND onednn_cc_srcs onednn/${ONEDNN_FILE}.cc)
      endif()
    endif()
    if(WITH_XPU)
      string(REPLACE "_op" "_op_xpu" XPU_FILE "${TARGET}")
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${XPU_FILE}.cc)
        list(APPEND xpu_cc_srcs ${XPU_FILE}.cc)
      endif()
    endif()
    if(WITH_XPU_KP)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.xpu)
        list(APPEND xpu_kp_cc_srcs ${TARGET}.xpu)
      endif()
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.kps)
        list(APPEND xpu_kp_cc_srcs ${TARGET}.kps)
      endif()
    endif()
  else()
    foreach(src ${op_library_SRCS})
      if(WITH_ROCM AND ${src} MATCHES ".*_cudnn_op.cu$")
        list(APPEND miopen_cu_srcs ${src})
      elseif(WITH_ROCM AND ${src} MATCHES ".*\\.cu$")
        list(APPEND hip_srcs ${src})
      elseif(WITH_ROCM AND ${src} MATCHES ".*_cudnn_op.cu.cc$")
        list(APPEND miopen_cu_cc_srcs ${src})
      elseif(WITH_ROCM AND ${src} MATCHES ".*\\.cu.cc$")
        list(APPEND hip_cc_srcs ${src})
      elseif(WITH_GPU AND ${src} MATCHES ".*_cudnn_op.cu$")
        list(APPEND cudnn_cu_srcs ${src})
      elseif(WITH_GPU AND ${src} MATCHES ".*\\.cu$")
        list(APPEND cu_srcs ${src})
      elseif(WITH_GPU AND ${src} MATCHES ".*_cudnn_op.cu.cc$")
        list(APPEND cudnn_cu_cc_srcs ${src})
      elseif(WITH_GPU AND ${src} MATCHES ".*\\.cu.cc$")
        list(APPEND cu_cc_srcs ${src})
      elseif(WITH_ONEDNN AND ${src} MATCHES ".*_onednn_op.cc$")
        list(APPEND onednn_cc_srcs ${src})
      elseif(WITH_XPU AND ${src} MATCHES ".*_op_xpu.cc$")
        list(APPEND xpu_cc_srcs ${src})
      elseif(WITH_XPU_KP AND ${src} MATCHES ".*\\.xpu$")
        list(APPEND xpu_kp_cc_srcs ${src})
      elseif(WITH_XPU_KP AND ${src} MATCHES ".*\\.kps$")
        list(APPEND xpu_kp_cc_srcs ${src})
      elseif(${src} MATCHES ".*\\.cc$")
        list(APPEND cc_srcs ${src})
      elseif((WITH_ROCM OR WITH_GPU) AND ${src} MATCHES ".*\\.kps$")
        string(REPLACE ".kps" ".cu" src_cu ${src})
        file(COPY ${src} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/${src}
             ${CMAKE_CURRENT_BINARY_DIR}/${src_cu})
        if(WITH_ROCM)
          list(APPEND hip_srcs ${CMAKE_CURRENT_BINARY_DIR}/${src_cu})
        else()
          list(APPEND cu_srcs ${CMAKE_CURRENT_BINARY_DIR}/${src_cu})
        endif()
      else()
        message(
          FATAL_ERROR
            "${TARGET} Source file ${src} should only be .cc or .cu or .xpu")
      endif()
    endforeach()
  endif()

  list(LENGTH xpu_cc_srcs xpu_cc_srcs_len)
  list(LENGTH xpu_kp_cc_srcs xpu_kp_cc_srcs_len)
  list(LENGTH cc_srcs cc_srcs_len)
  if(${cc_srcs_len} EQUAL 0)
    message(
      FATAL_ERROR
        "The op library ${TARGET} should contains at least one .cc file")
  endif()
  if(WIN32)
    # remove windows unsupported op, because windows has no nccl, no warpctc such ops.
    foreach(windows_unsupported_op "nccl_op" "gen_nccl_id_op")
      if("${TARGET}" STREQUAL "${windows_unsupported_op}")
        return()
      endif()
    endforeach()
  endif()

  # Unity Build relies on global option `WITH_UNITY_BUILD` and local option `UNITY`.
  if(WITH_UNITY_BUILD AND op_library_UNITY)
    # Generate the unity target name by the directory where source files located.
    string(REPLACE "${PADDLE_SOURCE_DIR}/paddle/fluid/" "" UNITY_TARGET
                   ${CMAKE_CURRENT_SOURCE_DIR})
    string(REPLACE "/" "_" UNITY_TARGET ${UNITY_TARGET})
    set(UNITY_TARGET "paddle_${UNITY_TARGET}_unity")
    if(NOT ${UNITY_TARGET} IN_LIST OP_LIBRARY)
      set(OP_LIBRARY
          ${UNITY_TARGET} ${OP_LIBRARY}
          CACHE INTERNAL "op libs")
    endif()
  else()
    set(OP_LIBRARY
        ${TARGET} ${OP_LIBRARY}
        CACHE INTERNAL "op libs")
  endif()

  list(LENGTH op_library_DEPS op_library_DEPS_len)
  if(${op_library_DEPS_len} GREATER 0)
    set(DEPS_OPS
        ${TARGET} ${DEPS_OPS}
        PARENT_SCOPE)
  endif()
  if(WITH_GPU)
    # Unity Build relies on global option `WITH_UNITY_BUILD` and local option `UNITY`.
    if(WITH_UNITY_BUILD AND op_library_UNITY)
      # Combine the cc and cu source files.
      compose_unity_target_sources(${UNITY_TARGET} cc ${cc_srcs} ${cu_cc_srcs}
                                   ${cudnn_cu_cc_srcs} ${onednn_cc_srcs})
      compose_unity_target_sources(${UNITY_TARGET} cu ${cudnn_cu_srcs}
                                   ${cu_srcs})
      if(TARGET ${UNITY_TARGET})
        # If `UNITY_TARGET` exists, add source files to `UNITY_TARGET`.
        target_sources(${UNITY_TARGET} PRIVATE ${unity_target_cc_sources}
                                               ${unity_target_cu_sources})
      else()
        # If `UNITY_TARGET` does not exist, create `UNITY_TARGET` with source files.
        nv_library(
          ${UNITY_TARGET}
          SRCS ${unity_target_cc_sources} ${unity_target_cu_sources}
          DEPS ${op_library_DEPS} ${op_common_deps})
      endif()
      # Add alias library to handle dependencies.
      add_library(${TARGET} ALIAS ${UNITY_TARGET})
    else()
      nv_library(
        ${TARGET}
        SRCS ${cc_srcs} ${cu_cc_srcs} ${cudnn_cu_cc_srcs} ${cudnn_cu_srcs}
             ${onednn_cc_srcs} ${cu_srcs}
        DEPS ${op_library_DEPS} ${op_common_deps})
    endif()
  elseif(WITH_ROCM)
    list(REMOVE_ITEM miopen_cu_cc_srcs "affine_grid_cudnn_op.cu.cc")
    list(REMOVE_ITEM miopen_cu_cc_srcs "grid_sampler_cudnn_op.cu.cc")
    list(REMOVE_ITEM hip_srcs "cholesky_op.cu")
    list(REMOVE_ITEM hip_srcs "cholesky_solve_op.cu")
    list(REMOVE_ITEM hip_srcs "lu_op.cu")
    list(REMOVE_ITEM hip_srcs "matrix_rank_op.cu")
    list(REMOVE_ITEM hip_srcs "svd_op.cu")
    list(REMOVE_ITEM hip_srcs "eigvalsh_op.cu")
    list(REMOVE_ITEM hip_srcs "qr_op.cu")
    list(REMOVE_ITEM hip_srcs "eigh_op.cu")
    list(REMOVE_ITEM hip_srcs "lstsq_op.cu")
    list(REMOVE_ITEM hip_srcs "multinomial_op.cu")
    list(REMOVE_ITEM hip_srcs "multiclass_nms3_op.cu")
    hip_library(
      ${TARGET}
      SRCS ${cc_srcs} ${hip_cc_srcs} ${miopen_cu_cc_srcs} ${miopen_cu_srcs}
           ${onednn_cc_srcs} ${hip_srcs}
      DEPS ${op_library_DEPS} ${op_common_deps})
  elseif(WITH_XPU_KP AND ${xpu_kp_cc_srcs_len} GREATER 0)
    xpu_library(
      ${TARGET}
      SRCS ${cc_srcs} ${onednn_cc_srcs} ${xpu_cc_srcs} ${xpu_kp_cc_srcs}
      DEPS ${op_library_DEPS} ${op_common_deps})
  else()
    # Unity Build relies on global option `WITH_UNITY_BUILD` and local option `UNITY`.
    if(WITH_UNITY_BUILD AND op_library_UNITY)
      # Combine the cc source files.
      compose_unity_target_sources(${UNITY_TARGET} cc ${cc_srcs}
                                   ${onednn_cc_srcs} ${xpu_cc_srcs})
      if(TARGET ${UNITY_TARGET})
        # If `UNITY_TARGET` exists, add source files to `UNITY_TARGET`.
        target_sources(${UNITY_TARGET} PRIVATE ${unity_target_cc_sources})
      else()
        # If `UNITY_TARGET` does not exist, create `UNITY_TARGET` with source files.
        cc_library(
          ${UNITY_TARGET}
          SRCS ${unity_target_cc_sources}
          DEPS ${op_library_DEPS} ${op_common_deps})
      endif()
      # Add alias library to handle dependencies.
      add_library(${TARGET} ALIAS ${UNITY_TARGET})
    else()
      cc_library(
        ${TARGET}
        SRCS ${cc_srcs} ${onednn_cc_srcs} ${xpu_cc_srcs}
        DEPS ${op_library_DEPS} ${op_common_deps})
    endif()
  endif()

  list(LENGTH cu_srcs cu_srcs_len)
  list(LENGTH hip_srcs hip_srcs_len)
  list(LENGTH cu_cc_srcs cu_cc_srcs_len)
  list(LENGTH hip_cc_srcs hip_cc_srcs_len)
  list(LENGTH onednn_cc_srcs onednn_cc_srcs_len)
  list(LENGTH xpu_cc_srcs xpu_cc_srcs_len)
  list(LENGTH miopen_cu_cc_srcs miopen_cu_cc_srcs_len)

  # Define operators that don't need pybind here.
  foreach(
    manual_pybind_op
    "compare_all_op"
    "compare_op"
    "logical_op"
    "bitwise_op"
    "nccl_op"
    "tensor_array_read_write_op"
    "tensorrt_engine_op")

    if("${TARGET}" STREQUAL "${manual_pybind_op}")
      set(pybind_flag 1)
    endif()
  endforeach()

  # The registration of USE_OP, please refer to paddle/fluid/framework/op_registry.h.
  # Note that it's enough to just adding one operator to pybind in a *_op.cc file.
  # And for detail pybind information, please see generated paddle/pybind/pybind.h.
  set(ORIGINAL_TARGET ${TARGET})
  string(REGEX REPLACE "_op" "" TARGET "${TARGET}")

  foreach(cc_src ${cc_srcs})
    # pybind USE_OP_ITSELF
    set(op_name "")
    # Add PHI Kernel Registry Message
    find_phi_register(${cc_src} ${pybind_file} "PD_REGISTER_KERNEL")
    find_phi_register(${cc_src} ${pybind_file} "PD_REGISTER_STRUCT_KERNEL")
    find_phi_register(${cc_src} ${pybind_file}
                      "PD_REGISTER_KERNEL_FOR_ALL_DTYPE")
    find_register(${cc_src} "REGISTER_OPERATOR" op_name)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_ITSELF(${op_name});\n")
      # hack: for example, the target in conv_transpose_op.cc is conv2d_transpose, used in onednn
      set(TARGET ${op_name})
      set(pybind_flag 1)
    endif()

    # pybind USE_OP_ITSELF
    set(op_name "")
    # Add PHI Kernel Registry Message
    find_register(${cc_src} "REGISTER_ACTIVATION_OP" op_name)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_ITSELF(${op_name});\n")
      # hack: for example, the target in conv_transpose_op.cc is conv2d_transpose, used in onednn
      set(TARGET ${op_name})
      set(pybind_flag 1)
    endif()

    set(op_name "")
    find_register(${cc_src} "REGISTER_OP_WITHOUT_GRADIENT" op_name)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_ITSELF(${op_name});\n")
      # hack: for example, the target in conv_transpose_op.cc is conv2d_transpose, used in onednn
      set(TARGET ${op_name})
      set(pybind_flag 1)
    endif()

    # pybind USE_OP_DEVICE_KERNEL for CPU
    set(op_name "")
    find_register(${cc_src} "REGISTER_OP_CPU_KERNEL" op_name)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, CPU);\n")
      # why change TARGET here?
      # when building paddle with on_infer, the REGISTER_OPERATOR(*_grad) will be removed before compiling (see details in remove_grad_op_and_kernel.py)
      # in elementwise_op.cc, it will find REGISTER_OPERATOR(grad_add) and set TARGET to grad_add
      # and, in the following "onednn" part, it will add USE_OP_DEVICE_KERNEL(grad_add, MKLDNN) to pybind.h
      # however, grad_add has no onednn kernel.
      set(TARGET ${op_name})
      set(pybind_flag 1)
    endif()
  endforeach()

  # pybind USE_OP_DEVICE_KERNEL for CUDA
  list(APPEND cu_srcs ${cu_cc_srcs})
  # message("cu_srcs ${cu_srcs}")
  foreach(cu_src ${cu_srcs})
    set(op_name "")
    # Add PHI Kernel Registry Message
    find_phi_register(${cu_src} ${pybind_file} "PD_REGISTER_KERNEL")
    find_phi_register(${cu_src} ${pybind_file} "PD_REGISTER_STRUCT_KERNEL")
    find_phi_register(${cu_src} ${pybind_file}
                      "PD_REGISTER_KERNEL_FOR_ALL_DTYPE")
    find_register(${cu_src} "REGISTER_OP_CUDA_KERNEL" op_name)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, CUDA);\n")
      set(pybind_flag 1)
    endif()
  endforeach()

  # pybind USE_OP_DEVICE_KERNEL for operators/onednn/*
  list(APPEND onednn_srcs ${onednn_cc_srcs})
  foreach(onednn_src ${onednn_srcs})
    set(op_name "")
    # Add PHI Kernel Registry Message
    find_phi_register(${onednn_src} ${pybind_file} "PD_REGISTER_KERNEL")
    find_phi_register(${onednn_src} ${pybind_file} "PD_REGISTER_STRUCT_KERNEL")
    find_phi_register(${onednn_src} ${pybind_file}
                      "PD_REGISTER_KERNEL_FOR_ALL_DTYPE")
    find_register(${onednn_src} "REGISTER_OP_CUDA_KERNEL" op_name)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, CUDA);\n")
      set(pybind_flag 1)
    endif()
  endforeach()
  # pybind USE_OP_DEVICE_KERNEL for ROCm
  list(APPEND hip_srcs ${hip_cc_srcs})
  # message("hip_srcs ${hip_srcs}")
  foreach(hip_src ${hip_srcs})
    set(op_name "")
    find_register(${hip_src} "REGISTER_OP_CUDA_KERNEL" op_name)
    find_phi_register(${hip_src} ${pybind_file} "PD_REGISTER_KERNEL")
    find_phi_register(${hip_src} ${pybind_file} "PD_REGISTER_STRUCT_KERNEL")
    find_phi_register(${hip_src} ${pybind_file}
                      "PD_REGISTER_KERNEL_FOR_ALL_DTYPE")
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, CUDA);\n")
      set(pybind_flag 1)
    endif()
  endforeach()

  # pybind USE_OP_DEVICE_KERNEL for CUDNN/MIOPEN
  list(APPEND cudnn_cu_srcs ${cudnn_cu_cc_srcs})
  list(APPEND cudnn_cu_srcs ${miopen_cu_cc_srcs})
  list(APPEND cudnn_cu_srcs ${miopen_cu_srcs})
  list(LENGTH cudnn_cu_srcs cudnn_cu_srcs_len)
  #message("cudnn_cu_srcs ${cudnn_cu_srcs}")
  if(${cudnn_cu_srcs_len} GREATER 0 AND ${ORIGINAL_TARGET} STREQUAL
                                        "activation_op")
    file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(relu, CUDNN);\n")
  else()
    foreach(cudnn_src ${cudnn_cu_srcs})
      set(op_name "")
      find_register(${cudnn_src} "REGISTER_OP_KERNEL" op_name)
      if(NOT ${op_name} EQUAL "")
        file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, CUDNN);\n")
        set(pybind_flag 1)
      endif()
    endforeach()
  endif()

  if(WITH_XPU AND ${xpu_cc_srcs_len} GREATER 0)
    if(${ORIGINAL_TARGET} STREQUAL "activation_op")
      file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(relu, XPU);\n")
    else()
      foreach(xpu_src ${xpu_cc_srcs})
        set(op_name "")
        find_register(${xpu_src} "REGISTER_OP_XPU_KERNEL" op_name)
        find_phi_register(${xpu_src} ${pybind_file} "PD_REGISTER_STRUCT_KERNEL")
        if(NOT ${op_name} EQUAL "")
          file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, XPU);\n")
          set(pybind_flag 1)
        else()
          find_register(${xpu_src} "REGISTER_OP_XPU_KERNEL_FUNCTOR" op_name)
          if(NOT ${op_name} EQUAL "")
            file(APPEND ${pybind_file}
                 "USE_OP_DEVICE_KERNEL(${op_name}, XPU);\n")
            set(pybind_flag 1)
          endif()
        endif()
      endforeach()
    endif()
  endif()

  # pybind USE_OP_DEVICE_KERNEL for XPU KP
  if(WITH_XPU_KP AND ${xpu_kp_cc_srcs_len} GREATER 0)
    foreach(xpu_kp_src ${xpu_kp_cc_srcs})
      set(op_name "")
      find_register(${xpu_kp_src} "REGISTER_OP_KERNEL" op_name)
      find_phi_register(${xpu_kp_src} ${pybind_file}
                        "PD_REGISTER_STRUCT_KERNEL")
      if(NOT ${op_name} EQUAL "")
        file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(${op_name}, KP);\n")
        message(STATUS "Building KP Target: ${op_name}")
        set(pybind_flag 1)
      endif()
    endforeach()
  endif()

  # pybind USE_OP_DEVICE_KERNEL for MKLDNN
  if(WITH_ONEDNN AND ${onednn_cc_srcs_len} GREATER 0)
    # Append first implemented MKLDNN activation operator
    if(${ONEDNN_FILE} STREQUAL "activation_onednn_op")
      file(APPEND ${pybind_file} "USE_OP_DEVICE_KERNEL(softplus, MKLDNN);\n")
    else()
      foreach(onednn_src ${onednn_cc_srcs})
        set(op_name "")
        find_register(${onednn_src} "REGISTER_OP_KERNEL" op_name)
        if(NOT ${op_name} EQUAL "")
          file(APPEND ${pybind_file}
               "USE_OP_DEVICE_KERNEL(${op_name}, MKLDNN);\n")
          set(pybind_flag 1)
        endif()
      endforeach()
    endif()
  endif()

  # pybind USE_NO_KERNEL_OP
  # HACK: if REGISTER_OP_CPU_KERNEL presents the operator must have kernel
  string(REGEX MATCH "REGISTER_OP_CPU_KERNEL" regex_result "${TARGET_CONTENT}")
  string(REPLACE "_op" "" TARGET "${TARGET}")
  if(${pybind_flag} EQUAL 0 AND regex_result STREQUAL "")
    file(APPEND ${pybind_file} "USE_NO_KERNEL_OP(${TARGET});\n")
    set(pybind_flag 1)
  endif()

  # pybind USE_OP
  if(${pybind_flag} EQUAL 0)
    # NOTE(*): activation use macro to register the kernels, set use_op manually.
    if(${TARGET} STREQUAL "activation")
      file(APPEND ${pybind_file} "USE_OP_ITSELF(relu);\n")
    elseif(${TARGET} STREQUAL "fake_dequantize")
      file(APPEND ${pybind_file} "USE_OP(fake_dequantize_max_abs);\n")
    elseif(${TARGET} STREQUAL "fake_quantize")
      file(APPEND ${pybind_file} "USE_OP(fake_quantize_abs_max);\n")
    elseif(${TARGET} STREQUAL "tensorrt_engine_op")
      message(
        STATUS
          "Pybind skips [tensorrt_engine_op], for this OP is only used in inference"
      )
    else()
      file(APPEND ${pybind_file} "USE_OP(${TARGET});\n")
    endif()
  endif()
endfunction()

function(register_operators)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs EXCLUDES DEPS)
  cmake_parse_arguments(register_operators "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  file(
    GLOB OPS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*_op.cc")
  string(REPLACE "_onednn" "" OPS "${OPS}")
  string(REPLACE "_xpu" "" OPS "${OPS}")
  string(REPLACE ".cc" "" OPS "${OPS}")
  list(REMOVE_DUPLICATES OPS)
  list(LENGTH register_operators_DEPS register_operators_DEPS_len)

  foreach(src ${OPS})
    list(FIND register_operators_EXCLUDES ${src} _index)
    if(${_index} EQUAL -1)
      if(${register_operators_DEPS_len} GREATER 0)
        op_library(${src} UNITY DEPS ${register_operators_DEPS})
      else()
        op_library(${src} UNITY)
      endif()
    endif()
  endforeach()

  # Complete the processing of `UNITY_TARGET`.
  if(WITH_UNITY_BUILD)
    finish_unity_target(cc)
    if(WITH_GPU)
      finish_unity_target(cu)
    endif()
  endif()
endfunction()

function(prune_pybind_h)
  set(op_list ${OP_LIST})

  list(APPEND op_list "load_combine")
  list(APPEND op_list "tensorrt_engine")

  # TODO(ming1753): conditional_block_infer is temporarily reserved here to avoid link errors in functions of standalone_executor
  list(APPEND op_list "conditional_block_infer")

  # add fused_op in op_list
  list(APPEND op_list "fc")
  list(APPEND op_list "fused_conv2d_add_act")
  list(APPEND op_list "fusion_seqconv_eltadd_relu")
  list(APPEND op_list "fusion_seqpool_cvm_concat")
  list(APPEND op_list "fusion_gru")
  list(APPEND op_list "fusion_seqexpand_concat_fc")
  list(APPEND op_list "fusion_repeated_fc_relu")
  list(APPEND op_list "fusion_squared_mat_sub")

  # add plugin_op in op_list
  list(APPEND op_list "anchor_generator")

  file(STRINGS ${pybind_file} op_registry_list)

  file(WRITE ${pybind_file_prune} "")
  file(
    APPEND ${pybind_file_prune}
    "// Generated by the paddle/fluid/operators/CMakeLists.txt.  DO NOT EDIT!\n"
  )

  # add USE_OP_ITSELF for all op in op_list
  foreach(op_name IN LISTS op_list)
    file(APPEND ${pybind_file_prune} "USE_OP_ITSELF(${op_name});\n")
  endforeach()

  foreach(op_registry IN LISTS op_registry_list)
    if(NOT ${op_registry} EQUAL "")
      foreach(op_name IN LISTS op_list)
        string(FIND ${op_registry} "(${op_name})" index1)
        string(FIND ${op_registry} "(${op_name}," index2)
        string(FIND ${op_registry} "USE_OP_ITSELF" index3)
        if(((NOT ${index1} EQUAL "-1") OR (NOT ${index2} EQUAL "-1"))
           AND (${index3} EQUAL "-1"))
          file(APPEND ${pybind_file_prune} "${op_registry}\n")
        endif()
      endforeach()
    endif()
  endforeach()

  file(WRITE ${pybind_file} "")
  file(STRINGS ${pybind_file_prune} op_registry_list_tmp)
  foreach(op_name IN LISTS op_registry_list_tmp)
    if(NOT ${op_name} EQUAL "")
      file(APPEND ${pybind_file} "${op_name}\n")
    endif()
  endforeach()
endfunction()

function(append_op_util_declare TARGET)
  file(READ ${TARGET} target_content)
  string(REGEX MATCH "(PD_REGISTER_ARG_MAPPING_FN)\\([ \t\r\n]*[a-z0-9_]*"
               util_registrar "${target_content}")
  if(NOT ${util_registrar} EQUAL "")
    string(REPLACE "PD_REGISTER_ARG_MAPPING_FN" "PD_DECLARE_ARG_MAPPING_FN"
                   util_declare "${util_registrar}")
    string(APPEND util_declare ");\n")
    file(APPEND ${op_utils_header} "${util_declare}")
  endif()
endfunction()

function(append_op_kernel_map_declare TARGET)
  file(READ ${TARGET} target_content)
  string(
    REGEX
      MATCH
      "(PD_REGISTER_BASE_KERNEL_NAME)\\([ \t\r\n]*[a-z0-9_]*,[ \\\t\r\n]*[a-z0-9_]*"
      kernel_mapping_registrar
      "${target_content}")
  if(NOT ${kernel_mapping_registrar} EQUAL "")
    string(REPLACE "PD_REGISTER_BASE_KERNEL_NAME" "PD_DECLARE_BASE_KERNEL_NAME"
                   kernel_mapping_declare "${kernel_mapping_registrar}")
    string(APPEND kernel_mapping_declare ");\n")
    file(APPEND ${op_utils_header} "${kernel_mapping_declare}")
  endif()
endfunction()

function(register_op_utils TARGET_NAME)
  set(utils_srcs)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs EXCLUDES DEPS)
  cmake_parse_arguments(register_op_utils "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  file(GLOB SIGNATURES
       "${PADDLE_SOURCE_DIR}/paddle/fluid/operators/ops_signature/*_sig.cc")
  foreach(target ${SIGNATURES})
    append_op_util_declare(${target})
    append_op_kernel_map_declare(${target})
    list(APPEND utils_srcs ${target})
  endforeach()

  cc_library(
    ${TARGET_NAME}
    SRCS ${utils_srcs}
    DEPS ${register_op_utils_DEPS})
endfunction()
