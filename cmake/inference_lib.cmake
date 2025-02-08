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

# make package for paddle inference shared and static library
set(PADDLE_INFERENCE_INSTALL_DIR
    "${CMAKE_BINARY_DIR}/paddle_inference_install_dir"
    CACHE STRING "A path setting paddle inference shared and static libraries")

# At present, the size of static lib in Windows is very large,
# so we need to crop the library size.
if(WIN32)
  #todo: remove the option
  option(WITH_STATIC_LIB
         "Compile demo with static/shared library, default use dynamic." OFF)
  if(NOT PYTHON_EXECUTABLE)
    find_package(PythonInterp REQUIRED)
  endif()
endif()

set(COPY_SCRIPT_DIR ${PADDLE_SOURCE_DIR}/cmake)
function(copy TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DSTS)
  cmake_parse_arguments(copy_lib "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  list(LENGTH copy_lib_SRCS copy_lib_SRCS_len)
  list(LENGTH copy_lib_DSTS copy_lib_DSTS_len)
  if(NOT ${copy_lib_SRCS_len} EQUAL ${copy_lib_DSTS_len})
    message(
      FATAL_ERROR
        "${TARGET} source numbers are not equal to destination numbers")
  endif()
  math(EXPR len "${copy_lib_SRCS_len} - 1")
  foreach(index RANGE ${len})
    list(GET copy_lib_SRCS ${index} src)
    list(GET copy_lib_DSTS ${index} dst)
    if(WIN32) #windows
      file(TO_NATIVE_PATH ${src} native_src)
      file(TO_NATIVE_PATH ${dst} native_dst)
      add_custom_command(
        TARGET ${TARGET}
        POST_BUILD
        COMMAND ${PYTHON_EXECUTABLE} ${COPY_SCRIPT_DIR}/copyfile.py
                ${native_src} ${native_dst})
    else() #not windows
      add_custom_command(
        TARGET ${TARGET}
        POST_BUILD
        COMMAND mkdir -p "${dst}"
        COMMAND cp -r "${src}" "${dst}"
        COMMENT "copying ${src} -> ${dst}")
    endif() # not windows
  endforeach()
endfunction()

function(copy_part_of_third_party TARGET DST)
  if(${CBLAS_PROVIDER} STREQUAL MKLML)
    set(dst_dir "${DST}/third_party/install/mklml")
    if(WIN32)
      copy(
        ${TARGET}
        SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_SHARED_LIB}
             ${MKLML_SHARED_IOMP_LIB} ${MKLML_INC_DIR}
        DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir}/lib ${dst_dir}/lib
             ${dst_dir})
    else()
      copy(
        ${TARGET}
        SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_INC_DIR}
        DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir})
      if(WITH_STRIP)
        add_custom_command(
          TARGET ${TARGET}
          POST_BUILD
          COMMAND strip -s ${dst_dir}/lib/libiomp5.so
          COMMAND strip -s ${dst_dir}/lib/libmklml_intel.so
          COMMENT "striping libiomp5.so\nstriping libmklml_intel.so")
      endif()
    endif()
  elseif(${CBLAS_PROVIDER} STREQUAL EXTERN_OPENBLAS)
    set(dst_dir "${DST}/third_party/install/openblas")
    if(WIN32)
      copy(
        ${TARGET}
        SRCS ${CBLAS_INSTALL_DIR}/lib ${OPENBLAS_SHARED_LIB}
             ${CBLAS_INSTALL_DIR}/include
        DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir})
    else()
      copy(
        ${TARGET}
        SRCS ${CBLAS_INSTALL_DIR}/lib ${CBLAS_INSTALL_DIR}/include
        DSTS ${dst_dir} ${dst_dir})
    endif()

    if(WITH_SPARSELT)
      set(dst_dir "${DST}/third_party/install/cusparselt")
      copy(
        ${TARGET}
        SRCS ${CUSPARSELT_INC_DIR} ${CUSPARSELT_LIB_DIR}
        DSTS ${dst_dir} ${dst_dir})
    endif()
  endif()

  if(WITH_ONEDNN)
    set(dst_dir "${DST}/third_party/install/onednn")
    if(WIN32)
      copy(
        ${TARGET}
        SRCS ${ONEDNN_INC_DIR} ${ONEDNN_SHARED_LIB} ${ONEDNN_LIB}
        DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir}/lib)
    else()
      copy(
        ${TARGET}
        SRCS ${ONEDNN_INC_DIR} ${ONEDNN_SHARED_LIB}
        DSTS ${dst_dir} ${dst_dir}/lib)
      if(WITH_STRIP)
        add_custom_command(
          TARGET ${TARGET}
          POST_BUILD
          COMMAND strip -s ${dst_dir}/lib/libdnnl.so.3
          COMMENT "striping libdnnl.so.3")
      endif()
    endif()
  endif()

  if(WITH_ONNXRUNTIME)
    set(dst_dir "${DST}/third_party/install/onnxruntime")
    copy(
      ${TARGET}
      SRCS ${ONNXRUNTIME_INC_DIR} ${ONNXRUNTIME_LIB_DIR}
      DSTS ${dst_dir} ${dst_dir})

    set(dst_dir "${DST}/third_party/install/paddle2onnx")
    copy(
      ${TARGET}
      SRCS ${PADDLE2ONNX_INC_DIR}/paddle2onnx ${PADDLE2ONNX_LIB_DIR}
      DSTS ${dst_dir}/include ${dst_dir})
  endif()

  if(WITH_OPENVINO)
    set(dst_dir "${DST}/third_party/install/openvino")
    copy(
      ${TARGET}
      SRCS ${OPENVINO_INC_DIR} ${OPENVINO_LIB_DIR}
      DSTS ${dst_dir} ${dst_dir})

    set(dst_dir "${DST}/third_party/install/tbb")
    copy(
      ${TARGET}
      SRCS ${TBB_INC_DIR} ${TBB_LIB_DIR}
      DSTS ${dst_dir} ${dst_dir})
  endif()

  set(dst_dir "${DST}/third_party/install/gflags")
  copy(
    ${TARGET}
    SRCS ${GFLAGS_INCLUDE_DIR} ${GFLAGS_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib)

  set(dst_dir "${DST}/third_party/install/glog")
  copy(
    ${TARGET}
    SRCS ${GLOG_INCLUDE_DIR} ${GLOG_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib)

  set(dst_dir "${DST}/third_party/install/utf8proc")
  copy(
    ${TARGET}
    SRCS ${UTF8PROC_INSTALL_DIR}/include ${UTF8PROC_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib)

  set(dst_dir "${DST}/third_party/install/yaml-cpp")
  copy(
    ${TARGET}
    SRCS ${YAML_INSTALL_DIR}/include ${YAML_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib)

  if(WITH_CRYPTO)
    set(dst_dir "${DST}/third_party/install/cryptopp")
    copy(
      ${TARGET}
      SRCS ${CRYPTOPP_INCLUDE_DIR} ${CRYPTOPP_LIBRARIES}
      DSTS ${dst_dir} ${dst_dir}/lib)
  endif()

  set(dst_dir "${DST}/third_party/install/xxhash")
  copy(
    ${TARGET}
    SRCS ${XXHASH_INCLUDE_DIR} ${XXHASH_LIBRARIES}
    DSTS ${dst_dir} ${dst_dir}/lib)

  if(WITH_FLASHATTN)
    set(dst_dir "${DST}/third_party/install/flashattn")
    copy(
      ${TARGET}
      SRCS ${FLASHATTN_INCLUDE_DIR} ${FLASHATTN_LIBRARIES}
      DSTS ${dst_dir} ${dst_dir}/lib)
  endif()

  if(WITH_FLASHATTN_V3)
    set(dst_dir "${DST}/third_party/install/flashattn")
    copy(
      ${TARGET}
      SRCS ${FLASHATTN_INCLUDE_DIR} ${FLASHATTN_V3_LIBRARIES}
      DSTS ${dst_dir} ${dst_dir}/lib)
  endif()

  if(NOT PROTOBUF_FOUND OR WIN32)
    set(dst_dir "${DST}/third_party/install/protobuf")
    copy(
      ${TARGET}
      SRCS ${PROTOBUF_INCLUDE_DIR} ${PROTOBUF_LIBRARY}
      DSTS ${dst_dir} ${dst_dir}/lib)
  endif()

  if(LITE_BINARY_DIR)
    set(dst_dir "${DST}/third_party/install/lite")
    copy(
      ${TARGET}
      SRCS ${LITE_BINARY_DIR}/${LITE_OUTPUT_BIN_DIR}/*
      DSTS ${dst_dir})
  endif()
endfunction()

# inference library for only inference 
# paddle_inference paddle_inference_c
if(WITH_DYNAMIC_INFER_ON)
  set(inference_lib_deps third_party paddle_inference_shared)
else()
  set(inference_lib_deps third_party paddle_inference paddle_inference_c
paddle_inference_shared paddle_inference_c_shared)
endif()

add_custom_target(inference_lib_dist ALL DEPENDS ${inference_lib_deps})

set(dst_dir "${PADDLE_INFERENCE_INSTALL_DIR}/third_party/threadpool")
copy(
  inference_lib_dist
  SRCS ${THREADPOOL_INCLUDE_DIR}/ThreadPool.h
  DSTS ${dst_dir})

# GPU must copy externalErrorMsg.pb
if(WITH_GPU)
  set(dst_dir "${PADDLE_INFERENCE_INSTALL_DIR}/third_party/externalError/data")
  copy(
    inference_lib_dist
    SRCS ${externalError_INCLUDE_DIR}
    DSTS ${dst_dir})
endif()

if(WITH_XPU)
  set(dst_dir "${PADDLE_INFERENCE_INSTALL_DIR}/third_party/install/xpu")
  copy(
    inference_lib_dist
    SRCS ${XPU_INC_DIR} ${XPU_LIB_DIR}
    DSTS ${dst_dir} ${dst_dir})
endif()

# CMakeCache Info
copy(
  inference_lib_dist
  SRCS ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR})

copy_part_of_third_party(inference_lib_dist ${PADDLE_INFERENCE_INSTALL_DIR})

set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")

if(WIN32)
  set(paddle_common_lib ${PADDLE_BINARY_DIR}/paddle/common/common.*)
else()
  set(paddle_common_lib ${PADDLE_BINARY_DIR}/paddle/common/libcommon.*)
endif()
copy(
  inference_lib_dist
  SRCS ${paddle_common_lib}
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib)

if(WIN32)
  if(WITH_STATIC_LIB)
    set(paddle_inference_lib
        $<TARGET_FILE_DIR:paddle_inference>/libpaddle_inference.lib
        $<TARGET_FILE_DIR:paddle_inference>/paddle_inference.*)
  else()
    set(paddle_inference_lib
        $<TARGET_FILE_DIR:paddle_inference_shared>/paddle_inference.dll
        $<TARGET_FILE_DIR:paddle_inference_shared>/paddle_inference.lib)
  endif()
  copy(
    inference_lib_dist
    SRCS ${src_dir}/inference/api/paddle_*.h ${paddle_inference_lib}
    DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include
         ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib
         ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib)
else()
  set(paddle_inference_lib
      ${PADDLE_BINARY_DIR}/paddle/fluid/inference/libpaddle_inference.*)
  copy(
    inference_lib_dist
    SRCS ${src_dir}/inference/api/paddle_*.h ${paddle_inference_lib}
    DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include
         ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib)
  if(WITH_SHARED_PHI)
    set(paddle_phi_libs ${PADDLE_BINARY_DIR}/paddle/phi/libphi*)
    copy(
      inference_lib_dist
      SRCS ${paddle_phi_libs}
      DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib)
    if(WITH_GPU OR WITH_ROCM)
      set(paddle_phi_kernel_gpu_lib
          ${PADDLE_BINARY_DIR}/paddle/phi/libphi_gpu.*)
      copy(
        inference_lib_dist
        SRCS ${paddle_phi_kernel_gpu_lib}
        DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib)
    endif()
  endif()
endif()

copy(
  inference_lib_dist
  SRCS ${CMAKE_BINARY_DIR}/paddle/phi/core/framework/framework.pb.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/internal)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/fluid/framework/io/crypto/cipher.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/crypto/)
include_directories(${CMAKE_BINARY_DIR}/../paddle/fluid/framework/io)

# copy api headers for phi & custom op
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/common/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/common/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/api/ext/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/api/ext/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/api/include/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/api/include/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/api/all.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/api/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/common/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/common/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/core/enforce.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/core/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/utils/string/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/utils/string/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/utils/string/tinyformat/tinyformat.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/utils/string/tinyformat/
)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/core/visit_type.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/core/)

copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/core/distributed/type_defs.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/core/distributed/
)

copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/phi/core/distributed/auto_parallel/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/core/distributed/auto_parallel/
)

copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/fluid/platform/init_phi.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/utils/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/utils/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/extension.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/)

copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/pir/include/core/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/core/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/pir/include/dialect/control_flow/ir/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/dialect/control_flow/ir/
)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/pir/include/dialect/shape/ir/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/dialect/shape/ir/
)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/pir/include/dialect/shape/utils/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/dialect/shape/utils/
)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/dialect/shape/interface/infer_symbolic_shape/
)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/pir/include/pass/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/pass/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/pir/include/pattern_rewrite/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/pattern_rewrite/
)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/drr/include/*.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/drr/)
copy(
  inference_lib_dist
  SRCS ${PADDLE_SOURCE_DIR}/paddle/fluid/pir/utils/general_functions.h
  DSTS ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/utils/)

# the include path of paddle needs to be changed to adapt to inference api path
add_custom_command(
  TARGET inference_lib_dist
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -P
          "${PADDLE_SOURCE_DIR}/cmake/export_paddle_header.cmake"
  COMMENT "Change paddle header include path to adapt to inference api path")

# CAPI inference library for only inference
set(PADDLE_INFERENCE_C_INSTALL_DIR
    "${CMAKE_BINARY_DIR}/paddle_inference_c_install_dir"
    CACHE STRING "A path setting CAPI paddle inference shared")
copy_part_of_third_party(inference_lib_dist ${PADDLE_INFERENCE_C_INSTALL_DIR})

set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
if(WIN32)
  set(paddle_inference_c_lib
      $<TARGET_FILE_DIR:paddle_inference_c>/paddle_inference_c.*)
else()
  set(paddle_inference_c_lib
      ${PADDLE_BINARY_DIR}/paddle/fluid/inference/capi_exp/libpaddle_inference_c.*
  )
endif()

copy(
  inference_lib_dist
  SRCS ${src_dir}/inference/capi_exp/pd_*.h ${paddle_inference_c_lib}
  DSTS ${PADDLE_INFERENCE_C_INSTALL_DIR}/paddle/include
       ${PADDLE_INFERENCE_C_INSTALL_DIR}/paddle/lib)

if(WITH_STRIP AND NOT WIN32)
  add_custom_command(
    TARGET inference_lib_dist
    POST_BUILD
    COMMAND
      strip -s
      ${PADDLE_INFERENCE_C_INSTALL_DIR}/paddle/lib/libpaddle_inference_c.so
    COMMAND strip -s
            ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.so
    COMMENT "striping libpaddle_inference_c.so\nstriping libpaddle_inference.so"
  )
endif()

version(${PADDLE_INFERENCE_INSTALL_DIR}/version.txt)
version(${PADDLE_INFERENCE_C_INSTALL_DIR}/version.txt)
