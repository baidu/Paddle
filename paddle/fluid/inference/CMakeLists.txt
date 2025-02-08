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
#

cc_library(
  paddle_inference_io
  SRCS io.cc
  DEPS paddle_framework ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})

# analysis and tensorrt must be added before creating static library,
# otherwise, there would be undefined reference to them in static library.
add_subdirectory(analysis)
add_subdirectory(utils)
if(TENSORRT_FOUND)
  add_subdirectory(tensorrt)
endif()

if(WITH_OPENVINO)
  add_subdirectory(openvino)
endif()

# fluid_modules exclude API-interface of inference/api and inference/capi_exp
get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
get_property(phi_modules GLOBAL PROPERTY PHI_MODULES)
get_property(ir_targets GLOBAL PROPERTY IR_TARGETS)
get_property(not_infer_modules GLOBAL PROPERTY NOT_INFER_MODULES)
set(utils_modules pretty_log string_helper utf8proc)

add_subdirectory(api)

# Create static inference library if needed
# All static libs in inference/api
set(STATIC_INFERENCE_API
    paddle_inference_api analysis_predictor zero_copy_tensor reset_tensor_array
    analysis_config paddle_pass_builder)

set(OP_LIST
    ""
    CACHE STRING "The list of operators that will be compiled")

set(KERNEL_LIST
    ""
    CACHE STRING "The list of phi kernels that will be compiled")

# shared inference library deps
list(REMOVE_DUPLICATES fluid_modules)
# windows static library（both CPU and GPU）over the limit, so no longer create_static_lib,
# and cc_library is dummy
if(WIN32)
  cc_library(paddle_inference DEPS ${fluid_modules} ${STATIC_INFERENCE_API}
                                   ${utils_modules})
else()
  create_static_lib(paddle_inference ${phi_modules} ${fluid_modules}
                    ${ir_targets} ${STATIC_INFERENCE_API} ${utils_modules})
endif()

if(NOT APPLE)
  # TODO(liuyiqu: Temporarily disable the link flag because it is not support on Mac.
  set(LINK_FLAGS
      "-Wl,--retain-symbols-file ${CMAKE_CURRENT_SOURCE_DIR}/paddle_inference.sym"
  )
  set_target_properties(paddle_inference PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
endif()
# C inference API
add_subdirectory(capi_exp)

set(SHARED_INFERENCE_SRCS
    io.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/api/api.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/api/api_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/io_utils.cc)

# NOTE(Aurelius84): For inference library, some DEPS is useless
# such as non-infer operator related targets et.al.
list(REMOVE_ITEM fluid_modules cinn_op_dialect)
# NOTE(Aurelisu84): Remove pir dialect related target DEPS for inference
# shared library to prune library size.
# list(REMOVE_ITEM fluid_modules ${not_infer_modules})

set(SHARED_INFERENCE_DEPS phi phi_core common ${fluid_modules}
                          analysis_predictor ${utils_modules})
if(WITH_GPU OR WITH_ROCM)
  list(APPEND SHARED_INFERENCE_DEPS phi_gpu)
endif()
if(NOT WIN32)
  list(APPEND SHARED_INFERENCE_DEPS ${ir_targets})
endif()

if(WITH_CRYPTO)
  set(SHARED_INFERENCE_DEPS ${SHARED_INFERENCE_DEPS} framework_io)
endif()

if(WITH_PSCORE)
  set(SHARED_INFERENCE_DEPS ${SHARED_INFERENCE_DEPS} fleet ps_service)
endif()

if(WITH_ONNXRUNTIME)
  set(SHARED_INFERENCE_SRCS
      ${SHARED_INFERENCE_SRCS}
      ${CMAKE_CURRENT_SOURCE_DIR}/api/onnxruntime_predictor.cc)
endif()

# Create shared inference library
cc_library(
  paddle_inference_shared SHARED
  SRCS ${SHARED_INFERENCE_SRCS}
  DEPS ${SHARED_INFERENCE_DEPS})

get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
target_link_libraries(paddle_inference_shared ${os_dependency_modules})
if(WIN32)
  set_property(TARGET paddle_inference_shared
               PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
  target_link_libraries(paddle_inference_shared phi common)
endif()
set(INFERENCE_SHARED_LINKFLAGS "-Wl,-rpath,'$ORIGIN'")
set_target_properties(paddle_inference_shared
                      PROPERTIES LINK_FLAGS "${INFERENCE_SHARED_LINKFLAGS}")
set_target_properties(paddle_inference_shared PROPERTIES OUTPUT_NAME
                                                         paddle_inference)
if(NOT APPLE
   AND NOT WIN32
   AND NOT WITH_TESTING
   AND NOT WITH_INFERENCE_API_TEST)
  # TODO(liuyiqun): Temporarily disable the link flag because it is not support on Mac.
  set(INFERENCE_SHARED_LINKFLAGS
      "${INFERENCE_SHARED_LINKFLAGS},--version-script=${CMAKE_CURRENT_SOURCE_DIR}/paddle_inference.map"
  )
  set_target_properties(paddle_inference_shared
                        PROPERTIES LINK_FLAGS "${INFERENCE_SHARED_LINKFLAGS}")
  # check symbol hidden
  file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/check_symbol.cmake
    "execute_process(COMMAND sh -c \"${CMAKE_CURRENT_SOURCE_DIR}/check_symbol.sh"
    " ${CMAKE_CURRENT_BINARY_DIR}/libpaddle_inference.so\" RESULT_VARIABLE symbol_res)\n"
    "if(NOT \"\${symbol_res}\" STREQUAL \"0\")\n"
    "  message(FATAL_ERROR \"Check symbol failed.\")\n"
    "endif()\n")
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol"
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/check_symbol.cmake"
    DEPENDS paddle_inference_shared)
  add_custom_target(check_symbol ALL
                    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol")
endif()
