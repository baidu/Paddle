# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

set(C_API_SRCS pd_config.cc pd_predictor.cc pd_tensor.cc pd_utils.cc)

if(NOT WITH_DYNAMIC_INFER_ON)
  cc_library(
    paddle_inference_c
    SRCS ${C_API_SRCS}
    DEPS paddle_inference)
endif()

# Create inference capi shared library
cc_library(
  paddle_inference_c_shared SHARED
  SRCS ${C_API_SRCS})
set_target_properties(paddle_inference_c_shared PROPERTIES OUTPUT_NAME
                                                           paddle_inference_c)

if(APPLE)
  target_link_libraries(
    paddle_inference_c_shared
    xxhash
    utf8proc
    cryptopp
    protobuf
    phi
    common
    pir
    cblas)
endif()

if(WIN32)
  target_link_libraries(paddle_inference_c_shared shlwapi.lib)
endif()
