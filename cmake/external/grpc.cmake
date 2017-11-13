# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

include (ExternalProject)

SET(GRPC_SOURCES_DIR ${THIRD_PARTY_PATH}/grpc)
SET(GRPC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/grpc)
SET(GRPC_INCLUDE_DIR "${GRPC_SOURCES_DIR}/src/extern_grpc/include/" CACHE PATH "grpc include directory." FORCE)
SET(GRPC_LIBRARIES "${GRPC_SOURCES_DIR}/src/extern_grpc-build/libgrpc++_unsecure.a" CACHE FILEPATH "GRPC_LIBRARIES" FORCE)
SET(GRPC_CPP_PLUGIN "${GRPC_SOURCES_DIR}/src/extern_grpc-build/grpc_cpp_plugin" CACHE FILEPATH "GRPC_CPP_PLUGIN" FORCE)

ExternalProject_Add(
    extern_grpc
    DEPENDS protobuf zlib
    GIT_REPOSITORY "https://github.com/grpc/grpc.git"
    GIT_TAG "v1.7.x"
    PREFIX          ${GRPC_SOURCES_DIR}
    UPDATE_COMMAND  ""
    BUILD_COMMAND   make grpc_cpp_plugin grpc++_unsecure
    # TODO(typhoonzero): install into third_party/install
    INSTALL_COMMAND ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_INSTALL_PREFIX=${GRPC_INSTALL_DIR}
                    -DCMAKE_INSTALL_LIBDIR=${GRPC_INSTALL_DIR}/lib
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DgRPC_BUILD_TESTS=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DPROTOBUF_INCLUDE_DIRS:STRING=${PROTOBUF_INCLUDE_DIR}
        -DPROTOBUF_LIBRARIES:STRING=${PROTOBUF_LIBRARY}
        -DZLIB_ROOT:STRING=${ZLIB_INSTALL_DIR}
	    -DgRPC_SSL_PROVIDER:STRING=NONE
)

ADD_LIBRARY(grpc STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET grpc PROPERTY IMPORTED_LOCATION ${GRPC_LIBRARIES})
include_directories(${GRPC_INCLUDE_DIR})
ADD_DEPENDENCIES(grpc extern_grpc)

