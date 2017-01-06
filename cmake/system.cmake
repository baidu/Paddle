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

IF(WIN32)
    SET(HOST_SYSTEM "win32")
ELSE(WIN32)
    IF(APPLE)
        EXEC_PROGRAM (sw_vers ARGS -productVersion OUTPUT_VARIABLE MACOSX_VERSION)
        STRING(REGEX MATCH "[0-9]+.[0-9]+" VERSION "${MACOSX_VERSION}")
        SET(MACOS_VERSION ${VERSION})
        SET(HOST_SYSTEM "macosx")
    ELSE(APPLE)
        IF(EXISTS "/etc/issue")
            FILE(READ "/etc/issue" LINUX_ISSUE)
            IF(LINUX_ISSUE MATCHES "CentOS")
                SET(HOST_SYSTEM "centos")
            ELSEIF(LINUX_ISSUE MATCHES "Debian")
                SET(HOST_SYSTEM "debian")
            ELSEIF(LINUX_ISSUE MATCHES "Ubuntu")
                SET(HOST_SYSTEM "ubuntu")
            ENDIF()
        ENDIF(EXISTS "/etc/issue")
    ENDIF(APPLE)
ENDIF(WIN32)

# query number of logical cores
CMAKE_HOST_SYSTEM_INFORMATION(RESULT CPU_CORES QUERY NUMBER_OF_LOGICAL_CORES)

MARK_AS_ADVANCED(HOST_SYSTEM CPU_CORES)

MESSAGE(STATUS "Found Paddle host system: ${HOST_SYSTEM}")
MESSAGE(STATUS "Found Paddle host system's CPU: ${CPU_CORES} cores")

# external dependencies log output
SET(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD    0     # Wrap download in script to log output
    LOG_UPDATE      1     # Wrap update in script to log output
    LOG_CONFIGURE   1     # Wrap configure in script to log output
    LOG_BUILD       1     # Wrap build in script to log output
    LOG_TEST        1     # Wrap test in script to log output
    LOG_INSTALL     1     # Wrap install in script to log output
)
