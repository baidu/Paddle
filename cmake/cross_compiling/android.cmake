message(STATUS "--------------------------------here 1")

if(NOT ANDROID)
    return()
endif()

message(STATUS "--------------------------------here 2")
if(NOT DEFINED PLATFORM)
    set(PLATFORM "arm-v7a" CACHE STRING "arm-v7a or arm-v8a.")
endif()





if(${PLATFORM} STREQUAL "arm-v7a")
    set(ANDROID_ABI "armeabi-v7a with NEON")
    # set(PLATFORM_CXX_FLAGS "-march=armv7-a -mfpu=neon -mfloat-abi=softfp -pie -fPIE -w -Wno-error=format-security")
elseif(${PLATFORM} STREQUAL "arm-v8a")
    set(ANDROID_ABI "arm64-v8a")
    # set(PLATFORM_CXX_FLAGS "-march=armv8-a  -pie -fPIE -w -Wno-error=format-security -llog")
else()
    message(FATAL_ERROR "Not supported platform: ${PLATFORM}")
endif()

set(ANDROID_API_LEVEL "22")



#set(CMAKE_CXX_FLAGS ${PLATFORM_CXX_FLAGS})
set(ANDROID_ARM_MODE arm)
set(ANDROID_ARM_NEON ON)
set(ANDROID_PIE TRUE)
set(ANDROID_STL "c++_static")
set(ANDROID_PLATFORM "android-22")

set(CMAKE_TOOLCHAIN_FILE "android.toolchain.cmake")
