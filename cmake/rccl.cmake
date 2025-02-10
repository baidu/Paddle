if(NOT WITH_ROCM)
  return()
endif()

# Now we don't support RCCL on windows
if(WIN32)
  return()
endif()

if(WITH_RCCL)
  if(ROCM_6)
    set(RCCL_INCLUDE_DIR ${ROCM_PATH}/include/rccl)
  else()
    set(RCCL_ROOT
        ${ROCM_PATH}/rccl
        CACHE PATH "RCCL ROOT")
    find_path(
      RCCL_INCLUDE_DIR rccl.h
      PATHS ${RCCL_ROOT} ${RCCL_ROOT}/include ${RCCL_ROOT}/local/include
            $ENV{RCCL_ROOT} $ENV{RCCL_ROOT}/include
            $ENV{RCCL_ROOT}/local/include
      NO_DEFAULT_PATH)
  endif()

  file(READ ${RCCL_INCLUDE_DIR}/rccl.h RCCL_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "define NCCL_VERSION_CODE +([0-9]+)" RCCL_VERSION
               "${RCCL_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define NCCL_VERSION_CODE +([0-9]+)" "\\1" RCCL_VERSION
                       "${RCCL_VERSION}")

  # 2604 for ROCM3.5 and 2708 for ROCM 3.9
  message(STATUS "Current RCCL header is ${RCCL_INCLUDE_DIR}/rccl.h. "
                 "Current RCCL version is v${RCCL_VERSION}. ")
endif()
