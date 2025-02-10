if(NOT WITH_DISTRIBUTE OR NOT WITH_UCC)
  return()
endif()

find_package(UCC REQUIRED)
find_package(UCX REQUIRED)

message(STATUS "UCC include path: " ${UCC_INCLUDE_DIRS})
message(STATUS "UCC libraries: " ${UCC_LIBRARIES})

include_directories(SYSTEM ${UCC_INCLUDE_DIRS})

# generate a static dummy target to track ucclib dependencies
# for cc_library(xxx SRCS xxx.c DEPS ucclib)
generate_dummy_static_lib(LIB_NAME "ucclib" GENERATOR "ucc.cmake")
target_link_libraries(ucclib ucx::ucs ucx::ucp ucc::ucc)
