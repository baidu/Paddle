add_subdirectory(auto_parallel)
add_subdirectory(collective)
if(WITH_PYTHON)
  py_proto_compile(pslib_py_proto SRCS ps.proto)
  py_proto_compile(ps_py_proto SRCS the_one_ps.proto)
  add_custom_target(ps_py_proto_init)
  add_custom_command(
    TARGET ps_py_proto_init
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${PADDLE_BINARY_DIR}/python/paddle/distributed/fleet/proto)
  add_dependencies(ps_py_proto ps_py_proto_init)
  set(PSLIB_PROTO_DSTPATH
      "${PADDLE_SOURCE_DIR}/python/paddle/incubate/distributed/fleet/parameter_server/pslib/"
  )
  if(NOT WIN32)
    add_custom_command(
      TARGET ps_py_proto
      POST_BUILD
      COMMAND mv the_one_ps_pb2.py
              ${PADDLE_BINARY_DIR}/python/paddle/distributed/fleet/proto/)
    add_custom_command(
      TARGET pslib_py_proto
      POST_BUILD
      COMMAND mv ps_pb2.py "${PSLIB_PROTO_DSTPATH}")
  else()
    string(
      REPLACE "/" "\\" fleet_proto_dstpath
              "${PADDLE_BINARY_DIR}/python/paddle/distributed/fleet/proto/")
    add_custom_command(
      TARGET ps_py_proto
      POST_BUILD
      COMMAND copy /Y the_one_ps_pb2.py ${fleet_proto_dstpath}
      COMMENT
        "Copy generated python the_one_ps_pb2 into directory ${fleet_proto_dstpath}."
    )
    string(REPLACE "/" "\\" PSLIB_PROTO_DSTPATH "${PSLIB_PROTO_DSTPATH}")
    add_custom_command(
      TARGET pslib_py_proto
      POST_BUILD
      COMMAND copy /Y ps_pb2.py ${PSLIB_PROTO_DSTPATH})
  endif()
  message(
    STATUS
      "Copy generated python ps_pb2.py into directory ${PSLIB_PROTO_DSTPATH}")
endif()

if(WITH_RPC)
  add_subdirectory(rpc)
endif()

if(NOT WITH_PSCORE)
  return()
endif()

proto_library(ps_framework_proto SRCS the_one_ps.proto)

set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-error=unused-value -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=parentheses -Wno-error=unused-result"
)

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()
add_subdirectory(common)
add_subdirectory(ps)
add_subdirectory(test)
add_subdirectory(index_dataset)
