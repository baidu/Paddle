cc_library(
  process_group
  SRCS process_group.cc
  DEPS phi common xxhash)

cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api process_group phi common string_helper)

if(WITH_DISTRIBUTE)
  cc_library(
    process_group_gloo
    SRCS process_group_gloo.cc gloo_send_recv.cc
    DEPS phi common eager_api gloo_wrapper)
endif()

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    process_group_nccl
    SRCS process_group_nccl.cc common.cc
    DEPS process_group phi ${DEVICE_EVENT_LIBS})

  cc_library(
    async_load
    SRCS async_load.cc
    DEPS phi ${DEVICE_EVENT_LIBS})

endif()

if(WITH_XPU_BKCL)
  cc_library(
    process_group_bkcl
    SRCS process_group_bkcl.cc bkcl_tools.cc common.cc
    DEPS process_group phi)
endif()

if(WITH_MPI)
  cc_library(
    process_group_mpi
    SRCS process_group_mpi.cc mpi_tools.cc common.cc
    DEPS phi)
endif()

if(WITH_UCC)
  cc_library(
    process_group_ucc
    SRCS process_group_ucc.cc common.cc #ucc_tools.cc
    DEPS process_group phi)
endif()

if(WITH_CUSTOM_DEVICE)
  cc_library(
    process_group_custom
    SRCS process_group_custom.cc custom_ccl_tools.cc common.cc
    DEPS process_group phi)
endif()

set(COMM_UTILS_DEPS process_group)
if(WITH_NCCL OR WITH_RCCL)
  set(COMM_UTILS_DEPS ${PROCESS_GROUP_UTILS_DEPS} process_group_nccl)
endif()
if(WITH_CUSTOM_DEVICE)
  set(COMM_UTILS_DEPS ${PROCESS_GROUP_UTILS_DEPS} process_group_custom)
endif()
cc_library(
  processgroup_comm_utils
  SRCS processgroup_comm_utils.cc
  DEPS ${COMM_UTILS_DEPS})
