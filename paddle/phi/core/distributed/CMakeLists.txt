add_subdirectory(check)
add_subdirectory(store)
add_subdirectory(auto_parallel)
add_subdirectory(collective)

set(DISTRIBUTED_COMMON_SRCS comm_context_manager.cc)

if(WITH_NCCL OR WITH_RCCL)
  list(APPEND DISTRIBUTED_COMMON_SRCS comm_task_manager.cc)
  list(APPEND DISTRIBUTED_COMMON_SRCS nccl_comm_context.cc nccl_comm_task.cc
       nccl_tools.cc)
endif()

if(WITH_GLOO)
  list(APPEND DISTRIBUTED_COMMON_SRCS gloo_utils.cc gloo_comm_context.cc)
endif()

if(WITH_CUSTOM_DEVICE)
  list(APPEND DISTRIBUTED_COMMON_SRCS xccl_comm_context.cc)
endif()

if(WITH_XPU_BKCL)
  list(APPEND DISTRIBUTED_COMMON_SRCS bkcl_comm_context.cc)
endif()

if(WITH_UCC)
  list(APPEND DISTRIBUTED_COMMON_SRCS ucc_comm_context.cc ucc_tools.cc)
endif()

collect_srcs(core_srcs SRCS ${DISTRIBUTED_COMMON_SRCS})
